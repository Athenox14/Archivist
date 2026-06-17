use anyhow::Result;
use archivist::config::Config;
use archivist::{index, search};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

/// Lit jusqu'à `buf.len()` octets (ou EOF). Renvoie le nb lu.
fn read_block(f: &mut std::fs::File, buf: &mut [u8]) -> std::io::Result<usize> {
    use std::io::Read;
    let mut filled = 0;
    while filled < buf.len() {
        match f.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

/// (A) Hash du préfixe ET du suffixe (16 Ko chacun) — pré-tri bon marché.
fn prefix_suffix(path: &Path, size: u64) -> Option<(String, String)> {
    use std::io::{Seek, SeekFrom};
    const K: u64 = 16 * 1024;
    let mut f = std::fs::File::open(path).ok()?;
    let plen = K.min(size).max(1) as usize;
    let mut pbuf = vec![0u8; plen];
    let pn = read_block(&mut f, &mut pbuf).ok()?;
    let p = blake3::hash(&pbuf[..pn]).to_hex().to_string();
    let s = if size > K {
        f.seek(SeekFrom::Start(size - K)).ok()?;
        let mut sbuf = vec![0u8; K as usize];
        let sn = read_block(&mut f, &mut sbuf).ok()?;
        blake3::hash(&sbuf[..sn]).to_hex().to_string()
    } else {
        p.clone()
    };
    Some((p, s))
}

/// Contexte d'écriture en flux (thread-safe) + suivi d'avancement/ETA.
struct Emitter<'a> {
    w: std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
    done: std::sync::atomic::AtomicUsize,
    files: &'a [(PathBuf, u64, String)],
}

impl<'a> Emitter<'a> {
    fn emit(&self, idx: usize, hash: &str) {
        use std::io::Write;
        use std::sync::atomic::Ordering;
        let (_, size, rel) = &self.files[idx];
        let mut w = self.w.lock().unwrap();
        let _ = writeln!(w, "{hash}\t{size}\t{rel}");
        // flush périodique → écriture temps réel, survit à un crash
        if self.done.fetch_add(1, Ordering::Relaxed) % 2000 == 0 {
            let _ = w.flush();
        }
    }
}

/// Suivi d'avancement + ETA d'UNE phase (chaque phase a son propre compteur).
struct Phase {
    n: std::sync::atomic::AtomicUsize,
    total: usize,
    start: std::time::Instant,
    label: &'static str,
}

impl Phase {
    fn new(label: &'static str, total: usize) -> Self {
        Phase {
            n: std::sync::atomic::AtomicUsize::new(0),
            total,
            start: std::time::Instant::now(),
            label,
        }
    }
    fn tick(&self) {
        use std::sync::atomic::Ordering;
        let d = self.n.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 500 == 0 || d == self.total {
            let el = self.start.elapsed().as_secs_f64();
            let eta = if d > 0 {
                el / d as f64 * (self.total - d) as f64
            } else {
                0.0
            };
            eprintln!("  [{}] {d}/{} · ETA ~{:.0}s", self.label, self.total, eta);
        }
    }
}

/// (B+C) Hash progressif par blocs avec élimination précoce : on ne lit chaque
/// fichier que jusqu'à ce qu'il devienne UNIQUE dans son groupe ; les vrais
/// doublons (identiques) sont lus en entier et reçoivent leur hash complet.
/// Émet chaque résultat en flux via `em` ; `ph` suit l'avancement.
fn progressive_group(em: &Emitter, ph: &Phase, group: &[usize]) {
    use std::collections::{HashMap, HashSet};
    const BLOCK: usize = 1 << 20; // 1 Mo

    struct A {
        idx: usize,
        f: std::fs::File,
        h: blake3::Hasher,
    }
    let files = em.files;
    let put = |idx: usize, hash: &str| {
        em.emit(idx, hash);
        ph.tick();
    };
    let mut active: Vec<A> = Vec::new();
    for &i in group {
        match std::fs::File::open(&files[i].0) {
            Ok(f) => active.push(A {
                idx: i,
                f,
                h: blake3::Hasher::new(),
            }),
            Err(_) => put(i, &format!("u:{i}")),
        }
    }
    let size = files[group[0]].1;
    let mut pos: u64 = 0;
    let mut buf = vec![0u8; BLOCK];

    loop {
        if active.len() <= 1 {
            if let Some(a) = active.first() {
                put(a.idx, &format!("u:{}", a.idx));
            }
            break;
        }
        for a in active.iter_mut() {
            let n = read_block(&mut a.f, &mut buf).unwrap_or(0);
            a.h.update(&buf[..n]);
        }
        pos += BLOCK as u64;
        let eof = pos >= size;

        let mut buckets: HashMap<String, Vec<usize>> = HashMap::new();
        for (j, a) in active.iter().enumerate() {
            buckets
                .entry(a.h.clone().finalize().to_hex().to_string())
                .or_default()
                .push(j);
        }
        let mut keep: HashSet<usize> = HashSet::new();
        for (digest, js) in buckets {
            if eof {
                if js.len() >= 2 {
                    for &j in &js {
                        put(active[j].idx, &digest);
                    }
                } else {
                    put(active[js[0]].idx, &format!("u:{}", active[js[0]].idx));
                }
            } else if js.len() == 1 {
                put(active[js[0]].idx, &format!("u:{}", active[js[0]].idx));
            } else {
                keep.extend(js);
            }
        }
        if eof {
            break;
        }
        active = active
            .into_iter()
            .enumerate()
            .filter(|(j, _)| keep.contains(j))
            .map(|(_, a)| a)
            .collect();
    }
}

#[derive(Parser)]
#[command(
    name = "archivist",
    about = "Backup/archivage standalone (zstd) + recherche sémantique",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Indexe : compresse la source vers l'archive + construit la DB de recherche.
    Index {
        /// Dossier source (lecture).
        #[arg(long)]
        source: PathBuf,
        /// Dossier archive (écriture).
        #[arg(long)]
        archive: PathBuf,
        /// Niveau zstd 1..=22 (défaut 9).
        #[arg(long, default_value_t = 9)]
        level: i32,
        /// Chemin DB (défaut <archive>/.archivist.db).
        #[arg(long)]
        db: Option<PathBuf>,
        /// Dossier des modèles ONNX + tokenizers.
        #[arg(long, default_value = "models")]
        models: PathBuf,
        /// Stocke les fichiers TELS QUELS (copie, sans compression) + hash complet.
        #[arg(long)]
        no_compress: bool,
    },
    /// Interne : extrait le texte d'UN pdf et l'imprime (isolé en sous-processus).
    #[command(hide = true)]
    ExtractPdf {
        #[arg(long)]
        path: PathBuf,
    },
    /// Vérifie l'intégrité de l'archive (re-hash vs hash enregistré → corruption).
    Verify {
        #[arg(long)]
        archive: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
    },
    /// Hashe (blake3) tous les fichiers d'une source → fichier TSV local.
    Hashdump {
        /// Dossier à hasher.
        #[arg(long)]
        source: PathBuf,
        /// Fichier TSV de sortie. Si omis : `hashes_<nom_dossier>.tsv`.
        /// Recréé à neuf à chaque exécution.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Reconstruit la base de recherche depuis l'archive (sans la source).
    Populate {
        /// Racine de l'archive (contient les .zst).
        #[arg(long)]
        archive: PathBuf,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long, default_value = "models")]
        models: PathBuf,
    },
    /// Recherche sémantique : résultats Images + Documents.
    Search {
        /// Dossier archive (pour localiser la DB).
        #[arg(long)]
        archive: PathBuf,
        /// Requête en langage naturel.
        #[arg(long)]
        query: String,
        /// Nombre de résultats par catégorie.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        #[arg(long)]
        db: Option<PathBuf>,
        #[arg(long, default_value = "models")]
        models: PathBuf,
    },
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Index {
            source,
            archive,
            level,
            db,
            models,
            no_compress,
        } => {
            std::fs::create_dir_all(&archive)?;
            let db_path = Config::db_path_for(&archive, db);
            // archive partagée : range sous <nom_source>/ et déduplique cross-source.
            let label = source
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string());
            let cfg = Config {
                source,
                archive,
                zstd_level: level,
                db_path,
                models_dir: models,
                label,
                store: no_compress,
            };
            let st = index::run(&cfg)?;
            println!(
                "OK : {} fichiers | {} canoniques compressés | {} hardlinks | {} images | {} chunks texte",
                st.files_total, st.canonicals, st.hardlinks, st.images_embedded, st.chunks_embedded
            );
        }
        Cmd::Hashdump { source, out } => {
            use rayon::prelude::*;
            use std::collections::HashMap;
            use std::io::Write;

            // nom de sortie dédié au dossier si non fourni ; recréé à neuf.
            let outpath = out.unwrap_or_else(|| {
                let leaf = source
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("source")
                    .replace([' ', '/', '\\'], "_");
                PathBuf::from(format!("hashes_{leaf}.tsv"))
            });
            let _ = std::fs::remove_file(&outpath);

            // 1. walk + tailles (stat seulement)
            let files: Vec<(PathBuf, u64, String)> = walkdir::WalkDir::new(&source)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
                .filter_map(|e| {
                    let p = e.path().to_path_buf();
                    let size = e.metadata().ok()?.len();
                    let rel = p
                        .strip_prefix(&source)
                        .ok()?
                        .to_string_lossy()
                        .replace('\\', "/");
                    Some((p, size, rel))
                })
                .collect();
            let total = files.len();

            let em = Emitter {
                w: std::sync::Mutex::new(std::io::BufWriter::new(std::fs::File::create(&outpath)?)),
                done: std::sync::atomic::AtomicUsize::new(0),
                files: &files,
            };

            // 2. groupe par TAILLE (0 lecture). Tailles uniques → sentinelle (émise direct).
            let mut by_size: HashMap<u64, Vec<usize>> = HashMap::new();
            for (i, (_, size, _)) in files.iter().enumerate() {
                by_size.entry(*size).or_default().push(i);
            }
            let mut cand: Vec<usize> = Vec::new();
            for idxs in by_size.values() {
                if idxs.len() == 1 {
                    em.emit(idxs[0], &format!("u:{}", idxs[0]));
                } else {
                    cand.extend_from_slice(idxs);
                }
            }
            cand.sort_by(|&a, &b| files[a].2.cmp(&files[b].2)); // (D) localité disque
            eprintln!(
                "{total} fichiers · {} en collision de taille → préfixe+suffixe…",
                cand.len()
            );

            // 3. (A) préfixe + suffixe → sous-groupes ; singletons → sentinelle
            let pf = Phase::new("préfixe+suffixe", cand.len());
            let ps: Vec<(usize, Option<(String, String)>)> = cand
                .par_iter()
                .map(|&i| {
                    let r = prefix_suffix(&files[i].0, files[i].1);
                    pf.tick();
                    (i, r)
                })
                .collect();
            let mut groups: HashMap<(u64, String, String), Vec<usize>> = HashMap::new();
            for (i, k) in ps {
                match k {
                    Some((p, s)) => groups.entry((files[i].1, p, s)).or_default().push(i),
                    None => em.emit(i, &format!("u:{i}")),
                }
            }
            let mut survivors: Vec<Vec<usize>> = Vec::new();
            for (_, g) in groups {
                if g.len() == 1 {
                    em.emit(g[0], &format!("u:{}", g[0]));
                } else {
                    survivors.push(g);
                }
            }
            let n_surv: usize = survivors.iter().map(|g| g.len()).sum();
            eprintln!(
                "{n_surv} survivants ({} groupes) → hash progressif…",
                survivors.len()
            );

            // 4. (B+C) hash progressif par groupe, en parallèle ; émission en flux
            let pg = Phase::new("progressif", n_surv);
            survivors
                .par_iter()
                .for_each(|g| progressive_group(&em, &pg, g));

            em.w.lock().unwrap().flush()?;
            println!("OK : {total} fichiers → {}", outpath.display());
        }
        Cmd::ExtractPdf { path } => {
            use std::io::Write;
            let text = archivist::extract::extract_pdf_raw(&path);
            let _ = std::io::stdout().write_all(text.as_bytes());
        }
        Cmd::Verify { archive, db } => {
            let db_path = Config::db_path_for(&archive, db);
            let cfg = Config {
                source: PathBuf::new(),
                archive,
                zstd_level: 9,
                db_path,
                models_dir: PathBuf::new(),
                label: None,
                store: false,
            };
            let rep = index::verify_integrity(&cfg, &|d, t| {
                if d == t {
                    eprintln!("  {d}/{t}");
                }
            })?;
            println!(
                "Intégrité : {} OK · {} CORROMPUS · {} MANQUANTS",
                rep.ok,
                rep.corrupted.len(),
                rep.missing.len()
            );
            for c in &rep.corrupted {
                println!("  CORROMPU : {c}");
            }
            for m in &rep.missing {
                println!("  MANQUANT : {m}");
            }
        }
        Cmd::Populate {
            archive,
            db,
            models,
        } => {
            let db_path = Config::db_path_for(&archive, db);
            let cfg = Config {
                source: PathBuf::new(),
                archive,
                zstd_level: 9,
                db_path,
                models_dir: models,
                label: None,
                store: false,
            };
            let st = index::populate_from_archive(&cfg, &|_| {})?;
            println!(
                "Base peuplée : {} fichiers | {} images | {} chunks texte",
                st.files_total, st.images_embedded, st.chunks_embedded
            );
        }
        Cmd::Search {
            archive,
            query,
            top_k,
            db,
            models,
        } => {
            let db_path = Config::db_path_for(&archive, db);
            let cfg = Config {
                source: PathBuf::new(),
                archive,
                zstd_level: 9,
                db_path,
                models_dir: models,
                label: None,
                store: false,
            };
            let res = search::search(&cfg, &query, top_k)?;

            println!("\n=== Images ===");
            if res.images.is_empty() {
                println!("(aucune)");
            }
            for h in &res.images {
                println!("  {:.4}  {}", h.score, h.rel_path);
            }

            println!("\n=== Documents ===");
            if res.docs.is_empty() {
                println!("(aucun)");
            }
            for h in &res.docs {
                println!(
                    "  {:.4}  {}#{}  — {}",
                    h.score, h.rel_path, h.chunk_idx, h.snippet
                );
            }
        }
    }
    Ok(())
}
