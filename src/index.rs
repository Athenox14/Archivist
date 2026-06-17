//! Orchestration de l'indexation : dedup → archive (zstd + hardlinks) → DB recherche.

use crate::config::{self, Config};
use crate::db::Db;
use crate::dedup;
use crate::embed::{clip::ClipEncoder, text::TextEncoder};
use crate::{archive, extract};
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use walkdir::WalkDir;

pub struct IndexStats {
    pub files_total: usize,
    pub canonicals: usize,
    pub hardlinks: usize,
    pub images_embedded: usize,
    pub chunks_embedded: usize,
}

/// Étapes de progression rapportées pendant l'indexation.
pub enum Progress {
    Scanning,
    Compress { done: usize, total: usize },
    Embed { done: usize, total: usize },
    Done,
}

/// Indexe une source vers une archive. Idempotent.
pub fn run(cfg: &Config) -> Result<IndexStats> {
    run_with_progress(cfg, &|_| {})
}

/// Variante avec callback de progression. Le callback est appelé depuis
/// plusieurs threads (phase compression) → il doit être `Sync`.
pub fn run_with_progress(cfg: &Config, progress: &(dyn Fn(Progress) + Sync)) -> Result<IndexStats> {
    // DB ouverte tôt : alimente la dédup cross-source (tailles + hash connus).
    let db = Db::open(&cfg.db_path)?;

    log::info!("scan + hash sélectif…");
    progress(Progress::Scanning);
    let known_sizes = db.distinct_sizes()?;
    // hashe tout si archive partagée OU mode stockage (intégrité/corruption).
    let hash_all = cfg.label.is_some() || cfg.store;
    let mut entries = dedup::scan_and_hash(&cfg.source, &known_sizes, hash_all)?;

    // Archive PARTAGÉE : préfixe les rel_path par le label de la source pour
    // éviter les collisions et permettre la dédup inter-sources.
    if let Some(label) = &cfg.label {
        for e in &mut entries {
            e.rel_path = format!("{label}/{}", e.rel_path);
        }
    }

    let groups = dedup::group_by_hash(&entries);
    let mut canonical_of: HashMap<usize, usize> = HashMap::new();
    let mut is_canonical = vec![true; entries.len()];
    for idxs in groups.values() {
        let canon = idxs[0];
        for &i in &idxs[1..] {
            canonical_of.insert(i, canon);
            is_canonical[i] = false;
        }
    }

    // (1) Dédup CROSS-SOURCE : un canonique dont le hash existe déjà sous une
    // AUTRE source (et dont le .zst est présent) devient un hardlink.
    let not_prefix = cfg
        .label
        .as_ref()
        .map(|l| format!("{l}/"))
        .unwrap_or_default();
    let mut cross_links: Vec<(usize, String)> = Vec::new();
    // destination selon le mode (stockage brut vs .zst)
    let store = cfg.store;
    let archive_dir = cfg.archive.clone();
    let dest = move |rel: &str| {
        if store {
            archive::dest_plain(&archive_dir, rel)
        } else {
            archive::dest_zst(&archive_dir, rel)
        }
    };

    let mut to_compress: Vec<usize> = Vec::new();
    for i in (0..entries.len()).filter(|&i| is_canonical[i]) {
        let e = &entries[i];
        if !not_prefix.is_empty() {
            if let Some(h) = &e.hash {
                if let Some(prev) = db.find_other_with_hash(h, &not_prefix)? {
                    if dest(&prev).exists() {
                        cross_links.push((i, prev));
                        continue;
                    }
                }
            }
        }
        to_compress.push(i);
    }

    // (3) Dictionnaire zstd pour le texte (mode compression seulement).
    let dict: Option<Vec<u8>> = if cfg.store {
        None
    } else {
        let dict_path = cfg.archive.join(".archivist.dict");
        if dict_path.exists() {
            std::fs::read(&dict_path).ok()
        } else {
            let d = train_text_dict(&entries, &to_compress);
            if let Some(b) = &d {
                let _ = std::fs::create_dir_all(&cfg.archive);
                let _ = std::fs::write(&dict_path, b);
                log::info!("dictionnaire zstd entraîné ({} octets)", b.len());
            }
            d
        }
    };

    // --- écriture archive (copie brute OU compression) ---
    if cfg.store {
        log::info!("stockage sans compression…");
    } else {
        log::info!("compression zstd (base {})…", cfg.zstd_level);
    }
    let comp_total = to_compress.len();
    let comp_done = AtomicUsize::new(0);
    progress(Progress::Compress {
        done: 0,
        total: comp_total,
    });
    to_compress.par_iter().try_for_each(|&i| -> Result<()> {
        let e = &entries[i];
        let dst = dest(&e.rel_path);
        if cfg.store {
            archive::store_to(&e.abs_path, &dst)?;
        } else {
            let lvl = config::effective_level(&e.abs_path, cfg.zstd_level);
            let d = if config::is_text(&e.abs_path) {
                dict.as_deref()
            } else {
                None
            };
            archive::compress_to_opt(&e.abs_path, &dst, lvl, d)?;
        }
        let n = comp_done.fetch_add(1, Ordering::Relaxed) + 1;
        progress(Progress::Compress {
            done: n,
            total: comp_total,
        });
        Ok(())
    })?;

    // Hardlinks : cross-source d'abord (cible déjà sur disque), puis doublons intra.
    let mut hardlinks = 0;
    for (i, prev) in &cross_links {
        archive::hardlink_or_copy(&dest(prev), &dest(&entries[*i].rel_path))?;
        hardlinks += 1;
    }
    for (&dup, &canon) in &canonical_of {
        archive::hardlink_or_copy(
            &dest(&entries[canon].rel_path),
            &dest(&entries[dup].rel_path),
        )?;
        hardlinks += 1;
    }

    // --- modèles ---
    let mut clip = ClipEncoder::load(&cfg.models_dir).ok();
    let mut text_enc = TextEncoder::load(&cfg.models_dir).ok();
    if clip.is_none() {
        log::warn!("CLIP indisponible — pas d'embeddings image");
    }
    if text_enc.is_none() {
        log::warn!("encodeur texte indisponible — pas d'embeddings texte");
    }

    let mut images_embedded = 0;
    let mut chunks_embedded = 0;

    // total embeddable (pour le %) : images + fichiers texte.
    let embed_total = entries
        .iter()
        .filter(|e| config::is_image(&e.abs_path) || config::is_text(&e.abs_path))
        .count();
    let mut embed_done = 0;
    progress(Progress::Embed {
        done: 0,
        total: embed_total,
    });

    // Passe 1 : upsert tous les fichiers, classe les jobs d'embedding.
    let mut image_jobs: Vec<(i64, std::path::PathBuf)> = Vec::new();
    let mut text_jobs: Vec<(i64, std::path::PathBuf)> = Vec::new();
    for e in &entries {
        let hash = e.hash.as_deref();
        let (file_id, needs_reindex) = db.upsert_file(&e.rel_path, hash, e.size, e.mtime)?;
        let path = e.abs_path.as_path();
        let embeddable = config::is_image(path) || config::is_text(path);
        if !needs_reindex {
            if embeddable {
                embed_done += 1; // déjà indexé → compte pour le %
            }
            continue;
        }
        if config::is_image(path) {
            image_jobs.push((file_id, e.abs_path.clone()));
        } else if config::is_text(path) {
            text_jobs.push((file_id, e.abs_path.clone()));
        }
    }
    progress(Progress::Embed {
        done: embed_done,
        total: embed_total,
    });

    // Passe 2 : images PAR LOT (prétraitement parallèle + 1 passe ONNX/lot).
    const BATCH: usize = 16;
    if let Some(enc) = clip.as_mut() {
        for chunk in image_jobs.chunks(BATCH) {
            let paths: Vec<std::path::PathBuf> = chunk.iter().map(|(_, p)| p.clone()).collect();
            match enc.embed_images(&paths) {
                Ok(vecs) => {
                    for ((file_id, path), v) in chunk.iter().zip(vecs) {
                        match v {
                            Some(vec) => {
                                db.insert_image_embedding(*file_id, &vec)?;
                                images_embedded += 1;
                            }
                            None => log::warn!("image illisible : {}", path.display()),
                        }
                    }
                }
                Err(e) => log::warn!("batch images : {e}"),
            }
            embed_done += chunk.len();
            progress(Progress::Embed {
                done: embed_done,
                total: embed_total,
            });
        }
    }

    // Passe 3 : textes (séquentiel).
    for (file_id, path) in &text_jobs {
        chunks_embedded += embed_text_file(&db, text_enc.as_mut(), *file_id, path)?;
        embed_done += 1;
        progress(Progress::Embed {
            done: embed_done,
            total: embed_total,
        });
    }

    progress(Progress::Done);
    Ok(IndexStats {
        files_total: entries.len(),
        canonicals: to_compress.len(),
        hardlinks,
        images_embedded,
        chunks_embedded,
    })
}

/// Entraîne un dictionnaire zstd sur les fichiers texte canoniques (hors PDF).
/// `None` si trop peu d'échantillons (le dict n'apporterait rien).
fn train_text_dict(entries: &[dedup::FileEntry], to_compress: &[usize]) -> Option<Vec<u8>> {
    let mut samples: Vec<Vec<u8>> = Vec::new();
    for &i in to_compress {
        let p = &entries[i].abs_path;
        if config::is_text(p) && config::ext_lower(p).as_deref() != Some("pdf") {
            if let Ok(b) = std::fs::read(p) {
                if !b.is_empty() {
                    samples.push(b);
                }
            }
        }
    }
    if samples.len() < 8 {
        return None; // pas assez de matière → dict inutile
    }
    zstd::dict::from_samples(&samples, 16 * 1024).ok()
}

pub struct IntegrityReport {
    pub ok: usize,
    pub corrupted: Vec<String>,
    pub missing: Vec<String>,
}

/// Vérifie l'intégrité de l'archive : re-hash chaque fichier (déstocké/décompressé)
/// et compare au hash enregistré → détecte corruption (bit-rot) et fichiers manquants.
pub fn verify_integrity(
    cfg: &Config,
    progress: &(dyn Fn(usize, usize) + Sync),
) -> Result<IntegrityReport> {
    let db = Db::open_ro(&cfg.db_path)?;
    let rows = db.all_files_with_hash()?;
    let dict = std::fs::read(cfg.archive.join(".archivist.dict")).ok();
    let total = rows.len();
    let done = AtomicUsize::new(0);

    enum St {
        Ok,
        Corrupt(String),
        Missing(String),
    }
    let res: Vec<St> = rows
        .par_iter()
        .map(|(rel, expected)| {
            let plain = cfg.archive.join(rel);
            let bytes = if plain.is_file() {
                std::fs::read(&plain).ok()
            } else {
                let z = archive::dest_zst(&cfg.archive, rel);
                if z.is_file() {
                    let d = if config::is_text(Path::new(rel)) {
                        dict.as_deref()
                    } else {
                        None
                    };
                    archive::decompress_bytes(&z, d).ok()
                } else {
                    None
                }
            };
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 500 == 0 {
                progress(n, total);
            }
            match bytes {
                None => St::Missing(rel.clone()),
                Some(b) => {
                    if blake3::hash(&b).to_hex().to_string() == *expected {
                        St::Ok
                    } else {
                        St::Corrupt(rel.clone())
                    }
                }
            }
        })
        .collect();

    let mut rep = IntegrityReport {
        ok: 0,
        corrupted: Vec::new(),
        missing: Vec::new(),
    };
    for s in res {
        match s {
            St::Ok => rep.ok += 1,
            St::Corrupt(r) => rep.corrupted.push(r),
            St::Missing(r) => rep.missing.push(r),
        }
    }
    rep.corrupted.sort();
    rep.missing.sort();
    progress(total, total);
    Ok(rep)
}

/// Lit le contenu d'un fichier d'archive (décompresse si `.zst`).
fn read_archive_bytes(path: &Path, compressed: bool, dict: Option<&[u8]>) -> Option<Vec<u8>> {
    if compressed {
        archive::decompress_bytes(path, dict).ok()
    } else {
        std::fs::read(path).ok()
    }
}

/// Reprend/reconstruit la base de recherche À PARTIR DE L'ARCHIVE (sans la
/// source, sans re-hash). Gère les archives compressées (`.zst`) ET le stockage
/// brut. NON destructif : ne ré-embedde QUE ce qui manque → vraie reprise.
pub fn populate_from_archive(
    cfg: &Config,
    progress: &(dyn Fn(Progress) + Sync),
) -> Result<IndexStats> {
    let db = Db::open(&cfg.db_path)?;
    let dict = std::fs::read(cfg.archive.join(".archivist.dict")).ok();

    progress(Progress::Scanning);
    // tous les fichiers de l'archive, hors métadonnées internes
    let mut files: Vec<PathBuf> = WalkDir::new(&cfg.archive)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.path().to_path_buf())
        .filter(|p| {
            let n = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            !n.starts_with(".archivist.db")
                && n != ".archivist.dict"
                && !n.ends_with(".tmp_copy")
                && !n.ends_with(".zst.tmp")
        })
        .collect();
    files.sort();

    // Passe 1 : enregistre les fichiers (non destructif) + classe le manquant.
    let mut image_jobs: Vec<(i64, PathBuf, bool)> = Vec::new();
    let mut text_jobs: Vec<(i64, PathBuf, bool, String)> = Vec::new();
    let mut embed_total = 0usize;
    for path in &files {
        let rel_raw = path.strip_prefix(&cfg.archive)?.to_string_lossy().replace('\\', "/");
        let compressed = rel_raw.ends_with(".zst");
        let rel = if compressed {
            rel_raw.trim_end_matches(".zst").to_string()
        } else {
            rel_raw.clone()
        };
        let meta = std::fs::metadata(path)?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let file_id = db.get_or_insert_file(&rel, meta.len(), mtime)?;
        let orig = Path::new(&rel);
        if config::is_image(orig) {
            embed_total += 1;
            if !db.has_image_embedding(file_id)? {
                image_jobs.push((file_id, path.clone(), compressed));
            }
        } else if config::is_text(orig) {
            embed_total += 1;
            if !db.has_text_chunks(file_id)? {
                let ext = config::ext_lower(orig).unwrap_or_default();
                text_jobs.push((file_id, path.clone(), compressed, ext));
            }
        }
    }
    log::info!(
        "{} fichiers | {} images à embedder | {} textes (reste)",
        files.len(),
        image_jobs.len(),
        text_jobs.len()
    );

    let mut clip = ClipEncoder::load(&cfg.models_dir).ok();
    let mut text_enc = TextEncoder::load(&cfg.models_dir).ok();
    let mut images_embedded = 0;
    let mut chunks_embedded = 0;
    let mut embed_done = embed_total - image_jobs.len() - text_jobs.len();
    progress(Progress::Embed { done: embed_done, total: embed_total });

    // Passe 2 : images par lot (lecture/décompression parallèle → embed octets).
    const BATCH: usize = 16;
    if let Some(enc) = clip.as_mut() {
        for chunk in image_jobs.chunks(BATCH) {
            let datas: Vec<Option<Vec<u8>>> = chunk
                .par_iter()
                .map(|(_, p, c)| read_archive_bytes(p, *c, None))
                .collect();
            let present: Vec<Vec<u8>> = datas.iter().filter_map(|d| d.clone()).collect();
            match enc.embed_images_bytes(&present) {
                Ok(embs) => {
                    let mut ei = 0;
                    for ((file_id, _, _), d) in chunk.iter().zip(&datas) {
                        if d.is_some() {
                            if let Some(v) = &embs[ei] {
                                db.insert_image_embedding(*file_id, v)?;
                                images_embedded += 1;
                            }
                            ei += 1;
                        }
                    }
                }
                Err(e) => log::warn!("batch images : {e}"),
            }
            embed_done += chunk.len();
            progress(Progress::Embed { done: embed_done, total: embed_total });
        }
    }

    // Passe 3 : textes.
    for (file_id, path, compressed, ext) in &text_jobs {
        let bytes = read_archive_bytes(path, *compressed, dict.as_deref()).unwrap_or_default();
        let text = if ext == "pdf" {
            let tmp = std::env::temp_dir().join("archivist_pdf_tmp.pdf");
            std::fs::write(&tmp, &bytes).ok();
            extract::extract_text(&tmp).unwrap_or_default()
        } else {
            String::from_utf8_lossy(&bytes).into_owned()
        };
        chunks_embedded += embed_text_str(&db, text_enc.as_mut(), *file_id, &text)?;
        embed_done += 1;
        progress(Progress::Embed { done: embed_done, total: embed_total });
    }

    progress(Progress::Done);
    Ok(IndexStats {
        files_total: files.len(),
        canonicals: 0,
        hardlinks: 0,
        images_embedded,
        chunks_embedded,
    })
}

fn embed_text_file(
    db: &Db,
    enc: Option<&mut TextEncoder>,
    file_id: i64,
    path: &Path,
) -> Result<usize> {
    let text = match extract::extract_text(path) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("extract {}: {e}", path.display());
            return Ok(0);
        }
    };
    embed_text_str(db, enc, file_id, &text)
}

fn embed_text_str(
    db: &Db,
    enc: Option<&mut TextEncoder>,
    file_id: i64,
    text: &str,
) -> Result<usize> {
    let Some(enc) = enc else { return Ok(0) };
    let chunks = extract::chunk_text(text, 1000);
    let mut n = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        match enc.embed_passage(chunk) {
            Ok(v) => {
                db.insert_text_chunk(file_id, i, chunk, &v)?;
                n += 1;
            }
            Err(e) => log::warn!("embed chunk {i} (file_id {file_id}): {e}"),
        }
    }
    Ok(n)
}
