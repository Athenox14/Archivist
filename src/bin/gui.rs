//! GUI native (eframe/egui) pour Archivist : indexer, voir l'état, rechercher.
//!
//! - Archive = dossier PARENT : recherche/état agrègent récursivement les bases.
//! - Modèles auto-détectés. Recherche images (CLIP) + documents (e5) + noms.
//! - Worker thread qui garde les encodeurs en cache → requêtes instantanées.
//! - Vérifie une mise à jour au démarrage.
#![windows_subsystem = "windows"] // pas de fenêtre console sous Windows

#[path = "../update_mod.rs"]
mod update;

use archivist::config::Config;
use archivist::db::Db;
use archivist::embed::{clip::ClipEncoder, text::TextEncoder};
use archivist::{index, search};
use eframe::egui;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use walkdir::WalkDir;

// ---------- résolution des modèles ----------

fn resolve_models() -> Option<PathBuf> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("ARCHIVIST_MODELS") {
        cands.push(PathBuf::from(p));
    }
    if let Ok(exe) = std::env::current_exe() {
        // target/release/gui.exe → remonte vers la racine projet
        for up in [1usize, 2, 3] {
            let mut p = exe.clone();
            for _ in 0..up {
                p.pop();
            }
            cands.push(p.join("models"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        cands.push(cwd.join("models"));
    }
    cands
        .into_iter()
        .find(|d| d.join("clip_image.onnx").exists())
}

fn find_dbs(parent: &Path) -> Vec<PathBuf> {
    if parent.is_file() {
        return vec![parent.to_path_buf()];
    }
    // les bases vivent à la racine (prof. 1) ou racine/<source>/ (prof. 2).
    // Limiter la profondeur évite de scanner tout l'arbre (lent sur HDD).
    WalkDir::new(parent)
        .max_depth(3)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && e.file_name() == ".archivist.db")
        .map(|e| e.path().to_path_buf())
        .collect()
}

/// Liste des bases, mise en cache par chemin (évite de re-walker à chaque requête).
fn dbs_cached(cache: &mut Option<(PathBuf, Vec<PathBuf>)>, archive: &Path) -> Vec<PathBuf> {
    if let Some((p, v)) = cache.as_ref() {
        if p == archive {
            return v.clone();
        }
    }
    let v = find_dbs(archive);
    *cache = Some((archive.to_path_buf(), v.clone()));
    v
}

/// Vecteurs chargés en RAM pour une archive (évite de relire le HDD par requête).
struct ImgVec {
    rel: String,
    zst: PathBuf,
    vec: Vec<f32>,
}
struct DocVec {
    disp: String,
    idx: i64,
    snippet: String,
    vec: Vec<f32>,
}
struct EmbCache {
    archive: PathBuf,
    imgs: Vec<ImgVec>,
    docs: Vec<DocVec>,
    rels: Vec<String>, // pour la recherche par nom de fichier
}

// ---------- messages worker <-> UI ----------

enum Cmd {
    Index {
        source: PathBuf,
        archive: PathBuf,
        level: i32,
        store: bool,
    },
    Populate {
        archive: PathBuf,
    },
    Verify {
        archive: PathBuf,
    },
    Status {
        archive: PathBuf,
    },
    Search {
        archive: PathBuf,
        query: String,
        top_k: usize,
    },
}

struct ImgRes {
    rel: String,
    score: f32,
    zst: PathBuf,
    bytes: Option<Vec<u8>>,
}

enum Msg {
    Log(String),
    Busy(bool),
    Progress(f32, String),        // fraction 0..1, label
    Status(i64, i64, i64, usize), // fichiers, images, chunks, nb_bases
    Search {
        images: Vec<ImgRes>,
        docs: Vec<(String, i64, f32, String)>,
        files: Vec<String>,
    },
}

fn decompress(zst: &Path) -> Option<Vec<u8>> {
    let f = std::fs::File::open(zst).ok()?;
    zstd::stream::decode_all(std::io::BufReader::new(f)).ok()
}

fn worker(rx: Receiver<Cmd>, tx: Sender<Msg>, ctx: egui::Context) {
    let models = resolve_models();
    let mut clip: Option<ClipEncoder> = None;
    let mut text: Option<TextEncoder> = None;
    let mut db_cache: Option<(PathBuf, Vec<PathBuf>)> = None;
    let mut emb_cache: Option<EmbCache> = None;

    let send = |m: Msg| {
        let _ = tx.send(m);
        ctx.request_repaint();
    };

    if models.is_none() {
        send(Msg::Log(
            "Modèles introuvables. Lance `python scripts/export_models.py` (dossier projet)."
                .into(),
        ));
    } else {
        send(Msg::Log(format!(
            "Modèles : {}",
            models.as_ref().unwrap().display()
        )));
    }

    for cmd in rx {
        match cmd {
            Cmd::Index {
                source,
                archive,
                level,
                store,
            } => {
                send(Msg::Busy(true));
                // archive PARTAGÉE : tout va dans la racine `archive`, préfixé
                // par le nom de la source → dédup cross-source + une seule base.
                let label = source
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("source")
                    .to_string();
                let mode = if store { " (sans compression)" } else { "" };
                send(Msg::Log(format!(
                    "Indexation{mode} {} → {}/{label}…",
                    source.display(),
                    archive.display()
                )));
                let _ = std::fs::create_dir_all(&archive);
                let cfg = Config {
                    source,
                    archive: archive.clone(),
                    zstd_level: level,
                    db_path: Config::db_path_for(&archive, None),
                    models_dir: models.clone().unwrap_or_else(|| PathBuf::from("models")),
                    label: Some(label),
                    store,
                };
                // callback de progression (appelé depuis plusieurs threads → Sync via Mutex)
                let ptx = std::sync::Mutex::new(tx.clone());
                let pctx = ctx.clone();
                let prog = move |p: index::Progress| {
                    let (frac, label) = match p {
                        index::Progress::Scanning => (0.0, "Analyse + hash…".to_string()),
                        index::Progress::Scan { done, total } => (
                            if total == 0 {
                                0.0
                            } else {
                                done as f32 / total as f32
                            },
                            format!("Analyse {done}/{total}"),
                        ),
                        index::Progress::Compress { done, total } => (
                            if total == 0 {
                                1.0
                            } else {
                                done as f32 / total as f32
                            },
                            format!("Compression {done}/{total}"),
                        ),
                        index::Progress::Embed { done, total } => (
                            if total == 0 {
                                1.0
                            } else {
                                done as f32 / total as f32
                            },
                            format!("Embeddings {done}/{total}"),
                        ),
                        index::Progress::Done => (1.0, "Terminé".to_string()),
                    };
                    if let Ok(g) = ptx.lock() {
                        let _ = g.send(Msg::Progress(frac, label));
                    }
                    pctx.request_repaint();
                };
                match index::run_with_progress(&cfg, &prog) {
                    Ok(st) => send(Msg::Log(format!(
                        "OK : {} fichiers | {} compressés | {} hardlinks | {} images | {} chunks",
                        st.files_total,
                        st.canonicals,
                        st.hardlinks,
                        st.images_embedded,
                        st.chunks_embedded
                    ))),
                    Err(e) => send(Msg::Log(format!("ERREUR index : {e}"))),
                }
                db_cache = None;
                emb_cache = None;
                send(Msg::Busy(false));
            }

            Cmd::Populate { archive } => {
                send(Msg::Busy(true));
                send(Msg::Log(format!(
                    "Peuplement de la base depuis {}…",
                    archive.display()
                )));
                let ptx = std::sync::Mutex::new(tx.clone());
                let pctx = ctx.clone();
                let prog = move |p: index::Progress| {
                    let (frac, label) = match p {
                        index::Progress::Scanning => (0.0, "Recensement des fichiers…".to_string()),
                        index::Progress::Scan { done, total } => (
                            if total == 0 {
                                0.0
                            } else {
                                done as f32 / total as f32
                            },
                            format!("Recensement {done}/{total}"),
                        ),
                        index::Progress::Compress { done, total } => (
                            if total == 0 {
                                1.0
                            } else {
                                done as f32 / total as f32
                            },
                            format!("Compression {done}/{total}"),
                        ),
                        index::Progress::Embed { done, total } => (
                            if total == 0 {
                                1.0
                            } else {
                                done as f32 / total as f32
                            },
                            format!("Embeddings {done}/{total}"),
                        ),
                        index::Progress::Done => (1.0, "Terminé".to_string()),
                    };
                    if let Ok(g) = ptx.lock() {
                        let _ = g.send(Msg::Progress(frac, label));
                    }
                    pctx.request_repaint();
                };
                let cfg = Config {
                    source: PathBuf::new(),
                    archive: archive.clone(),
                    zstd_level: 9,
                    db_path: Config::db_path_for(&archive, None),
                    models_dir: models.clone().unwrap_or_else(|| PathBuf::from("models")),
                    label: None,
                    store: false,
                };
                match index::populate_from_archive(&cfg, &prog) {
                    Ok(st) => send(Msg::Log(format!(
                        "Base peuplée : {} fichiers | {} images | {} chunks",
                        st.files_total, st.images_embedded, st.chunks_embedded
                    ))),
                    Err(e) => send(Msg::Log(format!("ERREUR peuplement : {e}"))),
                }
                db_cache = None;
                emb_cache = None;
                send(Msg::Busy(false));
            }

            Cmd::Verify { archive } => {
                send(Msg::Busy(true));
                send(Msg::Log(format!(
                    "Vérification d'intégrité de {}…",
                    archive.display()
                )));
                let ptx = std::sync::Mutex::new(tx.clone());
                let pctx = ctx.clone();
                let prog = move |d: usize, t: usize| {
                    let frac = if t == 0 { 1.0 } else { d as f32 / t as f32 };
                    if let Ok(g) = ptx.lock() {
                        let _ = g.send(Msg::Progress(frac, format!("Vérif {d}/{t}")));
                    }
                    pctx.request_repaint();
                };
                let cfg = Config {
                    source: PathBuf::new(),
                    archive: archive.clone(),
                    zstd_level: 9,
                    db_path: Config::db_path_for(&archive, None),
                    models_dir: PathBuf::new(),
                    label: None,
                    store: false,
                };
                match index::verify_integrity(&cfg, &prog) {
                    Ok(rep) => {
                        send(Msg::Log(format!(
                            "Intégrité : {} OK · {} CORROMPUS · {} MANQUANTS",
                            rep.ok,
                            rep.corrupted.len(),
                            rep.missing.len()
                        )));
                        for c in rep.corrupted.iter().take(100) {
                            send(Msg::Log(format!("  CORROMPU : {c}")));
                        }
                        for m in rep.missing.iter().take(100) {
                            send(Msg::Log(format!("  MANQUANT : {m}")));
                        }
                    }
                    Err(e) => send(Msg::Log(format!("ERREUR vérif : {e}"))),
                }
                send(Msg::Busy(false));
            }

            Cmd::Status { archive } => {
                let dbs = dbs_cached(&mut db_cache, &archive);
                let (mut f, mut i, mut c) = (0i64, 0i64, 0i64);
                for db_path in &dbs {
                    if let Ok((a, b, d)) = Db::open_ro(db_path).and_then(|db| db.counts()) {
                        f += a;
                        i += b;
                        c += d;
                    }
                }
                send(Msg::Status(f, i, c, dbs.len()));
            }

            Cmd::Search {
                archive,
                query,
                top_k,
            } => {
                send(Msg::Busy(true));
                let dbs = dbs_cached(&mut db_cache, &archive);
                if dbs.is_empty() {
                    send(Msg::Log(format!(
                        "Aucune base .archivist.db sous {}",
                        archive.display()
                    )));
                    send(Msg::Busy(false));
                    continue;
                }

                // encodeurs (chargés une seule fois)
                if let Some(m) = &models {
                    if clip.is_none() {
                        clip = ClipEncoder::load(m).ok();
                    }
                    if text.is_none() {
                        text = TextEncoder::load(m).ok();
                    }
                }

                // charge les vecteurs en RAM une fois par archive (lecture HDD unique)
                let need_reload = emb_cache
                    .as_ref()
                    .map(|c| c.archive != archive)
                    .unwrap_or(true);
                if need_reload {
                    send(Msg::Log(
                        "Chargement des vecteurs en mémoire (1ʳᵉ requête)…".into(),
                    ));
                    let mut imgs = Vec::new();
                    let mut docs = Vec::new();
                    let mut rels = Vec::new();
                    for db_path in &dbs {
                        let root = db_path.parent().unwrap_or(Path::new(".")).to_path_buf();
                        let rname = root
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_string();
                        let Ok(db) = Db::open_ro(db_path) else {
                            continue;
                        };
                        if let Ok(v) = db.all_image_embeddings() {
                            for (rel, vec) in v {
                                let zst = root.join(format!("{rel}.zst"));
                                imgs.push(ImgVec { rel, zst, vec });
                            }
                        }
                        if let Ok(v) = db.all_text_chunks() {
                            for (rel, idx, text, vec) in v {
                                docs.push(DocVec {
                                    disp: format!("{rname}/{rel}"),
                                    idx,
                                    snippet: text.chars().take(160).collect(),
                                    vec,
                                });
                            }
                        }
                        if let Ok(paths) = db.all_rel_paths() {
                            for rel in paths {
                                rels.push(format!("{rname}/{rel}"));
                            }
                        }
                    }
                    send(Msg::Log(format!(
                        "{} images, {} chunks en cache",
                        imgs.len(),
                        docs.len()
                    )));
                    emb_cache = Some(EmbCache {
                        archive: archive.clone(),
                        imgs,
                        docs,
                        rels,
                    });
                }
                let cache = emb_cache.as_ref().unwrap();

                use rayon::prelude::*;
                let clip_qv = clip.as_mut().and_then(|c| c.embed_text(&query).ok());
                let text_qv = text.as_mut().and_then(|t| t.embed_query(&query).ok());
                let tokens = search::query_tokens(&query);

                // images
                let mut images = Vec::new();
                if let Some(qv) = &clip_qv {
                    let mut scored: Vec<(usize, f32)> = cache
                        .imgs
                        .par_iter()
                        .enumerate()
                        .map(|(i, e)| (i, archivist::embed::cosine_normalized(qv, &e.vec)))
                        .collect();
                    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    scored.truncate(top_k);
                    images = scored
                        .into_iter()
                        .map(|(i, score)| {
                            let e = &cache.imgs[i];
                            ImgRes {
                                rel: e.rel.clone(),
                                score,
                                zst: e.zst.clone(),
                                bytes: decompress(&e.zst),
                            }
                        })
                        .collect();
                }

                // documents
                let mut docs = Vec::new();
                if let Some(qv) = &text_qv {
                    let mut scored: Vec<(usize, f32)> = cache
                        .docs
                        .par_iter()
                        .enumerate()
                        .map(|(i, e)| (i, archivist::embed::cosine_normalized(qv, &e.vec)))
                        .collect();
                    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    scored.truncate(top_k);
                    docs = scored
                        .into_iter()
                        .map(|(i, score)| {
                            let e = &cache.docs[i];
                            (e.disp.clone(), e.idx, score, e.snippet.clone())
                        })
                        .collect();
                }

                // noms de fichiers
                let mut files: Vec<String> = cache
                    .rels
                    .iter()
                    .filter(|r| search::name_matches(r, &tokens))
                    .take(50)
                    .cloned()
                    .collect();
                files.sort();

                send(Msg::Search {
                    images,
                    docs,
                    files,
                });
                send(Msg::Busy(false));
            }
        }
    }
}

// ---------- app ----------

#[derive(PartialEq)]
enum Tab {
    Search,
    Index,
    Status,
}

struct ImgItem {
    rel: String,
    score: f32,
    zst: PathBuf,
    tex: Option<egui::TextureHandle>,
}

/// Décompresse le .zst dans un dossier temp et ouvre le fichier avec
/// l'application par défaut de Windows (double-clic sur une vignette).
fn open_archived(zst: &Path) {
    let Some(stem) = zst.file_stem().and_then(|s| s.to_str()) else {
        return;
    }; // "x.JPG.zst" → "x.JPG"
    let dir = std::env::temp_dir().join("archivist_view");
    let _ = std::fs::create_dir_all(&dir);
    let out = dir.join(stem);
    if !out.exists() {
        if let Some(bytes) = decompress(zst) {
            let _ = std::fs::write(&out, bytes);
        } else {
            return;
        }
    }
    let _ = std::process::Command::new("explorer").arg(&out).spawn();
}

struct App {
    tab: Tab,
    source: String,
    archive: String,
    level: i32,
    store: bool,
    query: String,
    top_k: usize,
    busy: bool,
    show_log: bool,
    progress: Option<(f32, String)>,
    log: Vec<String>,
    status: Option<(i64, i64, i64, usize)>,
    images: Vec<ImgItem>,
    docs: Vec<(String, i64, f32, String)>,
    files: Vec<String>,
    pending: Option<Vec<ImgRes>>,
    tx: Sender<Cmd>,
    rx: Receiver<Msg>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<Cmd>();
        let (msg_tx, msg_rx) = std::sync::mpsc::channel::<Msg>();
        cc.egui_ctx.style_mut(|s| {
            s.spacing.item_spacing = egui::vec2(8.0, 8.0);
            s.spacing.button_padding = egui::vec2(10.0, 5.0);
        });
        let ctx = cc.egui_ctx.clone();
        std::thread::spawn(move || worker(cmd_rx, msg_tx, ctx));
        App {
            tab: Tab::Search,
            source: String::new(),
            archive: String::new(),
            level: 9,
            store: false,
            query: String::new(),
            top_k: 12,
            busy: false,
            show_log: false,
            progress: None,
            log: Vec::new(),
            status: None,
            images: Vec::new(),
            docs: Vec::new(),
            files: Vec::new(),
            pending: None,
            tx: cmd_tx,
            rx: msg_rx,
        }
    }

    fn push_log(&mut self, s: String) {
        self.log.push(s);
        if self.log.len() > 200 {
            let d = self.log.len() - 200;
            self.log.drain(0..d);
        }
    }

    fn drain(&mut self, ctx: &egui::Context) {
        while let Ok(m) = self.rx.try_recv() {
            match m {
                Msg::Log(s) => self.push_log(s),
                Msg::Busy(b) => {
                    self.busy = b;
                    if !b {
                        self.progress = None;
                    }
                }
                Msg::Progress(frac, label) => self.progress = Some((frac, label)),
                Msg::Status(f, i, c, n) => self.status = Some((f, i, c, n)),
                Msg::Search {
                    images,
                    docs,
                    files,
                } => {
                    self.images.clear();
                    self.pending = Some(images);
                    self.docs = docs;
                    self.files = files;
                }
            }
        }
        if let Some(pend) = self.pending.take() {
            for r in pend {
                let tex = r
                    .bytes
                    .as_ref()
                    .and_then(|b| decode_texture(ctx, &r.rel, b));
                self.images.push(ImgItem {
                    rel: r.rel,
                    score: r.score,
                    zst: r.zst,
                    tex,
                });
            }
        }
    }
}

fn decode_texture(ctx: &egui::Context, name: &str, bytes: &[u8]) -> Option<egui::TextureHandle> {
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    let color = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &img);
    Some(ctx.load_texture(name, color, egui::TextureOptions::LINEAR))
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain(ctx);

        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("🪶 Archivist")
                        .strong()
                        .size(17.0)
                        .color(egui::Color32::from_rgb(90, 170, 230)),
                );
                ui.separator();
                ui.selectable_value(&mut self.tab, Tab::Search, "🔍 Recherche");
                ui.selectable_value(&mut self.tab, Tab::Index, "📦 Indexer");
                ui.selectable_value(&mut self.tab, Tab::Status, "📊 État");
                ui.separator();
                if self.busy {
                    ui.spinner();
                }
                if let Some((frac, label)) = &self.progress {
                    ui.add(
                        egui::ProgressBar::new(*frac)
                            .desired_width(260.0)
                            .text(format!("{label}  ({:.0}%)", frac * 100.0)),
                    );
                } else if self.busy {
                    ui.label("traitement…");
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let lbl = if self.show_log {
                        "Journal ▾"
                    } else {
                        "Journal ▸"
                    };
                    if ui.selectable_label(self.show_log, lbl).clicked() {
                        self.show_log = !self.show_log;
                    }
                });
            });
        });

        if self.show_log {
            egui::TopBottomPanel::bottom("log")
                .resizable(true)
                .default_height(140.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for line in &self.log {
                                ui.monospace(line);
                            }
                        });
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| match self.tab {
            Tab::Index => self.ui_index(ui),
            Tab::Search => self.ui_search(ui),
            Tab::Status => self.ui_status(ui),
        });
    }
}

impl App {
    fn path_row(ui: &mut egui::Ui, label: &str, val: &mut String) {
        ui.horizontal(|ui| {
            ui.label(label);
            ui.add(egui::TextEdit::singleline(val).desired_width(440.0));
        });
    }

    fn ui_index(&mut self, ui: &mut egui::Ui) {
        ui.heading("Indexer une source");
        Self::path_row(ui, "Source ", &mut self.source);
        Self::path_row(ui, "Archive", &mut self.archive);
        ui.horizontal(|ui| {
            ui.add_enabled(!self.store, egui::Slider::new(&mut self.level, 1..=22))
                .on_disabled_hover_text("désactivé en mode sans compression");
            ui.label("zstd");
        });
        ui.checkbox(
            &mut self.store,
            "Sans compression (copie brute + hash complet → vérif intégrité)",
        );
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!self.busy, egui::Button::new("▶ Lancer l'indexation"))
                .clicked()
            {
                let _ = self.tx.send(Cmd::Index {
                    source: PathBuf::from(&self.source),
                    archive: PathBuf::from(&self.archive),
                    level: self.level,
                    store: self.store,
                });
            }
            if ui
                .add_enabled(!self.busy, egui::Button::new("⏵ Reprendre (sans re-hash)"))
                .on_hover_text(
                    "Reprend une indexation stoppée : lit l'archive telle quelle et \
                     n'embedde QUE ce qui manque. Pas de re-hash, pas besoin de la source.",
                )
                .clicked()
            {
                let _ = self.tx.send(Cmd::Populate {
                    archive: PathBuf::from(&self.archive),
                });
            }
        });
        ui.add_space(6.0);
        ui.label("« Archive » = racine PARTAGÉE. La source est rangée sous <nom_source>/ et dédupliquée contre les autres sources. Modèles auto-détectés.");
        ui.label("Idempotent : ne recompresse que le nouveau/modifié.");
    }

    fn ui_status(&mut self, ui: &mut egui::Ui) {
        ui.heading("État (récursif)");
        Self::path_row(ui, "Dossier", &mut self.archive);
        ui.horizontal(|ui| {
            if ui.button("🔄 Rafraîchir").clicked() {
                let _ = self.tx.send(Cmd::Status { archive: PathBuf::from(&self.archive) });
            }
            if ui
                .add_enabled(!self.busy, egui::Button::new("🛡 Vérifier l'intégrité (corruption)"))
                .on_hover_text("Re-hash chaque fichier et compare au hash enregistré. Résultat dans le journal.")
                .clicked()
            {
                self.show_log = true;
                let _ = self.tx.send(Cmd::Verify { archive: PathBuf::from(&self.archive) });
            }
        });
        ui.add_space(10.0);
        match self.status {
            Some((f, i, c, n)) => {
                ui.label(format!("Bases trouvées       : {n}"));
                ui.label(format!("Fichiers indexés     : {f}"));
                ui.label(format!("Embeddings image     : {i}"));
                ui.label(format!("Chunks texte         : {c}"));
            }
            None => {
                ui.label("Clique « Rafraîchir ».");
            }
        }
    }

    fn ui_search(&mut self, ui: &mut egui::Ui) {
        ui.heading("Recherche sémantique (récursive)");
        Self::path_row(ui, "Dossier", &mut self.archive);
        ui.horizontal(|ui| {
            ui.label("Requête");
            let resp = ui.add(egui::TextEdit::singleline(&mut self.query).desired_width(380.0));
            ui.label("top-k");
            ui.add(egui::DragValue::new(&mut self.top_k).range(1..=50));
            let go = ui
                .add_enabled(!self.busy, egui::Button::new("🔍 Chercher"))
                .clicked();
            let entered = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if (go || entered) && !self.busy {
                let _ = self.tx.send(Cmd::Search {
                    archive: PathBuf::from(&self.archive),
                    query: self.query.clone(),
                    top_k: self.top_k,
                });
            }
        });
        ui.separator();

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if !self.images.is_empty() {
                    ui.label(egui::RichText::new("Images").strong().size(16.0));
                    ui.add_space(4.0);
                    // grille multi-rangées : cellules de largeur fixe, retour à la ligne auto
                    let cell = 190.0;
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(10.0, 10.0);
                        for it in &self.images {
                            ui.allocate_ui(egui::vec2(cell, cell + 56.0), |ui| {
                                ui.vertical(|ui| {
                                    let name = it.rel.rsplit('/').next().unwrap_or(&it.rel);
                                    match &it.tex {
                                        Some(tex) => {
                                            let st = egui::load::SizedTexture::from_handle(tex);
                                            let resp = ui
                                                .add(
                                                    egui::Image::new(st)
                                                        .max_size(egui::vec2(cell, cell))
                                                        .maintain_aspect_ratio(true)
                                                        .sense(egui::Sense::click()),
                                                )
                                                .on_hover_text("Double-clic : ouvrir");
                                            if resp.double_clicked() {
                                                open_archived(&it.zst);
                                            }
                                        }
                                        None => {
                                            ui.label("(aperçu indispo)");
                                        }
                                    }
                                    // nom en grand, score petit et lisible
                                    ui.label(egui::RichText::new(name).size(15.0).strong());
                                    ui.label(
                                        egui::RichText::new(format!("score {:.2}", it.score))
                                            .size(11.0)
                                            .weak(),
                                    );
                                });
                            });
                        }
                    });
                }

                if !self.docs.is_empty() {
                    ui.add_space(12.0);
                    ui.label(egui::RichText::new("Documents").strong().size(16.0));
                    for (path, idx, score, snip) in &self.docs {
                        ui.group(|ui| {
                            ui.label(format!("{:.3}  {}#{}", score, path, idx));
                            ui.small(snip);
                        });
                    }
                }

                if !self.files.is_empty() {
                    ui.add_space(12.0);
                    ui.label(egui::RichText::new("Noms de fichiers").strong().size(16.0));
                    for f in &self.files {
                        ui.monospace(f);
                    }
                }

                if self.images.is_empty() && self.docs.is_empty() && self.files.is_empty() {
                    ui.label("Aucun résultat (lance une recherche).");
                }
            });
    }
}

fn load_icon() -> egui::IconData {
    let bytes = include_bytes!("../../assets/icon.png");
    match image::load_from_memory(bytes) {
        Ok(img) => {
            let img = img.to_rgba8();
            let (w, h) = img.dimensions();
            egui::IconData {
                rgba: img.into_raw(),
                width: w,
                height: h,
            }
        }
        Err(_) => egui::IconData {
            rgba: vec![0; 4],
            width: 1,
            height: 1,
        },
    }
}

/// Chemin du log de crash (à côté de l'exe, sinon dossier temp).
fn crash_log_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(std::env::temp_dir)
        .join("archivist_crash.log")
}

/// Écrit toute panic (toutes threads) dans un fichier — sinon invisible (pas de console).
fn install_panic_logger() {
    let path = crash_log_path();
    std::panic::set_hook(Box::new(move |info| {
        use std::io::Write;
        let bt = std::backtrace::Backtrace::force_capture();
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(f, "\n=== PANIC @ epoch {secs} ===\n{info}\n{bt}");
        }
    }));
}

fn main() -> eframe::Result<()> {
    install_panic_logger();
    // mise à jour auto au démarrage
    if update::try_update() {
        update::restart();
    }
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1040.0, 780.0])
            .with_min_inner_size([720.0, 480.0])
            .with_icon(std::sync::Arc::new(load_icon())),
        ..Default::default()
    };
    eframe::run_native(
        "Archivist",
        opts,
        Box::new(|cc| Ok(Box::new(App::new(cc)) as Box<dyn eframe::App>)),
    )
}
