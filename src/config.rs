use std::path::PathBuf;

/// Configuration partagée d'un run (index ou search).
#[derive(Clone, Debug)]
pub struct Config {
    pub source: PathBuf,
    pub archive: PathBuf,
    /// Niveau zstd (1..=22). Défaut 9.
    pub zstd_level: i32,
    /// Chemin DB SQLite. Défaut: <archive>/.archivist.db
    pub db_path: PathBuf,
    /// Répertoire des modèles ONNX + tokenizers.
    pub models_dir: PathBuf,
    /// Préfixe de chemin (= nom de la source) pour une archive PARTAGÉE
    /// multi-sources. `None` = archive mono-source (rel bruts).
    pub label: Option<String>,
    /// `true` = stocke les fichiers TELS QUELS (copie, pas de .zst). Permet la
    /// vérif d'intégrité directe et force le hash complet de tout.
    pub store: bool,
}

impl Config {
    pub fn db_path_for(archive: &std::path::Path, explicit: Option<PathBuf>) -> PathBuf {
        explicit.unwrap_or_else(|| archive.join(".archivist.db"))
    }
}

/// Extensions images supportées (recherche texte→image).
pub const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "gif", "bmp"];
/// Extensions texte extractibles (recherche texte→texte).
pub const TEXT_EXTS: &[&str] = &["txt", "md", "pdf"];

pub fn ext_lower(path: &std::path::Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
}

pub fn is_image(path: &std::path::Path) -> bool {
    ext_lower(path).map_or(false, |e| IMAGE_EXTS.contains(&e.as_str()))
}

pub fn is_text(path: &std::path::Path) -> bool {
    ext_lower(path).map_or(false, |e| TEXT_EXTS.contains(&e.as_str()))
}

/// Formats DÉJÀ compressés : zstd n'y gagne ~rien en taille. On les compresse
/// quand même (critère standalone) mais à bas niveau pour ne pas gâcher de CPU.
pub const PRECOMPRESSED_EXTS: &[&str] = &[
    "jpg", "jpeg", "png", "gif", "webp", "mp4", "mov", "avi", "mkv", "m4v", "mp3", "aac", "ogg",
    "flac", "opus", "zip", "gz", "xz", "7z", "rar", "zst", "bz2", "lz4", "heic", "webm",
];

pub fn is_precompressed(path: &std::path::Path) -> bool {
    ext_lower(path).map_or(false, |e| PRECOMPRESSED_EXTS.contains(&e.as_str()))
}

/// Formats très compressibles : on pousse zstd au max, le gain de taille vaut
/// le surcoût CPU (texte, logs, images non compressées, audio PCM…).
pub const HIGH_COMPRESS_EXTS: &[&str] = &[
    "txt", "md", "csv", "tsv", "log", "json", "jsonl", "ndjson", "xml", "html", "htm", "svg",
    "yaml", "yml", "toml", "ini", "bmp", "tiff", "tif", "wav", "ppm", "pgm", "pbm", "dib",
];

pub fn is_high_compress(path: &std::path::Path) -> bool {
    ext_lower(path).map_or(false, |e| HIGH_COMPRESS_EXTS.contains(&e.as_str()))
}

/// Niveau zstd effectif : abaissé (≤3) pour le déjà-compressé, poussé (≥19)
/// pour le très compressible, sinon niveau de base.
pub fn effective_level(path: &std::path::Path, base: i32) -> i32 {
    if is_precompressed(path) {
        base.min(3)
    } else if is_high_compress(path) {
        base.max(19)
    } else {
        base
    }
}
