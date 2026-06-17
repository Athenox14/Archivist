//! archivist — backup/archivage standalone (zstd par fichier) + recherche sémantique.
//!
//! Deux couches indépendantes :
//!  1. L'ARCHIVE : `archive/<arbo>/fichier.ext.zst`. Utilisable sans ce binaire
//!     (zstd standard + visionneuse d'images). C'est le critère "standalone".
//!  2. La DB SQLite : couche de recherche UNIQUEMENT. Optionnelle ; enrichit l'archive.

pub mod archive;
pub mod config;
pub mod db;
pub mod dedup;
pub mod embed;
pub mod extract;
pub mod index;
pub mod search;

pub use config::Config;
