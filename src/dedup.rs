//! Déduplication économe en compute.
//!
//! 1. Parcours source, regroupe par TAILLE EXACTE.
//! 2. Hash (blake3) UNIQUEMENT les groupes de 2+ fichiers de même taille.
//! 3. Un canonique par hash : compressé une fois ; doublons = HARDLINKS vers le .zst.
//! 4. Fichiers de taille unique : pas de hash.
//! 5. Idempotent : relancer re-déduplique sans casser.

use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Clone, Debug)]
pub struct FileEntry {
    pub abs_path: PathBuf,
    pub rel_path: String,
    pub size: u64,
    pub mtime: i64,
    /// Some si hashé (groupe de doublons potentiels), None si taille unique.
    pub hash: Option<String>,
}

/// mtime en secondes epoch (i64), 0 si indisponible.
fn mtime_secs(meta: &std::fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Hash blake3 en streaming (mémoire bornée).
pub fn hash_file(path: &Path) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    let mut f = std::fs::File::open(path)?;
    std::io::copy(&mut f, &mut hasher)?;
    Ok(hasher.finalize().to_hex().to_string())
}

/// Scanne la source et applique la stratégie de hash sélectif.
///
/// On hashe un fichier si son groupe de taille contient 2+ éléments DANS la
/// source, OU si sa taille est déjà connue ailleurs (`known_sizes`, venant de
/// la DB) — nécessaire pour dédupliquer entre sources distinctes.
pub fn scan_and_hash(
    source: &Path,
    known_sizes: &std::collections::HashSet<u64>,
    hash_all: bool,
) -> Result<Vec<FileEntry>> {
    // Phase 1 : collecte + groupage par taille.
    let mut by_size: HashMap<u64, Vec<(PathBuf, String, i64)>> = HashMap::new();
    for entry in WalkDir::new(source).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let abs = entry.path().to_path_buf();
        let rel = abs
            .strip_prefix(source)?
            .to_string_lossy()
            .replace('\\', "/"); // chemins relatifs portables
        let meta = entry.metadata()?;
        by_size
            .entry(meta.len())
            .or_default()
            .push((abs, rel, mtime_secs(&meta)));
    }

    // Phase 2 : hash UNIQUEMENT les groupes 2+, en parallèle.
    let mut entries: Vec<FileEntry> = Vec::new();
    let mut to_hash: Vec<(PathBuf, String, i64, u64)> = Vec::new();

    for (size, group) in by_size {
        // hashe si : mode dédup-globale, OU doublon de taille intra-source, OU
        // taille déjà connue ailleurs (cross-source).
        let must_hash = hash_all || group.len() > 1 || known_sizes.contains(&size);
        if must_hash {
            for (abs, rel, mtime) in group {
                to_hash.push((abs, rel, mtime, size));
            }
        } else {
            let (abs, rel, mtime) = group.into_iter().next().unwrap();
            entries.push(FileEntry {
                abs_path: abs,
                rel_path: rel,
                size,
                mtime,
                hash: None,
            });
        }
    }

    let hashed: Vec<FileEntry> = to_hash
        .par_iter()
        .map(|(abs, rel, mtime, size)| {
            let hash = hash_file(abs).ok();
            FileEntry {
                abs_path: abs.clone(),
                rel_path: rel.clone(),
                size: *size,
                mtime: *mtime,
                hash,
            }
        })
        .collect();

    entries.extend(hashed);
    Ok(entries)
}

/// Regroupe les entrées hashées par hash → liste des rel_path partageant ce contenu.
/// Premier (ordre lexicographique du rel_path) = canonique.
pub fn group_by_hash(entries: &[FileEntry]) -> HashMap<String, Vec<usize>> {
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, e) in entries.iter().enumerate() {
        if let Some(h) = &e.hash {
            groups.entry(h.clone()).or_default().push(i);
        }
    }
    for v in groups.values_mut() {
        v.sort_by(|&a, &b| entries[a].rel_path.cmp(&entries[b].rel_path));
    }
    groups
}
