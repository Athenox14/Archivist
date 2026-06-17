//! Écriture de l'archive standalone : `archive/<arbo>/fichier.ext.zst`.
//!
//! Doublons : hardlink vers le .zst canonique (pas de recompression).
//! Idempotent : skip si .zst déjà présent et à jour (mtime source <= mtime .zst).

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Chemin .zst destination pour un rel_path donné.
pub fn dest_zst(archive: &Path, rel_path: &str) -> PathBuf {
    archive.join(format!("{rel_path}.zst"))
}

/// Chemin destination SANS compression (octets exacts).
pub fn dest_plain(archive: &Path, rel_path: &str) -> PathBuf {
    archive.join(rel_path)
}

/// Copie src → dst tel quel (mode stockage). Idempotent (skip si à jour).
pub fn store_to(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    if up_to_date(src, dst) {
        return Ok(());
    }
    let tmp = dst.with_extension("tmp_copy");
    fs::copy(src, &tmp).with_context(|| format!("copy {}", src.display()))?;
    fs::rename(&tmp, dst)?;
    Ok(())
}

/// Le .zst est-il à jour vs la source ?
fn up_to_date(src: &Path, dst: &Path) -> bool {
    let (Ok(sm), Ok(dm)) = (fs::metadata(src), fs::metadata(dst)) else {
        return false;
    };
    match (sm.modified(), dm.modified()) {
        (Ok(s), Ok(d)) => s <= d,
        _ => false,
    }
}

/// Compresse src → dst (.zst), en streaming. Crée les dossiers parents.
pub fn compress_to(src: &Path, dst: &Path, level: i32) -> Result<()> {
    compress_to_opt(src, dst, level, None)
}

/// Variante avec dictionnaire zstd optionnel (option « dictionnaire partagé »).
/// Restauration : `zstd -D <dict> -d fichier.zst`.
pub fn compress_to_opt(src: &Path, dst: &Path, level: i32, dict: Option<&[u8]>) -> Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    if up_to_date(src, dst) {
        return Ok(()); // idempotent
    }
    let infile = fs::File::open(src).with_context(|| format!("open {}", src.display()))?;
    let tmp = dst.with_extension("zst.tmp");
    {
        let outfile = fs::File::create(&tmp)?;
        let mut encoder = match dict {
            Some(d) => zstd::stream::write::Encoder::with_dictionary(outfile, level, d)?,
            None => zstd::stream::Encoder::new(outfile, level)?,
        };
        let _ = num_cpus_hint();
        let mut reader = std::io::BufReader::new(infile);
        std::io::copy(&mut reader, &mut encoder)?;
        encoder.finish()?;
    }
    fs::rename(&tmp, dst)?; // atomique
    Ok(())
}

fn num_cpus_hint() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

/// Décompresse un .zst en mémoire (dict optionnel pour le texte).
pub fn decompress_bytes(zst: &Path, dict: Option<&[u8]>) -> Result<Vec<u8>> {
    let f = fs::File::open(zst)?;
    let r = std::io::BufReader::new(f);
    match dict {
        Some(d) => {
            let mut dec = zstd::stream::read::Decoder::with_dictionary(r, d)?;
            let mut out = Vec::new();
            std::io::copy(&mut dec, &mut out)?;
            Ok(out)
        }
        None => Ok(zstd::stream::decode_all(r)?),
    }
}

/// Crée un hardlink dup_dst → canonical_dst. Si hardlink échoue
/// (cross-device, FS sans support), retombe sur une copie.
pub fn hardlink_or_copy(canonical_dst: &Path, dup_dst: &Path) -> Result<()> {
    if let Some(parent) = dup_dst.parent() {
        fs::create_dir_all(parent)?;
    }
    if dup_dst.exists() {
        // idempotent : déjà lié/copié. On suppose à jour si présent.
        return Ok(());
    }
    match fs::hard_link(canonical_dst, dup_dst) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(canonical_dst, dup_dst)?;
            Ok(())
        }
    }
}
