//! Extraction de texte (.txt/.md/.pdf) et découpage en chunks.

use anyhow::Result;
use std::path::Path;

/// Extrait le texte brut d'un fichier supporté.
pub fn extract_text(path: &Path) -> Result<String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "txt" | "md" => Ok(std::fs::read_to_string(path)?),
        "pdf" => Ok(pdf_extract::extract_text(path)?),
        _ => Ok(String::new()),
    }
}

/// Découpe en chunks par paragraphe (séparés par lignes vides), en
/// fusionnant les courts pour viser ~`target_chars`. Filtre le vide.
pub fn chunk_text(text: &str, target_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for para in text.split("\n\n") {
        let p = para.trim();
        if p.is_empty() {
            continue;
        }
        if cur.len() + p.len() + 1 > target_chars && !cur.is_empty() {
            chunks.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(p);
        // paragraphe géant seul → on le pousse tel quel
        if cur.len() >= target_chars {
            chunks.push(std::mem::take(&mut cur));
        }
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }
    chunks
}
