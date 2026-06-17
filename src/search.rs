//! Recherche : cosine brute-force parallélisé (rayon). La DB n'est qu'un fichier
//! de données — tout le calcul vit en Rust.

use crate::config::Config;
use crate::db::Db;
use crate::embed::{clip::ClipEncoder, cosine_normalized, text::TextEncoder};
use anyhow::Result;
use rayon::prelude::*;

/// Top-k par sélection partielle O(N) (select_nth_unstable) + tri des k seuls,
/// plutôt qu'un tri complet O(N log N) de toute la collection.
fn truncate_top_k<T, F>(mut v: Vec<T>, k: usize, mut cmp: F) -> Vec<T>
where
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    let k = k.min(v.len());
    if k == 0 {
        return Vec::new();
    }
    v.select_nth_unstable_by(k - 1, &mut cmp);
    v.truncate(k);
    v.sort_unstable_by(&mut cmp);
    v
}

#[derive(Debug, Clone)]
pub struct ImageHit {
    pub rel_path: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct DocHit {
    pub rel_path: String,
    pub chunk_idx: i64,
    pub score: f32,
    pub snippet: String,
}

pub struct SearchResults {
    pub images: Vec<ImageHit>,
    pub docs: Vec<DocHit>,
}

/// Score les images d'UNE base contre un vecteur requête CLIP déjà embeddé.
pub fn score_images(db: &Db, qv: &[f32], top_k: usize) -> Result<Vec<ImageHit>> {
    let rows = db.all_image_embeddings()?;
    let scored: Vec<ImageHit> = rows
        .par_iter()
        .map(|(path, v)| ImageHit {
            rel_path: path.clone(),
            score: cosine_normalized(qv, v),
        })
        .collect();
    Ok(truncate_top_k(scored, top_k, |a, b| {
        b.score.total_cmp(&a.score)
    }))
}

/// Score les chunks texte d'UNE base contre un vecteur requête e5 déjà embeddé.
pub fn score_docs(db: &Db, qv: &[f32], top_k: usize) -> Result<Vec<DocHit>> {
    let rows = db.all_text_chunks()?;
    let scored: Vec<DocHit> = rows
        .par_iter()
        .map(|(path, idx, text, v)| DocHit {
            rel_path: path.clone(),
            chunk_idx: *idx,
            score: cosine_normalized(qv, v),
            snippet: snippet(text),
        })
        .collect();
    Ok(truncate_top_k(scored, top_k, |a, b| {
        b.score.total_cmp(&a.score)
    }))
}

/// Fusionne et garde le top-k global (résultats de plusieurs bases).
pub fn merge_top_k<T, F>(items: Vec<T>, k: usize, cmp: F) -> Vec<T>
where
    F: FnMut(&T, &T) -> std::cmp::Ordering,
{
    truncate_top_k(items, k, cmp)
}

/// Barre unique mono-base (CLI) : embedde la requête puis score images + docs.
pub fn search(cfg: &Config, query: &str, top_k: usize) -> Result<SearchResults> {
    let db = Db::open(&cfg.db_path)?;
    let mut images = Vec::new();
    if let Ok(mut clip) = ClipEncoder::load(&cfg.models_dir) {
        let qv = clip.embed_text(query)?;
        images = score_images(&db, &qv, top_k)?;
    }
    let mut docs = Vec::new();
    if let Ok(mut enc) = TextEncoder::load(&cfg.models_dir) {
        let qv = enc.embed_query(query)?;
        docs = score_docs(&db, &qv, top_k)?;
    }
    Ok(SearchResults { images, docs })
}

/// Découpe une requête en tokens minuscules pour le matching de noms.
pub fn query_tokens(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Un nom de fichier matche-t-il (substring d'au moins un token) ?
pub fn name_matches(rel_path: &str, tokens: &[String]) -> bool {
    if tokens.is_empty() {
        return false;
    }
    let low = rel_path.to_lowercase();
    tokens.iter().any(|t| low.contains(t.as_str()))
}

pub fn snippet(text: &str) -> String {
    let s: String = text.chars().take(160).collect();
    if text.chars().count() > 160 {
        format!("{s}…")
    } else {
        s
    }
}
