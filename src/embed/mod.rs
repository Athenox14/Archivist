//! Embeddings via ONNX Runtime (`ort`).
//!
//! Pourquoi `ort` plutôt que `candle` :
//!  - Charge directement les exports ONNX HuggingFace de CLIP ViT-B/32 et
//!    multilingual-e5-small sans réécrire l'architecture ni convertir les poids.
//!  - Backend mûr, perfs CPU solides, quantization dispo.
//!  - candle = pur Rust mais demande de mapper manuellement chaque tenseur de
//!    poids ; plus fragile pour un livrable reproductible.
//!
//! Modèles attendus dans `--models` (voir README pour l'export) :
//!   clip_image.onnx, clip_text.onnx, clip_tokenizer.json
//!   e5_small.onnx,    e5_tokenizer.json

pub mod clip;
pub mod text;

use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

static EP_LOGGED: AtomicBool = AtomicBool::new(false);

/// Construit une session ONNX. CPU par défaut (stable). GPU opt-in via les
/// features `cuda` (NVIDIA) ou `directml` — ordre : CUDA → DirectML → CPU.
/// ort tombe sur l'EP suivant si l'EP demandé échoue à s'initialiser.
pub fn build_session(path: &Path) -> Result<Session> {
    use ort::ep::CPUExecutionProvider;

    let mut eps = Vec::new();
    #[cfg(feature = "cuda")]
    {
        eps.push(ort::ep::CUDAExecutionProvider::default().build());
    }
    #[cfg(feature = "directml")]
    {
        eps.push(ort::ep::DirectMLExecutionProvider::default().build());
    }
    eps.push(CPUExecutionProvider::default().build());

    if !EP_LOGGED.swap(true, Ordering::Relaxed) {
        let mut order = String::new();
        if cfg!(feature = "cuda") {
            order.push_str("CUDA → ");
        }
        if cfg!(feature = "directml") {
            order.push_str("DirectML → ");
        }
        order.push_str("CPU");
        log::info!("EP demandés : {order}");
    }

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let b = Session::builder().map_err(to_anyhow)?;
    let b = b.with_execution_providers(eps).map_err(to_anyhow)?;
    let b = b
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(to_anyhow)?;
    let mut b = b.with_intra_threads(threads).map_err(to_anyhow)?;
    b.commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("load onnx {}: {e}", path.display()))
}

/// Convertit n'importe quelle `ort::Error<R>` (payload non-Send) en `anyhow`.
fn to_anyhow<R>(e: ort::Error<R>) -> anyhow::Error {
    anyhow::anyhow!("ort: {e}")
}

/// Normalise L2 en place.
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine de deux vecteurs déjà normalisés = produit scalaire.
#[inline]
pub fn cosine_normalized(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
