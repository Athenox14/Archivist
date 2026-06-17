//! CLIP OpenAI ViT-B/32 — encodeurs image et texte (ort).
//!
//! Image : resize 224, center-crop, normalisation CLIP, NCHW f32.
//! Texte : tokenizer CLIP (BPE), contexte 77, encodeur texte.
//! Sortie : embedding 512-d, normalisé L2 → comparable en cosine.

use anyhow::{Context, Result};
use image::imageops::FilterType;
use ort::session::{Session, SessionOutputs};
use ort::value::Value;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Plafonne la mémoire d'un décodage image → un en-tête géant/corrompu échoue
/// proprement au lieu d'allouer des Go (évite le pic RAM / blocage).
fn decode_limits() -> image::Limits {
    let mut l = image::Limits::no_limits();
    l.max_alloc = Some(256 * 1024 * 1024); // 256 Mo max par image
    l.max_image_width = Some(30_000);
    l.max_image_height = Some(30_000);
    l
}

const SIZE: u32 = 224;
const CTX: usize = 77;
// Constantes de normalisation CLIP (RGB).
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_72];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

pub struct ClipEncoder {
    image_session: Session,
    text_session: Session,
    tokenizer: Tokenizer,
}

impl ClipEncoder {
    pub fn load(models_dir: &Path) -> Result<Self> {
        let image_session = super::build_session(&models_dir.join("clip_image.onnx"))?;
        let text_session = super::build_session(&models_dir.join("clip_text.onnx"))?;
        let tokenizer = Tokenizer::from_file(models_dir.join("clip_tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("clip tokenizer: {e}"))?;
        Ok(Self {
            image_session,
            text_session,
            tokenizer,
        })
    }

    /// Prétraite une image disque → Vec f32 NCHW [1,3,224,224].
    fn preprocess_image(path: &Path) -> Result<Vec<f32>> {
        let mut r = image::ImageReader::open(path)
            .with_context(|| format!("open image {}", path.display()))?
            .with_guessed_format()?;
        r.limits(decode_limits());
        Ok(Self::preprocess_rgb(&r.decode()?.to_rgb8()))
    }

    /// Idem depuis des octets en mémoire (image décompressée d'une archive).
    fn preprocess_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
        let mut r = image::ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()
            .context("format image")?;
        r.limits(decode_limits());
        Ok(Self::preprocess_rgb(&r.decode()?.to_rgb8()))
    }

    fn preprocess_rgb(img: &image::RgbImage) -> Vec<f32> {
        // resize plus petit côté à 224 puis center-crop
        let (w, h) = img.dimensions();
        let scale = SIZE as f32 / w.min(h) as f32;
        let nw = (w as f32 * scale).round().max(SIZE as f32) as u32;
        let nh = (h as f32 * scale).round().max(SIZE as f32) as u32;
        let resized = image::imageops::resize(img, nw, nh, FilterType::CatmullRom);
        let x0 = (nw - SIZE) / 2;
        let y0 = (nh - SIZE) / 2;

        let n = (SIZE * SIZE) as usize;
        let mut arr = vec![0f32; 3 * n];
        for y in 0..SIZE {
            for x in 0..SIZE {
                let px = resized.get_pixel(x0 + x, y0 + y);
                let idx = (y * SIZE + x) as usize;
                for c in 0..3 {
                    let v = px[c] as f32 / 255.0;
                    arr[c * n + idx] = (v - MEAN[c]) / STD[c]; // plan c, NCHW
                }
            }
        }
        arr
    }

    /// Embedding image (512-d, L2-normalisé).
    pub fn embed_image(&mut self, path: &Path) -> Result<Vec<f32>> {
        Ok(self
            .embed_images(std::slice::from_ref(&path.to_path_buf()))?
            .into_iter()
            .next()
            .flatten()
            .context("embedding image vide")?)
    }

    /// Embedding image PAR LOT : prétraitement parallèle (rayon) + une seule
    /// passe ONNX sur le batch → amortit l'overhead, exploite le GPU.
    /// Renvoie un `None` aux positions dont l'image n'a pas pu être lue.
    pub fn embed_images(&mut self, paths: &[PathBuf]) -> Result<Vec<Option<Vec<f32>>>> {
        let pre: Vec<Option<Vec<f32>>> = paths
            .par_iter()
            .map(|p| Self::preprocess_image(p).ok())
            .collect();
        self.run_batch(pre)
    }

    /// Idem mais depuis des images en mémoire (octets décompressés d'archive).
    pub fn embed_images_bytes(&mut self, images: &[Vec<u8>]) -> Result<Vec<Option<Vec<f32>>>> {
        let pre: Vec<Option<Vec<f32>>> = images
            .par_iter()
            .map(|b| Self::preprocess_bytes(b).ok())
            .collect();
        self.run_batch(pre)
    }

    /// Lance une passe ONNX sur un lot de tenseurs prétraités.
    fn run_batch(&mut self, pre: Vec<Option<Vec<f32>>>) -> Result<Vec<Option<Vec<f32>>>> {
        let mut batch: Vec<f32> = Vec::new();
        let mut orig_idx: Vec<usize> = Vec::new();
        for (i, pp) in pre.iter().enumerate() {
            if let Some(v) = pp {
                batch.extend_from_slice(v);
                orig_idx.push(i);
            }
        }
        let mut out = vec![None; pre.len()];
        let n = orig_idx.len();
        if n == 0 {
            return Ok(out);
        }

        let shape = vec![n as i64, 3, SIZE as i64, SIZE as i64];
        let input = Value::from_array((shape, batch))?;
        let outputs = self
            .image_session
            .run(ort::inputs!["pixel_values" => input])?;
        let (_, first) = outputs.iter().next().context("no output")?;
        let (_shape, data) = first.try_extract_tensor::<f32>()?;
        let dim = data.len() / n;

        for (k, &orig) in orig_idx.iter().enumerate() {
            let mut v = data[k * dim..(k + 1) * dim].to_vec();
            super::l2_normalize(&mut v);
            out[orig] = Some(v);
        }
        Ok(out)
    }

    /// Embedding texte CLIP (512-d, L2-normalisé). Sert aux requêtes.
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("clip encode: {e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        let mut mask: Vec<i64> = enc.get_attention_mask().iter().map(|&i| i as i64).collect();
        ids.truncate(CTX);
        mask.truncate(CTX);
        ids.resize(CTX, 0);
        mask.resize(CTX, 0);

        let shape = vec![1i64, CTX as i64];
        let ids_t = Value::from_array((shape.clone(), ids))?;
        let mask_t = Value::from_array((shape, mask))?;
        let outputs = self
            .text_session
            .run(ort::inputs!["input_ids" => ids_t, "attention_mask" => mask_t])?;
        let mut v = first_embedding(&outputs)?;
        super::l2_normalize(&mut v);
        Ok(v)
    }
}

/// Extrait le premier tenseur de sortie aplati en Vec<f32>.
fn first_embedding(outputs: &SessionOutputs) -> Result<Vec<f32>> {
    let (_, first) = outputs.iter().next().context("no output")?;
    let (_shape, data) = first.try_extract_tensor::<f32>()?;
    Ok(data.to_vec())
}
