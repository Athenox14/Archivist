//! multilingual-e5-small — encodeur texte (ort).
//!
//! e5 attend des préfixes : "query: " pour les requêtes, "passage: " pour les
//! documents indexés. Pooling = moyenne masquée des hidden states. Sortie 384-d,
//! L2-normalisée → cosine.
//!
//! Alternative plus simple notée au README : static-similarity-mrl-multilingual-v1
//! via model2vec-rs (pas d'ONNX runtime, lookup + moyenne). Choix retenu ici : e5
//! pour la qualité contextuelle ; model2vec garde une porte de sortie si la chaîne
//! ONNX pose problème sur la cible.

use anyhow::{Context, Result};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use tokenizers::Tokenizer;

const MAX_LEN: usize = 512;

pub struct TextEncoder {
    session: Session,
    tokenizer: Tokenizer,
}

impl TextEncoder {
    pub fn load(models_dir: &Path) -> Result<Self> {
        let session = super::build_session(&models_dir.join("e5_small.onnx"))?;
        let tokenizer = Tokenizer::from_file(models_dir.join("e5_tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("e5 tokenizer: {e}"))?;
        Ok(Self { session, tokenizer })
    }

    pub fn embed_passage(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed(&format!("passage: {text}"))
    }

    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed(&format!("query: {text}"))
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("e5 encode: {e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&i| i as i64).collect();
        let mut mask: Vec<i64> = enc.get_attention_mask().iter().map(|&i| i as i64).collect();
        ids.truncate(MAX_LEN);
        mask.truncate(MAX_LEN);
        let len = ids.len();
        let shape = vec![1i64, len as i64];

        let mask_v = mask.clone();
        let ids_t = Value::from_array((shape.clone(), ids))?;
        let mask_t = Value::from_array((shape.clone(), mask))?;
        let type_ids = Value::from_array((shape, vec![0i64; len]))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
            "token_type_ids" => type_ids,
        ])?;

        // last_hidden_state : [1, len, hidden]
        let (_, first) = outputs.iter().next().context("no output")?;
        let (shape, data) = first.try_extract_tensor::<f32>()?;
        let hidden = shape[shape.len() - 1] as usize;

        // mean pooling masqué
        let mut pooled = vec![0f32; hidden];
        let mut count = 0f32;
        for (t, &m) in mask_v.iter().enumerate() {
            if m == 0 {
                continue;
            }
            count += 1.0;
            let base = t * hidden;
            for h in 0..hidden {
                pooled[h] += data[base + h];
            }
        }
        if count > 0.0 {
            for v in pooled.iter_mut() {
                *v /= count;
            }
        }
        super::l2_normalize(&mut pooled);
        Ok(pooled)
    }
}
