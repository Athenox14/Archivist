//! Bench latence de requête : cosine brute-force parallélisé (rayon) + top-k.
//! C'est le cœur du coût de recherche une fois la requête embeddée.

use archivist::embed::cosine_normalized;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rayon::prelude::*;

const DIM: usize = 512;

fn synth_vectors(n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            let mut v: Vec<f32> = (0..DIM)
                .map(|d| ((i * 131 + d * 17) % 1000) as f32)
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect()
}

/// BASELINE : score tout + tri complet O(N log N).
fn topk_full_sort(query: &[f32], db: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = db
        .par_iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_normalized(query, v)))
        .collect();
    scored.par_sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored
}

/// OPTIMISÉ : score tout + sélection partielle O(N) (select_nth_unstable),
/// puis tri des k seuls. Évite de trier toute la collection.
fn topk_partial(query: &[f32], db: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = db
        .par_iter()
        .enumerate()
        .map(|(i, v)| (i, cosine_normalized(query, v)))
        .collect();
    let k = k.min(scored.len());
    if k == 0 {
        return Vec::new();
    }
    scored.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

fn bench_query(c: &mut Criterion) {
    for &n in &[1_000usize, 10_000, 100_000] {
        let db = synth_vectors(n);
        let query = db[0].clone();

        let mut g = c.benchmark_group("query_cosine_baseline");
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| topk_full_sort(&query, &db, 10));
        });
        g.finish();

        let mut g = c.benchmark_group("query_cosine_partial");
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| topk_partial(&query, &db, 10));
        });
        g.finish();
    }
}

criterion_group!(benches, bench_query);
criterion_main!(benches);
