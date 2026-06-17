//! Bench débit d'indexation : fichiers/s et Mo/s (dedup + compression zstd).
//! Les modèles ONNX sont absents → les embeddings sont sautés, on mesure la
//! couche archive pure.

use archivist::config::Config;
use archivist::index;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::fs;
use std::path::PathBuf;

/// Génère un corpus synthétique : `n` fichiers de `size` octets, dont une
/// fraction de doublons pour exercer la dédup.
fn make_corpus(dir: &std::path::Path, n: usize, size: usize) -> u64 {
    fs::create_dir_all(dir).unwrap();
    let unique: Vec<u8> = (0..size).map(|i| (i * 31 + 7) as u8).collect();
    let dup: Vec<u8> = vec![0xABu8; size];
    let mut total = 0u64;
    for i in 0..n {
        let mut content = if i % 3 == 0 {
            dup.clone()
        } else {
            unique.clone()
        };
        if i % 3 != 0 {
            content[0] = i as u8; // rend unique
            content[1] = (i >> 8) as u8;
        }
        let p = dir.join(format!("file_{i:05}.bin"));
        fs::write(&p, &content).unwrap();
        total += content.len() as u64;
    }
    total
}

fn bench_index(c: &mut Criterion) {
    let n = 200;
    let size = 64 * 1024; // 64 Kio/fichier
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("src");
    let total_bytes = make_corpus(&source, n, size);

    let mut group = c.benchmark_group("indexing");
    group.throughput(Throughput::Bytes(total_bytes));
    group.sample_size(10);
    // Compare l'impact du niveau zstd sur le débit (ratio/vitesse).
    for level in [3i32, 9, 19] {
        group.bench_function(format!("index_zstd{level}"), |b| {
            b.iter_batched(
                || {
                    let out = tempfile::tempdir().unwrap();
                    let archive: PathBuf = out.path().join("arc");
                    (out, archive)
                },
                |(_, archive)| {
                    fs::create_dir_all(&archive).unwrap();
                    let cfg = Config {
                        source: source.clone(),
                        archive: archive.clone(),
                        zstd_level: level,
                        db_path: archive.join(".archivist.db"),
                        models_dir: PathBuf::from("___absent___"),
                        label: None,
                    };
                    index::run(&cfg).unwrap();
                },
                criterion::BatchSize::PerIteration,
            );
        });
    }
    group.finish();
    eprintln!("corpus: {n} fichiers, {} Mio", total_bytes / (1024 * 1024));
}

criterion_group!(benches, bench_index);
criterion_main!(benches);
