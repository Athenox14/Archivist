# Benchmarks

Mesures `criterion` sur le poste de dev (Windows 11, x86_64, build `--release`,
LTO thin). Reproduire :

```bash
cargo bench --bench indexing
cargo bench --bench query
```

Rapports HTML détaillés : `target/criterion/`.

> Données de bench **synthétiques** (corpus généré, vecteurs déterministes). Les
> *débits* et *latences* sont représentatifs ; les *ratios de compression* ne le
> sont pas (le corpus de bench est partiellement incompressible).

## 1. Débit d'indexation (dedup + compression zstd)

Corpus : 200 fichiers × 64 Kio = 12 Mio, ~1/3 de doublons (exerce dédup + hardlinks).
Embeddings désactivés (modèles absents) → mesure la couche archive pure.

| Niveau zstd | Temps médian | Débit       | Fichiers/s |
|-------------|--------------|-------------|------------|
| 3           | ~183 ms      | 65.5 MiB/s  | ~1090      |
| **9 (défaut)** | **244 ms** | **51.1 MiB/s** | **~818** |
| 19          | 913 ms       | 13.7 MiB/s  | ~219       |

**Lecture** : passer de 9→3 gagne ~+28 % de débit ; 9→19 coûte ~3.7× en temps.
Le niveau 9 par défaut est un bon compromis ; baisser à 3 pour des sources
volumineuses peu compressibles, monter à 19 pour archivage froid.

## 2. Latence de requête (cosine brute-force + top-k)

Vecteurs 512-d normalisés, top-k = 10, parallélisé rayon.

| N vecteurs | Baseline (tri complet) | Optimisé (sélection partielle) | Gain   |
|------------|------------------------|--------------------------------|--------|
| 1 000      | ~112 µs                | ~110 µs                        | ~0 %   |
| 10 000     | 645 µs                 | **507 µs**                     | **−21 %** |
| 100 000    | 7.32 ms                | 7.27 ms                        | ~0 %   |

## Passes d'optimisation

### Baseline
Scoring `par_iter` + **tri complet** `par_sort_unstable` O(N log N) puis `truncate(k)`.

### Passe A — sélection partielle du top-k (appliquée à `search.rs`)
Remplace le tri complet par `select_nth_unstable_by` O(N) + tri des `k` éléments
seuls. **Avant/après** : 10 000 vecteurs 645 µs → 507 µs (**−21 %**).

- Neutre à N=1 000 (dominé par l'overhead rayon / allocation du `Vec`).
- Neutre à N=100 000 : à grande échelle, le coût est **dominé par les produits
  scalaires eux-mêmes** (memory-bound, ~51M mult pour 100k×512), pas par le tri.
  Le tri ne pesait que ~2 % du temps total → l'optimiser n'aide plus.

### Passe B — niveau zstd (paramètre, pas code)
Bench des niveaux 3 / 9 / 19 (tableau §1). Confirme que le débit d'indexation est
**dominé par la compression** ; le niveau est le levier principal. Choix exposé via
`--level`.

### Piste C (non retenue ici)
À N≫100k, viser le kernel cosine : layout aplati contigu (`Vec<f32>` unique +
offsets) pour la localité cache, et SIMD explicite. Gain attendu sur la passe
memory-bound ; non implémenté faute de corpus à cette échelle dans les tests.
