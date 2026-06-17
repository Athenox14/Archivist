# Archivist

Outil d'archivage **standalone** avec déduplication, **recherche sémantique**
(texte→image et texte→texte) et **vérification d'intégrité**.

- L'archive est lisible **sans Archivist** : chaque fichier est soit stocké tel
  quel, soit compressé en zstd standard (`fichier.ext.zst`). Une simple commande
  `zstd -d` et une visionneuse suffisent à tout restaurer.
- Une base SQLite séparée porte **uniquement** la couche de recherche et les
  hashs d'intégrité. L'archive fonctionne sans elle ; la base l'enrichit.
- Interface en ligne de commande **et** interface graphique.

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Modèles d'embedding](#modèles-dembedding)
- [Utilisation — CLI](#utilisation--cli)
- [Utilisation — GUI](#utilisation--gui)
- [Comment ça marche](#comment-ça-marche)
- [Scripts d'analyse](#scripts-danalyse)
- [Avertissement hardlinks](#avertissement-hardlinks)
- [Développement](#développement)

## Fonctionnalités

- **Deux modes de stockage** : compression zstd (niveau adaptatif) ou copie brute
  sans compression.
- **Déduplication** par contenu (blake3) : un fichier identique présent plusieurs
  fois n'est stocké qu'une fois, les autres deviennent des *hardlinks*. Fonctionne
  entre sources et entre exécutions.
- **Recherche sémantique** :
  - texte → image via CLIP ViT-B/32 ;
  - texte → texte via multilingual-e5-small ;
  - recherche par nom de fichier.
- **Vérification d'intégrité** : re-hash et comparaison au hash enregistré pour
  détecter corruption (bit-rot) et fichiers manquants.
- **Reconstruction de la base** depuis l'archive seule (sans la source).
- **Mise à jour automatique** au démarrage de l'interface graphique.

## Installation

```bash
cargo build --release
```

Binaires produits : `archivist` (CLI) et `gui` (interface graphique), dans
`target/release/`.

Prérequis : toolchain Rust et, sous Windows, les Build Tools MSVC. ONNX Runtime
est téléchargé automatiquement par le crate `ort` à la compilation.

### Accélération GPU (optionnelle)

```bash
cargo build --release --features cuda      # GPU NVIDIA (CUDA/cuDNN compatibles requis)
cargo build --release --features directml  # tout GPU Windows DX12
```

Par défaut, l'inférence tourne sur CPU (stable sur tous les modèles).

## Modèles d'embedding

Les modèles ONNX et leurs tokenizers sont attendus dans `models/` :

```
clip_image.onnx  clip_text.onnx  clip_tokenizer.json
e5_small.onnx    e5_tokenizer.json
```

Pour les générer :

```bash
pip install torch transformers fpdf2
python scripts/export_models.py
```

Sans modèles, l'indexation fonctionne quand même (compression, déduplication,
intégrité) ; seuls les embeddings sont sautés.

## Utilisation — CLI

### Indexer

```bash
# Compression zstd (défaut, niveau 9)
archivist index --source /data/photos --archive /backup --level 9 --models models

# Stockage brut (sans compression) + hash complet de tout
archivist index --source /data/photos --archive /backup --no-compress
```

La source est rangée sous `archive/<nom_source>/…` avec une base unique, ce qui
permet la déduplication entre plusieurs sources indexées dans la même racine.

### Rechercher

```bash
archivist search --archive /backup --query "chat roux sur un canapé" --top-k 10
```

Deux jeux de résultats : Images (CLIP) et Documents (e5).

### Vérifier l'intégrité

```bash
archivist verify --archive /backup
# → Intégrité : N OK · M CORROMPUS · K MANQUANTS
```

### Reconstruire la base depuis l'archive

```bash
archivist populate --archive /backup --models models
```

### Lister les hashs (déduplication / corruption hors ligne)

```bash
archivist hashdump --source /data --out hashes.tsv
```

## Utilisation — GUI

```bash
./target/release/gui
```

Trois onglets :

- **Recherche** : barre unique, résultats Images (vignettes, double-clic = ouvrir)
  + Documents + Noms de fichiers. Recherche récursive sur toutes les bases sous le
  dossier indiqué.
- **Indexer** : source, archive, niveau zstd ou case « sans compression ».
- **État** : compteurs (fichiers, embeddings, chunks) et bouton **Vérifier
  l'intégrité**.

Au démarrage, l'application vérifie une éventuelle mise à jour et l'installe.

## Comment ça marche

### Déduplication (économe en lecture)

1. Parcours de la source, regroupement par **taille exacte** (`stat`, aucune
   lecture de contenu).
2. Hash (blake3) **uniquement** des fichiers en collision de taille, en cascade :
   préfixe (16 Kio) → suffixe (16 Kio) → hash progressif par blocs avec
   élimination précoce (un fichier n'est lu que jusqu'à ce qu'il devienne unique).
3. Un canonique par hash : stocké/compressé une fois ; les doublons deviennent des
   hardlinks.
4. Idempotent : relancer ne retraite que le nouveau ou le modifié.

### Niveau de compression adaptatif

- Formats déjà compressés (jpg, mp4, zip…) → niveau bas (gain nul, rapide).
- Formats très compressibles (txt, csv, bmp, wav…) → niveau élevé.
- Reste → niveau demandé (`--level`).

Un dictionnaire zstd partagé est entraîné sur les fichiers texte quand ils sont
assez nombreux (restauration : `zstd -D archive/.archivist.dict -d fichier.zst`).

### Schéma SQLite (chemins relatifs)

```
files(id, rel_path, hash, size, mtime)
image_embeddings(file_id, vec)          -- f32[]
text_chunks(file_id, chunk_idx, text, vec)
```

### Recherche

La requête est embeddée par CLIP-texte et par e5 ; la similarité cosinus est
calculée en force brute, parallélisée. La base reste un simple fichier de données.

## Scripts d'analyse

Dans `scripts/` (lisent un TSV produit par `hashdump`) :

- `top_folders.py` — top des dossiers par nombre de fichiers (`--no-media`,
  `--recursive`).
- `short_files.py` — fichiers au nom ou au contenu très court.
- `dups_folders.py` — dossiers entièrement redondants et fichiers en double, avec
  suppression sûre (`--delete-all`, vérification d'existence des copies).
- `dups_browser.py` — parcours interactif des groupes de doublons.
- `analyze_similar.py` — similarité entre dossiers par (nom, taille).
- `compare_hashes.py` — rapport de doublons par contenu.

## Avertissement hardlinks

L'archive utilise des hardlinks pour les doublons. Une copie naïve (sans
préservation des liens) casse la déduplication et fait exploser la taille.
Pour déplacer l'archive, utiliser un outil qui préserve les hardlinks
(`cp -a`, `rsync -aH`, `robocopy`) ou relancer la déduplication à destination.

## Développement

```bash
cargo test
cargo bench          # criterion : débit d'indexation + latence de requête
cargo clippy --all-targets
```

Benchmarks détaillés : voir `BENCHMARKS.md`.

## Licence

MIT.
