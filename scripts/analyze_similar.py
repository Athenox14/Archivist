#!/usr/bin/env python3
"""Analyse de redondance d'un dossier de backups (métadonnées seules, sans lire
le contenu des fichiers).

- Doublons EXACTS estimés par (nom de fichier, taille).
- Similarité entre dossiers (jusqu'à une profondeur donnée) via les fichiers
  communs (chemin relatif + taille), avec overlap % / Jaccard % / octets partagés.

Sort un rapport Markdown sur stdout. Usage :
    python analyze_similar.py <racine> [profondeur_max=2]
"""
import os
import sys
from collections import defaultdict
from itertools import combinations

ROOT = sys.argv[1]
MAX_DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 2

IGNORE_NAMES = {".ds_store", "thumbs.db", "desktop.ini", ".localized"}
IGNORE_PREFIX = ("._",)
IGNORE_DIRS = {".spotlight-v100", ".fseventsd", ".trashes", ".documentrevisions-v100",
               "$recycle.bin", "system volume information"}


def human(n):
    for u in ("o", "Ko", "Mo", "Go", "To"):
        if n < 1024 or u == "To":
            return f"{n:.1f} {u}" if u != "o" else f"{int(n)} o"
        n /= 1024


def keep(name):
    low = name.lower()
    if low in IGNORE_NAMES or low.startswith(IGNORE_PREFIX):
        return False
    return True


# --- 1. inventaire complet (un seul walk) ---
all_files = []  # (relpath_lower, size, parent_dir_abs)
total_size = 0
n_files = 0
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d.lower() not in IGNORE_DIRS]
    for fn in filenames:
        if not keep(fn):
            continue
        full = os.path.join(dirpath, fn)
        try:
            size = os.path.getsize(full)
        except OSError:
            continue
        rel = os.path.relpath(full, ROOT).replace("\\", "/").lower()
        all_files.append((rel, size, dirpath))
        total_size += size
        n_files += 1

# --- 2. doublons exacts par (basename, taille) ---
by_keysig = defaultdict(list)
for rel, size, _ in all_files:
    base = rel.rsplit("/", 1)[-1]
    if size > 0:
        by_keysig[(base, size)].append(rel)

dup_bytes = 0
dup_groups = []
for (base, size), rels in by_keysig.items():
    if len(rels) > 1:
        wasted = size * (len(rels) - 1)
        dup_bytes += wasted
        dup_groups.append((wasted, base, size, len(rels)))
dup_groups.sort(reverse=True)

# --- 3. signatures récursives par dossier (profondeur <= MAX_DEPTH) ---
def depth_of(absdir):
    rel = os.path.relpath(absdir, ROOT)
    if rel == ".":
        return 0
    return rel.count(os.sep) + 1

# sélectionne les dossiers candidats
candidate_dirs = set()
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d.lower() not in IGNORE_DIRS]
    d = depth_of(dirpath)
    if 1 <= d <= MAX_DEPTH:
        candidate_dirs.add(dirpath)

# signature[d] = set de (relpath_dans_d, size) ; size totale du dossier
sig = {d: set() for d in candidate_dirs}
bytes_of = defaultdict(int)
for rel_root, size, parent in all_files:
    full = os.path.join(ROOT, rel_root.replace("/", os.sep))
    for d in candidate_dirs:
        dn = d + os.sep
        if full.lower().startswith(dn.lower()):
            inner = os.path.relpath(full, d).replace("\\", "/").lower()
            sig[d].add((inner, size))
            bytes_of[d] += size

# ne garde que les dossiers non triviaux
dirs = [d for d in candidate_dirs if len(sig[d]) >= 5]


def is_ancestor(a, b):
    a2, b2 = a.lower().rstrip(os.sep) + os.sep, b.lower().rstrip(os.sep) + os.sep
    return b2.startswith(a2) or a2.startswith(b2)


pairs = []
for a, b in combinations(dirs, 2):
    if is_ancestor(a, b):
        continue
    sa, sb = sig[a], sig[b]
    # filtre rapide : tailles très différentes → peu d'intérêt
    if len(sa) == 0 or len(sb) == 0:
        continue
    inter = sa & sb
    if len(inter) < 5:
        continue
    shared_bytes = sum(s for _, s in inter)
    overlap = len(inter) / min(len(sa), len(sb))
    jacc = len(inter) / len(sa | sb)
    if overlap >= 0.30:
        pairs.append((shared_bytes, overlap, jacc, a, b, len(inter), len(sa), len(sb)))
pairs.sort(reverse=True)


def relname(d):
    return os.path.relpath(d, ROOT).replace("\\", "/")


# --- rapport markdown ---
out = []
out.append(f"# Rapport de redondance — {ROOT}\n")
out.append(f"- Fichiers analysés : **{n_files:,}**".replace(",", " "))
out.append(f"- Taille totale : **{human(total_size)}**")
out.append(f"- Dossiers comparés (prof. ≤ {MAX_DEPTH}) : **{len(dirs)}**\n")

out.append("## Doublons exacts (même nom + même taille)\n")
out.append(f"Espace récupérable estimé par dédup fichier-entier : **{human(dup_bytes)}** "
           f"(~{100*dup_bytes/total_size:.1f} % du total).\n")
out.append("Top 15 fichiers dupliqués (octets gaspillés) :\n")
out.append("| Gaspillé | Fichier | Taille unité | Copies |")
out.append("|---|---|---|---|")
for wasted, base, size, cnt in dup_groups[:15]:
    out.append(f"| {human(wasted)} | `{base}` | {human(size)} | {cnt} |")

out.append("\n## Dossiers similaires\n")
if not pairs:
    out.append("_Aucune paire de dossiers avec recouvrement ≥ 30 %._")
else:
    out.append("overlap = part du plus petit dossier présente dans l'autre ; "
               "Jaccard = intersection/union.\n")
    out.append("| Octets communs | overlap | Jaccard | Dossier A | Dossier B |")
    out.append("|---|---|---|---|---|")
    for shb, ov, ja, a, b, ni, na, nb in pairs[:40]:
        flag = " ⊂" if ov >= 0.97 else ""
        out.append(f"| {human(shb)} | {ov*100:.0f}% | {ja*100:.0f}% | "
                   f"`{relname(a)}`{flag} | `{relname(b)}` |")
    out.append("\n⊂ = le plus petit dossier est quasi entièrement inclus dans l'autre.")

print("\n".join(out))
print(f"\n<<<DUPBYTES={dup_bytes};TOTAL={total_size};PAIRS={len(pairs)}>>>", file=sys.stderr)
