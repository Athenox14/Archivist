#!/usr/bin/env python3
"""Compare les hash d'un dump TSV (hash<TAB>taille<TAB>chemin) et produit un
rapport de doublons EXACTS par contenu (blake3), bien plus fiable que nom+taille.

Usage : python compare_hashes.py <fichier.tsv> <rapport_sortie.md>
"""
import sys
from collections import defaultdict

TSV = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "RAPPORT_HASH.md"


def human(n):
    for u in ("o", "Ko", "Mo", "Go", "To"):
        if n < 1024 or u == "To":
            return f"{n:.1f} {u}" if u != "o" else f"{int(n)} o"
        n /= 1024


by_hash = defaultdict(list)  # hash -> [(size, rel)]
total_files = 0
total_size = 0
with open(TSV, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        h, size, rel = parts[0], int(parts[1]), parts[2]
        by_hash[h].append((size, rel))
        total_files += 1
        total_size += size

dup_bytes = 0
groups = []  # (wasted, size, count, [rels])
for h, items in by_hash.items():
    if len(items) > 1:
        size = items[0][0]
        wasted = size * (len(items) - 1)
        dup_bytes += wasted
        groups.append((wasted, size, len(items), [r for _, r in items]))
groups.sort(reverse=True)

# doublons par dossier de 1er niveau (qui se recoupe avec qui)
folder_pair = defaultdict(int)  # (a,b) -> octets dupliqués partagés
def top1(rel):
    return rel.split("/", 1)[0]
for wasted, size, cnt, rels in groups:
    tops = sorted(set(top1(r) for r in rels))
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            folder_pair[(tops[i], tops[j])] += size  # 1 copie partagée

out = []
out.append("# Rapport de doublons par HASH de contenu (blake3) — F:\\BACKUPS\n")
out.append(f"- Fichiers : **{total_files:,}**".replace(",", " "))
out.append(f"- Taille totale : **{human(total_size)}**")
out.append(f"- **Doublons exacts (contenu identique) : {human(dup_bytes)} récupérables** "
           f"(~{100*dup_bytes/total_size:.1f} %)\n")

out.append("## Top 25 contenus dupliqués\n")
out.append("| Gaspillé | Taille unité | Copies | Exemple de chemins |")
out.append("|---|---|---|---|")
for wasted, size, cnt, rels in groups[:25]:
    ex = " · ".join(rels[:2])
    if len(rels) > 2:
        ex += f" · (+{len(rels)-2})"
    ex = ex.replace("|", "/")
    out.append(f"| {human(wasted)} | {human(size)} | {cnt} | {ex} |")

out.append("\n## Dossiers (niveau 1) qui partagent le plus de contenu\n")
out.append("| Octets dupliqués communs | Dossier A | Dossier B |")
out.append("|---|---|---|")
for (a, b), bytes_ in sorted(folder_pair.items(), key=lambda x: -x[1])[:25]:
    out.append(f"| {human(bytes_)} | `{a}` | `{b}` |")

out.append(f"\n_{len(groups)} groupes de doublons au total._")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print(f"Rapport écrit : {OUT}  ({human(dup_bytes)} dupliqués)")
