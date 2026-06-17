#!/usr/bin/env python3
"""Parcours interactif des doublons (depuis hashes_backups.tsv).

Affiche les groupes un par un (du plus gros gaspillage au plus petit), avec le
DOSSIER PARENT de chaque copie (pas le fichier).

Usage :
    python dups_browser.py [tsv] [--min-mb N] [--sort waste|count|size] [--all]

Commandes pendant le parcours :
    Entrée = groupe suivant · a = tout dérouler · q = quitter
"""
import argparse
from collections import defaultdict


def human(n):
    for u in ("o", "Ko", "Mo", "Go", "To"):
        if n < 1024 or u == "To":
            return f"{n:.1f} {u}" if u != "o" else f"{int(n)} o"
        n /= 1024


def parent(rel):
    return rel.rsplit("/", 1)[0] if "/" in rel else "(racine)"


def basename(rel):
    return rel.rsplit("/", 1)[-1]


def main():
    ap = argparse.ArgumentParser(description="Parcours interactif des doublons")
    ap.add_argument("tsv", nargs="?", default="hashes_backups.tsv")
    ap.add_argument("--min-mb", type=float, default=0, help="ignorer les fichiers < N Mo")
    ap.add_argument("--sort", choices=["waste", "count", "size"], default="waste")
    ap.add_argument("--all", action="store_true", help="tout afficher sans pause")
    args = ap.parse_args()

    groups = defaultdict(list)  # hash -> [(size, rel)]
    with open(args.tsv, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 3:
                continue
            h, size, rel = p[0], int(p[1]), p[2]
            groups[h].append((size, rel))

    minb = args.min_mb * 1024 * 1024
    items = []  # (waste, size, count, [rels])
    for h, lst in groups.items():
        if len(lst) < 2:
            continue
        size = lst[0][0]
        if size < minb:
            continue
        items.append((size * (len(lst) - 1), size, len(lst), [r for _, r in lst]))

    key = {"waste": lambda x: x[0], "size": lambda x: x[1], "count": lambda x: x[2]}[args.sort]
    items.sort(key=key, reverse=True)

    total_waste = sum(w for w, _, _, _ in items)
    print(f"\n{len(items)} groupes de doublons · gaspillage total {human(total_waste)}")
    print(f"(tri par {args.sort}, min {args.min_mb} Mo)\n")

    for i, (waste, size, cnt, rels) in enumerate(items, 1):
        print(f"[{i}/{len(items)}]  {basename(rels[0])}")
        print(f"   {human(size)} l'unité · {cnt} copies · {human(waste)} gaspillé")
        for r in sorted(set(parent(r) for r in rels)):
            print(f"     - {r}")
        print()
        if not args.all:
            c = input("[Entrée] suivant · a = tout · q = quitter > ").strip().lower()
            if c == "q":
                break
            if c == "a":
                args.all = True
            print()


if __name__ == "__main__":
    main()
