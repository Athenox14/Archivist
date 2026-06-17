#!/usr/bin/env python3
"""Liste les fichiers au NOM trop court ou au CONTENU trop court (peu d'octets).

Pratique pour repérer des fichiers douteux/vides/temporaires.

Usage :
    python short_files.py [--tsv hashes_backups.tsv] [--name-max N] [--size-max N]
                          [--top N] [--root F:\\BACKUPS]

    --name-max N  nom (sans extension) de N caractères ou moins (def 3)
    --size-max N  contenu de N octets ou moins (def 16)
    --top N       limite d'affichage par section (def 80)
"""
import argparse
import os


def human(n):
    for u in ("o", "Ko", "Mo", "Go"):
        if n < 1024 or u == "Go":
            return f"{n:.0f} {u}" if u == "o" else f"{n:.1f} {u}"
        n /= 1024


def stem(rel):
    base = rel.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="hashes_backups.tsv")
    ap.add_argument("--name-max", type=int, default=3)
    ap.add_argument("--size-max", type=int, default=16)
    ap.add_argument("--top", type=int, default=80)
    ap.add_argument("--root", default="F:\\BACKUPS")
    args = ap.parse_args()

    short_name = []  # (namelen, size, rel)
    short_size = []  # (size, rel)
    total = 0
    with open(args.tsv, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 3:
                continue
            size, rel = int(p[1]), p[2]
            total += 1
            nl = len(stem(rel))
            if nl <= args.name_max:
                short_name.append((nl, size, rel))
            if size <= args.size_max:
                short_size.append((size, rel))

    def full(rel):
        return args.root.rstrip("\\/") + "\\" + rel.replace("/", "\\")

    print(f"{total} fichiers analysés\n")

    print(f"=== NOM court (≤ {args.name_max} car. hors extension) : {len(short_name)} ===")
    for nl, size, rel in sorted(short_name)[: args.top]:
        print(f"  nom={nl:>2}c  {human(size):>8}  {full(rel)}")
    if len(short_name) > args.top:
        print(f"  … (+{len(short_name)-args.top})")

    print(f"\n=== CONTENU court (≤ {args.size_max} octets) : {len(short_size)} ===")
    for size, rel in sorted(short_size)[: args.top]:
        print(f"  {size:>4} o  {full(rel)}")
    if len(short_size) > args.top:
        print(f"  … (+{len(short_size)-args.top})")


if __name__ == "__main__":
    main()
