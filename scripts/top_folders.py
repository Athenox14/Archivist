#!/usr/bin/env python3
"""Top des dossiers par nombre de fichiers (depuis hashes_backups.tsv).

Par défaut : compte les fichiers DIRECTEMENT dans chaque dossier.
--recursive : compte aussi tous les sous-dossiers.

Usage :
    python top_folders.py [tsv] [N] [--recursive] [--root F:\\BACKUPS]
"""
import argparse
from collections import defaultdict


MEDIA_EXTS = {
    # images
    "jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "tif", "heic", "heif", "raw",
    "cr2", "nef", "arw", "dng", "svg", "ico", "psd",
    # vidéos
    "mp4", "mov", "avi", "mkv", "m4v", "wmv", "flv", "webm", "mpg", "mpeg", "3gp", "ts", "mts",
}


def is_media(rel):
    ext = rel.rsplit(".", 1)[-1].lower() if "." in rel else ""
    return ext in MEDIA_EXTS


def human(n):
    for u in ("o", "Ko", "Mo", "Go", "To"):
        if n < 1024 or u == "To":
            return f"{n:.1f} {u}" if u != "o" else f"{int(n)} o"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("top", nargs="?", type=int, default=10, help="nombre de dossiers (def 10)")
    ap.add_argument("--tsv", default="hashes_backups.tsv")
    ap.add_argument("--recursive", action="store_true", help="inclure les sous-dossiers")
    ap.add_argument("--no-media", action="store_true", help="exclure images et vidéos du compte")
    ap.add_argument("--root", default="F:\\BACKUPS")
    args = ap.parse_args()

    count = defaultdict(int)
    size = defaultdict(int)
    total = 0
    with open(args.tsv, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 3:
                continue
            sz, rel = int(p[1]), p[2]
            if args.no_media and is_media(rel):
                continue
            total += 1
            if "/" not in rel:
                dirs = ["(racine)"]
            elif args.recursive:
                parts = rel.split("/")[:-1]
                dirs = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
            else:
                dirs = [rel.rsplit("/", 1)[0]]
            for d in dirs:
                count[d] += 1
                size[d] += sz

    mode = "récursif" if args.recursive else "direct"
    flt = " · hors média" if args.no_media else ""
    print(f"Top {args.top} dossiers par nb de fichiers ({mode}{flt}) — {total} fichiers comptés\n")
    ordered = sorted(count.items(), key=lambda x: -x[1])[: args.top]
    for i, (d, c) in enumerate(ordered, 1):
        full = args.root.rstrip("\\/") + "\\" + d.replace("/", "\\")
        print(f"{i:>3}. {c:>7} fichiers  {human(size[d]):>10}  {full}")


if __name__ == "__main__":
    main()
