#!/usr/bin/env python3
"""Dossiers entièrement redondants (supprimables sans perte) puis fichiers en
double restants. Chemins ABSOLUS. Calcul en tâche de fond (le 1er s'affiche
pendant que la suite se prépare). Vérifie l'existence avant d'afficher.

Usage :
    python dups_folders.py [tsv] [--root F:\\BACKUPS] [--min-mb N]
                           [--files-top N] [--files-min-mb N] [--paths] [--yes]

Pendant le parcours :
    Entrée = suivant · s = SUPPRIMER l'élément affiché · a = tout afficher · q = quitter
"""
import argparse
import os
import queue
import shutil
import threading
from bisect import bisect_left
from collections import defaultdict


def human(n):
    for u in ("o", "Ko", "Mo", "Go", "To"):
        if n < 1024 or u == "To":
            return f"{n:.1f} {u}" if u != "o" else f"{int(n)} o"
        n /= 1024


def ancestors_incl(d):
    parts = d.split("/")
    for i in range(1, len(parts) + 1):
        yield "/".join(parts[:i])


def lca(dirs):
    common = []
    for tup in zip(*[d.split("/") for d in dirs]):
        if len(set(tup)) == 1:
            common.append(tup[0])
        else:
            break
    return "/".join(common)


def parent(p):
    return p.rsplit("/", 1)[0] if "/" in p else ""


def under(q, p):
    return q == p or q.startswith(p + "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="?", default="hashes_backups.tsv")
    ap.add_argument("--root", default="F:\\BACKUPS", help="racine réelle des chemins")
    ap.add_argument("--min-mb", type=float, default=1)
    ap.add_argument("--files-top", type=int, default=100)
    ap.add_argument("--files-min-mb", type=float, default=0)
    ap.add_argument("--paths", action="store_true")
    ap.add_argument("--yes", action="store_true", help="ne pas reconfirmer chaque suppression")
    ap.add_argument("--delete-all", action="store_true",
                    help="supprime TOUT le set redondant (après vérif d'existence des copies)")
    ap.add_argument("--no-verify", action="store_true",
                    help="avec --delete-all : ne pas revérifier les copies sur disque (+ rapide, - sûr)")
    args = ap.parse_args()

    root = args.root.rstrip("\\/")

    def absp(rel):
        return os.path.join(root, rel.replace("/", os.sep))

    # --- chargement ---
    hashes = defaultdict(list)
    info = {}
    paths = []
    with open(args.tsv, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 3:
                continue
            h, size, rel = p[0], int(p[1]), p[2]
            hashes[h].append(rel)
            info[rel] = (size, h)
            paths.append(rel)
    paths.sort()

    dir_size = defaultdict(int)
    dirs_with_files = set()
    for rel, (size, _) in info.items():
        par = parent(rel)
        if par:
            for a in ancestors_incl(par):
                dir_size[a] += size
                dirs_with_files.add(a)

    blocked = set()
    for h, plist in hashes.items():
        L = lca([parent(p) for p in plist])
        if L:
            for a in ancestors_incl(L):
                blocked.add(a)

    redundant = {d for d in dirs_with_files if d not in blocked}
    maximal = [d for d in redundant if parent(d) not in redundant]
    maximal.sort(key=lambda d: dir_size[d], reverse=True)
    minb = args.min_mb * 1024 * 1024
    fminb = args.files_min_mb * 1024 * 1024

    def files_under(D):
        pref = D + "/"
        i = bisect_left(paths, pref)
        while i < len(paths) and paths[i].startswith(pref):
            yield paths[i]
            i += 1

    # --- producteur : calcule en tâche de fond, pousse les éléments prêts ---
    q = queue.Queue(maxsize=8)

    def produce():
        deleted = []
        for D in maximal:
            if dir_size[D] < minb:
                continue
            flist = list(files_under(D))
            if not flist:
                continue
            safe = True
            twin_dirs = defaultdict(int)
            for path in flist:
                _, h = info[path]
                twin = None
                for cand in hashes[h]:
                    if cand == path or under(cand, D) or any(under(cand, p) for p in deleted):
                        continue
                    twin = cand
                    break
                if twin is None:
                    safe = False
                    break
                twin_dirs[parent(twin)] += 1
            if safe:
                deleted.append(D)
                q.put(("folder", D, dir_size[D], len(flist), dict(twin_dirs)))
        # fichiers en double restants
        rem = []
        for h, plist in hashes.items():
            if len(plist) < 2:
                continue
            r = [p for p in plist if not any(under(p, d) for d in deleted)]
            if len(r) > 1 and info[r[0]][0] >= fminb:
                r.sort()
                rem.append((info[r[0]][0] * (len(r) - 1), info[r[0]][0], r[0], r[1:]))
        rem.sort(reverse=True)
        for item in rem[: args.files_top]:
            q.put(("file", *item))
        q.put(None)

    threading.Thread(target=produce, daemon=True).start()

    # ----- mode suppression de masse -----
    if args.delete_all:
        folders, files = [], []
        while True:
            it = q.get()
            if it is None:
                break
            (folders if it[0] == "folder" else files).append(it)
        del_set = [it[1] for it in folders]  # rel des dossiers à supprimer

        def twin_ok(rel, exclude_dir):
            """Une copie de ce contenu existe sur disque, hors dossiers supprimés."""
            _, h = info[rel]
            for c in hashes[h]:
                if c == rel or under(c, exclude_dir):
                    continue
                if any(under(c, d) for d in del_set):
                    continue
                if os.path.isfile(absp(c)):
                    return True
            return False

        nf = len(folders)
        nfi = len(files)
        approx = sum(it[2] for it in folders) + sum(it[1] * len(it[4]) for it in files)
        print(f"\n--delete-all : {nf} dossiers + {nfi} groupes de fichiers → ~{human(approx)}")
        if not args.no_verify:
            print("(vérif d'existence des copies activée — peut stater beaucoup de fichiers)")
        if not args.yes:
            if input('Tape "OUI" pour tout supprimer > ').strip() != "OUI":
                print("annulé")
                return

        freed = 0
        for _, D, sz, n, _t in folders:
            full = absp(D)
            if not os.path.isdir(full):
                continue
            if not args.no_verify:
                bad = next((p for p in files_under(D) if not twin_ok(p, D)), None)
                if bad is not None:
                    print(f"  ⚠ gardé (copie manquante pour {bad}) : {full}")
                    continue
            try:
                shutil.rmtree(full)
                freed += sz
                print(f"  ✗ {human(sz):>9}  {full}")
            except OSError as e:
                print(f"  erreur {full} : {e}")
        # poids par source (top-niveau) → on garde la copie dans la plus grosse
        src_total = defaultdict(int)
        for rel, (sz, _) in info.items():
            src_total[rel.split("/", 1)[0]] += sz

        def src_rank(rel):
            return src_total.get(rel.split("/", 1)[0], 0)

        for _, waste, size, keeper, extras in files:
            present = [p for p in [keeper] + extras if os.path.isfile(absp(p))]
            if len(present) < 2:
                continue  # plus de doublon réel → on ne touche pas
            present.sort(key=src_rank, reverse=True)  # garde la + grosse source
            for p in present[1:]:
                try:
                    os.remove(absp(p))
                    freed += size
                except OSError as e:
                    print(f"  erreur {absp(p)} : {e}")

        # balayage final : supprime les dossiers devenus vides (bottom-up)
        rmdirs = 0
        for dp, _dn, _fn in os.walk(root, topdown=False):
            try:
                if not os.listdir(dp) and os.path.abspath(dp) != os.path.abspath(root):
                    os.rmdir(dp)
                    rmdirs += 1
            except OSError:
                pass
        print(f"\nTerminé. Espace libéré : {human(freed)} · dossiers vides supprimés : {rmdirs}")
        return

    print("Analyse en cours… (le premier résultat s'affiche dès qu'il est prêt)\n")

    show_all = False
    freed = 0
    n = 0
    while True:
        item = q.get()
        if item is None:
            break

        if item[0] == "folder":
            _, D, sz, nf, tdirs = item
            full = absp(D)
            if not os.path.isdir(full):  # vérif existence
                continue
            n += 1
            print(f"\n[{n}] DOSSIER redondant — {human(sz)} · {nf} fichiers")
            print(f"   À SUPPRIMER : {full}")
            ordered = sorted(tdirs.items(), key=lambda x: -x[1])
            show = ordered if args.paths else ordered[:6]
            print("   copies sûres dans :")
            for d, c in show:
                print(f"     {c:>6}×  {absp(d)}")
            if not args.paths and len(ordered) > 6:
                print(f"     … (+{len(ordered)-6}, --paths pour tout)")
            target, is_dir, tsize = full, True, sz
        else:
            _, waste, size, keeper, extras = item
            existing = [e for e in extras if os.path.isfile(absp(e))]
            if not existing:
                continue
            n += 1
            print(f"\n[{n}] FICHIER en double — {human(size)} ×{len(existing)+1}")
            print(f"   GARDER      : {absp(keeper)}")
            for e in existing:
                print(f"   supprimable : {absp(e)}")
            target, is_dir, tsize = None, False, size  # suppression fichier gérée ci-dessous

        if show_all:
            continue
        c = input("   [Entrée]=suivant · s=SUPPRIMER · a=tout · q=quitter > ").strip().lower()
        if c == "q":
            break
        if c == "a":
            show_all = True
        elif c == "s":
            try:
                if item[0] == "folder":
                    if not args.yes:
                        ok = input(f"   confirmer rmtree {target} ? (o/N) ").strip().lower()
                        if ok != "o":
                            print("   annulé")
                            continue
                    shutil.rmtree(target)
                    freed += tsize
                    print(f"   ✗ SUPPRIMÉ ({human(tsize)})")
                else:
                    for e in existing:
                        os.remove(absp(e))
                    freed += size * len(existing)
                    print(f"   ✗ {len(existing)} copies supprimées ({human(size*len(existing))})")
            except OSError as e:
                print(f"   erreur suppression : {e}")

    print(f"\nFini. Espace libéré cette session : {human(freed)}")


if __name__ == "__main__":
    main()
