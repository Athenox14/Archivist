#!/usr/bin/env python3
"""Génère assets/icon.png : une plume stylisée (logo Archivist)."""
import math
from pathlib import Path
from PIL import Image, ImageDraw

S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

COL = (90, 170, 230, 255)   # bleu plume
SHAFT = (60, 130, 190, 255)

# rachis (tige) du bas-gauche vers le haut-droite
tip = (200, 48)
base = (60, 210)


def lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


# barbes de chaque côté, plus courtes vers la pointe
for i in range(1, 22):
    t = i / 22.0
    p = lerp(base, tip, t)
    length = 60 * (1 - t) + 8
    ang = math.atan2(tip[1] - base[1], tip[0] - base[0])
    # côté "haut-gauche"
    a1 = ang - math.radians(52)
    q1 = (p[0] + length * math.cos(a1), p[1] + length * math.sin(a1))
    d.line([p, q1], fill=COL, width=5)
    # côté "bas-droite"
    a2 = ang + math.radians(52)
    q2 = (p[0] + length * math.cos(a2), p[1] + length * math.sin(a2))
    d.line([p, q2], fill=COL, width=5)

# rachis par-dessus
d.line([base, tip], fill=SHAFT, width=7)
d.ellipse([base[0] - 6, base[1] - 6, base[0] + 6, base[1] + 6], fill=SHAFT)

out = Path(__file__).resolve().parent.parent / "assets"
out.mkdir(exist_ok=True)
img.save(out / "icon.png")
print("écrit", out / "icon.png")
