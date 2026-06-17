#!/usr/bin/env python3
"""Génère un jeu de données SYNTHÉTIQUE pour tester archivist.

Produit dans <out>/ :
  images/ : PNG de formes/couleurs nettes (cibles CLIP texte→image)
  docs/   : .txt / .md / .pdf thématiques (cibles e5 texte→texte, multilingue)
  + des doublons exacts (même contenu) pour exercer la dédup/hardlinks.

Usage : python scripts/gen_synthetic.py <dossier_sortie>
"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")

# (nom, couleur, forme) — descriptions matchables par CLIP.
IMAGES = [
    ("red_square", (220, 30, 30), "square"),
    ("blue_circle", (30, 60, 220), "circle"),
    ("green_triangle", (30, 180, 60), "triangle"),
    ("yellow_circle", (240, 220, 20), "circle"),
    ("black_square", (15, 15, 15), "square"),
    ("orange_triangle", (240, 140, 20), "triangle"),
]

DOCS = {
    "rust_async.md": (
        "# Programmation asynchrone en Rust\n\n"
        "Tokio est un runtime asynchrone pour Rust. Il fournit un executor, "
        "des primitives I/O non bloquantes et des canaux. Le mot-clef async "
        "transforme une fonction en future, await suspend son execution.\n\n"
        "Les futures sont paresseuses : rien ne s'execute tant qu'on ne les "
        "poll pas via un executor comme tokio ou async-std."
    ),
    "cuisine.txt": (
        "Recette de la tarte aux pommes\n\n"
        "Ingredients : pate brisee, six pommes, sucre, beurre, cannelle. "
        "Etalez la pate, disposez les pommes en rosace, saupoudrez de sucre "
        "et de cannelle. Enfournez trente minutes a deux cents degres.\n\n"
        "Servez tiede avec une boule de glace vanille."
    ),
    "astronomy.txt": (
        "The life cycle of stars\n\n"
        "A star forms from a collapsing cloud of gas and dust. Nuclear fusion "
        "in its core converts hydrogen into helium, releasing energy. Massive "
        "stars end as supernovae, leaving neutron stars or black holes.\n\n"
        "Our Sun is a main-sequence star and will become a red giant in "
        "roughly five billion years."
    ),
    "finance.md": (
        "# Interet compose\n\n"
        "L'interet compose fait croitre un capital de maniere exponentielle. "
        "Les interets generes sont reinvestis et produisent a leur tour des "
        "interets. Plus l'horizon de placement est long, plus l'effet est fort.\n\n"
        "Formule : capital final = capital initial multiplie par (1 + taux) "
        "puissance nombre d'annees."
    ),
}

PDF_TEXT = (
    "Manuel de jardinage biologique\n\n"
    "Le compost enrichit le sol en matiere organique. Melangez dechets verts "
    "et bruns, aerez regulierement, maintenez l'humidite. Le paillage limite "
    "l'evaporation et protege les racines du gel.\n\n"
    "La rotation des cultures previent l'epuisement des sols et reduit les "
    "maladies."
)


def gen_images(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for name, color, shape in IMAGES:
        img = Image.new("RGB", (224, 224), (245, 245, 245))
        dr = ImageDraw.Draw(img)
        if shape == "square":
            dr.rectangle([50, 50, 174, 174], fill=color)
        elif shape == "circle":
            dr.ellipse([50, 50, 174, 174], fill=color)
        else:  # triangle
            dr.polygon([(112, 40), (40, 184), (184, 184)], fill=color)
        img.save(d / f"{name}.png")
    # doublon EXACT (meme bytes) pour la dedup : copie de red_square
    (d / "red_square_COPY.png").write_bytes((d / "red_square.png").read_bytes())


def gen_docs(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for name, text in DOCS.items():
        (d / name).write_text(text, encoding="utf-8")
    # doublon exact d'un .txt
    (d / "astronomy_COPY.txt").write_text(DOCS["astronomy.txt"], encoding="utf-8")
    # PDF avec vraie couche texte
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    for line in PDF_TEXT.split("\n"):
        if line.strip():
            pdf.multi_cell(0, 8, line)
        else:
            pdf.ln(5)
    pdf.output(str(d / "jardinage.pdf"))


if __name__ == "__main__":
    gen_images(OUT / "images")
    gen_docs(OUT / "docs")
    print("Donnees synthetiques ecrites dans", OUT.resolve())
