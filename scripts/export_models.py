#!/usr/bin/env python3
"""Exporte les modèles ONNX + tokenizers attendus par archivist dans ./models/.

Produit :
  models/clip_image.onnx, clip_text.onnx, clip_tokenizer.json
  models/e5_small.onnx,    e5_tokenizer.json

Dépendances :
  pip install torch transformers optimum[onnxruntime] open_clip_torch
"""
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "models"
OUT.mkdir(exist_ok=True)


def export_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    proc.tokenizer.save_pretrained(OUT)
    # le fichier s'appelle tokenizer.json
    os.replace(OUT / "tokenizer.json", OUT / "clip_tokenizer.json")

    # encodeur image
    pixel = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model.vision_model, (pixel,), OUT / "_vis.onnx",
        input_names=["pixel_values"], output_names=["last_hidden_state"],
        dynamic_axes={"pixel_values": {0: "b"}}, opset_version=17,
    )
    # NB : pour des embeddings projetés 512-d, wrappez get_image_features /
    # get_text_features (voir doc). Ici on illustre la mécanique d'export.

    ids = torch.randint(0, 1000, (1, 77))
    mask = torch.ones(1, 77, dtype=torch.long)

    # transformers 5.x : get_*_features renvoie un objet ; l'embedding projeté
    # dans l'espace joint (512-d) est `pooler_output`.
    class Img(torch.nn.Module):
        def __init__(s): super().__init__(); s.m = model
        def forward(s, pixel_values):
            return s.m.get_image_features(pixel_values=pixel_values).pooler_output

    class Txt(torch.nn.Module):
        def __init__(s): super().__init__(); s.m = model
        def forward(s, input_ids, attention_mask):
            return s.m.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            ).pooler_output

    torch.onnx.export(
        Img(), (pixel,), OUT / "clip_image.onnx",
        input_names=["pixel_values"], output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "b"}}, opset_version=17,
    )
    torch.onnx.export(
        Txt(), (ids, mask), OUT / "clip_text.onnx",
        input_names=["input_ids", "attention_mask"], output_names=["text_embeds"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}},
        opset_version=17,
    )
    (OUT / "_vis.onnx").unlink(missing_ok=True)


def export_e5():
    # Export direct via torch.onnx (optimum incompatible transformers 5.x).
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = "intfloat/multilingual-e5-small"
    model = AutoModel.from_pretrained(name).eval()
    tok = AutoTokenizer.from_pretrained(name)
    tok.save_pretrained(OUT)
    os.replace(OUT / "tokenizer.json", OUT / "e5_tokenizer.json")

    # Wrapper : 3 entrées (input_ids, attention_mask, token_type_ids) →
    # last_hidden_state, pour matcher exactement le code Rust.
    class E5(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.m = model

        def forward(s, input_ids, attention_mask, token_type_ids):
            out = s.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return out.last_hidden_state

    ids = torch.randint(0, 1000, (1, 16))
    mask = torch.ones(1, 16, dtype=torch.long)
    tt = torch.zeros(1, 16, dtype=torch.long)
    torch.onnx.export(
        E5(), (ids, mask, tt), OUT / "e5_small.onnx",
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "b", 1: "s"},
            "attention_mask": {0: "b", 1: "s"},
            "token_type_ids": {0: "b", 1: "s"},
            "last_hidden_state": {0: "b", 1: "s"},
        },
        opset_version=17,
    )


if __name__ == "__main__":
    export_clip()
    export_e5()
    print("Modèles écrits dans", OUT)
