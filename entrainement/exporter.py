"""Étape 5 : convertit le classifieur au format ONNX (quantifié 8 bits) pour qu'il tourne dans le navigateur,
puis le publie sur Hugging Face, d'où le site le télécharge.

  python exporter.py --depot votre-nom/detecteur-ia-fr      # jeton dans HF_TOKEN

Le dossier publié a la structure attendue par Transformers.js :
  config.json, tokenizer.json, ..., onnx/model.onnx, onnx/model_quantized.onnx

La conversion utilise directement PyTorch et ONNX Runtime (pas de bibliothèque « optimum »),
pour ne pas imposer de versions de transformers incompatibles avec vLLM.
"""
import argparse
import json
import shutil
from pathlib import Path

ICI = Path(__file__).resolve().parent


class _Logits:
    """Enveloppe qui renvoie uniquement les logits, sortie nommée « logits » comme l'attend Transformers.js."""

    def __new__(cls, model):
        import torch

        class W(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask, token_type_ids=None):
                kw = {"input_ids": input_ids, "attention_mask": attention_mask}
                if token_type_ids is not None:
                    kw["token_type_ids"] = token_type_ids
                return self.m(**kw).logits

        return W(model)


def exporter(src, dst):
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "onnx").mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(src)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(src, attn_implementation="eager")
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(src)
    model.eval()
    model.config.save_pretrained(dst)
    tok.save_pretrained(dst)

    enc = tok(["Ceci est un exemple de texte pour l'export.", "Un autre, un peu plus long, pour les tailles variables."],
              padding=True, return_tensors="pt")
    noms = ["input_ids", "attention_mask"] + (["token_type_ids"] if "token_type_ids" in enc else [])
    args = tuple(enc[n] for n in noms)
    axes = {n: {0: "batch", 1: "sequence"} for n in noms}
    axes["logits"] = {0: "batch"}
    brut = dst / "onnx" / "model.onnx"
    with torch.no_grad():
        try:
            torch.onnx.export(_Logits(model), args, str(brut), input_names=noms, output_names=["logits"],
                              dynamic_axes=axes, opset_version=17, do_constant_folding=True, dynamo=False)
        except TypeError:  # anciennes versions de PyTorch sans l'option dynamo
            torch.onnx.export(_Logits(model), args, str(brut), input_names=noms, output_names=["logits"],
                              dynamic_axes=axes, opset_version=17, do_constant_folding=True)
    quantize_dynamic(str(brut), str(dst / "onnx" / "model_quantized.onnx"), weight_type=QuantType.QInt8)

    if (Path(src) / "metriques.json").exists():
        shutil.copy(Path(src) / "metriques.json", dst / "metriques.json")
    return dst, model, tok


def verifier(dst, model, tok, textes=("Bonjour, ceci est un petit test.", "Il est important de noter que ce sujet joue un rôle clé.")):
    """Compare le modèle ONNX (brut et quantifié) au modèle PyTorch d'origine."""
    import numpy as np
    import onnxruntime as ort
    import torch
    enc = tok(list(textes), padding=True, return_tensors="pt")
    with torch.no_grad():
        ref = model(**enc).logits.numpy()
    for f in ("model.onnx", "model_quantized.onnx"):
        sess = ort.InferenceSession(str(Path(dst) / "onnx" / f))
        noms = {i.name for i in sess.get_inputs()}
        out = sess.run(None, {k: v.numpy() for k, v in enc.items() if k in noms})[0]
        ecart = float(np.abs(out - ref).max())
        print(f"   {f} : écart max avec PyTorch {ecart:.4f}")
        if f == "model.onnx" and ecart > 1e-2:
            raise SystemExit("La conversion ONNX donne des résultats différents du modèle d'origine.")


def carte(dst, info):
    m = info.get("historique", [{}])[-1]
    txt = f"""---
language: [fr, en]
license: mit
library_name: transformers.js
pipeline_tag: text-classification
tags: [ai-detection, onnx]
base_model: {info.get('base', '')}
---

# Détecteur de texte IA (français)

Classifieur humain / IA entraîné pour le site [Détecteur de texte IA](https://github.com/alexis-peche-project/detecteur-ia),
à partir de `{info.get('base', '')}`, sur des textes humains antérieurs à 2023 et leurs miroirs générés par plusieurs modèles,
avec recherche d'exemples difficiles. Labels : 0 = humain, 1 = IA.

Résultats sur la partie validation : AUROC {m.get('auroc', float('nan')):.4f},
part de textes d'IA repérés à 1 % de faux positifs : {m.get('tpr_a_1pc_fpr', float('nan')):.3f}.
Voir `metriques.json` et le banc d'essai du site pour le détail.
"""
    (Path(dst) / "README.md").write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    import os
    modeles = Path(os.environ.get("DETIA_MODELES", ICI / "modeles"))
    ap.add_argument("--source", default=str(modeles / "classifieur"))
    ap.add_argument("--sortie", default=str(modeles / "classifieur-onnx"))
    ap.add_argument("--depot", help="dépôt Hugging Face de destination (ex. votre-nom/detecteur-ia-fr)")
    a = ap.parse_args()
    dst, model, tok = exporter(a.source, a.sortie)
    verifier(dst, model, tok)
    info = json.loads((Path(a.source) / "metriques.json").read_text()) if (Path(a.source) / "metriques.json").exists() else {}
    carte(dst, info)
    print(f"Modèle ONNX → {dst}")
    if a.depot:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.depot, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(dst), repo_id=a.depot, repo_type="model")
        print(f"Publié sur https://huggingface.co/{a.depot}")


if __name__ == "__main__":
    main()
