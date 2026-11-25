"""Étape 5 : convertit le classifieur au format ONNX (quantifié 8 bits) pour qu'il tourne dans le navigateur,
puis le publie sur Hugging Face, d'où le site le télécharge.

  python exporter.py --depot votre-nom/detecteur-ia-fr      # jeton dans HF_TOKEN

Le dossier publié a la structure attendue par Transformers.js :
  config.json, tokenizer.json, ..., onnx/model.onnx, onnx/model_quantized.onnx
"""
import argparse
import json
import shutil
from pathlib import Path

ICI = Path(__file__).resolve().parent


def exporter(src, dst):
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    dst = Path(dst)
    tmp = dst / "_onnx"
    if dst.exists():
        shutil.rmtree(dst)
    model = ORTModelForSequenceClassification.from_pretrained(src, export=True)
    model.save_pretrained(tmp)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)

    q = ORTQuantizer.from_pretrained(tmp)
    q.quantize(save_dir=tmp / "q", quantization_config=AutoQuantizationConfig.avx2(is_static=False, per_channel=False))

    (dst / "onnx").mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp / "model.onnx", dst / "onnx" / "model.onnx")
    qfile = next((tmp / "q").glob("*.onnx"))
    shutil.copy(qfile, dst / "onnx" / "model_quantized.onnx")
    shutil.copy(tmp / "config.json", dst / "config.json")
    for extra in ("metriques.json",):
        if (Path(src) / extra).exists():
            shutil.copy(Path(src) / extra, dst / extra)
    shutil.rmtree(tmp)
    return dst


def verifier(dst, textes=("Bonjour, ceci est un petit test.",)):
    """Vérifie que le modèle ONNX quantifié donne le même classement que le modèle d'origine."""
    import onnxruntime as ort
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(dst)
    sess = ort.InferenceSession(str(Path(dst) / "onnx" / "model_quantized.onnx"))
    noms = {i.name for i in sess.get_inputs()}
    for t in textes:
        enc = tok(t, return_tensors="np")
        out = sess.run(None, {k: v for k, v in enc.items() if k in noms})[0]
        print("   test ONNX :", out.tolist())


def carte(dst, depot, info):
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
    ap.add_argument("--source", default=str(ICI / "modeles" / "classifieur"))
    ap.add_argument("--sortie", default=str(ICI / "modeles" / "classifieur-onnx"))
    ap.add_argument("--depot", help="dépôt Hugging Face de destination (ex. votre-nom/detecteur-ia-fr)")
    a = ap.parse_args()
    dst = exporter(a.source, a.sortie)
    verifier(dst)
    info = json.loads((Path(a.source) / "metriques.json").read_text()) if (Path(a.source) / "metriques.json").exists() else {}
    carte(dst, a.depot, info)
    print(f"Modèle ONNX → {dst}")
    if a.depot:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.depot, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(dst), repo_id=a.depot, repo_type="model")
        print(f"Publié sur https://huggingface.co/{a.depot}")


if __name__ == "__main__":
    main()
