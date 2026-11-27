"""Compression du classifieur ONNX pour le navigateur, avec contrôle de précision.

La compression par défaut d'onnxruntime (quantize_dynamic sur tout le modèle) abîme mmBERT :
la table des mots (256 000 lignes) compressée d'un seul bloc donne des scores très différents
du modèle d'origine. Ce module essaie plusieurs réglages, mesure chacun sur de vrais textes
par rapport au modèle ONNX non compressé, et garde le plus léger qui reste fidèle.
Si aucun ne l'est, le site utilisera le modèle non compressé (plus lourd à télécharger).

Utilisation seule, pour corriger un modèle déjà publié (pas besoin de GPU) :
    python quantifier.py --depot alexis974/detecteur-ia-fr --textes humains.jsonl ia.jsonl
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

# Écart maximal toléré sur la marge (logit IA − logit humain) par rapport au modèle non compressé.
TOLERANCE = 0.5

EXEMPLES = [
    "Bonjour, ceci est un petit test.",
    "Il est important de noter que ce sujet joue un rôle clé dans notre société moderne.",
    "On est partis vers 8h, la route était bouchée jusqu'à Lyon, bref on est arrivés crevés.",
    "En conclusion, cette approche offre de nombreux avantages tout en présentant certains défis.",
    "La commune compte 1 245 habitants au recensement de 2019 et fait partie du canton de Vienne.",
    "J'ai testé la recette hier soir : trop de sel, mais la cuisson était parfaite.",
    "Dans un monde en constante évolution, il est essentiel de s'adapter aux nouvelles technologies.",
    "Le film est long, parfois maladroit, mais la dernière demi-heure vaut vraiment le détour.",
]


def textes_de_test(fichiers, n=160, graine=0):
    textes = list(EXEMPLES)
    lus = []
    for f in fichiers or []:
        with open(f, encoding="utf-8") as fh:
            for ligne in fh:
                try:
                    t = json.loads(ligne).get("text", "")
                except Exception:
                    continue
                if len(t) > 200:
                    lus.append(t)
    random.Random(graine).shuffle(lus)
    return textes + lus[:n]


def entrees(tok, textes, noms, longueur=512):
    """Encode chaque texte séparément (pas de remplissage), comme le site."""
    sys.path.insert(0, str(Path(__file__).parent))
    from commun import normaliser
    import numpy as np
    out = []
    for t in textes:
        enc = tok(normaliser(t), truncation=True, max_length=longueur, return_tensors="np")
        out.append({k: v.astype(np.int64) for k, v in enc.items() if k in noms})
    return out


def marges(chemin, lots):
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"])
    noms = {i.name for i in sess.get_inputs()}
    res = []
    for e in lots:
        lg = sess.run(None, {k: v for k, v in e.items() if k in noms})[0][0]
        res.append(float(lg[1] - lg[0]) if len(lg) > 1 else float(lg[0]))
    return np.array(res)


def mots_en_fp16(src, dst):
    """Table des mots en demi-précision (deux fois plus légère), relue en précision normale juste après."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    m = onnx.load(str(src))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    nouveaux = []
    for node in list(g.node):
        nouveaux.append(node)
        if node.op_type != "Gather" or node.input[0] not in inits:
            continue
        init = inits[node.input[0]]
        if init.data_type != onnx.TensorProto.FLOAT or np.prod(init.dims) < 1_000_000:
            continue
        init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name))
        sortie = node.output[0]
        node.output[0] = sortie + "_f16"
        nouveaux.append(helper.make_node("Cast", [node.output[0]], [sortie], to=onnx.TensorProto.FLOAT, name=node.name + "_vers_f32"))
    del g.node[:]
    g.node.extend(nouveaux)
    onnx.save(m, str(dst))
    return dst


def poids_4bits(src, dst):
    """Poids des multiplications en 4 bits par blocs de 32 (MatMulNBits). Les activations restent
    en précision normale : contrairement à la compression dynamique 8 bits, les valeurs extrêmes
    des activations de mmBERT ne sont pas écrasées."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    q = MatMulNBitsQuantizer(onnx.load(str(src)), block_size=32, is_symmetric=True, bits=4)
    q.process()
    q.model.save_model_to_file(str(dst), False)
    return dst


def candidats(brut, dossier):
    """Fabrique plusieurs versions compressées. Retourne {nom: (fichier, dtype pour le site)}."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    dossier = Path(dossier)
    noeuds = [n.name for n in onnx.load(str(brut), load_external_data=False).graph.node if n.op_type in ("MatMul", "Gemm")]
    tete = noeuds[-2:]  # couche dense de la tête et classifieur final
    q = lambda src, dst, **kw: quantize_dynamic(str(src), str(dossier / dst), weight_type=QuantType.QInt8, **kw) or dossier / dst
    essais = {}

    def essai(nom, dtype, fabrique):
        try:
            essais[nom] = (fabrique(), dtype)
        except Exception as e:
            print(f"   {nom} : impossible ({type(e).__name__}: {e})", file=sys.stderr)

    # Compression des poids seuls (4 bits), la plus sûre pour mmBERT.
    essai("poids4+mots16", "q4", lambda: mots_en_fp16(poids_4bits(brut, dossier / "_p4.onnx"), dossier / "poids4+mots16.onnx"))
    essai("poids4", "q4", lambda: poids_4bits(brut, dossier / "poids4.onnx"))
    essai("mots16", "q4", lambda: mots_en_fp16(brut, dossier / "mots16.onnx"))
    # Compression dynamique 8 bits (poids et activations) : plus légère, mais abîme souvent mmBERT.
    def mots_ligne_puis_calcul():
        g = q(brut, "_mots.onnx", per_channel=True, op_types_to_quantize=["Gather"])
        return q(g, "mots_ligne+calcul.onnx", op_types_to_quantize=["MatMul"], nodes_to_exclude=tete)
    essai("dyn8_mots_ligne+calcul", "q8", mots_ligne_puis_calcul)
    essai("dyn8_par_canal_sans_tete", "q8", lambda: q(brut, "tout_par_canal_sans_tete.onnx", per_channel=True, nodes_to_exclude=tete))
    essai("dyn8_defaut", "q8", lambda: q(brut, "defaut.onnx"))
    return essais


FICHIERS = {"q8": "model_quantized.onnx", "q4": "model_q4.onnx"}


def choisir(brut, lots, dossier_onnx, tolerance=TOLERANCE):
    """Garde la version compressée la plus légère qui reste fidèle ; retourne le dtype à charger
    ('q4', 'q8' ou 'fp32') et écrit le fichier correspondant dans dossier_onnx."""
    import numpy as np
    dossier_onnx = Path(dossier_onnx)
    ref = marges(brut, lots)
    # Tolérance : au moins `tolerance`, ou 5 % de l'étendue des marges (les logits de mmBERT vont loin).
    tol = max(tolerance, 0.05 * float(ref.max() - ref.min()))
    tmp = Path(tempfile.mkdtemp(prefix="quantif-"))
    try:
        bilan = []
        for nom, (f, dtype) in candidats(brut, tmp).items():
            try:
                m = marges(f, lots)
            except Exception as e:
                print(f"   {nom:26s} illisible par onnxruntime ({type(e).__name__})")
                continue
            ecart = float(np.abs(m - ref).max())
            accord = float(np.mean((m > 0) == (ref > 0)))
            taille = os.path.getsize(f) / 1e6
            bilan.append((nom, f, dtype, ecart, accord, taille))
            print(f"   {nom:26s} {taille:7.1f} Mo   écart max {ecart:7.3f}   même verdict {100 * accord:5.1f} %")
        print(f"   (modèle complet : {os.path.getsize(brut) / 1e6:.1f} Mo, marges de {ref.min():.1f} à {ref.max():.1f}, tolérance {tol:.2f})")
        for d in FICHIERS.values():
            if (dossier_onnx / d).exists():
                (dossier_onnx / d).unlink()
        bons = sorted([b for b in bilan if b[3] <= tol and b[4] >= 0.99], key=lambda b: b[5])
        if bons:
            nom, f, dtype, ecart, _, taille = bons[0]
            shutil.copy(f, dossier_onnx / FICHIERS[dtype])
            print(f"→ Version retenue : {nom} ({taille:.0f} Mo, écart max {ecart:.3f}), chargée par le site en « {dtype} »")
            return dtype
        print("→ Aucune version compressée n'est assez fidèle : le site utilisera le modèle complet.")
        return "fp32"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ecrire_dtype(config_json, dtype):
    """Indique au site (transformers.js) quelle version du modèle charger."""
    p = Path(config_json)
    cfg = json.loads(p.read_text())
    cfg["transformers.js_config"] = {**cfg.get("transformers.js_config", {}), "dtype": dtype}
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depot", required=True, help="modèle publié sur Hugging Face, ex. alexis974/detecteur-ia-fr")
    ap.add_argument("--textes", nargs="*", default=[], help="fichiers .jsonl de textes (champ text) pour la mesure")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--sans-envoi", action="store_true", help="mesurer sans republier sur Hugging Face")
    a = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer
    import onnxruntime as ort

    dossier = Path(snapshot_download(a.depot, allow_patterns=["*.json", "*.txt", "*.model", "onnx/model.onnx"]))
    brut = dossier / "onnx" / "model.onnx"
    tok = AutoTokenizer.from_pretrained(dossier)
    noms = {i.name for i in ort.InferenceSession(str(brut), providers=["CPUExecutionProvider"]).get_inputs()}
    lots = entrees(tok, textes_de_test(a.textes, a.n), noms)
    print(f"Mesure sur {len(lots)} textes")

    travail = Path(tempfile.mkdtemp(prefix="republier-"))
    (travail / "onnx").mkdir()
    dtype = choisir(brut, lots, travail / "onnx", a.tolerance)
    shutil.copy(dossier / "config.json", travail / "config.json")
    ecrire_dtype(travail / "config.json", dtype)
    if a.sans_envoi:
        print("Rien n'a été envoyé (--sans-envoi).")
        return
    api = HfApi()
    api.upload_folder(repo_id=a.depot, folder_path=str(travail), commit_message=f"Compression corrigée (dtype {dtype})")
    for d, f in FICHIERS.items():
        if d != dtype:  # retire les anciennes versions compressées non retenues
            try:
                api.delete_file(f"onnx/{f}", repo_id=a.depot, commit_message="Version compressée non retenue retirée")
            except Exception:
                pass
    print(f"Publié sur https://huggingface.co/{a.depot} — le site chargera la version « {dtype} ».")


if __name__ == "__main__":
    main()
