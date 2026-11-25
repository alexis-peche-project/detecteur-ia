"""Fonctions partagées par le pipeline : lecture/écriture JSONL, normalisation, découpe et score."""
import hashlib
import json
import os
import math
import re
from pathlib import Path

DATA = Path(os.environ.get("DETIA_DATA", Path(__file__).resolve().parent / "donnees"))


def lire_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def ecrire_jsonl(path, rows, mode="w"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def nb_mots(t):
    return len(re.findall(r"\w+", t))


def split_de(doc_id, val=0.1, test=0.2):
    """Répartition déterministe : un document et ses miroirs restent toujours dans la même partie."""
    h = int(hashlib.sha1(doc_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


_MD = [
    (re.compile(r"\*\*([^*\n]+)\*\*"), r"\1"),
    (re.compile(r"__([^_\n]+)__"), r"\1"),
    (re.compile(r"^#{1,6}\s+", re.M), ""),
    (re.compile(r"^\s*[-*•]\s+", re.M), ""),
    (re.compile(r"[“”«»]"), '"'),
    (re.compile(r"[‘’]"), "'"),
    (re.compile(r"[​-‏⁠﻿]"), ""),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def normaliser(t):
    """Retire la mise en forme Markdown et unifie la typographie.
    Le classifieur doit apprendre le style d'écriture, pas des raccourcis de mise en forme.
    Le même traitement est appliqué dans le site (js/classifier-worker.js)."""
    for rx, rep in _MD:
        t = rx.sub(rep, t)
    return t.strip()


def fenetres(ids, longueur, max_fen=None):
    """Découpe une liste de tokens en fenêtres consécutives."""
    out = [ids[i:i + longueur] for i in range(0, max(len(ids), 1), longueur)]
    out = [w for w in out if len(w) >= min(32, len(ids))]
    if max_fen and len(out) > max_fen:
        pas = (len(out) - 1) / (max_fen - 1) if max_fen > 1 else 0
        out = [out[round(k * pas)] for k in range(max_fen)]
    return out or [ids]


def sigmoide(x):
    return 1 / (1 + math.exp(-x))


def scorer_docs(model, tokenizer, textes, longueur=512, max_fen=16, batch=16, device="cpu"):
    """Score document = moyenne des logits (IA - humain) sur les fenêtres. Renvoie des logits."""
    import torch
    model.eval()
    res = []
    for t in textes:
        ids = tokenizer(normaliser(t), add_special_tokens=False)["input_ids"]
        fen = fenetres(ids, longueur - 2, max_fen)
        logits = []
        for i in range(0, len(fen), batch):
            lot = [tokenizer.build_inputs_with_special_tokens(w) for w in fen[i:i + batch]]
            m = max(len(x) for x in lot)
            pad = tokenizer.pad_token_id or 0
            inp = torch.tensor([x + [pad] * (m - len(x)) for x in lot], device=device)
            att = torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x in lot], device=device)
            with torch.no_grad():
                out = model(input_ids=inp, attention_mask=att).logits.float()
            logits += (out[:, 1] - out[:, 0]).tolist()
        res.append(sum(logits) / len(logits))
    return res
