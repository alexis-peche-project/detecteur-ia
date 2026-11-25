"""Serveur MELD pour le Détecteur de texte IA.

MELD (Li, Wan & Paetzold, 2026) est un détecteur entraîné, open source (licence MIT),
premier des modèles open source sur le banc d'essai RAID. Il ne traite que l'anglais.
Ce serveur le rend accessible au site, qui est statique.

Déploiement : voir README.md de ce dossier (Hugging Face Space, SDK Docker, gratuit).
"""
import importlib.util
import os
import sys
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
from pydantic import BaseModel

MODEL_REPO = os.environ.get("MELD_REPO", "anon-review-meld-2026/meld")
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS", "https://alexis-peche-project.github.io").split(",") if o.strip()]
MAX_CHARS = int(os.environ.get("MAX_CHARS", "60000"))
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", "12"))

app = FastAPI(title="Serveur MELD")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["POST", "GET"], allow_headers=["content-type"])

_scorer = None
_lock = threading.Lock()
_hits = defaultdict(deque)


def get_scorer():
    """Télécharge MELD une fois, puis garde le modèle en mémoire."""
    global _scorer
    if _scorer is None:
        with _lock:
            if _scorer is None:
                path = snapshot_download(MODEL_REPO)
                spec = importlib.util.spec_from_file_location("meld", os.path.join(path, "meld.py"))
                meld = importlib.util.module_from_spec(spec)
                sys.modules["meld"] = meld
                spec.loader.exec_module(meld)
                _scorer = meld.Scorer(model_dir=path)
    return _scorer


class Demande(BaseModel):
    text: str


def limite(ip: str):
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        raise HTTPException(429, "Trop de demandes, réessayez dans une minute.")
    q.append(now)


@app.on_event("startup")
def prechargement():
    threading.Thread(target=get_scorer, daemon=True).start()


@app.get("/")
def etat():
    return {"service": "MELD", "modele": MODEL_REPO, "pret": _scorer is not None}


@app.post("/score")
def score(d: Demande, request: Request):
    limite(request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0])
    text = d.text.strip()
    if len(text.split()) < 30:
        raise HTTPException(400, "Texte trop court pour MELD (30 mots minimum, 100 conseillés).")
    text = text[:MAX_CHARS]
    with _lock:
        r = get_scorer().score(text)
    return {
        "p_ai": float(r["p_ai"]),
        "score": float(r["score"]),
        "seuil_fpr_1pc": float(r["threshold_fpr_0.01"]),
        "signale": bool(r["flagged"]),
        "famille": r.get("family"),
        "tache": r.get("task"),
        "tokens": int(r.get("n_tokens", 0)),
        "tronque": bool(r.get("truncated", False)),
    }
