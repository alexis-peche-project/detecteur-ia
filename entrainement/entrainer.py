"""Étape 3 : entraîne le classifieur humain / IA, puis l'améliore par recherche d'exemples difficiles.

1. Entraînement d'un encodeur multilingue (mmBERT-small par défaut) sur les textes humains
   et leurs miroirs IA (partie « train »). Le meilleur état est choisi sur la partie « val ».
2. Recherche d'exemples difficiles (« hard negative mining », méthode Pangram) :
   on fait passer la réserve de textes humains jamais vus (partie « pool ») dans le modèle,
   on garde ceux qu'il prend à tort pour de l'IA, on leur fabrique des miroirs IA
   (generer.py --ids), et on reprend l'entraînement avec ces cas difficiles. On répète --tours fois.

Sortie : modeles/classifieur/ (modèle + tokenizer + metriques.json)

Exemple (Colab, GPU) :
  python entrainer.py --tours 2
"""
import argparse
import json
import os
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from commun import DATA, MODELES, avec_speciaux, fenetres, lire_jsonl, normaliser, scorer_docs
from metriques import resume, seuil_pour_fpr

ICI = Path(__file__).resolve().parent


def charger(exclure=None):
    docs = lire_jsonl(DATA / "humains.jsonl") + lire_jsonl(DATA / "ia.jsonl")
    if exclure:
        docs = [d for d in docs if d.get("generator") not in exclure]
    return docs


def exemples(docs, tok, longueur, max_fen, rng, poids_difficiles=None):
    out = []
    for d in docs:
        ids = tok(normaliser(d["text"]), add_special_tokens=False)["input_ids"]
        fen = fenetres(ids, longueur - 2, max_fen)
        rep = (poids_difficiles or {}).get(d.get("pair_id") or d["id"], 1)
        for w in fen:
            for _ in range(rep):
                out.append((avec_speciaux(tok, w), d["label"]))
    rng.shuffle(out)
    return out


def collate(pad):
    def f(lot):
        m = max(len(x) for x, _ in lot)
        ids = torch.tensor([x + [pad] * (m - len(x)) for x, _ in lot])
        att = torch.tensor([[1] * len(x) + [0] * (m - len(x)) for x, _ in lot])
        y = torch.tensor([l for _, l in lot])
        return ids, att, y
    return f


def evaluer(model, tok, docs, a, device):
    s = scorer_docs(model, tok, [d["text"] for d in docs], a.longueur, a.max_fen_eval, a.batch * 2, device)
    y = [d["label"] for d in docs]
    return resume(y, s), s


def entrainer_epoques(model, tok, train, val, a, device, epoques, lr, rng, poids_difficiles=None, meilleur=None):
    ex = exemples(train, tok, a.longueur, a.max_fen, rng, poids_difficiles)
    n_ia = sum(l for _, l in ex)
    w = torch.tensor([len(ex) / (2 * max(1, len(ex) - n_ia)), len(ex) / (2 * max(1, n_ia))], dtype=torch.float, device=device)
    dl = DataLoader(ex, batch_size=a.batch, shuffle=True, collate_fn=collate(tok.pad_token_id or 0))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    pas_total = math.ceil(len(dl) / a.accum) * epoques
    chauffe = max(1, int(0.06 * pas_total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda p: min(1, (p + 1) / chauffe) * max(0.0, (pas_total - p) / max(1, pas_total - chauffe)))
    use_amp = device == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    meilleur = meilleur or {"cle": -1}
    for ep in range(epoques):
        model.train()
        t0, tot = time.time(), 0.0
        for i, (ids, att, y) in enumerate(dl):
            ids, att, y = ids.to(device), att.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                loss = lossf(model(input_ids=ids, attention_mask=att).logits.float(), y) / a.accum
            scaler.scale(loss).backward()
            tot += loss.item() * a.accum
            if (i + 1) % a.accum == 0 or i + 1 == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if (i + 1) % 200 == 0:
                print(f"   époque {ep + 1} lot {i + 1}/{len(dl)} perte {tot / (i + 1):.4f}", file=sys.stderr)
        m, _ = evaluer(model, tok, val, a, device)
        cle = m["tpr_a_1pc_fpr"] + m["auroc"]
        print(f"→ époque {ep + 1} ({time.time() - t0:.0f} s) val : AUROC {m['auroc']:.4f}, "
              f"IA repérée à 1 % de faux positifs {m['tpr_a_1pc_fpr']:.3f}", file=sys.stderr)
        if cle > meilleur["cle"]:
            meilleur = {"cle": cle, "metriques": m}
            model.save_pretrained(a.sortie)
            tok.save_pretrained(a.sortie)
    return meilleur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="jhu-clsp/mmBERT-small", help="encodeur de départ")
    ap.add_argument("--sortie", default=str(MODELES / "classifieur"))
    ap.add_argument("--longueur", type=int, default=512)
    ap.add_argument("--max-fen", type=int, default=4, help="fenêtres max par document à l'entraînement")
    ap.add_argument("--max-fen-eval", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epoques", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--tours", type=int, default=2, help="tours de recherche d'exemples difficiles")
    ap.add_argument("--difficiles-par-tour", type=int, default=400)
    ap.add_argument("--generateurs", default=None, help="générateurs pour les miroirs des exemples difficiles")
    ap.add_argument("--exclure-generateur", default="", help="générateur(s) gardé(s) hors entraînement pour tester la généralisation")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--force", action="store_true", help="réentraîner même si un modèle terminé existe déjà")
    a = ap.parse_args()
    if (Path(a.sortie) / "metriques.json").exists() and not a.force:
        print(f"Déjà fait : modèle entraîné dans {a.sortie} (--force pour réentraîner).")
        return
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(a.sortie).mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.base, num_labels=2, id2label={0: "humain", 1: "ia"}, label2id={"humain": 0, "ia": 1}).to(device)

    exclure = {g for g in a.exclure_generateur.split(",") if g}
    docs = charger(exclure)
    train = [d for d in docs if d["split"] == "train"]
    val = [d for d in docs if d["split"] == "val"]
    print(f"train : {len(train)} documents, val : {len(val)}", file=sys.stderr)
    if not train or not val or len({d['label'] for d in val}) < 2:
        sys.exit("Données insuffisantes : lancez donnees.py puis generer.py.")

    meilleur = entrainer_epoques(model, tok, train, val, a, device, a.epoques, a.lr, rng)
    historique = [{"tour": 0, **meilleur["metriques"]}]
    difficiles = {}

    for tour in range(1, a.tours + 1):
        model = AutoModelForSequenceClassification.from_pretrained(a.sortie).to(device)
        pool = [d for d in lire_jsonl(DATA / "humains.jsonl") if d["split"] == "pool" and d["id"] not in difficiles]
        if not pool:
            break
        _, s_val = evaluer(model, tok, val, a, device)
        seuil = seuil_pour_fpr([d["label"] for d in val], s_val, 0.05)
        s_pool = scorer_docs(model, tok, [d["text"] for d in pool], a.longueur, a.max_fen_eval, a.batch * 2, device)
        classes = sorted(zip(s_pool, pool), key=lambda x: -x[0])
        nouveaux = [d for s, d in classes if s > seuil][:a.difficiles_par_tour]
        print(f"→ tour {tour} : {len(nouveaux)} textes humains pris pour de l'IA sur {len(pool)} "
              f"({100 * len(nouveaux) / len(pool):.1f} %)", file=sys.stderr)
        if not nouveaux:
            break
        ids_path = DATA / f"difficiles_tour{tour}.txt"
        ids_path.write_text("\n".join(d["id"] for d in nouveaux))
        cmd = [os.environ.get("DETIA_GEN_PYTHON", sys.executable), str(ICI / "generer.py"), "--ids", str(ids_path)]
        if a.generateurs:
            cmd += ["--generateurs", a.generateurs]
        subprocess.run(cmd, check=False)
        for d in nouveaux:
            difficiles[d["id"]] = 3   # ces cas sont vus 3 fois plus souvent
        tous = charger(exclure)
        mines = [d for d in tous if d["split"] == "pool" and (d["id"] in difficiles or d.get("pair_id") in difficiles)]
        train_tour = train + mines
        # Le tour est toujours gardé : la partie « val » ne contient pas ces cas difficiles,
        # elle ne peut donc pas mesurer le gain (qui est mesuré ensuite par evaluer.py).
        meilleur = entrainer_epoques(model, tok, train_tour, val, a, device, 1, a.lr / 2, rng, difficiles, None)
        train = train_tour
        historique.append({"tour": tour, "difficiles": len(nouveaux), **meilleur["metriques"]})

    info = {"base": a.base, "longueur": a.longueur, "exclus": sorted(exclure), "historique": historique,
            "n_train": len(train), "n_val": len(val), "date": time.strftime("%Y-%m-%d")}
    (Path(a.sortie) / "metriques.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(json.dumps(historique[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
