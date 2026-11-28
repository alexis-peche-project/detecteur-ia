"""Test indépendant du détecteur : des textes qu'il n'a jamais vus, d'autres genres et d'autres IA.

Ce test mesure la fiabilité « dans la vraie vie », là où le banc d'essai (evaluer.py) mesure la fiabilité
sur des textes du même type que l'entraînement. Rien n'est réutilisé de l'entraînement :
  - textes humains : vos propres fichiers (copies, courriels, écrits personnels, articles…),
    rangés si possible par genre dans des sous-dossiers, et en option des extraits de livres anciens ;
  - textes d'IA : produits par des modèles absents de l'entraînement (Claude Sonnet, Gemini, Mistral, GPT,
    Qwen3, Phi-4…), avec des consignes nouvelles, sur les mêmes sujets que vos textes ;
  - variantes difficiles : textes d'IA retouchés (fautes, phrases supprimées) et textes mixtes
    (moitié humaine, moitié IA).
Le détecteur est mesuré tel que le site l'utilise : même modèle ONNX allégé, mêmes réglages (js/calibration.json).

Les textes restent dans le dossier de travail : le fichier de résultats publié ne contient que des chiffres.

Exemple :
    python test_independant.py --humains /kaggle/input/mes-textes \\
        --generateurs "anthropic:claude-sonnet-5-5,gemini:gemini-3.5-flash,mistral:mistral-small-latest"
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
from commun import DATA, ecrire_jsonl, fenetres, lire_jsonl, nb_mots, normaliser  # noqa: E402
from metriques import auroc  # noqa: E402

DOSSIER = DATA / "test_independant"
RACINE = ICI.parent

# ---------- 1. Textes humains ----------

def lire_fichier(f):
    ext = f.suffix.lower()
    if ext in (".txt", ".md"):
        return f.read_text(encoding="utf-8", errors="ignore")
    if ext == ".docx":
        import docx
        return "\n\n".join(p.text for p in docx.Document(str(f)).paragraphs)
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(p.extract_text() or "" for p in PdfReader(str(f)).pages)
    if ext == ".odt":
        from odf import teletype
        from odf.opendocument import load
        from odf.text import P
        return "\n\n".join(teletype.extractText(p) for p in load(str(f)).getElementsByType(P))
    return ""


def tronquer(t, maxi=900):
    """Garde au plus ~maxi mots, en paragraphes entiers quand c'est possible."""
    if nb_mots(t) <= maxi:
        return t.strip()
    out, n = [], 0
    for p in re.split(r"\n\s*\n", t):
        if n and n + nb_mots(p) > maxi:
            break
        out.append(p)
        n += nb_mots(p)
    return "\n\n".join(out).strip() if out else " ".join(t.split()[:maxi])


def humains_fichiers(dossiers):
    rows = []
    for d in dossiers:
        d = Path(d)
        fichiers = [d] if d.is_file() else sorted(x for x in d.rglob("*") if x.is_file())
        for f in fichiers:
            try:
                brut = lire_fichier(f)
            except Exception as e:
                print(f"   « {f.name} » illisible : {e}", file=sys.stderr)
                continue
            # Le genre est le nom du dossier qui contient le fichier (copies, courriels…).
            genre = f.parent.name.replace("_", " ").replace("-", " ") or "texte personnel"
            for k, t in enumerate(x for x in re.split(r"^\s*---\s*$", brut, flags=re.M) if nb_mots(x) >= 80):
                rows.append({"text": tronquer(t), "genre": genre, "origine": f"{f.name}#{k}"})
    return rows


def humains_livres(n, rng):
    """Extraits de livres français du domaine public (avant 1955, numérisés : quelques fautes d'OCR)."""
    if not n:
        return []
    from datasets import load_dataset
    ds = load_dataset("PleIAs/French-PD-Books", split="train", streaming=True).shuffle(seed=rng.randint(0, 10**6), buffer_size=200)
    rows = []
    for x in ds:
        t = x.get("complete_text") or ""
        mots = t.split()
        if len(mots) < 3000:
            continue
        i = rng.randint(1000, len(mots) - 1500)
        extrait = " ".join(mots[i:i + 400])
        if sum(c.isalpha() for c in extrait) / max(1, len(extrait)) < 0.75:
            continue  # page trop abîmée par l'OCR
        rows.append({"text": extrait, "genre": "livre ancien (domaine public)", "origine": x.get("title", "")[:80]})
        if len(rows) >= n:
            break
    return rows


def deja_vus():
    """Empreintes des textes de l'entraînement, pour écarter tout recouvrement."""
    vus = set()
    for f in (DATA / "humains.jsonl", DATA / "ia.jsonl"):
        for r in lire_jsonl(f):
            vus.add(hashlib.sha1(normaliser(r["text"])[:300].encode()).hexdigest())
    return vus


# ---------- 2. Textes d'IA (modèles et consignes absents de l'entraînement) ----------

CONSIGNES = {
    "ecrire": "Voici le début d'un texte, pour te donner le sujet et le genre (c'est un(e) {genre}) :\n« {debut} »\n\n"
              "Écris ton propre texte complet sur le même sujet, dans le même genre, en français, d'environ {n} mots. "
              "Ne recopie pas ce début. Réponds uniquement par le texte.",
    "naturel": "Tu dois écrire un(e) {genre} en français, d'environ {n} mots, sur le sujet de ce début :\n« {debut} »\n\n"
               "Écris comme une vraie personne : style naturel, phrases de longueurs variées, pas de formules toutes faites, "
               "pas de titres ni de listes. Il ne faut pas qu'on devine qu'une IA l'a écrit. Réponds uniquement par le texte.",
    "paraphrase": "Reformule entièrement ce texte en français, avec tes propres mots, en gardant toutes les idées "
                  "et à peu près la même longueur. Réponds uniquement par le texte reformulé.\n\n{texte}",
}


def travaux_ia(humains, gens, par_gen, rng, faits):
    par = {}
    for g in gens:
        jobs, ordre = [], humains[:]
        rng.shuffle(ordre)
        for k in range(par_gen):
            h = ordre[k % len(ordre)]
            tache = ["ecrire", "naturel", "paraphrase"][k % 3]
            jid = "t-" + hashlib.sha1(f"{g}|{h['id']}|{tache}|{k}".encode()).hexdigest()[:12]
            if jid in faits:
                continue
            debut = " ".join(h["text"].split()[:60])
            n = max(120, min(700, nb_mots(h["text"])))
            jobs.append({"id": jid, "h": h, "tache": tache, "temperature": round(rng.uniform(0.7, 1.0), 2), "seed": rng.randint(0, 10**6),
                         "prompt": CONSIGNES[tache].format(genre=h["genre"], debut=debut, n=n, texte=h["text"])})
        par[g] = jobs
    return par


def generer_ia(humains, gens, par_gen, rng):
    import generer
    sortie = DOSSIER / "ia.jsonl"
    faits = {r["id"] for r in lire_jsonl(sortie)}
    for g, jobs in travaux_ia(humains, gens, par_gen, rng, faits).items():
        if not jobs:
            print(f"→ {g} : déjà fait", file=sys.stderr)
            continue
        print(f"→ {g} : {len(jobs)} textes à créer", file=sys.stderr, flush=True)

        def sauver(lot, textes, g=g):
            rows = []
            for j, t in zip(lot, textes):
                t = generer.nettoyer_sortie(t or "")
                if nb_mots(t) >= 60:
                    rows.append({"id": j["id"], "text": t, "label": 1, "genre": j["h"]["genre"], "pair_id": j["h"]["id"],
                                 "generator": g.split(":", 1)[-1], "task": j["tache"]})
            ecrire_jsonl(sortie, rows, mode="a")
            print(f"   {g} : {len(rows)}/{len(lot)} gardés", file=sys.stderr, flush=True)

        try:
            generer.generer(jobs, g, sauver)
        except Exception as e:
            print(f"   {g} ignoré ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
    return lire_jsonl(sortie)


# ---------- 3. Variantes difficiles ----------

def retoucher(t, rng):
    """Ce que fait un élève pressé : supprime une phrase sur dix, ajoute des fautes de frappe et des minuscules."""
    phrases = re.split(r"(?<=[.!?])\s+", t)
    phrases = [p for p in phrases if rng.random() > 0.1] or phrases
    out = []
    for p in phrases:
        mots = p.split(" ")
        for i, m in enumerate(mots):
            r = rng.random()
            if len(m) > 3 and r < 0.03:
                k = rng.randint(1, len(m) - 2)
                mots[i] = m[:k] + m[k + 1] + m[k] + m[k + 2:]  # lettres inversées
            elif len(m) > 4 and r < 0.05:
                mots[i] = m.replace("é", "e", 1)  # accent oublié
        p = " ".join(mots)
        if rng.random() < 0.15 and p:
            p = p[0].lower() + p[1:]
        out.append(p)
    return " ".join(out)


def variantes(humains, ia, rng):
    par_id = {h["id"]: h for h in humains}
    rows = []
    for x in ia:
        if x["task"] == "paraphrase":
            continue
        rows.append({**x, "id": x["id"] + "-r", "text": retoucher(x["text"], rng), "task": "retouché à la main"})
        h = par_id.get(x["pair_id"])
        if h and rng.random() < 0.5:
            ph, pi = re.split(r"(?<=[.!?])\s+", h["text"]), re.split(r"(?<=[.!?])\s+", x["text"])
            mixte = " ".join(ph[:max(1, len(ph) // 2)]) + "\n\n" + " ".join(pi[len(pi) // 2:])
            rows.append({**x, "id": x["id"] + "-m", "text": mixte, "label": 0.5, "task": "mixte (moitié humain)"})
    return rows


# ---------- 4. Mesure, comme sur le site ----------

def scorer_onnx(textes, depot, longueur=512, max_fen=16, local=None):
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from commun import avec_speciaux
    d = Path(local) if local else Path(snapshot_download(depot, allow_patterns=["*.json", "*.txt", "*.model"]))
    dtype = json.loads((d / "config.json").read_text()).get("transformers.js_config", {}).get("dtype", "fp32")
    fichier = {"q8": "model_quantized.onnx", "q4": "model_q4.onnx"}.get(dtype, "model.onnx")
    if not local:
        d = Path(snapshot_download(depot, allow_patterns=["*.json", "*.txt", "*.model", f"onnx/{fichier}"]))
    print(f"Détecteur : {depot}, version « {dtype} » ({fichier}), comme sur le site", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(d)
    sess = ort.InferenceSession(str(d / "onnx" / fichier), providers=["CPUExecutionProvider"])
    noms = {i.name for i in sess.get_inputs()}
    res = []
    for k, t in enumerate(textes):
        ids = tok(normaliser(t), add_special_tokens=False)["input_ids"]
        logits = []
        for w in fenetres(ids, longueur - 2, max_fen):
            x = np.array([avec_speciaux(tok, w)], dtype=np.int64)
            feed = {"input_ids": x, "attention_mask": np.ones_like(x), "token_type_ids": np.zeros_like(x)}
            o = sess.run(None, {n: v for n, v in feed.items() if n in noms})[0][0]
            logits.append(float(o[1] - o[0]))
        res.append(sum(logits) / len(logits))
        if (k + 1) % 50 == 0:
            print(f"   {k + 1}/{len(textes)} textes mesurés", file=sys.stderr, flush=True)
    return res


def score_site(logit, cal):
    """Score affiché par le site (0 à 100) à partir du logit du détecteur et des réglages publiés."""
    e = cal["ensembles"]["clf"]
    p = 1 / (1 + math.exp(-(e["intercept"] + e["coef"][0] * logit)))
    s5, s1 = e["seuil_5pc"], e["seuil_1pc"]
    if p <= s5:
        q = 0.4 * p / s5
    elif p <= s1:
        q = 0.4 + 0.2 * (p - s5) / (s1 - s5)
    else:
        q = 0.6 + 0.4 * (p - s1) / (1 - s1)
    return round(100 * q)


def taux(liste):
    return {"n": len(liste), "taux": (sum(liste) / len(liste)) if liste else None}


def bilan(docs, scores, logits):
    out = {}
    hum = [s for d, s in zip(docs, scores) if d["label"] == 0]
    ia = [s for d, s in zip(docs, scores) if d["label"] == 1]
    y = [d["label"] for d in docs if d["label"] in (0, 1)]
    l = [x for d, x in zip(docs, logits) if d["label"] in (0, 1)]
    out["n_humains"], out["n_ia"] = len(hum), len(ia)
    out["auroc"] = auroc(y, l) if hum and ia else None
    out["ia_reperee"] = taux([s >= 60 for s in ia])                      # « probablement IA » sur le site
    out["ia_reperee_ou_incertain"] = taux([s >= 40 for s in ia])
    out["humains_accuses"] = taux([s >= 60 for s in hum])                # fausses alertes du site
    out["humains_incertains"] = taux([40 <= s < 60 for s in hum])
    groupes = defaultdict(lambda: defaultdict(list))
    for d, s in zip(docs, scores):
        n = nb_mots(d["text"])
        long_ = "moins de 150 mots" if n < 150 else "150 à 400 mots" if n <= 400 else "plus de 400 mots"
        if d["label"] == 0:
            groupes["fausses_alertes_par_genre"][d["genre"]].append(s >= 60)
            groupes["fausses_alertes_par_longueur"][long_].append(s >= 60)
        elif d["label"] == 1:
            groupes["ia_reperee_par_generateur"][d["generator"]].append(s >= 60)
            groupes["ia_reperee_par_consigne"][d["task"]].append(s >= 60)
            groupes["ia_reperee_par_longueur"][long_].append(s >= 60)
        else:
            groupes["textes_mixtes"]["signalés « probablement IA »"].append(s >= 60)
            groupes["textes_mixtes"]["signalés au moins « incertain »"].append(s >= 40)
    for k, g in groupes.items():
        out[k] = {kk: taux(v) for kk, v in sorted(g.items())}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--humains", nargs="*", default=[], help="dossiers ou fichiers de textes humains (txt, md, docx, pdf, odt)")
    ap.add_argument("--livres", type=int, default=40, help="extraits de livres anciens du domaine public (0 = aucun)")
    ap.add_argument("--generateurs", default="anthropic:claude-sonnet-5-5")
    ap.add_argument("--par-generateur", type=int, default=60)
    ap.add_argument("--depot", default="")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--modele-local", default=None, help="dossier ONNX local (tests), au lieu du modèle publié")
    ap.add_argument("--sortie", default=str(RACINE / "resultats" / "test_independant.json"))
    ap.add_argument("--etape", choices=["tout", "humains", "generer", "mesurer"], default="tout",
                    help="pour lancer la génération dans l'environnement vLLM et le reste ailleurs")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    DOSSIER.mkdir(parents=True, exist_ok=True)
    cal = json.loads((RACINE / "js" / "calibration.json").read_text())
    depot = a.depot or cal["classifieur"]["id"]

    # 1. Humains (sans recouvrement avec l'entraînement).
    fh = DOSSIER / "humains.jsonl"
    humains = lire_jsonl(fh)
    if not humains:
        brut = humains_fichiers(a.humains) + humains_livres(a.livres, rng)
        vus = deja_vus()
        for r in brut:
            r["id"] = "th-" + hashlib.sha1(r["text"][:300].encode()).hexdigest()[:12]
            r["label"] = 0
        humains = [r for r in {r["id"]: r for r in brut}.values()
                   if hashlib.sha1(normaliser(r["text"])[:300].encode()).hexdigest() not in vus]
        ecrire_jsonl(fh, humains)
    if not humains:
        sys.exit("Aucun texte humain : indiquez un dossier avec --humains (ou laissez --livres).")
    from collections import Counter
    print(f"{len(humains)} textes humains : {dict(Counter(h['genre'] for h in humains))}", file=sys.stderr)
    if a.etape == "humains":
        return

    # 2. IA, 3. variantes.
    gens = [g.strip() for g in a.generateurs.split(",") if g.strip()]
    ia = generer_ia(humains, gens, a.par_generateur, rng) if a.etape in ("tout", "generer") else lire_jsonl(DOSSIER / "ia.jsonl")
    if a.etape == "generer":
        return
    var = variantes(humains, ia, random.Random(a.seed + 1))
    docs = humains + ia + var
    print(f"{len(ia)} textes d'IA, {len(var)} variantes (retouchés, mixtes)", file=sys.stderr)

    # 4. Mesure.
    logits = scorer_onnx([d["text"] for d in docs], depot, local=a.modele_local)
    scores = [score_site(x, cal) for x in logits]
    for d, x, s in zip(docs, logits, scores):
        d["logit"], d["score"] = x, s
    ecrire_jsonl(DOSSIER / "scores.jsonl", docs)  # textes compris : reste dans le dossier de travail

    res = {"date": __import__("datetime").date.today().isoformat(), "detecteur": depot, "seuil": "score du site ≥ 60 % (« probablement IA »)",
           "generateurs": sorted({d["generator"] for d in ia}), "genres_humains": sorted({h["genre"] for h in humains}),
           **bilan(docs, scores, logits)}
    Path(a.sortie).parent.mkdir(parents=True, exist_ok=True)
    Path(a.sortie).write_text(json.dumps(res, indent=2, ensure_ascii=False))

    pc = lambda t: "—" if t["taux"] is None else f"{100 * t['taux']:.1f} % (sur {t['n']})"
    print("\n=== Test indépendant ===")
    print(f"AUROC : {res['auroc']:.4f}" if res["auroc"] is not None else "AUROC : —")
    print(f"Textes d'IA repérés (score ≥ 60 %) : {pc(res['ia_reperee'])}")
    print(f"Textes humains accusés à tort (score ≥ 60 %) : {pc(res['humains_accuses'])}")
    for k in ("ia_reperee_par_generateur", "ia_reperee_par_consigne", "fausses_alertes_par_genre", "textes_mixtes"):
        print(f"\n{k.replace('_', ' ')} :")
        for kk, v in res.get(k, {}).items():
            print(f"   {kk:45s} {pc(v)}")
    erreurs = sorted([d for d in docs if d["label"] == 0 and d["score"] >= 60], key=lambda d: -d["score"])[:8]
    if erreurs:
        print("\nTextes humains accusés à tort (à relire) :")
        for d in erreurs:
            print(f"   {d['score']:3d} %  [{d['genre']}] {d.get('origine', '')} : {d['text'][:110]!r}")
    rates = sorted([d for d in docs if d["label"] == 1 and d["score"] < 60], key=lambda d: d["score"])[:8]
    if rates:
        print("\nTextes d'IA non repérés (à relire) :")
        for d in rates:
            print(f"   {d['score']:3d} %  [{d['generator']} / {d['task']}] {d['text'][:110]!r}")
    print(f"\nRésultats (chiffres seulement) → {a.sortie}")


if __name__ == "__main__":
    main()
