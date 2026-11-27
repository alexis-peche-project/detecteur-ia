"""Étape 1 : rassemble des textes humains en français, écrits avant l'arrivée de ChatGPT.

Sortie : donnees/humains.jsonl
Champs : id, text, label=0, source, domain, split (train/val/test/pool), lang, sujet

Exemples :
  python donnees.py --par-source 4000
  python donnees.py --local mes_textes.jsonl --domaine copie   # vos propres textes humains
"""
import argparse
import hashlib
import random
import re
import sys

from commun import DATA, ecrire_jsonl, lire_jsonl, nb_mots, split_de

# Sources publiques (Hugging Face, format Parquet). On ne garde que des textes antérieurs à fin 2022
# quand la date est connue, pour éviter les textes d'IA déjà présents sur le web.
SOURCES = [
    {"nom": "web", "repo": "HuggingFaceFW/fineweb-2", "config": "fra_Latn", "split": "train",
     "champ": "text", "domaine": "page web", "annee_max": 2022},
    {"nom": "wikipedia", "repo": "wikimedia/wikipedia", "config": "20231101.fr", "split": "train",
     "champ": "text", "titre": "title", "domaine": "article encyclopédique"},
    {"nom": "critiques", "repo": "tblard/allocine", "config": None, "split": "train",
     "champ": "review", "domaine": "critique de film"},
    {"nom": "faq", "repo": "almanach/hc3_french_ood", "config": "faq_fr_gouv", "split": "test",
     "champ": "answer", "titre": "question", "domaine": "réponse à une question"},
]

MIN_MOTS, MAX_MOTS = 120, 1200


def nettoyer(t):
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def couper(t, max_mots=MAX_MOTS):
    """Garde le début du texte, coupé à la fin d'un paragraphe ou d'une phrase."""
    mots = t.split()
    if len(mots) <= max_mots:
        return t
    court = " ".join(mots[:max_mots])
    fin = max(court.rfind("\n"), court.rfind(". "))
    return court[:fin + 1].strip() if fin > len(court) * 0.6 else court


def annee(row):
    for k in ("date", "dump"):
        v = str(row.get(k) or "")
        m = re.search(r"(19|20)\d\d", v)
        if m:
            return int(m.group(0))
    return None


def sujet_de(texte, titre=None):
    if titre:
        return str(titre).strip()[:200]
    premiere = re.split(r"(?<=[.!?])\s", texte.strip(), maxsplit=1)[0]
    return " ".join(premiere.split()[:30])


def depuis_hub(src, n, rng):
    from datasets import load_dataset
    print(f"→ {src['nom']} ({src['repo']})", file=sys.stderr)
    ds = load_dataset(src["repo"], src["config"], split=src["split"], streaming=True)
    ds = ds.shuffle(seed=rng.randint(0, 10**6), buffer_size=10_000)
    out = []
    for row in ds:
        if "filtre_label" in src and row.get("label") != src["filtre_label"]:
            continue
        if src.get("annee_max"):
            a = annee(row)
            if a is None or a > src["annee_max"]:
                continue
        t = nettoyer(str(row.get(src["champ"]) or ""))
        if nb_mots(t) < MIN_MOTS:
            continue
        t = couper(t)
        out.append({"text": t, "source": src["nom"], "domain": src["domaine"],
                    "sujet": sujet_de(t, row.get(src.get("titre", ""), None))})
        if len(out) >= n:
            break
    return out


def depuis_local(path, domaine):
    import json
    out = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            r = json.loads(l)
            t = nettoyer(r["text"] if isinstance(r, dict) else str(r))
            if nb_mots(t) >= MIN_MOTS // 2:
                out.append({"text": couper(t), "source": "local", "domain": r.get("domain", domaine),
                            "sujet": sujet_de(t, r.get("title"))})
    return out


def ajouter(a, deja):
    import json
    marque = DATA / "ajouts.json"
    faits = json.loads(marque.read_text()) if marque.exists() else []
    if a.etiquette in faits:
        print(f"Déjà fait : ajout « {a.etiquette} » ({len(deja)} textes humains).")
        return
    rng = random.Random(f"{a.seed}-{a.etiquette}")  # autre tirage que la collecte d'origine
    connus = {r["id"] for r in deja}
    rows = []
    voulues = set(a.sources.split(","))
    for src in SOURCES:
        if src["nom"] not in voulues:
            continue
        try:
            # On lit plus loin dans le flux pour trouver assez de textes nouveaux.
            docs = depuis_hub(src, a.ajouter * 2, rng)
        except Exception as e:
            print(f"   source ignorée : {e}", file=sys.stderr)
            continue
        n = 0
        for d in docs:
            cle = d["text"][:200]
            doc_id = f"h-{d['source']}-{hashlib.sha1(cle.encode()).hexdigest()[:12]}"
            if doc_id in connus:
                continue
            connus.add(doc_id)
            split = split_de(doc_id)
            if split == "train" and rng.random() < a.pool:
                split = "pool"
            rows.append({"id": doc_id, "label": 0, "lang": "fr", "split": split, "generator": "humain", **d})
            n += 1
            if n >= a.ajouter:
                break
    ecrire_jsonl(DATA / "humains.jsonl", rows, mode="a")
    marque.write_text(json.dumps(faits + [a.etiquette]))
    from collections import Counter
    print(f"{len(rows)} textes humains ajoutés (total {len(deja) + len(rows)})")
    print("par source :", dict(Counter(r["source"] for r in rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--par-source", type=int, default=3000, help="textes par source publique")
    ap.add_argument("--sources", default=",".join(s["nom"] for s in SOURCES))
    ap.add_argument("--local", help="fichier JSONL de textes humains ({'text': ...} par ligne)")
    ap.add_argument("--domaine", default="texte personnel")
    ap.add_argument("--pool", type=float, default=0.25,
                    help="part des textes d'entraînement gardés en réserve pour la recherche d'exemples difficiles")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--force", action="store_true", help="refaire la collecte même si elle existe déjà")
    ap.add_argument("--ajouter", type=int, default=0,
                    help="ajoute ce nombre de nouveaux textes par source à une collecte existante")
    ap.add_argument("--etiquette", default="ajout-1",
                    help="nom de l'ajout : un ajout déjà fait sous ce nom n'est pas refait (reprise sans doublon)")
    a = ap.parse_args()
    deja = lire_jsonl(DATA / "humains.jsonl")
    if deja and a.ajouter:
        return ajouter(a, deja)
    if deja and not a.force:
        print(f"Déjà fait : {len(deja)} textes humains dans {DATA / 'humains.jsonl'} (--force pour refaire).")
        return
    rng = random.Random(a.seed)

    docs = []
    voulues = set(a.sources.split(","))
    for src in SOURCES:
        if src["nom"] not in voulues:
            continue
        try:
            docs += depuis_hub(src, a.par_source, rng)
        except Exception as e:  # une source indisponible ne bloque pas les autres
            print(f"   source ignorée : {e}", file=sys.stderr)
    if a.local:
        docs += depuis_local(a.local, a.domaine)
    if not docs:
        sys.exit("Aucun texte récupéré.")

    vus, rows = set(), []
    for d in docs:
        cle = d["text"][:200]
        if cle in vus:
            continue
        vus.add(cle)
        doc_id = f"h-{d['source']}-{hashlib.sha1(cle.encode()).hexdigest()[:12]}"
        split = split_de(doc_id)
        if split == "train" and rng.random() < a.pool:
            split = "pool"
        rows.append({"id": doc_id, "label": 0, "lang": "fr", "split": split, "generator": "humain", **d})

    ecrire_jsonl(DATA / "humains.jsonl", rows)
    from collections import Counter
    print(f"{len(rows)} textes humains → {DATA / 'humains.jsonl'}")
    print("par split :", dict(Counter(r["split"] for r in rows)))
    print("par source :", dict(Counter(r["source"] for r in rows)))


if __name__ == "__main__":
    main()
