"""Étape 2 : crée les textes d'IA « miroirs » des textes humains (méthode Pangram).

Chaque texte humain reçoit un ou plusieurs miroirs : même sujet, même genre, même longueur,
écrits par différents modèles. Le classifieur ne peut donc pas se fier au sujet ou à la longueur :
il doit apprendre le style.

Trois consignes, pour couvrir les usages réels :
  ecrire     : texte écrit à partir du sujet ;
  humaniser  : même chose, avec la demande d'écrire « comme un humain » (tentative de contournement) ;
  reecrire   : réécriture complète d'un texte humain (paraphrase par IA).

Générateurs (option --generateurs, séparés par des virgules) :
  vllm:<modèle Hugging Face>      modèle ouvert sur le GPU (Colab)
  anthropic:<modèle>              clé dans ANTHROPIC_API_KEY
  openai:<modèle>                 clé dans OPENAI_API_KEY
  mistral:<modèle>                clé dans MISTRAL_API_KEY
  factice                         pour les tests, sans modèle

Sortie : donnees/ia.jsonl (ajout à la suite ; les miroirs déjà créés sont sautés)
"""
import argparse
import gc
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from commun import DATA, ecrire_jsonl, lire_jsonl, nb_mots

GENERATEURS_COLAB = [
    "vllm:Qwen/Qwen2.5-7B-Instruct-AWQ",
    "vllm:hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "vllm:Qwen/Qwen2.5-3B-Instruct",
    "vllm:HuggingFaceTB/SmolLM3-3B",
    "vllm:microsoft/Phi-3.5-mini-instruct",
]

TACHES = [("ecrire", 0.45), ("humaniser", 0.30), ("reecrire", 0.25)]

STYLES = {
    "page web": "pour un site web",
    "article encyclopédique": "dans le style d'un article d'encyclopédie",
    "critique de film": "comme un spectateur qui donne son avis sur un film",
    "réponse à une question": "comme une réponse à une question posée sur une page d'aide",
}

CONSIGNE_HUMAIN = ("Écris de façon naturelle, comme le ferait une vraie personne : varie la longueur des phrases, "
                   "évite les listes, le gras, les titres et les formules toutes faites, et ne conclus pas par un résumé.")


def prompt(doc, tache):
    n = max(80, min(900, nb_mots(doc["text"])))
    style = STYLES.get(doc["domain"], f"de type « {doc['domain']} »")
    if tache == "reecrire":
        return (f"Réécris entièrement le texte suivant avec tes propres mots, en français, en gardant le sens "
                f"et une longueur proche (environ {n} mots). Donne uniquement le texte réécrit.\n\n{doc['text']}")
    p = (f"Écris un texte en français {style}, d'environ {n} mots, sur le sujet suivant : {doc['sujet']}\n"
         f"Donne uniquement le texte, sans introduction ni commentaire.")
    return p + ("\n" + CONSIGNE_HUMAIN if tache == "humaniser" else "")


def nettoyer_sortie(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    lignes = t.split("\n")
    if lignes and re.match(r"^(voici|bien sûr|certainement|d'accord|absolument)\b.*:\s*$", lignes[0].strip(), re.I):
        t = "\n".join(lignes[1:]).strip()
    return t.strip().strip('"').strip()


def choisir_tache(rng):
    r, acc = rng.random(), 0
    for t, p in TACHES:
        acc += p
        if r < acc:
            return t
    return TACHES[-1][0]


# ---------- Backends ----------
# Chaque backend traite les travaux par lots de TAILLE_LOT et appelle `sauver(lot, textes)` après chaque lot :
# si la session s'arrête, seuls les textes du lot en cours sont perdus.

TAILLE_LOT = 150


def lots(jobs, n=TAILLE_LOT):
    for i in range(0, len(jobs), n):
        yield jobs[i:i + n]


def backend_factice(jobs, sauver, **_):
    rng = random.Random(0)
    phrases = ["Il est important de noter que ce sujet joue un rôle clé.",
               "Cette évolution souligne l'importance cruciale de la question.",
               "En outre, plusieurs facteurs contribuent à cette dynamique.",
               "En conclusion, il s'agit d'un enjeu essentiel pour l'avenir."]
    for lot in lots(jobs):
        sauver(lot, [" ".join(rng.choice(phrases) for _ in range(12)) for _ in lot])


def backend_vllm(jobs, sauver, modele, max_tokens=1400, **_):
    import torch
    from vllm import LLM, SamplingParams
    cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    llm = LLM(model=modele, dtype="half" if cap < 8 else "auto", max_model_len=4096,
              gpu_memory_utilization=0.90, trust_remote_code=True, enforce_eager=cap < 8)
    tok = llm.get_tokenizer()
    contexte = 4096

    def ajuster(j):
        """Raccourcit une consigne trop longue (réécriture d'un long texte) pour tenir dans le contexte du modèle."""
        p = j["prompt"]
        n = len(tok.encode(p))
        while n > contexte - 600 and len(p) > 500:
            p = p[: int(len(p) * (contexte - 700) / n)]
            n = len(tok.encode(p))
        return p, max(200, min(max_tokens, contexte - n - 64))

    try:
        for lot in lots(jobs):
            prets = [ajuster(j) for j in lot]
            convs = [[{"role": "user", "content": p}] for p, _ in prets]
            params = [SamplingParams(temperature=j["temperature"], top_p=0.95, max_tokens=m, seed=j["seed"])
                      for j, (_, m) in zip(lot, prets)]
            try:
                outs = llm.chat(convs, params, use_tqdm=False)
                textes = [o.outputs[0].text for o in outs]
            except Exception as e:  # un lot en échec ne doit pas arrêter tout le générateur
                print(f"   lot ignoré ({type(e).__name__}: {str(e)[:200]})", file=sys.stderr, flush=True)
                textes = [""] * len(lot)
            sauver(lot, textes)
    finally:
        del llm
        gc.collect()
        try:
            torch.cuda.empty_cache()
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass


class CleRefusee(Exception):
    """Clé API invalide ou sans droits : inutile de réessayer."""


def _api_call(fournisseur, modele, j, max_tokens=1600):
    import requests
    for essai in range(5):
        try:
            if fournisseur == "anthropic":
                r = requests.post("https://api.anthropic.com/v1/messages", timeout=120, headers={
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                    "content-type": "application/json"},
                    json={"model": modele, "max_tokens": max_tokens, "temperature": j["temperature"],
                          "messages": [{"role": "user", "content": j["prompt"]}]})
                r.raise_for_status()
                return "".join(c.get("text", "") for c in r.json()["content"])
            url = {"openai": "https://api.openai.com/v1/chat/completions",
                   "mistral": "https://api.mistral.ai/v1/chat/completions"}[fournisseur]
            cle = os.environ["OPENAI_API_KEY" if fournisseur == "openai" else "MISTRAL_API_KEY"]
            body = {"model": modele, "messages": [{"role": "user", "content": j["prompt"]}]}
            if fournisseur == "mistral":
                body.update(temperature=j["temperature"], max_tokens=max_tokens)
            r = requests.post(url, timeout=180, headers={"authorization": f"Bearer {cle}"}, json=body)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            detail = (e.response.text[:300] if e.response is not None else "")
            if code in (400, 401, 402, 403, 404):
                # Erreur définitive (clé, crédit, modèle inconnu…) : inutile de réessayer.
                raise CleRefusee(f"{fournisseur} refuse la demande (erreur {code}) : {detail}") from e
            attente = 2 ** essai * 3
            print(f"   {fournisseur} : {e} ; nouvel essai dans {attente} s", file=sys.stderr)
            time.sleep(attente)
        except Exception as e:
            attente = 2 ** essai * 3
            print(f"   {fournisseur} : {e} ; nouvel essai dans {attente} s", file=sys.stderr)
            time.sleep(attente)
    return ""


def backend_api(jobs, sauver, fournisseur, modele, workers=6, **_):
    # Un premier appel seul vérifie la clé avant de lancer tous les travaux.
    _api_call(fournisseur, modele, {**jobs[0], "prompt": "Réponds seulement : ok", "temperature": 0}, max_tokens=5)
    with ThreadPoolExecutor(workers) as ex:
        for lot in lots(jobs):
            sauver(lot, list(ex.map(lambda j: _api_call(fournisseur, modele, j), lot)))


def generer(jobs, gen, sauver):
    if gen == "factice":
        return backend_factice(jobs, sauver)
    kind, _, modele = gen.partition(":")
    if kind == "vllm":
        return backend_vllm(jobs, sauver, modele)
    return backend_api(jobs, sauver, kind, modele)


# ---------- Programme ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generateurs", default=",".join(GENERATEURS_COLAB))
    ap.add_argument("--par-humain", type=int, default=1, help="miroirs par texte humain")
    ap.add_argument("--splits", default="train,val,test", help="parties concernées (pool : seulement via --ids)")
    ap.add_argument("--ids", help="fichier listant des id humains à traiter (exemples difficiles)")
    ap.add_argument("--max", type=int, default=0, help="limite de textes humains (0 = tous)")
    ap.add_argument("--sortie", default=str(DATA / "ia.jsonl"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    humains = lire_jsonl(DATA / "humains.jsonl")
    if a.ids:
        voulus = set(open(a.ids).read().split())
        humains = [h for h in humains if h["id"] in voulus]
    else:
        humains = [h for h in humains if h["split"] in a.splits.split(",")]
    rng.shuffle(humains)
    if a.max:
        humains = humains[:a.max]
    deja = {r["id"] for r in lire_jsonl(a.sortie)}
    gens = [g.strip() for g in a.generateurs.split(",") if g.strip()]

    # Répartit les travaux par générateur (un modèle chargé à la fois sur le GPU).
    par_gen = {g: [] for g in gens}
    k = 0
    for h in humains:
        for m in range(a.par_humain):
            g = gens[k % len(gens)]
            k += 1
            # Tous les tirages au sort sont faits avant de sauter les textes existants :
            # une reprise retombe exactement sur les mêmes travaux.
            tache = choisir_tache(rng)
            temperature, seed = round(rng.uniform(0.6, 1.0), 2), rng.randint(0, 10**6)
            hid = h["id"]
            jid = f"a-{hashlib.sha1(f'{hid}|{m}|{a.seed}'.encode()).hexdigest()[:12]}"
            if jid in deja:
                continue
            par_gen[g].append({"id": jid, "h": h, "tache": tache, "prompt": prompt(h, tache),
                               "temperature": temperature, "seed": seed})

    total = 0
    for g, jobs in par_gen.items():
        if not jobs:
            continue
        print(f"→ {g} : {len(jobs)} textes à créer", file=sys.stderr, flush=True)
        compte = {"gardes": 0, "faits": 0}

        def sauver(lot, textes, g=g, n=len(jobs), compte=compte):
            rows = []
            for j, t in zip(lot, textes):
                t = nettoyer_sortie(t or "")
                if nb_mots(t) < 40:
                    continue
                h = j["h"]
                rows.append({"id": j["id"], "text": t, "label": 1, "lang": h.get("lang", "fr"), "split": h["split"],
                             "source": h["source"], "domain": h["domain"], "pair_id": h["id"],
                             "generator": g.split(":", 1)[-1], "task": j["tache"]})
            ecrire_jsonl(a.sortie, rows, mode="a")
            compte["gardes"] += len(rows)
            compte["faits"] += len(lot)
            print(f"   {g} : {compte['faits']}/{n} ({compte['gardes']} gardés, sauvegardés)", file=sys.stderr, flush=True)

        try:
            generer(jobs, g, sauver)
        except Exception as e:
            print(f"   générateur ignoré ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
        total += compte["gardes"]
    print(f"{total} textes d'IA ajoutés → {a.sortie}")


if __name__ == "__main__":
    main()
