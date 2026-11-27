# Entraînement et banc d'essai

Ce dossier reproduit la méthode des détecteurs professionnels (GPTZero, Pangram) pour le français :
un classifieur entraîné sur des textes humains et leurs miroirs IA, amélioré par la recherche d'exemples difficiles,
puis mesuré sur des textes de test jamais vus.

**Le plus simple : ouvrez [`entrainement.ipynb`](entrainement.ipynb) dans Google Colab ou Kaggle (GPU gratuit) et lancez tout.**
[Ouvrir dans Colab](https://colab.research.google.com/github/alexis-peche-project/detecteur-ia/blob/main/entrainement/entrainement.ipynb)

## Les étapes

| Script | Rôle | Sortie |
|---|---|---|
| `donnees.py` | Textes humains français antérieurs à 2023 : web (FineWeb-2, filtré par date), Wikipédia, critiques AlloCiné, FAQ. Option `--local` pour ajouter vos propres textes humains. | `donnees/humains.jsonl` |
| `generer.py` | Miroirs IA de chaque texte humain : même sujet, genre et longueur. Trois consignes : écrire, écrire « comme un humain », réécrire. Plusieurs modèles ouverts (vLLM) et, si vous avez des clés, Claude, GPT et Mistral. | `donnees/ia.jsonl` |
| `entrainer.py` | Entraîne mmBERT-small, puis repère les textes humains de la réserve pris pour de l'IA, leur fabrique des miroirs et reprend l'entraînement (`--tours`). | `modeles/classifieur/` |
| `exporter.py` | Conversion ONNX quantifiée 8 bits, publication sur Hugging Face. | dépôt `votre-nom/detecteur-ia-fr` |
| `evaluer.py` | Banc d'essai : AUROC, part de textes d'IA repérés à 1 % et 5 % de faux positifs, détail par IA et par genre. Règle la combinaison des méthodes. | `../resultats/benchmark.json`, `../js/calibration.json` |

Les textes d'un même sujet (l'humain et ses miroirs) restent toujours dans la même partie (entraînement, réglage ou test).
Ainsi, le test ne contient aucun sujet déjà vu. Un générateur peut être exclu de l'entraînement
(`--exclure-generateur`) pour mesurer la détection d'une IA inconnue.

## Ce qui rend le détecteur fiable, par ordre d'importance

1. **La diversité des IA génératrices.** Des textes des IA actuelles (Claude, GPT, Mistral, Gemini) comptent plus que tout le reste.
2. **La diversité des textes humains.** Ajoutez avec `donnees.py --local` des textes proches de ceux que vous vérifierez,
   par exemple des copies d'élèves ou des rapports antérieurs à 2023. C'est le meilleur moyen de réduire les faux positifs.
3. **La quantité.** Pangram est parti d'environ 1 million de documents ; ce pipeline vise 10 000 à 20 000 textes sur un GPU gratuit.
   Augmentez `--par-source` si vous disposez de plus de temps de calcul.

## Améliorer un détecteur déjà entraîné

Ajoutez la sortie du run précédent comme entrée Kaggle (Add Input → votre notebook) : le notebook reprend
les textes déjà collectés et générés, puis
- ajoute `AJOUT_HUMAINS` nouveaux textes humains par source (`donnees.py --ajouter`), avec leurs miroirs ;
- réécrit `HUMANISES` textes d'IA dans les styles de l'outil de réécriture du site (`generer.py --humaniser`,
  consignes dans `js/reecriture.json`), pour que le détecteur reconnaisse les textes « humanisés » ;
- réentraîne sur l'ensemble et publie le résultat. Le banc d'essai donne le taux de textes repérés
  pour chaque consigne, dont chaque style de réécriture.

La compression du modèle pour le navigateur est vérifiée (`quantifier.py`) : la version compressée n'est publiée
que si ses scores restent proches de ceux du modèle complet.

## Tester sans GPU

```bash
python tests_locaux.py
```

Ce script fait tourner tout le pipeline avec un mini-modèle factice et un générateur factice,
pour vérifier le code. Ses scores n'ont aucune valeur.
