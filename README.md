# Détecteur de texte IA

Détecteur de textes générés par intelligence artificielle : gratuit, sans limite de mots, open source, et respectueux de la vie privée.

**Site : https://alexis-peche-project.github.io/detecteur-ia/**

## Ce qu'il fait

- Analyse un texte collé ou un document importé : PDF (texte), Word `.docx`, OpenDocument `.odt`, `.rtf`, `.txt`, `.md`, `.html`.
- Combine trois méthodes, comme les détecteurs commerciaux :

| Méthode | Principe | Poids |
|---|---|---|
| **Modèle de langage local** | Qwen2.5-0.5B-Instruct (multilingue) tourne dans le navigateur avec [Transformers.js](https://github.com/huggingface/transformers.js). Calcule la perplexité (comme le GPTZero d'origine), la burstiness, la part de mots du top 10 (GLTR) et surtout la **courbure de probabilité conditionnelle** de [Fast-DetectGPT](https://arxiv.org/abs/2310.05130) (ICLR 2024). | 50 % |
| **Vérification par IA** | Un grand modèle (Claude, GPT ou Mistral) relit le texte avec les critères Wikipédia, rend une probabilité et cite ses indices. | 30 % |
| **Signes Wikipédia et style** | Repère les signes du guide [« Signs of AI writing »](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing) du WikiProject AI Cleanup, adaptés au français : vocabulaire typique, parallélisme négatif, triplets, remarques éditoriales, conclusions, formules d'assistant, artefacts, mise en forme. Mesure aussi la régularité des phrases. | 20 % |

- Surligne chaque phrase selon sa probabilité d'avoir été générée et souligne chaque signe repéré.
- Exporte le rapport en texte ou en JSON.
- Propose une **calibration** : collez des textes dont vous connaissez l'origine, le seuil s'ajuste à vos données.

## Confidentialité

Le texte ne quitte pas le navigateur. Seule exception : si vous activez la vérification par IA, le texte est envoyé au fournisseur choisi (ou au relais du site).

## La courbure de probabilité, en bref

Pour chaque token, on compare la log-probabilité du mot réellement écrit à la log-probabilité moyenne attendue sous la distribution du modèle :

```
D = moyenne(log p(xₜ) − E[log p]) / √moyenne(Var[log p])
```

Un humain choisit souvent des mots moins attendus que ce que le modèle prédit (D proche de 0 ou négatif). Un modèle de langage choisit presque toujours des mots très attendus (D nettement positif). La probabilité affichée est `sigmoïde((D − seuil) / échelle)`, rapprochée de 50 % quand le texte compte moins de 120 tokens.

## Clés API et relais

Le site est statique : **ne mettez jamais de clé API dans le code**, elle serait publique. Deux possibilités :

1. Chaque visiteur entre sa propre clé dans « Réglages de la vérification par IA ». La clé reste dans son navigateur et n'est envoyée qu'au fournisseur.
2. Vous déployez le relais Cloudflare du dossier [`relais/`](relais/README.md), qui garde votre clé secrète, puis vous indiquez son adresse dans [`js/config.js`](js/config.js).

## Structure

```
index.html          page
css/style.css       styles
js/app.js           interface et orchestration
js/lm-worker.js     modèle de langage (Web Worker)
js/scoring.js       calcul des scores, combinaison, calibration
js/heuristics.js    signes Wikipédia et statistiques de style
js/llm-judge.js     vérification par IA (Anthropic, OpenAI, Mistral, relais)
js/prompt.js        consigne du vérificateur
js/docs.js          extraction de texte des documents
js/config.js        réglages (relais, poids, calibration)
relais/             relais Cloudflare Worker optionnel
```

## Développement local

Aucune compilation. Servez le dossier avec n'importe quel serveur statique :

```bash
python3 -m http.server 8000
```

puis ouvrez http://localhost:8000.

## Limites

Aucun détecteur n'est une preuve, y compris les services payants. Les résultats sont moins fiables sur les textes courts (moins de 150 mots), les textes d'IA retravaillés par un humain, et les textes humains très formels ou écrits par des non-natifs. Le modèle local est petit (0,5 milliard de paramètres) pour pouvoir tourner dans un navigateur : la calibration sur vos propres textes améliore nettement sa précision.

## Licence

MIT
