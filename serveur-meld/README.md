---
title: Detecteur IA - MELD
emoji: 🔎
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Serveur MELD (textes en anglais)

[MELD](https://arxiv.org/abs/2605.06903) est un détecteur entraîné (encodeur d'environ 1 milliard de paramètres, licence MIT). C'est le meilleur détecteur open source du banc d'essai RAID, au niveau des services commerciaux. Il ne traite que l'anglais : le site l'utilise automatiquement quand le texte est en anglais.

Il est trop lourd pour tourner dans un navigateur. Ce dossier le transforme en petit serveur gratuit sur Hugging Face Spaces.

## Déploiement

1. Créez un compte sur https://huggingface.co puis un Space : **New Space** → nom `detecteur-ia-meld` → SDK **Docker** → matériel **CPU basic (gratuit)**.
2. Envoyez les 4 fichiers de ce dossier (`app.py`, `Dockerfile`, `requirements.txt`, `README.md`) dans le Space (onglet **Files** → **Add file** → **Upload files**).
3. Attendez la fin de la construction (5 à 10 minutes). Le premier démarrage télécharge MELD (environ 4 Go).
4. L'adresse du serveur est `https://<votre-nom>-detecteur-ia-meld.hf.space`. Collez-la dans `MELD_URL` de `js/config.js`.

## Réglages (Settings → Variables)

| Variable | Défaut | Rôle |
|---|---|---|
| `ALLOWED_ORIGINS` | `https://alexis-peche-project.github.io` | Sites autorisés à appeler le serveur |
| `RATE_PER_MIN` | `12` | Analyses par minute et par adresse IP |
| `MAX_CHARS` | `60000` | Longueur maximale analysée |

Sur le processeur gratuit, une analyse prend de 5 à 30 secondes. Un Space gratuit se met en veille après 48 h sans visite ; la première analyse suivante attend son réveil (1 à 2 minutes).
