# Relais de vérification par IA (optionnel)

Le site est statique (GitHub Pages) : une clé API placée dans son code serait visible par tout le monde. Ce relais est un petit serveur gratuit sur Cloudflare Workers qui garde votre clé Anthropic secrète. Une fois déployé, tous les visiteurs peuvent utiliser la vérification par IA sans fournir de clé. Les appels sont facturés sur **votre** compte Anthropic : la limite de débit incluse (10 analyses par minute et par IP) et la restriction d'origine limitent les abus.

## Déploiement (5 minutes)

1. Créez un compte gratuit sur https://dash.cloudflare.com et installez Node.js.
2. Dans ce dossier :
   ```bash
   npx wrangler login
   npx wrangler secret put ANTHROPIC_API_KEY   # collez votre clé sk-ant-…
   npx wrangler deploy
   ```
3. Wrangler affiche une adresse du type `https://detecteur-ia-relais.<vous>.workers.dev`.
4. Ouvrez `js/config.js` à la racine du site et collez cette adresse dans `RELAY_URL`, puis poussez la modification sur GitHub.

## Réglages (`wrangler.toml`)

| Variable | Rôle |
|---|---|
| `ALLOWED_ORIGIN` | Adresse(s) du site autorisées, séparées par des virgules. |
| `MODEL` | Modèle Claude utilisé (`claude-haiku-5-5` est le moins cher, `claude-sonnet-5-5` le plus précis). |
| `MAX_CHARS` | Longueur maximale envoyée par appel. |
| `RATE_LIMITER` | Nombre d'analyses par minute et par IP. |
