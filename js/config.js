// Réglages du site.
// RELAY_URL : adresse de votre relais Cloudflare (voir relais/README.md).
// Laissez vide si vous n'avez pas déployé de relais : les visiteurs pourront
// toujours utiliser leur propre clé API.
export const RELAY_URL = '';

// Poids de chaque méthode dans le score final (renormalisés si une méthode est désactivée).
export const WEIGHTS = { lm: 0.5, judge: 0.3, heuristics: 0.2 };

// Calibration par défaut du modèle de langage local (modifiable sur le site, section Calibration).
// Score de courbure normalisé (Fast-DetectGPT) au-delà duquel un texte est jugé « IA » à 50 %.
export const DEFAULT_CALIBRATION = { center: 0.12, scale: 0.05 };
