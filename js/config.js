// Réglages du site.

// RELAY_URL : adresse de votre relais Cloudflare pour la vérification par IA (voir relais/README.md).
// Laissez vide si vous n'avez pas de relais : les visiteurs peuvent utiliser leur propre clé API.
export const RELAY_URL = '';

// MELD_URL : adresse de votre serveur MELD, détecteur entraîné pour l'anglais (voir serveur-meld/README.md).
export const MELD_URL = '';

// CLASSIFIER_ID : modèle entraîné publié sur Hugging Face (voir entrainement/README.md).
// Laissez vide pour utiliser celui indiqué par js/calibration.json, produit par le banc d'essai.
export const CLASSIFIER_ID = '';

// Poids utilisés tant que le banc d'essai n'a pas produit js/calibration.json.
export const WEIGHTS = { clf: 0.6, lm: 0.5, judge: 0.3, heuristics: 0.2, meld: 0.8 };

// Réglage par défaut du modèle de langage local (remplacé par js/calibration.json s'il existe).
export const DEFAULT_CALIBRATION = { center: 0.12, scale: 0.05 };
