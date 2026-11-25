// Calculs de score, sans dépendance au navigateur (testables avec Node).
import { cv } from './heuristics.js';

const sigmoid = x => 1 / (1 + Math.exp(-x));

// Résume les statistiques par phrase renvoyées par le modèle de langage.
// D : courbure normalisée par token = moyenne(log p - E[log p]) / racine(moyenne(Var[log p])).
// Proche de 0 ou négatif pour un humain, nettement positif pour une IA.
export function lmSummary(stats, calib, minTokensPerSentence = 6) {
  const T = stats.reduce((a, s) => ({ n: a.n + s.n, lp: a.lp + s.lp, mu: a.mu + s.mu, v: a.v + s.var, t10: a.t10 + s.top10, t1: a.t1 + s.top1 }), { n: 0, lp: 0, mu: 0, v: 0, t10: 0, t1: 0 });
  if (T.n < 2) return null;
  const D = ((T.lp - T.mu) / T.n) / Math.sqrt(T.v / T.n);
  const ppl = Math.exp(-T.lp / T.n);
  const sentPpl = stats.filter(s => s.n >= minTokensPerSentence).map(s => Math.exp(-s.lp / s.n));
  const burstiness = cv(sentPpl);
  const raw = sigmoid((D - calib.center) / calib.scale);
  const reliability = Math.min(1, T.n / 120);
  const prob = 0.5 + (raw - 0.5) * reliability;
  const perSentence = stats.map(s => s.n >= minTokensPerSentence
    ? sigmoid(((s.lp - s.mu) / s.n / Math.sqrt(s.var / s.n) - calib.center) / (calib.scale * 2.5))
    : null);
  return { D, ppl, burstiness, top10: T.t10 / T.n, top1: T.t1 / T.n, tokens: T.n, prob, perSentence, reliability };
}

// Combine les méthodes actives. Chaque entrée : { p: 0..1, w: poids }.
export function ensemble(parts, { strongSign = false, words = 0 } = {}) {
  const active = parts.filter(x => x && Number.isFinite(x.p));
  const W = active.reduce((a, x) => a + x.w, 0);
  let p = W ? active.reduce((a, x) => a + x.p * x.w, 0) / W : 0.5;
  if (strongSign) p = Math.max(p, 0.85);
  const sides = active.map(x => x.p >= 0.5);
  const agree = active.length > 1 && sides.every(s => s === sides[0]);
  const spread = active.length > 1 ? Math.max(...active.map(x => x.p)) - Math.min(...active.map(x => x.p)) : 0;
  let confidence = 'moyenne';
  if (words < 80 || spread > 0.45) confidence = 'faible';
  else if ((agree && words >= 250 && Math.abs(p - 0.5) > 0.25) || strongSign) confidence = 'élevée';
  if (active.length === 1 && confidence === 'élevée' && !strongSign) confidence = 'moyenne';
  return { p, score: Math.round(p * 100), confidence };
}

// Combinaison réglée par le banc d'essai (régression logistique sur la partie « val »).
// features : { clf: logit, lm: D, heur: 0..1, juge: 0..1 }. Renvoie null si aucun réglage ne correspond.
export function calibratedEnsemble(calibration, features) {
  const avail = Object.keys(features).filter(k => Number.isFinite(features[k])).sort();
  const ens = calibration?.ensembles?.[avail.join('+')];
  if (!ens) return null;
  const z = ens.features.reduce((a, f, i) => a + ens.coef[i] * features[f], ens.intercept);
  const p = sigmoid(z);
  return { p, key: avail.join('+'), seuil1: ens.seuil_1pc, seuil5: ens.seuil_5pc, flagged1: p > ens.seuil_1pc, flagged5: p > ens.seuil_5pc, tpr1: ens.tpr_1pc_test, auroc: ens.auroc_test };
}

// Détection simple de la langue (français, anglais ou autre) par mots outils.
export function detectLang(text) {
  const w = (text.toLowerCase().match(/[\p{L}']+/gu) || []).slice(0, 3000);
  const FR = new Set(['le', 'la', 'les', 'des', 'est', 'et', 'une', 'du', 'que', 'pour', 'dans', 'pas', 'qui', 'sur', 'au', 'avec', 'ce', 'il', 'elle', 'nous', 'vous', 'mais', 'ou', 'son', 'sa', 'ses', 'aux', 'par', 'plus', 'cette']);
  const EN = new Set(['the', 'and', 'is', 'of', 'to', 'that', 'in', 'it', 'for', 'with', 'as', 'was', 'are', 'this', 'be', 'by', 'not', 'or', 'have', 'from', 'at', 'which', 'an', 'they', 'you', 'we', 'but', 'his', 'her']);
  let fr = 0, en = 0;
  for (const x of w) { if (FR.has(x)) fr++; if (EN.has(x)) en++; }
  if (fr + en < 5) return 'autre';
  return fr >= en ? 'fr' : 'en';
}

export const verdict = s => s < 20 ? 'Très probablement écrit par un humain'
  : s < 40 ? 'Probablement écrit par un humain'
  : s < 60 ? 'Mixte ou incertain'
  : s < 80 ? 'Probablement généré par une IA'
  : 'Très probablement généré par une IA';

// Calibration : seuil qui sépare au mieux les deux groupes, et pente adaptée à leur écart.
export function calibrate(humanD, aiD) {
  if (!humanD.length || !aiD.length) throw new Error('Il faut au moins un texte de chaque type.');
  const all = [...humanD.map(d => [d, 0]), ...aiD.map(d => [d, 1])].sort((a, b) => a[0] - b[0]);
  let best = { acc: -1, t: 0 };
  for (let i = 0; i <= all.length; i++) {
    const t = i === 0 ? all[0][0] - 1e-3 : i === all.length ? all[i - 1][0] + 1e-3 : (all[i - 1][0] + all[i][0]) / 2;
    const ok = all.filter(([d, y]) => (d > t) === (y === 1)).length;
    const acc = ok / all.length;
    if (acc > best.acc) best = { acc, t };
  }
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const gap = Math.abs(mean(aiD) - mean(humanD));
  return { center: best.t, scale: Math.max(0.02, gap / 4), accuracy: best.acc };
}
