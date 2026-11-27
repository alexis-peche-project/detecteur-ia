// Calibrage par classe : scores de textes de référence écrits par les élèves, gardés sur cet appareil.
import { store } from './commun.js';

export function refClasse() {
  try { const r = JSON.parse(store.get('detia-classe')); return r?.scores?.length ? r : null; } catch { return null; }
}
export function enregistrerClasse(scores, nom) {
  store.set('detia-classe', JSON.stringify({ scores, nom: nom || 'Ma classe', date: new Date().toISOString().slice(0, 10) }));
}
export const oublierClasse = () => store.set('detia-classe', null);

// Part des textes de référence dont le score est inférieur à celui-ci.
export function rangClasse(score) {
  const r = refClasse();
  if (!r) return null;
  const sous = r.scores.filter(s => s < score).length + 0.5 * r.scores.filter(s => s === score).length;
  return { part: sous / r.scores.length, n: r.scores.length, nom: r.nom, max: Math.max(...r.scores) };
}
export function phraseClasse(score) {
  const r = rangClasse(score);
  if (!r) return '';
  const p = Math.round(r.part * 100);
  const depasse = score > r.max;
  return `Comparé aux ${r.n} textes de référence de « ${r.nom} » : score plus élevé que ${depasse ? 'tous' : `${p} % d'entre eux`}.`;
}
