// Test du perroquet : on coupe des passages en deux, une IA devine la suite, on compare avec la vraie suite.
// Un texte écrit par une IA suit souvent le chemin le plus prévisible : la suite devinée s'en rapproche
// (idée de la méthode DNA-GPT, Yang et al., 2023). Indicateur non mesuré par le banc d'essai.
import { splitSentences, words } from './heuristics.js';
import { appelIA, nettoyer } from './ia.js';
import { recouvrement } from './comparer.js';

function extraits(texte, max = 3) {
  let blocs = texte.split(/\n\s*\n/).map(p => p.trim()).filter(p => words(p).length >= 60);
  if (!blocs.length) {
    // Pas de paragraphe assez long : fenêtres d'environ 140 mots, en phrases entières.
    blocs = [];
    let cur = '';
    for (const s of splitSentences(texte)) {
      cur += (cur ? ' ' : '') + s.text;
      if (words(cur).length >= 140) { blocs.push(cur); cur = ''; }
    }
    if (words(cur).length >= 60) blocs.push(cur);
  }
  if (blocs.length > max) blocs = Array.from({ length: max }, (_, k) => blocs[Math.round(k * (blocs.length - 1) / (max - 1))]);
  return blocs.map(b => {
    const ph = splitSentences(b).map(s => s.text);
    let coupe = 1, n = 0;
    const total = words(b).length;
    for (let i = 0; i < ph.length - 1; i++) { n += words(ph[i]).length; if (n >= total / 2) { coupe = i + 1; break; } }
    return { debut: ph.slice(0, coupe).join(' '), suite: ph.slice(coupe).join(' ') };
  }).filter(e => words(e.debut).length >= 20 && words(e.suite).length >= 20);
}

export async function testPerroquet(cfg, texte, { signal, onProgress } = {}) {
  const ex = extraits(texte);
  if (!ex.length) throw new Error('Texte trop court pour le test du perroquet (il faut au moins un passage de 60 mots).');
  const res = [];
  for (let i = 0; i < ex.length; i++) {
    onProgress?.(i, ex.length);
    const n = words(ex[i].suite).length;
    const devine = nettoyer(await appelIA(cfg, {
      system: 'Tu continues des textes. Tu écris la suite la plus naturelle possible, dans la même langue et le même style.',
      user: `Voici le début d'un texte. Écris la suite directe, environ ${n} mots, sans répéter le début. Donne uniquement la suite.\n\n<debut>\n${ex[i].debut}\n</debut>`,
      maxTokens: 2000, signal
    }));
    res.push({ ...ex[i], devine, r2: recouvrement(ex[i].suite, devine, 2), r3: recouvrement(ex[i].suite, devine, 3) });
  }
  onProgress?.(ex.length, ex.length);
  const moy = res.reduce((a, r) => a + r.r3, 0) / res.length;
  const niveau = moy >= 0.15 ? { cls: 'ia', titre: 'Très prévisible pour une IA', texte: 'L\'IA a deviné une bonne partie des vraies formulations.' }
    : moy >= 0.07 ? { cls: 'doute', titre: 'Assez prévisible', texte: 'L\'IA retrouve certaines formulations, sans plus.' }
    : { cls: 'humain', titre: 'Peu prévisible', texte: 'La vraie suite part ailleurs que là où l\'IA l\'attendait.' };
  return { extraits: res, moyenne: moy, niveau };
}
