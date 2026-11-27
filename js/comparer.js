// Outils de comparaison de textes : mots, n-grammes, recouvrement, similarité TF-IDF,
// surlignage des passages communs, empreinte SHA-256. Tout est calculé dans le navigateur.
import { esc } from './commun.js';
import { splitSentences } from './heuristics.js';

const sansAccents = s => s.normalize('NFD').replace(/[̀-ͯ]/g, '');
export function mots(t) {
  const out = [];
  for (const m of String(t).matchAll(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu)) {
    out.push({ w: sansAccents(m[0].toLowerCase().replace(/’/g, "'")), start: m.index, end: m.index + m[0].length });
  }
  return out;
}
export function ngrammes(toks, n) {
  const s = new Set();
  for (let i = 0; i + n <= toks.length; i++) s.add(toks.slice(i, i + n).map(t => t.w).join(' '));
  return s;
}
// Part des n-grammes de a présents dans b.
export function recouvrement(a, b, n = 3) {
  const A = ngrammes(typeof a === 'string' ? mots(a) : a, n), B = b instanceof Set ? b : ngrammes(typeof b === 'string' ? mots(b) : b, n);
  if (!A.size) return 0;
  let k = 0;
  A.forEach(x => { if (B.has(x)) k++; });
  return k / A.size;
}

// Similarité cosinus TF-IDF entre tous les documents (mots de 3 lettres ou plus).
export function matriceTfidf(textes) {
  const docs = textes.map(t => {
    const tf = new Map();
    mots(t).forEach(({ w }) => { if (w.length >= 3) tf.set(w, (tf.get(w) || 0) + 1); });
    return tf;
  });
  const df = new Map();
  docs.forEach(d => d.forEach((_, w) => df.set(w, (df.get(w) || 0) + 1)));
  const N = docs.length;
  const vecs = docs.map(d => {
    const v = new Map();
    let norme = 0;
    d.forEach((c, w) => { const x = (1 + Math.log(c)) * Math.log((N + 1) / (df.get(w) + 0.5)); v.set(w, x); norme += x * x; });
    return { v, norme: Math.sqrt(norme) || 1 };
  });
  return vecs.map(a => vecs.map(b => {
    let s = 0;
    const [p, g] = a.v.size < b.v.size ? [a, b] : [b, a];
    p.v.forEach((x, w) => { const y = g.v.get(w); if (y) s += x * y; });
    return s / (a.norme * b.norme);
  }));
}

// Surligne dans `texte` les mots qui font partie d'un n-gramme présent dans `autre`.
export function surligner(texte, autre, n = 3, cls = 'commun') {
  const toks = mots(texte), B = autre instanceof Set ? autre : ngrammes(mots(autre), n);
  const marque = new Array(toks.length).fill(false);
  for (let i = 0; i + n <= toks.length; i++) {
    if (B.has(toks.slice(i, i + n).map(t => t.w).join(' '))) for (let k = i; k < i + n; k++) marque[k] = true;
  }
  let html = '', pos = 0, i = 0;
  while (i < toks.length) {
    if (!marque[i]) { i++; continue; }
    let j = i;
    while (j + 1 < toks.length && marque[j + 1]) j++;
    html += esc(texte.slice(pos, toks[i].start)) + `<mark class="${cls}">${esc(texte.slice(toks[i].start, toks[j].end))}</mark>`;
    pos = toks[j].end; i = j + 1;
  }
  return html + esc(texte.slice(pos));
}

// Pour chaque phrase du texte final : part reprise de la source et passage source le plus proche.
export function alignerPhrases(final, source) {
  const srcPhrases = splitSentences(source).map(s => ({ texte: s.text, toks: mots(s.text) })).filter(s => s.toks.length);
  const tous3 = ngrammes(mots(source), 3), tous2 = ngrammes(mots(source), 2);
  return splitSentences(final).map(s => {
    const toks = mots(s.text);
    const n = toks.length >= 8 ? 3 : 2;
    const part = toks.length < 3 ? 0 : recouvrement(toks, n === 3 ? tous3 : tous2, n);
    let proche = null, best = 0;
    if (part > 0.1) {
      const A = ngrammes(toks, 2);
      for (const p of srcPhrases) {
        const B = ngrammes(p.toks, 2);
        let k = 0; A.forEach(x => { if (B.has(x)) k++; });
        const sc = k / (A.size || 1);
        if (sc > best) { best = sc; proche = p.texte; }
      }
    }
    return { ...s, mots: toks.length, part, proche };
  });
}

// Empreinte SHA-256 du texte (espaces normalisés) : un seul caractère changé donne une autre empreinte.
export async function empreinte(texte) {
  const norm = String(texte).normalize('NFC').replace(/\s+/g, ' ').trim();
  const h = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(norm)));
  const hex = [...h].map(b => b.toString(16).padStart(2, '0')).join('');
  return { hex, court: hex.slice(0, 16).toUpperCase().match(/.{4}/g).join('-') };
}

export const separerTextes = s => String(s).split(/^\s*---\s*$/m).map(x => x.trim()).filter(Boolean);
