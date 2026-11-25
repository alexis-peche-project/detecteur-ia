// Web Worker : classifieur humain / IA entraîné (entrainement/), exécuté dans le navigateur.
// Même découpe et même normalisation que entrainement/commun.py.
import { AutoTokenizer, AutoModelForSequenceClassification, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';

env.allowLocalModels = false;
const MAX_TOKENS = 510;

let tok = null, model = null, loadedId = null, cancelled = false;

const MD = [
  [/\*\*([^*\n]+)\*\*/g, '$1'], [/__([^_\n]+)__/g, '$1'], [/^#{1,6}\s+/gm, ''], [/^\s*[-*•]\s+/gm, ''],
  [/[“”«»]/g, '"'], [/[‘’]/g, "'"], [/[​-‏⁠﻿]/g, ''], [/[ \t]+/g, ' '], [/\n{3,}/g, '\n\n']
];
const normaliser = t => MD.reduce((s, [r, x]) => s.replace(r, x), t).trim();

async function load(id) {
  if (model && loadedId === id) return;
  const progress_callback = p => {
    if (p.status === 'progress' && p.total) postMessage({ type: 'download', file: p.file, loaded: p.loaded, total: p.total });
  };
  let device = 'wasm';
  try { if (self.navigator?.gpu && await navigator.gpu.requestAdapter()) device = 'webgpu'; } catch { /* pas de WebGPU */ }
  tok = await AutoTokenizer.from_pretrained(id, { progress_callback });
  try {
    model = await AutoModelForSequenceClassification.from_pretrained(id, { device, dtype: 'q8', progress_callback });
  } catch (e) {
    model = await AutoModelForSequenceClassification.from_pretrained(id, { device: 'wasm', dtype: 'q8', progress_callback });
  }
  loadedId = id;
  postMessage({ type: 'ready' });
}

// Regroupe des phrases entières en fenêtres de MAX_TOKENS tokens au plus.
function windows(sentences) {
  const out = [];
  let cur = { text: '', n: 0, idx: [] };
  sentences.forEach((s, i) => {
    const t = normaliser(s);
    if (!t) return;
    const n = tok.encode(t, { add_special_tokens: false }).length;
    if (cur.n + n > MAX_TOKENS && cur.n) { out.push(cur); cur = { text: '', n: 0, idx: [] }; }
    cur.text += (cur.text ? ' ' : '') + t;
    cur.n += n;
    cur.idx.push(i);
  });
  if (cur.n) out.push(cur);
  return out;
}

function pick(n, max) {
  if (!max || n <= max) return [...Array(n).keys()];
  return [...new Set(Array.from({ length: max }, (_, k) => Math.round(k * (n - 1) / (max - 1))))];
}

async function analyse({ id, sentences, maxWindows }) {
  cancelled = false;
  await load(id);
  const wins = windows(sentences);
  const chosen = pick(wins.length, maxWindows);
  const perSentence = sentences.map(() => null);
  const logits = [];
  for (let k = 0; k < chosen.length; k++) {
    if (cancelled) { postMessage({ type: 'cancelled' }); return; }
    const w = wins[chosen[k]];
    const enc = await tok(w.text, { truncation: true, max_length: MAX_TOKENS + 2 });
    const out = await model(enc);
    const d = out.logits.data;
    const z = Number(d[1]) - Number(d[0]);
    logits.push(z);
    w.idx.forEach(i => { perSentence[i] = z; });
    postMessage({ type: 'progress', done: k + 1, total: chosen.length });
  }
  postMessage({ type: 'result', logit: logits.reduce((a, b) => a + b, 0) / logits.length, perSentence,
    windowsTotal: wins.length, windowsAnalysed: chosen.length });
}

self.onmessage = async e => {
  const m = e.data;
  try {
    if (m.type === 'analyse') await analyse(m);
    else if (m.type === 'cancel') cancelled = true;
  } catch (err) {
    postMessage({ type: 'error', message: String(err?.message || err) });
  }
};
