// Web Worker : analyse par modèle de langage, exécutée entièrement dans le navigateur.
// Méthode : log-probabilités des tokens (perplexité), rang des tokens (GLTR),
// et courbure de probabilité conditionnelle (Fast-DetectGPT, Bao et al., ICLR 2024).
import { AutoTokenizer, AutoModelForCausalLM, Tensor, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';

env.allowLocalModels = false;

const MODEL_ID = 'onnx-community/Qwen2.5-0.5B-Instruct';
const CHUNK_TOKENS = 192; // ~ 117 Mo de logits par bloc (vocabulaire de 151 936 tokens)

let tokenizer = null;
let model = null;
let backend = null;
let cancelled = false;

// Table de conversion float16 -> float32 (les logits peuvent arriver en fp16 sur WebGPU).
let F16 = null;
function f16Table() {
  if (F16) return F16;
  F16 = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    F16[h] = e === 0 ? s * 2 ** -14 * (f / 1024)
      : e === 31 ? (f ? NaN : s * Infinity)
      : s * 2 ** (e - 15) * (1 + f / 1024);
  }
  return F16;
}

async function pickBackend() {
  try {
    if (self.navigator?.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter && adapter.features.has('shader-f16')) return { device: 'webgpu', dtype: 'q4f16' };
    }
  } catch (e) { /* WebGPU indisponible */ }
  return { device: 'wasm', dtype: 'q8' };
}

async function load() {
  if (model) return;
  backend = await pickBackend();
  const progress_callback = p => {
    if (p.status === 'progress' && p.total) {
      postMessage({ type: 'download', file: p.file, loaded: p.loaded, total: p.total });
    }
  };
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback });
  try {
    model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, { ...backend, progress_callback });
  } catch (err) {
    if (backend.device === 'webgpu') {
      backend = { device: 'wasm', dtype: 'q8' };
      model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, { ...backend, progress_callback });
    } else throw err;
  }
  postMessage({ type: 'ready', backend });
}

// Découpe les phrases en blocs d'environ CHUNK_TOKENS tokens, sans couper une phrase.
function buildChunks(sentences) {
  const chunks = [];
  let cur = { ids: [], owner: [] };
  sentences.forEach((s, si) => {
    let ids = tokenizer.encode((si && !/^\s/.test(s) ? ' ' : '') + s, { add_special_tokens: false });
    if (ids.length > CHUNK_TOKENS) ids = ids.slice(0, CHUNK_TOKENS);
    if (cur.ids.length + ids.length > CHUNK_TOKENS && cur.ids.length) {
      chunks.push(cur);
      cur = { ids: [], owner: [] };
    }
    for (const id of ids) { cur.ids.push(id); cur.owner.push(si); }
  });
  if (cur.ids.length) chunks.push(cur);
  return chunks;
}

// Choisit des blocs répartis régulièrement dans le document (mode échantillon).
function sampleChunks(chunks, max) {
  if (!max || chunks.length <= max) return chunks.map((c, i) => i);
  const out = [];
  for (let k = 0; k < max; k++) out.push(Math.round(k * (chunks.length - 1) / (max - 1)));
  return [...new Set(out)];
}

async function scoreChunk(chunk, stats) {
  const T = chunk.ids.length;
  if (T < 2) return;
  const input_ids = new BigInt64Array(chunk.ids.map(BigInt));
  const ids = new Tensor('int64', input_ids, [1, T]);
  const mask = new Tensor('int64', new BigInt64Array(T).fill(1n), [1, T]);
  const out = await model({ input_ids: ids, attention_mask: mask });
  const logits = out.logits;
  const V = logits.dims[2];
  let raw;
  try { raw = logits.data; } catch { raw = null; }
  if (!raw || !raw.length) raw = await logits.ort_tensor?.getData?.();
  const lut = raw instanceof Uint16Array ? f16Table() : null;
  const row = new Float32Array(V);

  for (let t = 0; t < T - 1; t++) {
    const base = t * V;
    let max = -Infinity;
    for (let v = 0; v < V; v++) {
      const x = lut ? lut[raw[base + v]] : raw[base + v];
      row[v] = x;
      if (x > max) max = x;
    }
    let z = 0;
    for (let v = 0; v < V; v++) z += Math.exp(row[v] - max);
    const logZ = max + Math.log(z);
    const target = chunk.ids[t + 1];
    const xt = row[target];
    const lp = xt - logZ;
    let mu = 0, m2 = 0, rank = 0;
    for (let v = 0; v < V; v++) {
      const l = row[v] - logZ;
      const p = Math.exp(l);
      mu += p * l;
      m2 += p * l * l;
      if (row[v] > xt) rank++;
    }
    const varr = Math.max(m2 - mu * mu, 1e-6);
    const s = stats[chunk.owner[t + 1]];
    s.n++; s.lp += lp; s.mu += mu; s.var += varr;
    if (rank < 10) s.top10++;
    if (rank === 0) s.top1++;
    s.ent -= mu;
  }
  logits.dispose?.();
  ids.dispose?.(); mask.dispose?.();
}

async function analyse({ sentences, maxChunks }) {
  cancelled = false;
  await load();
  const chunks = buildChunks(sentences);
  const picked = sampleChunks(chunks, maxChunks);
  const stats = sentences.map(() => ({ n: 0, lp: 0, mu: 0, var: 0, top10: 0, top1: 0, ent: 0 }));
  const t0 = performance.now();
  for (let k = 0; k < picked.length; k++) {
    if (cancelled) { postMessage({ type: 'cancelled' }); return; }
    await scoreChunk(chunks[picked[k]], stats);
    postMessage({ type: 'progress', done: k + 1, total: picked.length, elapsed: performance.now() - t0 });
  }
  postMessage({
    type: 'result', stats, backend,
    chunksTotal: chunks.length, chunksAnalysed: picked.length,
    tokensTotal: chunks.reduce((a, c) => a + c.ids.length, 0)
  });
}

self.onmessage = async e => {
  const msg = e.data;
  try {
    if (msg.type === 'load') await load();
    else if (msg.type === 'analyse') await analyse(msg);
    else if (msg.type === 'cancel') cancelled = true;
  } catch (err) {
    postMessage({ type: 'error', message: String(err?.message || err) });
  }
};
