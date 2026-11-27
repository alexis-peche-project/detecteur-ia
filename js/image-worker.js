// Web Worker : détecteur d'images IA open source (SMOGY, Swin), exécuté dans le navigateur.
// Simple indice : ces détecteurs se trompent souvent sur les générateurs récents.
import { pipeline, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2';

env.allowLocalModels = false;
export const MODELE = 'onnx-community/SMOGY-Ai-images-detector-ONNX';
let clf = null;

async function charger() {
  if (clf) return;
  const progress_callback = p => {
    if (p.status === 'progress' && p.total) postMessage({ type: 'download', file: p.file, loaded: p.loaded, total: p.total });
  };
  let device = 'wasm';
  try { if (self.navigator?.gpu && await navigator.gpu.requestAdapter()) device = 'webgpu'; } catch { /* pas de WebGPU */ }
  const essais = [{ device, dtype: 'q8' }, { device, dtype: 'fp32' }, { device: 'wasm', dtype: 'fp32' }];
  let derniere;
  for (const o of essais) {
    try { clf = await pipeline('image-classification', MODELE, { ...o, progress_callback }); break; }
    catch (e) { derniere = e; }
  }
  if (!clf) throw derniere;
  postMessage({ type: 'ready' });
}

// Probabilité « IA » à partir des étiquettes du modèle, quel que soit leur nom.
function probaIA(sorties) {
  const ia = sorties.find(x => /artific|\bai\b|fake|generat|synth|^ia$/i.test(x.label));
  if (ia) return ia.score;
  const reel = sorties.find(x => /real|human|natur|authent/i.test(x.label));
  if (reel) return 1 - reel.score;
  throw new Error('Étiquettes du modèle inattendues : ' + sorties.map(x => x.label).join(', '));
}

self.onmessage = async e => {
  const { id, blob } = e.data;
  try {
    await charger();
    const img = await RawImage.fromBlob(blob);
    const sorties = await clf(img, { top_k: 5 });
    postMessage({ type: 'result', id, p: probaIA(sorties), labels: sorties });
  } catch (err) {
    postMessage({ type: 'error', id, message: err?.message || String(err) });
  }
};
