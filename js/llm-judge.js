// Vérification par un grand modèle de langage, via l'API du fournisseur choisi.
// La clé reste dans le navigateur du visiteur, ou sur le relais (relais/worker.js).
import { SYSTEM_PROMPT, userPrompt } from './prompt.js';

export const PROVIDERS = {
  anthropic: { label: 'Claude (Anthropic)', model: 'claude-sonnet-5-5', keyHint: 'sk-ant-…' },
  openai: { label: 'OpenAI', model: 'gpt-5-mini', keyHint: 'sk-…' },
  mistral: { label: 'Mistral AI', model: 'mistral-large-latest', keyHint: 'clé Mistral' },
  relais: { label: 'Relais du site (sans clé)', model: '', keyHint: '' }
};

const WORDS_PER_CALL = 2500;

function splitWords(text, size) {
  const parts = text.split(/(\s+)/);
  const chunks = [];
  let cur = '', n = 0;
  for (const p of parts) {
    cur += p;
    if (/\S/.test(p) && ++n >= size) { chunks.push(cur); cur = ''; n = 0; }
  }
  if (cur.trim()) chunks.push(cur);
  return chunks;
}

function pick(chunks, max) {
  if (chunks.length <= max) return chunks;
  const idx = [...new Set(Array.from({ length: max }, (_, k) => Math.round(k * (chunks.length - 1) / (max - 1))))];
  return idx.map(i => chunks[i]);
}

export function parseJson(raw) {
  const s = String(raw).replace(/```(?:json)?/g, '');
  const a = s.indexOf('{'), b = s.lastIndexOf('}');
  if (a < 0 || b < a) throw new Error('Réponse illisible du modèle vérificateur.');
  const j = JSON.parse(s.slice(a, b + 1));
  const p = Math.max(0, Math.min(100, Math.round(Number(j.probabilite_ia))));
  if (!Number.isFinite(p)) throw new Error('Score absent de la réponse.');
  const list = x => Array.isArray(x) ? x.filter(i => i && i.extrait).map(i => ({ extrait: String(i.extrait), raison: String(i.raison || '') })).slice(0, 6) : [];
  return {
    probabilite_ia: p,
    confiance: ['faible', 'moyenne', 'élevée'].includes(j.confiance) ? j.confiance : 'moyenne',
    verdict: String(j.verdict || ''),
    indices_ia: list(j.indices_ia),
    indices_humains: list(j.indices_humains)
  };
}

async function callOnce(cfg, text, signal) {
  const { provider, apiKey, model, relayUrl } = cfg;
  let res, body;
  if (provider === 'anthropic') {
    res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST', signal,
      headers: {
        'content-type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true'
      },
      body: JSON.stringify({
        model, max_tokens: 1500, temperature: 0,
        system: SYSTEM_PROMPT,
        messages: [{ role: 'user', content: userPrompt(text) }]
      })
    });
    body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body?.error?.message || `Erreur Anthropic ${res.status}`);
    return parseJson((body.content || []).map(c => c.text || '').join(''));
  }
  if (provider === 'openai' || provider === 'mistral') {
    const url = provider === 'openai' ? 'https://api.openai.com/v1/chat/completions' : 'https://api.mistral.ai/v1/chat/completions';
    const payload = {
      model,
      response_format: { type: 'json_object' },
      messages: [{ role: 'system', content: SYSTEM_PROMPT }, { role: 'user', content: userPrompt(text) }]
    };
    if (provider === 'mistral') payload.temperature = 0;
    res = await fetch(url, {
      method: 'POST', signal,
      headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(payload)
    });
    body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body?.error?.message || body?.message || `Erreur ${provider} ${res.status}`);
    return parseJson(body.choices?.[0]?.message?.content || '');
  }
  if (provider === 'relais') {
    if (!relayUrl) throw new Error('Aucune adresse de relais configurée.');
    res = await fetch(relayUrl, {
      method: 'POST', signal,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text })
    });
    body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body?.error || `Erreur du relais ${res.status}`);
    return parseJson(JSON.stringify(body));
  }
  throw new Error('Fournisseur inconnu.');
}

// Analyse un texte long en plusieurs appels répartis dans le document, puis fait la moyenne pondérée.
export async function judge(cfg, text, { maxCalls = 3, signal, onProgress } = {}) {
  if (cfg.provider !== 'relais' && !cfg.apiKey) throw new Error('Ajoutez une clé API dans les réglages de la vérification par IA.');
  const chunks = pick(splitWords(text, WORDS_PER_CALL), maxCalls);
  const results = [];
  for (let i = 0; i < chunks.length; i++) {
    onProgress?.(i, chunks.length);
    const r = await callOnce(cfg, chunks[i], signal);
    r.words = (chunks[i].match(/\S+/g) || []).length;
    results.push(r);
  }
  onProgress?.(chunks.length, chunks.length);
  const W = results.reduce((a, r) => a + r.words, 0) || 1;
  const order = ['faible', 'moyenne', 'élevée'];
  return {
    probabilite_ia: Math.round(results.reduce((a, r) => a + r.probabilite_ia * r.words, 0) / W),
    confiance: order[Math.min(...results.map(r => order.indexOf(r.confiance)))],
    verdict: results.map(r => r.verdict).filter(Boolean).join(' '),
    indices_ia: results.flatMap(r => r.indices_ia).slice(0, 8),
    indices_humains: results.flatMap(r => r.indices_humains).slice(0, 8),
    calls: results.length
  };
}
