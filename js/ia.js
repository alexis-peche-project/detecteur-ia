// Appel générique à un grand modèle (Claude, GPT, Mistral) avec la clé du visiteur, texte et images.
// Le relais du site n'accepte que ses consignes fixes (vérification, réécriture) : les autres outils demandent une clé.

export class IAError extends Error {}

export function verifierCfg(cfg) {
  if (cfg.provider === 'relais') throw new IAError('Cet outil a besoin de votre propre clé API (Claude, GPT ou Mistral) : choisissez un fournisseur dans les réglages de l\'IA.');
  if (!cfg.apiKey) throw new IAError('Ajoutez une clé API dans les réglages de l\'IA.');
}

// images : [{ type: 'image/jpeg', data: '<base64>' }]
export async function appelIA(cfg, { system = '', user, images = [], maxTokens = 8000, signal } = {}) {
  verifierCfg(cfg);
  const { provider, apiKey, model } = cfg;
  let res, body;
  if (provider === 'anthropic') {
    const content = [...images.map(i => ({ type: 'image', source: { type: 'base64', media_type: i.type, data: i.data } })), { type: 'text', text: user }];
    res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST', signal,
      headers: { 'content-type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' },
      // Pas de temperature : refusée par les modèles Claude récents.
      body: JSON.stringify({ model, max_tokens: maxTokens, ...(system ? { system } : {}), messages: [{ role: 'user', content }] })
    });
    body = await res.json().catch(() => ({}));
    if (!res.ok) throw new IAError(body?.error?.message || `Erreur Anthropic ${res.status}`);
    return (body.content || []).map(c => c.text || '').join('').trim();
  }
  if (provider === 'openai' || provider === 'mistral') {
    const url = provider === 'openai' ? 'https://api.openai.com/v1/chat/completions' : 'https://api.mistral.ai/v1/chat/completions';
    const content = images.length
      ? [...images.map(i => ({ type: 'image_url', image_url: provider === 'openai' ? { url: `data:${i.type};base64,${i.data}` } : `data:${i.type};base64,${i.data}` })), { type: 'text', text: user }]
      : user;
    const messages = [...(system ? [{ role: 'system', content: system }] : []), { role: 'user', content }];
    res = await fetch(url, {
      method: 'POST', signal,
      headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ model, messages })
    });
    body = await res.json().catch(() => ({}));
    if (!res.ok) throw new IAError(body?.error?.message || body?.message || `Erreur ${provider} ${res.status}`);
    return String(body.choices?.[0]?.message?.content || '').trim();
  }
  throw new IAError('Fournisseur inconnu.');
}

// Extrait le premier objet JSON d'une réponse.
export function json(raw) {
  const s = String(raw).replace(/```(?:json)?/g, '');
  const a = s.indexOf('{'), b = s.lastIndexOf('}');
  if (a < 0 || b < a) throw new IAError('Réponse illisible du modèle.');
  return JSON.parse(s.slice(a, b + 1));
}

export const nettoyer = t => String(t || '')
  .replace(/<think>[\s\S]*?<\/think>/g, '')
  .replace(/^(voici|bien sûr|certainement|d'accord)[^\n]*:\s*\n/i, '')
  .trim();

// Réduit une image (photo de copie) avant envoi : 2000 px au plus, JPEG.
export async function imageVersBase64(file, max = 2000) {
  const bmp = await createImageBitmap(file);
  const k = Math.min(1, max / Math.max(bmp.width, bmp.height));
  const c = new OffscreenCanvas(Math.round(bmp.width * k), Math.round(bmp.height * k));
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  const blob = await c.convertToBlob({ type: 'image/jpeg', quality: 0.88 });
  const buf = new Uint8Array(await blob.arrayBuffer());
  let s = '';
  for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  return { type: 'image/jpeg', data: btoa(s) };
}
