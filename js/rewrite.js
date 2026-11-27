// Réécriture d'un texte dans un style choisi, via le même fournisseur d'IA que la vérification.
// Les styles et les consignes sont dans js/reecriture.json, aussi utilisé par l'entraînement
// (entrainement/generer.py --humaniser) : le détecteur apprend à reconnaître ces réécritures.

let R = null;
export async function loadStyles() {
  if (!R) R = await (await fetch(new URL('./reecriture.json', import.meta.url))).json();
  return R;
}

export function buildPrompt(r, texte, styleId, intensiteId) {
  const s = r.styles.find(x => x.id === styleId) || r.styles[0];
  const i = r.intensites.find(x => x.id === intensiteId) || r.intensites[0];
  return {
    system: `${r.consigne}\n\n${s.consigne}\n${i.consigne}`,
    user: r.demande.replace('{texte}', texte)
  };
}

const WORDS_PER_CALL = 900;

// Découpe en morceaux de paragraphes entiers, pour garder la mise en page.
function chunks(text) {
  const paras = text.split(/\n\s*\n/).filter(p => p.trim());
  const out = [];
  let cur = [], n = 0;
  for (const p of paras) {
    const w = (p.match(/\S+/g) || []).length;
    if (cur.length && n + w > WORDS_PER_CALL) { out.push(cur.join('\n\n')); cur = []; n = 0; }
    cur.push(p); n += w;
  }
  if (cur.length) out.push(cur.join('\n\n'));
  return out;
}

const clean = t => String(t || '')
  .replace(/<think>[\s\S]*?<\/think>/g, '')
  .replace(/^\s*<texte>\s*|\s*<\/texte>\s*$/g, '')
  .replace(/^(voici|bien sûr|certainement|d'accord)[^\n]*:\s*\n/i, '')
  .trim();

async function callText(cfg, p, signal) {
  const { provider, apiKey, model, relayUrl } = cfg;
  let res, body;
  if (provider === 'anthropic') {
    res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST', signal,
      headers: {
        'content-type': 'application/json', 'x-api-key': apiKey,
        'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true'
      },
      // Pas de temperature : refusée par les modèles Claude récents.
      body: JSON.stringify({ model, max_tokens: 8000, system: p.system, messages: [{ role: 'user', content: p.user }] })
    });
    body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body?.error?.message || `Erreur Anthropic ${res.status}`);
    return clean((body.content || []).map(c => c.text || '').join(''));
  }
  if (provider === 'openai' || provider === 'mistral') {
    const url = provider === 'openai' ? 'https://api.openai.com/v1/chat/completions' : 'https://api.mistral.ai/v1/chat/completions';
    const payload = { model, messages: [{ role: 'system', content: p.system }, { role: 'user', content: p.user }] };
    if (provider === 'mistral') payload.temperature = 0.8;
    res = await fetch(url, {
      method: 'POST', signal,
      headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(payload)
    });
    body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body?.error?.message || body?.message || `Erreur ${provider} ${res.status}`);
    return clean(body.choices?.[0]?.message?.content);
  }
  if (provider === 'relais') {
    if (!relayUrl) throw new Error('Aucune adresse de relais configurée.');
    res = await fetch(relayUrl, {
      method: 'POST', signal,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ mode: 'reecrire', text: p.texte, style: p.style, intensite: p.intensite })
    });
    body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body?.error || `Erreur du relais ${res.status}`);
    return clean(body.texte);
  }
  throw new Error('Fournisseur inconnu.');
}

export async function rewrite(cfg, text, { style, intensite, signal, onProgress } = {}) {
  if (cfg.provider !== 'relais' && !cfg.apiKey) throw new Error('Ajoutez une clé API dans les réglages de l\'IA.');
  const r = await loadStyles();
  const parts = chunks(text);
  const out = [];
  for (let k = 0; k < parts.length; k++) {
    onProgress?.(k, parts.length);
    const p = { ...buildPrompt(r, parts[k], style, intensite), texte: parts[k], style, intensite };
    const t = await callText(cfg, p, signal);
    if (!t) throw new Error('Réponse vide du modèle.');
    out.push(t);
  }
  onProgress?.(parts.length, parts.length);
  return out.join('\n\n');
}
