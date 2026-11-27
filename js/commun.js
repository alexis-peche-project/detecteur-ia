// Outils partagés par toutes les pages du site.
import { PROVIDERS } from './llm-judge.js';
import { RELAY_URL } from './config.js';

export const $ = id => document.getElementById(id);
export const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const fmt = (x, d = 2) => Number(x).toFixed(d).replace('.', ',');
export const pct = x => Math.round(x * 100) + ' %';
export const store = {
  get(k, area = localStorage) { try { return area.getItem(k); } catch { return null; } },
  set(k, v, area = localStorage) { try { v == null ? area.removeItem(k) : area.setItem(k, v); } catch { /* stockage indisponible */ } }
};

export function download(name, content, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(content instanceof Blob ? content : new Blob([content], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// Envoie un texte à la page d'analyse, qui le lance dès son ouverture.
export function analyserSurPageTexte(text, nouvelOnglet = false) {
  store.set('detia-draft', text);
  store.set('detia-auto', String(Date.now()));
  if (nouvelOnglet) window.open('index.html', '_blank');
  else location.href = 'index.html';
}

/* ---------- Réglages de l'IA (vérification et réécriture), partagés entre les pages ---------- */
export function initReglagesIA() {
  const provSel = $('provider');
  if (!provSel) return null;
  provSel.innerHTML = Object.entries(PROVIDERS)
    .filter(([k]) => k !== 'relais' || RELAY_URL)
    .map(([k, p]) => `<option value="${k}">${esc(p.label)}</option>`).join('');
  const load = () => {
    const p = provSel.value;
    $('model').value = store.get('detia-model-' + p) || PROVIDERS[p].model;
    $('apiKey').placeholder = PROVIDERS[p].keyHint;
    $('apiKey').value = store.get('detia-key-' + p) || store.get('detia-key-' + p, sessionStorage) || '';
    $('remember').checked = !!store.get('detia-key-' + p);
    const relay = p === 'relais';
    $('keyRow').classList.toggle('hidden', relay);
    $('rememberRow').classList.toggle('hidden', relay);
    $('model').disabled = relay;
  };
  const save = () => {
    const p = provSel.value;
    store.set('detia-provider', p);
    store.set('detia-model-' + p, $('model').value.trim() || null);
    const key = $('apiKey').value.trim() || null;
    store.set('detia-key-' + p, $('remember').checked ? key : null);
    store.set('detia-key-' + p, key, sessionStorage);
  };
  const saved = store.get('detia-provider');
  provSel.value = saved && [...provSel.options].some(o => o.value === saved) ? saved : (RELAY_URL ? 'relais' : 'anthropic');
  load();
  provSel.addEventListener('change', () => { store.set('detia-provider', provSel.value); load(); });
  ['model', 'apiKey'].forEach(id => $(id).addEventListener('change', save));
  $('remember').addEventListener('change', save);
  return {
    save,
    cfg: () => ({ provider: provSel.value, apiKey: $('apiKey').value.trim(), model: $('model').value.trim(), relayUrl: RELAY_URL }),
    pret: () => provSel.value === 'relais' || !!$('apiKey').value.trim(),
    ouvrir: () => { $('apiBox').open = true; $('apiBox').scrollIntoView({ behavior: 'smooth', block: 'center' }); }
  };
}

/* ---------- Mâchou, la mascotte ---------- */
const BULLES = {
  idle: 'Donne-moi un texte à croquer.',
  chomp: 'Je mâche… je goûte chaque phrase.',
  humain: 'Ça a un goût d\'humain.',
  doute: 'Hmm, difficile à dire.',
  ia: 'Ça a un goût de robot !'
};
export const humeurDuScore = s => s >= 60 ? 'ia' : s >= 40 ? 'doute' : 'humain';
export function mascotte(bulles = BULLES) {
  const grande = $('mascotte'), mini = $('mascotteScore');
  if (grande && mini && !mini.innerHTML) mini.innerHTML = grande.querySelector('svg').outerHTML.replace('role="img"', 'aria-hidden="true"');
  return m => {
    if (grande) grande.dataset.mood = m;
    if (mini) mini.dataset.mood = m;
    if ($('bulle') && bulles[m]) $('bulle').textContent = bulles[m];
  };
}

// Zone de dépôt de fichiers (un ou plusieurs).
export function zoneDepot(el, onFiles) {
  ['dragenter', 'dragover'].forEach(ev => el.addEventListener(ev, e => { if (e.dataTransfer?.types?.includes('Files')) { e.preventDefault(); el.classList.add('over'); } }));
  ['dragleave', 'drop'].forEach(ev => el.addEventListener(ev, () => el.classList.remove('over')));
  el.addEventListener('drop', e => { const f = [...(e.dataTransfer?.files || [])]; if (f.length) { e.preventDefault(); onFiles(f); } });
}

// Sur mobile, la page en cours reste visible dans la barre de navigation qui défile.
document.querySelector('.top nav a[aria-current=page]')?.scrollIntoView({ block: 'nearest', inline: 'center' });
