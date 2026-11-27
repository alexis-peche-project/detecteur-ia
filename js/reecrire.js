// Page Réécrire : réécriture d'un texte dans un style choisi.
import { analyseHeuristics, words } from './heuristics.js';
import { rewrite, loadStyles } from './rewrite.js';
import { $, esc, store, initReglagesIA, analyserSurPageTexte } from './commun.js';

const ia = initReglagesIA();
const txt = $('rwSrc');
const compte = () => { const n = words(txt.value).length; $('rwCount').textContent = n.toLocaleString('fr-FR') + (n > 1 ? ' mots' : ' mot'); };
txt.value = store.get('detia-draft') || '';
compte();
txt.addEventListener('input', () => { compte(); if (txt.value.length < 200000) store.set('detia-draft', txt.value); });

let rwStyle = store.get('detia-rw-style') || 'naturel';
let rwAbort = null;
loadStyles().then(R => {
  if (!R.styles.some(x => x.id === rwStyle)) rwStyle = R.styles[0].id;
  const box = $('rwStyles');
  box.innerHTML = R.styles.map(x => `<button type="button" role="radio" data-id="${esc(x.id)}" aria-checked="${x.id === rwStyle}"><b>${esc(x.nom)}</b><small>${esc(x.description)}</small></button>`).join('');
  box.addEventListener('click', e => {
    const b = e.target.closest('button[data-id]');
    if (!b) return;
    rwStyle = b.dataset.id;
    store.set('detia-rw-style', rwStyle);
    box.querySelectorAll('button').forEach(x => x.setAttribute('aria-checked', String(x === b)));
  });
  $('rwInt').innerHTML = R.intensites.map(x => `<option value="${esc(x.id)}">${esc(x.nom)}</option>`).join('');
  $('rwInt').value = store.get('detia-rw-int') || R.intensites[0].id;
  $('rwInt').addEventListener('change', e => store.set('detia-rw-int', e.target.value));
}).catch(() => { $('rwState').textContent = 'Styles indisponibles.'; $('rwGo').disabled = true; });

const nbSignes = t => analyseHeuristics(t).found.reduce((a, f) => a + f.ex.length, 0);

$('rwGo').addEventListener('click', async () => {
  const text = txt.value.trim();
  if (words(text).length < 20) { $('rwState').textContent = 'Collez d\'abord un texte (20 mots au moins).'; txt.focus(); return; }
  if (!ia.pret()) { $('rwState').textContent = 'Ajoutez une clé API dans les réglages de l\'IA.'; ia.ouvrir(); return; }
  ia.save();
  rwAbort = new AbortController();
  $('rwGo').disabled = true; $('rwCancel').classList.remove('hidden');
  $('rwState').textContent = 'Réécriture…';
  try {
    const out = await rewrite(ia.cfg(), text, {
      style: rwStyle, intensite: $('rwInt').value, signal: rwAbort.signal,
      onProgress: (k, n) => { $('rwState').textContent = n > 1 ? `Réécriture… partie ${Math.min(k + 1, n)} sur ${n}` : 'Réécriture…'; }
    });
    $('rwText').textContent = out;
    const nom = $('rwStyles').querySelector(`[data-id="${rwStyle}"] b`)?.textContent || '';
    $('rwTitle').textContent = `Texte réécrit, style ${nom.toLowerCase()}`;
    $('rwSigns').textContent = `Signes Wikipédia : ${nbSignes(text)} dans l'original, ${nbSignes(out)} après réécriture`;
    $('rwOut').classList.remove('hidden');
    $('rwState').textContent = `${words(out).length.toLocaleString('fr-FR')} mots`;
  } catch (e) {
    $('rwState').textContent = e.name === 'AbortError' ? 'Annulé.' : `Erreur : ${e.message}`;
  } finally {
    $('rwGo').disabled = false; $('rwCancel').classList.add('hidden'); rwAbort = null;
  }
});
$('rwCancel').addEventListener('click', () => rwAbort?.abort());
$('rwCopy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('rwText').textContent); $('rwState').textContent = 'Copié.'; } catch { $('rwState').textContent = 'Copie impossible : sélectionnez le texte.'; }
});
$('rwReplace').addEventListener('click', () => { txt.value = $('rwText').textContent; compte(); store.set('detia-draft', txt.value); txt.focus(); });
$('rwUse').addEventListener('click', () => analyserSurPageTexte($('rwText').textContent));
