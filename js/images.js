// Page Images : marques d'origine (C2PA, IPTC, paramètres de génération, EXIF) et détecteur open source.
import { $, esc, store, zoneDepot, initReglagesIA } from './commun.js';
import { analyseVisuelle } from './vision-image.js';
import { lireMarques } from './meta-image.js';

const TYPES = 'image/jpeg,image/png,image/webp,image/avif,image/heic,image/heif,image/gif,.jpg,.jpeg,.png,.webp,.avif,.heic';
$('imgFile').accept = TYPES;
$('useImgModel').checked = store.get('detia-img-model') !== '0';
$('useImgModel').addEventListener('change', e => store.set('detia-img-model', e.target.checked ? '1' : '0'));
const ia = initReglagesIA();
$('useImgVision').checked = store.get('detia-img-vision') === '1';
$('useImgVision').addEventListener('change', e => {
  store.set('detia-img-vision', e.target.checked ? '1' : '0');
  if (e.target.checked && (!ia.pret() || $('provider').value === 'relais')) ia.ouvrir();
});

let worker = null;
const attentes = new Map();
const dl = {};
function modele(id, blob) {
  if (!worker) {
    worker = new Worker(new URL('./image-worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = e => {
      const m = e.data;
      if (m.type === 'download') {
        dl[m.file] = m;
        const v = Object.values(dl), L = v.reduce((a, x) => a + x.loaded, 0), T = v.reduce((a, x) => a + x.total, 0);
        $('imgEtat').textContent = `Téléchargement du détecteur d'images : ${Math.round(L / 1e6)} / ${Math.round(T / 1e6)} Mo`;
      } else if (m.type === 'ready') $('imgEtat').textContent = 'Détecteur prêt.';
      else if (attentes.has(m.id)) {
        const { ok, ko } = attentes.get(m.id);
        attentes.delete(m.id);
        m.type === 'result' ? ok(m) : ko(new Error(m.message));
      }
    };
    worker.onerror = e => { attentes.forEach(({ ko }) => ko(new Error(e.message || 'Le détecteur n\'a pas pu démarrer.'))); attentes.clear(); };
  }
  return new Promise((ok, ko) => { attentes.set(id, { ok, ko }); worker.postMessage({ id, blob }); });
}

/* Verdict prudent : une marque officielle l'emporte ; sinon, le détecteur ne donne qu'un indice. */
function verdictImage(m, p, v) {
  if (m.constats.some(c => c.force === 'ia')) return { cls: 'ia', titre: 'Marquée comme générée par IA', texte: 'Le fichier lui-même le déclare. C\'est l\'indice le plus sûr.' };
  const nv = v ? v.indicesIA.length : 0;
  if (v && v.p >= 0.8 && v.confiance !== 'faible') return { cls: 'ia', titre: 'Probablement générée par IA', texte: `L'analyse visuelle l'estime à ${Math.round(v.p * 100)} % et relève ${nv} détail(s) suspect(s). Aucune marque officielle ne le confirme.` };
  if (Number.isFinite(p) && p >= 0.85) return { cls: 'ia', titre: 'Probablement générée par IA', texte: `Le détecteur l'estime à ${Math.round(p * 100)} %. Aucune marque officielle ne le confirme.` };
  if ((v && v.p >= 0.5) || (Number.isFinite(p) && p >= 0.5)) return { cls: 'doute', titre: 'Indices d\'IA', texte: `${v ? `Analyse visuelle : ${Math.round(v.p * 100)} %. ` : ''}${Number.isFinite(p) ? `Détecteur : ${Math.round(p * 100)} %. ` : ''}À confirmer par d'autres moyens.` };
  if (m.constats.some(c => c.force === 'photo')) return { cls: 'humain', titre: 'Plutôt une vraie photo', texte: 'Données d\'appareil ou signature de prise de vue, sans signe d\'IA. Ces données peuvent toutefois être falsifiées.' };
  if (v || Number.isFinite(p)) return { cls: 'humain', titre: 'Pas d\'indice d\'IA', texte: `${v ? `Analyse visuelle : ${Math.round(v.p * 100)} %. ` : ''}${Number.isFinite(p) ? `Détecteur : ${Math.round(p * 100)} %. ` : ''}Ce n'est pas une preuve : les générateurs récents échappent souvent aux détecteurs.` };
  return { cls: 'neutre', titre: 'Rien de concluant', texte: 'Aucune marque d\'origine et pas d\'analyse par un détecteur.' };
}

let n = 0;
async function traiter(file) {
  const id = ++n;
  const url = URL.createObjectURL(file);
  const el = document.createElement('article');
  el.className = 'imgCard';
  el.innerHTML = `<div class="imgCadre"><img src="${url}" alt="${esc(file.name)}"><div class="zones"></div></div><div class="imgInfo"><h3>${esc(file.name)}</h3><p class="small">Analyse…</p></div>`;
  $('imgList').append(el);
  const info = el.querySelector('.imgInfo'), zones = el.querySelector('.zones');
  let m;
  try { m = await lireMarques(file); } catch (e) { m = { constats: [{ force: 'info', titre: 'Fichier illisible', detail: e.message }], generateurs: [], c2pa: false }; }
  const etat = { p: null, errP: null, v: null, errV: null, enCours: '' };
  const rendu = () => {
    const { p, v } = etat;
    const verdict = verdictImage(m, p, v);
    el.classList.toggle('large', !!v);
    zones.innerHTML = v ? v.indicesIA.filter(i => i.zone).map((i, k) => `<span class="zone" style="left:${i.zone.x * 100}%;top:${i.zone.y * 100}%;width:${Math.max(4, i.zone.l * 100)}%;height:${Math.max(4, i.zone.h * 100)}%" data-k="${k}"><b>${k + 1}</b></span>`).join('') : '';
    info.innerHTML = `<h3>${esc(file.name)}</h3>
      <div class="imgVerdict ${verdict.cls}"><b>${esc(verdict.titre)}</b><span>${esc(verdict.texte)}</span></div>
      <div class="kv">${Number.isFinite(p) ? `<div>Détecteur open source (SMOGY)<b>${Math.round(p * 100)} % IA</b></div>` : ''}${v ? `<div>Analyse visuelle par IA<b>${Math.round(v.p * 100)} % IA · confiance ${esc(v.confiance)}</b></div>` : ''}<div>Taille<b>${(file.size / 1e6).toFixed(1).replace('.', ',')} Mo</b></div></div>
      ${etat.enCours ? `<p class="small">${esc(etat.enCours)}</p>` : ''}
      ${etat.errP ? `<p class="small">Détecteur indisponible : ${esc(etat.errP)}</p>` : ''}
      ${etat.errV ? `<p class="small">Analyse visuelle impossible : ${esc(etat.errV)}</p>` : ''}
      ${v ? `<div class="vision"><p>${esc(v.verdict)}</p>
        ${v.indicesIA.length ? `<h4>Détails suspects</h4><ol class="indices">${v.indicesIA.map((i, k) => `<li data-k="${k}"><b>${i.zone ? `<span class="num">${k + 1}</span>` : ''}${esc(i.titre)}</b><span>${esc(i.detail)}</span></li>`).join('')}</ol>` : '<p class="small">Aucun détail suspect relevé.</p>'}
        ${v.indicesReels.length ? `<h4>Indices de vraie photo</h4><ul class="indices reels">${v.indicesReels.map(i => `<li><b>${esc(i.titre)}</b><span>${esc(i.detail)}</span></li>`).join('')}</ul>` : ''}
        <p class="small">Les zones sont approximatives. L'analyse visuelle se trompe aussi : vérifiez chaque détail en zoomant sur l'image d'origine.</p></div>` : ''}
      <ul class="constats">${m.constats.map(c => `<li class="${c.force}"><b>${esc(c.titre)}</b><span>${esc(c.detail)}</span></li>`).join('')}</ul>
      ${m.c2pa ? `<p class="small"><a href="https://contentcredentials.org/verify" target="_blank" rel="noopener">Vérifier la signature des Content Credentials</a> (déposez-y la même image).</p>` : ''}`;
  };
  // Survol d'un indice ↔ zone correspondante sur l'image.
  el.addEventListener('mouseover', e => {
    const k = e.target.closest('[data-k]')?.dataset.k;
    el.querySelectorAll('[data-k]').forEach(x => x.classList.toggle('actif', x.dataset.k === k));
  });
  rendu();
  const taches = [];
  if ($('useImgModel').checked && !m.constats.some(c => c.force === 'ia')) {
    taches.push(modele(id, file).then(r => { etat.p = r.p; }).catch(e => { etat.errP = e.message; }).finally(rendu));
  }
  if ($('useImgVision').checked) {
    if (!ia.pret() || $('provider').value === 'relais') { etat.errV = 'ajoutez votre clé API (Claude, GPT ou Mistral) dans les réglages de l\'IA.'; rendu(); }
    else {
      ia.save();
      etat.enCours = 'Analyse visuelle par l\'IA…'; rendu();
      taches.push(analyseVisuelle(ia.cfg(), file).then(v => { etat.v = v; }).catch(e => { etat.errV = e.message; }).finally(() => { etat.enCours = ''; rendu(); }));
    }
  }
  await Promise.all(taches);
}

async function ajouter(files) {
  const imgs = files.filter(f => f.type.startsWith('image/') || /\.(jpe?g|png|webp|avif|heic|gif)$/i.test(f.name));
  if (!imgs.length) { $('imgEtat').textContent = 'Aucune image dans la sélection.'; return; }
  $('imgVide').classList.add('hidden');
  for (const f of imgs) await traiter(f);
  $('imgEtat').textContent = `${imgs.length} image(s) analysée(s).`;
}
$('imgFile').addEventListener('change', e => { ajouter([...e.target.files]); e.target.value = ''; });
zoneDepot($('imgDrop'), ajouter);
document.addEventListener('paste', e => {
  const f = [...(e.clipboardData?.files || [])];
  if (f.length) ajouter(f);
});
