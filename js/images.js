// Page Images : marques d'origine (C2PA, IPTC, paramètres de génération, EXIF) et détecteur open source.
import { $, esc, store, zoneDepot } from './commun.js';
import { lireMarques } from './meta-image.js';

const TYPES = 'image/jpeg,image/png,image/webp,image/avif,image/heic,image/heif,image/gif,.jpg,.jpeg,.png,.webp,.avif,.heic';
$('imgFile').accept = TYPES;
$('useImgModel').checked = store.get('detia-img-model') !== '0';
$('useImgModel').addEventListener('change', e => store.set('detia-img-model', e.target.checked ? '1' : '0'));

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
function verdictImage(m, p) {
  if (m.constats.some(c => c.force === 'ia')) return { cls: 'ia', titre: 'Marquée comme générée par IA', texte: 'Le fichier lui-même le déclare. C\'est l\'indice le plus sûr.' };
  if (Number.isFinite(p) && p >= 0.85) return { cls: 'ia', titre: 'Probablement générée par IA', texte: `Le détecteur l'estime à ${Math.round(p * 100)} %. Aucune marque officielle ne le confirme.` };
  if (Number.isFinite(p) && p >= 0.5) return { cls: 'doute', titre: 'Indices d\'IA', texte: `Le détecteur l'estime à ${Math.round(p * 100)} %. À confirmer par d'autres moyens.` };
  if (m.constats.some(c => c.force === 'photo')) return { cls: 'humain', titre: 'Plutôt une vraie photo', texte: 'Données d\'appareil ou signature de prise de vue, sans signe d\'IA. Ces données peuvent toutefois être falsifiées.' };
  if (Number.isFinite(p)) return { cls: 'humain', titre: 'Pas d\'indice d\'IA', texte: `Le détecteur l'estime à ${Math.round(p * 100)} %. Ce n'est pas une preuve : les générateurs récents lui échappent souvent.` };
  return { cls: 'neutre', titre: 'Rien de concluant', texte: 'Aucune marque d\'origine et pas d\'analyse par le détecteur.' };
}

let n = 0;
async function traiter(file) {
  const id = ++n;
  const url = URL.createObjectURL(file);
  const el = document.createElement('article');
  el.className = 'imgCard';
  el.innerHTML = `<img src="${url}" alt="${esc(file.name)}"><div class="imgInfo"><h3>${esc(file.name)}</h3><p class="small">Analyse…</p></div>`;
  $('imgList').append(el);
  const info = el.querySelector('.imgInfo');
  let m;
  try { m = await lireMarques(file); } catch (e) { m = { constats: [{ force: 'info', titre: 'Fichier illisible', detail: e.message }], generateurs: [], c2pa: false }; }
  const rendu = (p, erreur) => {
    const v = verdictImage(m, p);
    info.innerHTML = `<h3>${esc(file.name)}</h3>
      <div class="imgVerdict ${v.cls}"><b>${esc(v.titre)}</b><span>${esc(v.texte)}</span></div>
      ${Number.isFinite(p) ? `<div class="kv"><div>Détecteur open source (SMOGY)<b>${Math.round(p * 100)} % IA</b></div><div>Taille<b>${(file.size / 1e6).toFixed(1).replace('.', ',')} Mo</b></div></div>` : ''}
      ${erreur ? `<p class="small">Détecteur indisponible : ${esc(erreur)}</p>` : ''}
      <ul class="constats">${m.constats.map(c => `<li class="${c.force}"><b>${esc(c.titre)}</b><span>${esc(c.detail)}</span></li>`).join('')}</ul>
      ${m.c2pa ? `<p class="small"><a href="https://contentcredentials.org/verify" target="_blank" rel="noopener">Vérifier la signature des Content Credentials</a> (déposez-y la même image).</p>` : ''}`;
  };
  rendu(null);
  if ($('useImgModel').checked && !m.constats.some(c => c.force === 'ia')) {
    info.querySelector('.small') && (info.querySelector('.small').textContent = 'Analyse par le détecteur…');
    try { const r = await modele(id, file); rendu(r.p); } catch (e) { rendu(null, e.message); }
  }
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
