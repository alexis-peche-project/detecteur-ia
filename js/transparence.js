// Page Transparence : carte de co-écriture (conversation avec l'IA → texte final) et étiquette d'usage de l'IA.
import { words } from './heuristics.js';
import { $, esc, store, download } from './commun.js';
import { alignerPhrases, empreinte } from './comparer.js';

const conv = $('trConv'), final = $('trFinal');
conv.value = store.get('detia-tr-conv') || '';
final.value = store.get('detia-tr-final') || '';
conv.addEventListener('input', () => store.set('detia-tr-conv', conv.value.length < 300000 ? conv.value : null));
final.addEventListener('input', () => store.set('detia-tr-final', final.value.length < 200000 ? final.value : null));

/* ---------- Carte de co-écriture ---------- */
const CLASSES = [
  { id: 'reprise', nom: 'Reprise de l\'IA', min: 0.5 },
  { id: 'retravail', nom: 'Retravaillée', min: 0.15 },
  { id: 'perso', nom: 'Personnelle', min: -1 }
];
let carte = null;

$('trCarte').addEventListener('click', () => {
  if (words(final.value).length < 10) { $('trCarteEtat').textContent = 'Collez le texte final (10 mots au moins).'; final.focus(); return; }
  if (words(conv.value).length < 10) { $('trCarteEtat').textContent = 'Collez la conversation avec l\'IA.'; conv.focus(); return; }
  const phrases = alignerPhrases(final.value, conv.value);
  const parts = { reprise: 0, retravail: 0, perso: 0 };
  let total = 0, html = '', pos = 0;
  const t = final.value;
  phrases.forEach(p => {
    const c = CLASSES.find(c => p.part >= c.min);
    p.classe = c.id;
    parts[c.id] += p.mots; total += p.mots;
    const titre = `${c.nom} · ${Math.round(p.part * 100)} % des expressions se trouvent dans la conversation${p.proche ? ` · passage le plus proche : « ${p.proche.slice(0, 180)} »` : ''}`;
    html += esc(t.slice(pos, p.start)) + `<span class="co ${c.id}" title="${esc(titre)}">${esc(t.slice(p.start, p.end))}</span>`;
    pos = p.end;
  });
  html += esc(t.slice(pos));
  const pc = k => total ? Math.round(parts[k] / total * 100) : 0;
  carte = { reprise: pc('reprise'), retravail: pc('retravail'), perso: pc('perso') };
  $('trCarteOut').innerHTML = `
    <div class="pile" role="img" aria-label="Reprise ${carte.reprise} %, retravaillée ${carte.retravail} %, personnelle ${carte.perso} %">
      <i class="reprise" style="width:${carte.reprise}%"></i><i class="retravail" style="width:${carte.retravail}%"></i><i class="perso" style="width:${carte.perso}%"></i></div>
    <div class="legend"><span><i class="lg co reprise"></i>Reprise de l'IA : ${carte.reprise} %</span><span><i class="lg co retravail"></i>Retravaillée : ${carte.retravail} %</span><span><i class="lg co perso"></i>Personnelle : ${carte.perso} %</span></div>
    <p class="small">Part des mots du texte final. Survolez une phrase pour voir le passage de la conversation le plus proche.</p>
    <div class="annot">${html}</div>`;
  $('trCarteEtat').textContent = '';
  $('trCarteDansEtiquette').checked = true;
});

/* ---------- Étiquette d'usage de l'IA ---------- */
const ETAPES = ['Idées et plan', 'Recherche d\'informations', 'Rédaction', 'Reformulation et style', 'Correction de la langue', 'Traduction'];
const CHOIX = ['Sans IA', 'Avec l\'aide d\'une IA', 'Fait par une IA', 'Non concerné'];
const sauve = (() => { try { return JSON.parse(store.get('detia-tr-etiquette')) || {}; } catch { return {}; } })();
$('trEtapes').innerHTML = ETAPES.map((e, i) => `<label>${esc(e)}<select data-etape="${i}">${CHOIX.map(c => `<option${sauve.etapes?.[i] === c ? ' selected' : ''}>${esc(c)}</option>`).join('')}</select></label>`).join('');
$('trAuteur').value = sauve.auteur || '';
$('trOutils').value = sauve.outils || '';

let etiquette = null;
$('trCreer').addEventListener('click', async () => {
  const texte = final.value.trim();
  if (words(texte).length < 10) { $('trEtiquetteEtat').textContent = 'Collez d\'abord le texte final, en haut de la page.'; final.focus(); return; }
  const etapes = [...$('trEtapes').querySelectorAll('select')].map(s => s.value);
  const auteur = $('trAuteur').value.trim(), outils = $('trOutils').value.trim();
  store.set('detia-tr-etiquette', JSON.stringify({ etapes, auteur, outils }));
  const e = await empreinte(texte);
  const date = new Date().toLocaleDateString('fr-FR', { day: 'numeric', month: 'long', year: 'numeric' });
  etiquette = { auteur, outils, etapes, date, mots: words(texte).length, code: e.court, carte: $('trCarteDansEtiquette').checked ? carte : null };
  const lignes = ETAPES.map((nom, i) => ({ nom, val: etapes[i] })).filter(l => l.val !== 'Non concerné');
  $('trEtiquetteOut').innerHTML = `<div class="etiquette" id="trEtiquette">
    <div class="etHead"><b>Usage de l'IA</b><span>${esc(date)}${auteur ? ` · ${esc(auteur)}` : ''}</span></div>
    ${lignes.map(l => `<div class="etLigne"><span>${esc(l.nom)}</span><b class="v${CHOIX.indexOf(l.val)}">${esc(l.val)}</b></div>`).join('')}
    ${outils ? `<div class="etLigne"><span>Outils</span><b>${esc(outils)}</b></div>` : ''}
    ${etiquette.carte ? `<div class="etCarte"><span>Mesuré sur la conversation fournie</span>
      <div class="pile"><i class="reprise" style="width:${carte.reprise}%"></i><i class="retravail" style="width:${carte.retravail}%"></i><i class="perso" style="width:${carte.perso}%"></i></div>
      <small>${carte.reprise} % repris de l'IA · ${carte.retravail} % retravaillé · ${carte.perso} % personnel</small></div>` : ''}
    <div class="etPied"><span>${etiquette.mots} mots</span><span>Empreinte <code>${e.court}</code></span></div>
  </div>
  <div class="bar"><button id="trCopier">Copier l'étiquette (texte)</button><button id="trPng">Télécharger l'image (.png)</button></div>
  <p class="small">L'empreinte est calculée à partir du texte : si une seule phrase change, elle ne correspond plus. Le lecteur peut la vérifier en bas de cette page.</p>`;
  $('trCopier').onclick = async () => { try { await navigator.clipboard.writeText(texteEtiquette()); $('trEtiquetteEtat').textContent = 'Étiquette copiée.'; } catch { $('trEtiquetteEtat').textContent = 'Copie refusée par le navigateur.'; } };
  $('trPng').onclick = () => imageEtiquette().then(b => download('etiquette-usage-ia.png', b));
  // Déclaration et mesure qui se contredisent : on prévient l'auteur (l'étiquette n'est pas modifiée).
  const redaction = etapes[2];
  $('trEtiquetteEtat').textContent = etiquette.carte && etiquette.carte.reprise >= 20 && redaction === 'Sans IA'
    ? `Attention : vous déclarez « Rédaction : Sans IA », mais la carte montre ${etiquette.carte.reprise} % de texte repris de l'IA.` : '';
});

function texteEtiquette() {
  const E = etiquette;
  return [
    `Étiquette d'usage de l'IA · ${E.date}${E.auteur ? ` · ${E.auteur}` : ''}`,
    ...ETAPES.map((n, i) => E.etapes[i] !== 'Non concerné' ? `${n} : ${E.etapes[i]}` : null).filter(Boolean),
    ...(E.outils ? [`Outils : ${E.outils}`] : []),
    ...(E.carte ? [`Mesuré sur la conversation fournie : ${E.carte.reprise} % repris de l'IA, ${E.carte.retravail} % retravaillé, ${E.carte.perso} % personnel`] : []),
    `Texte de ${E.mots} mots, empreinte ${E.code}`,
    `Vérification : ${new URL('transparence.html', location.href).href}`
  ].join('\n');
}

async function imageEtiquette() {
  const E = etiquette, W = 720, lignes = ETAPES.map((n, i) => [n, E.etapes[i]]).filter(([, v]) => v !== 'Non concerné');
  if (E.outils) lignes.push(['Outils', E.outils]);
  const H = 150 + lignes.length * 44 + (E.carte ? 96 : 0) + 70;
  const c = document.createElement('canvas'); c.width = W * 2; c.height = H * 2;
  const g = c.getContext('2d'); g.scale(2, 2);
  const encre = '#1D1A4E', police = 'system-ui, -apple-system, "Segoe UI", sans-serif';
  g.fillStyle = '#fff'; g.fillRect(0, 0, W, H);
  g.strokeStyle = encre; g.lineWidth = 4; g.strokeRect(2, 2, W - 4, H - 4);
  g.fillStyle = encre; g.font = `800 34px ${police}`; g.fillText('Usage de l\'IA', 28, 56);
  g.font = `500 17px ${police}`; g.fillText(`${E.date}${E.auteur ? ` · ${E.auteur}` : ''}`, 28, 86);
  g.fillRect(28, 104, W - 56, 6);
  let y = 146;
  const couleurs = ['#1E9E5E', '#B88A00', '#D6336C', '#5E5A8C'];
  lignes.forEach(([n, v]) => {
    g.font = `500 18px ${police}`; g.fillStyle = encre; g.fillText(n, 28, y);
    g.font = `800 18px ${police}`; g.fillStyle = couleurs[CHOIX.indexOf(v)] || encre;
    const tw = g.measureText(v).width; g.fillText(v.length > 48 ? v.slice(0, 47) + '…' : v, Math.max(300, W - 28 - tw), y);
    g.fillStyle = '#D9D6F5'; g.fillRect(28, y + 14, W - 56, 1.5);
    y += 44;
  });
  if (E.carte) {
    g.fillStyle = encre; g.font = `600 16px ${police}`; g.fillText('Mesuré sur la conversation fournie', 28, y + 4);
    let x = 28; const L = W - 56;
    [['#FFD3DC', E.carte.reprise], ['#FFE27A', E.carte.retravail], ['#C9F2DC', E.carte.perso]].forEach(([col, p]) => { g.fillStyle = col; g.fillRect(x, y + 18, L * p / 100, 22); x += L * p / 100; });
    g.strokeStyle = encre; g.lineWidth = 2; g.strokeRect(28, y + 18, L, 22);
    g.fillStyle = encre; g.font = `500 15px ${police}`; g.fillText(`${E.carte.reprise} % repris de l'IA · ${E.carte.retravail} % retravaillé · ${E.carte.perso} % personnel`, 28, y + 64);
    y += 96;
  }
  g.fillStyle = encre; g.fillRect(28, y - 10, W - 56, 3);
  g.font = `600 16px ${police}`; g.fillText(`${E.mots} mots`, 28, y + 22);
  const code = `Empreinte ${E.code}`; g.font = `700 16px ui-monospace, monospace`; g.fillText(code, W - 28 - g.measureText(code).width, y + 22);
  return new Promise(ok => c.toBlob(ok, 'image/png'));
}

/* ---------- Vérifier une étiquette ---------- */
$('trVerifier').addEventListener('click', async () => {
  const code = $('trCode').value.toUpperCase().replace(/[^0-9A-F]/g, '');
  const texte = $('trTexteVerif').value;
  if (code.length < 16 || !words(texte).length) { $('trVerifOut').textContent = 'Collez le texte et l\'empreinte de 16 caractères figurant sur l\'étiquette.'; return; }
  const e = await empreinte(texte);
  const ok = e.hex.slice(0, 16).toUpperCase() === code.slice(0, 16);
  $('trVerifOut').innerHTML = ok
    ? '<span class="pastille humain">Correspond</span> Ce texte est bien celui pour lequel l\'étiquette a été créée.'
    : '<span class="pastille ia">Ne correspond pas</span> Le texte a été modifié depuis la création de l\'étiquette, ou l\'étiquette concerne un autre texte.';
});
