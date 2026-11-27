// Page Plusieurs documents : analyse en série, tableau récapitulatif, export.
import { words } from './heuristics.js';
import { verdict } from './scoring.js';
import { extractText, ACCEPT } from './docs.js';
import { $, esc, store, download, zoneDepot, analyserSurPageTexte } from './commun.js';
import { analyserTexte, loadCalibration, classifierId, cancelWorkers } from './moteur.js';

let docs = [];          // { id, nom, file?, text?, etat, res?, erreur? }
let abort = null;
let tri = null;
let n = 0;

$('lotFile').accept = ACCEPT;
$('useLmLot').checked = store.get('detia-lot-lm') === '1';
$('useLmLot').addEventListener('change', e => store.set('detia-lot-lm', e.target.checked ? '1' : '0'));

function ajouter(files) {
  for (const f of files) docs.push({ id: ++n, nom: f.name, file: f, etat: 'attente' });
  dessiner();
}
$('lotFile').addEventListener('change', e => { ajouter([...e.target.files]); e.target.value = ''; });
zoneDepot($('lotDrop'), ajouter);
$('lotPaste').addEventListener('click', () => {
  const parts = $('lotTxt').value.split(/^\s*---\s*$/m).map(x => x.trim()).filter(x => words(x).length);
  parts.forEach(t => docs.push({ id: ++n, nom: `Texte collé ${n}`, text: t, etat: 'attente' }));
  if (parts.length) { $('lotTxt').value = ''; dessiner(); }
});
$('lotClear').addEventListener('click', () => { if (abort) return; docs = []; dessiner(); });

const ETATS = { attente: 'En attente', lecture: 'Lecture…', analyse: 'Analyse…', fini: '', erreur: 'Erreur', annule: 'Annulé' };
const classe = s => s >= 60 ? 'ia' : s >= 40 ? 'doute' : 'humain';

function dessiner() {
  const lignes = docs.slice();
  if (tri) lignes.sort((a, b) => ((b.res?.final.score ?? -1) - (a.res?.final.score ?? -1)) * tri);
  $('lotVide').classList.toggle('hidden', docs.length > 0);
  $('lotTable').classList.toggle('hidden', !docs.length);
  $('lotRows').innerHTML = lignes.map(d => {
    const E = d.res?.final, H = d.res?.heur;
    const signes = H ? H.found.reduce((a, f) => a + f.ex.length, 0) : '';
    return `<tr data-id="${d.id}">
      <td class="nom">${esc(d.nom)}</td>
      <td class="n">${H ? H.n.toLocaleString('fr-FR') : ''}</td>
      <td class="scorec">${E ? `<span class="pastille ${classe(E.score)}">${E.score} %</span>` : `<span class="small">${esc(d.erreur || ETATS[d.etat])}</span>`}</td>
      <td>${E ? esc(verdict(E.score)) : ''}</td>
      <td class="n">${signes}</td>
      <td class="n">${d.res ? `<button class="petit" data-voir="${d.id}">Détail</button>` : ''}</td></tr>`;
  }).join('');
  const faits = docs.filter(d => d.res);
  const c = { ia: 0, doute: 0, humain: 0 };
  faits.forEach(d => c[classe(d.res.final.score)]++);
  $('lotBilan').textContent = docs.length ? `${faits.length} / ${docs.length} analysé(s) · ${c.ia} probablement IA · ${c.doute} incertain(s) · ${c.humain} probablement humain(s)` : '';
  $('lotGo').disabled = !!abort || !docs.some(d => d.etat === 'attente' || d.etat === 'annule');
  $('lotCsv').disabled = !faits.length;
}

$('lotRows').addEventListener('click', e => {
  const id = Number(e.target.dataset?.voir);
  const d = docs.find(x => x.id === id);
  if (d) analyserSurPageTexte(d.res.text, true);
});
$('lotSort').addEventListener('click', () => { tri = tri === 1 ? -1 : 1; $('lotSort').textContent = tri === 1 ? 'Trier : scores croissants' : 'Trier : scores décroissants'; dessiner(); });

$('lotGo').addEventListener('click', async () => {
  await loadCalibration();
  abort = new AbortController();
  $('lotCancel').classList.remove('hidden');
  dessiner();
  const opts = { useClf: $('useClfLot').checked, useLm: $('useLmLot').checked, useJudge: false, signal: abort.signal };
  for (const d of docs) {
    if (abort.signal.aborted) break;
    if (d.etat !== 'attente' && d.etat !== 'annule') continue;
    try {
      if (!d.text) { d.etat = 'lecture'; dessiner(); d.text = (await extractText(d.file)).trim(); }
      if (!words(d.text).length) throw new Error('Aucun texte trouvé dans ce fichier.');
      d.etat = 'analyse'; dessiner();
      $('lotEtat').textContent = `Analyse de « ${d.nom} »…`;
      d.res = await analyserTexte(d.text, { ...opts, onStage: (id, s) => { if (s.msg?.startsWith('Téléchargement')) $('lotEtat').textContent = s.msg; } });
      d.etat = abort.signal.aborted ? 'annule' : 'fini';
      if (abort.signal.aborted) d.res = null;
    } catch (err) {
      d.etat = 'erreur'; d.erreur = err.message;
    }
    dessiner();
  }
  $('lotEtat').textContent = abort.signal.aborted ? 'Analyse arrêtée.' : 'Terminé.';
  abort = null;
  $('lotCancel').classList.add('hidden');
  dessiner();
});
$('lotCancel').addEventListener('click', () => { abort?.abort(); cancelWorkers(); });

$('lotCsv').addEventListener('click', () => {
  const q = s => `"${String(s).replace(/"/g, '""')}"`;
  const rows = [['Document', 'Mots', 'Score IA (%)', 'Verdict', 'Confiance', 'Signes Wikipédia', 'Détecteur entraîné (logit)'].map(q).join(';')];
  docs.filter(d => d.res).forEach(d => {
    const E = d.res.final, H = d.res.heur;
    rows.push([d.nom, H.n, E.score, verdict(E.score), E.confidence, H.found.reduce((a, f) => a + f.ex.length, 0), d.res.clf ? d.res.clf.logit.toFixed(2).replace('.', ',') : ''].map(q).join(';'));
  });
  download('analyse-documents.csv', '﻿' + rows.join('\n'), 'text/csv;charset=utf-8');
});

loadCalibration().then(() => {
  if (!classifierId()) { $('useClfLot').checked = false; $('useClfLot').disabled = true; }
});
dessiner();
