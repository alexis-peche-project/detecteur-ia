// Page Fiabilité et méthode : résultats du banc d'essai et calibration personnelle.
import { splitSentences, words } from './heuristics.js';
import { lmSummary, calibrate } from './scoring.js';
import { DEFAULT_CALIBRATION } from './config.js';
import { $, esc, fmt } from './commun.js';
import { loadCalibration, runLm, calib, setCalib } from './moteur.js';

const NOMS = { clf: 'Détecteur entraîné', lm: 'Modèle de langage (Fast-DetectGPT)', heur: 'Signes Wikipédia et style', juge: 'Vérification par IA', meld: 'MELD' };
const CONSIGNES = { ecrire: 'Écrit à partir d\'un sujet', humaniser: 'Demandé « comme un humain »', reecrire: 'Paraphrase d\'un texte humain' };
const pc = x => (Math.round(x * 1000) / 10).toLocaleString('fr-FR') + ' %';
const barre = x => `<td class="barc"><div class="hbar"><i style="width:${Math.round(x * 100)}%"></i></div></td>`;

function tableau(titre, entetes, lignes) {
  return `<h3 class="sous">${esc(titre)}</h3><div class="scroll"><table><thead><tr>${entetes}</tr></thead><tbody>${lignes}</tbody></table></div>`;
}

async function renderPerf() {
  let B = null;
  try { const r = await fetch('resultats/benchmark.json', { cache: 'no-cache' }); if (r.ok) B = await r.json(); } catch { /* absent */ }
  if (!B?.methodes || !Object.keys(B.methodes).length) {
    $('perfBody').innerHTML = `<p>Le banc d'essai n'a pas encore été lancé sur ce site : aucune mesure n'est publiée pour l'instant.</p>`;
    return;
  }
  const row = (nom, r) => `<tr><td>${esc(nom)}</td><td class="n">${fmt(r.auroc, 3)}</td>${barre(r.tpr_a_1pc_fpr)}<td class="n">${pc(r.tpr_a_1pc_fpr)}</td><td class="n">${Number.isFinite(r.faux_positifs) ? pc(r.faux_positifs) : '—'}</td></tr>`;
  const meth = Object.entries(B.methodes).map(([k, r]) => row(NOMS[k] || k, r)).join('');
  const best = Object.entries(B.combinaisons || {}).sort((a, b) => b[1].tpr_a_1pc_fpr - a[1].tpr_a_1pc_fpr)[0];
  const combo = best ? row('Combinaison : ' + best[0].split('+').map(k => NOMS[k] || k).join(' + '), best[1]) : '';
  let html = `<div class="scroll"><table><thead><tr><th>Méthode</th><th class="n">AUROC</th><th colspan="2">IA repérée à 1 % de faux positifs</th><th class="n">Faux positifs au seuil</th></tr></thead><tbody>${meth}${combo}</tbody></table></div>`;

  const C = B.methodes.clf;
  if (C?.ia_reperee_par_generateur) html += tableau('Détecteur entraîné : textes repérés pour chaque IA', '<th>IA</th><th class="n">Textes</th><th colspan="2">Repérés</th>',
    Object.entries(C.ia_reperee_par_generateur).sort((a, b) => b[1].taux - a[1].taux).map(([k, v]) => `<tr><td>${esc(k.split('/').pop())}</td><td class="n">${v.n}</td>${barre(v.taux)}<td class="n">${pc(v.taux)}</td></tr>`).join(''));
  if (C?.ia_reperee_par_consigne) html += tableau('Détecteur entraîné : selon la façon dont le texte a été produit', '<th>Consigne donnée à l\'IA</th><th class="n">Textes</th><th colspan="2">Repérés</th>',
    Object.entries(C.ia_reperee_par_consigne).map(([k, v]) => {
      const m = k.match(/^réécrit \((.+)\)$/);
      return `<tr><td>${esc(m ? `Réécrit par l'outil du site, style ${m[1]}` : (CONSIGNES[k] || k))}</td><td class="n">${v.n}</td>${barre(v.taux)}<td class="n">${pc(v.taux)}</td></tr>`;
    }).join(''));
  if (C?.faux_positifs_par_domaine) html += tableau('Détecteur entraîné : textes humains accusés à tort, par genre', '<th>Genre de texte</th><th class="n">Textes</th><th colspan="2">Fausses alertes</th>',
    Object.entries(C.faux_positifs_par_domaine).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="n">${v.n}</td>${barre(v.taux)}<td class="n">${pc(v.taux)}</td></tr>`).join(''));
  html += `<p class="small">Mesuré le ${esc(B.date)} sur ${B.donnees.test} textes de test en français (${esc(B.donnees.domaines.join(', '))}). Ces chiffres valent pour des textes semblables : ils seront moins bons sur des genres ou des IA absents du test.</p>`;
  $('perfBody').innerHTML = html;
}

/* ---------- Calibration ---------- */
function calState() {
  const custom = calib.center !== DEFAULT_CALIBRATION.center || calib.scale !== DEFAULT_CALIBRATION.scale;
  $('calState').textContent = `Seuil actuel : ${fmt(calib.center, 3)} (${custom ? 'personnalisé' : 'par défaut'})`;
}
$('calReset').addEventListener('click', () => { setCalib({ ...DEFAULT_CALIBRATION }); $('calOut').innerHTML = ''; calState(); });
$('calGo').addEventListener('click', async () => {
  const split = s => s.split(/^\s*---\s*$/m).map(x => x.trim()).filter(x => words(x).length >= 30);
  const hu = split($('calHu').value), ai = split($('calAi').value);
  if (!hu.length || !ai.length) { $('calOut').innerHTML = '<p class="small">Ajoutez au moins un texte de 30 mots ou plus dans chaque colonne.</p>'; return; }
  $('calGo').disabled = true;
  const rows = [];
  try {
    const all = [...hu.map(t => [t, 'Humain']), ...ai.map(t => [t, 'IA'])];
    for (let i = 0; i < all.length; i++) {
      $('calState').textContent = `Analyse du texte ${i + 1} sur ${all.length}…`;
      const [t, kind] = all[i];
      const r = await runLm(splitSentences(t).map(s => s.text), 6, m => {
        if (m.type === 'download') $('calState').textContent = `Téléchargement du modèle : ${Math.round(m.loaded / 1e6)} Mo…`;
      });
      const s = lmSummary(r.stats, calib);
      rows.push({ kind, D: s.D, words: words(t).length, start: t.slice(0, 60) });
    }
    const c = calibrate(rows.filter(r => r.kind === 'Humain').map(r => r.D), rows.filter(r => r.kind === 'IA').map(r => r.D));
    setCalib({ center: c.center, scale: c.scale });
    $('calOut').innerHTML = `<p>Nouveau seuil : <b>${fmt(c.center, 3)}</b>. ${Math.round(c.accuracy * 100)} % de vos textes sont bien classés avec ce seuil.</p>
      <table><thead><tr><th>Origine</th><th>Début du texte</th><th class="n">Mots</th><th class="n">Courbure</th></tr></thead><tbody>
      ${rows.map(r => `<tr><td>${r.kind}</td><td>${esc(r.start)}…</td><td class="n">${r.words}</td><td class="n">${fmt(r.D, 3)}</td></tr>`).join('')}</tbody></table>`;
  } catch (err) {
    $('calOut').innerHTML = `<p class="small">${esc(err.message)}</p>`;
  }
  $('calGo').disabled = false;
  calState();
});

/* ---------- Test indépendant (textes jamais vus, autres genres, autres IA) ---------- */
async function renderTest() {
  let T = null;
  try { const r = await fetch('resultats/test_independant.json', { cache: 'no-cache' }); if (r.ok) T = await r.json(); } catch { /* pas encore de test */ }
  const el = $('testBody');
  if (!el) return;
  if (!T) { el.innerHTML = '<p>Pas encore de test indépendant publié.</p>'; return; }
  const ligne = (nom, t) => t?.taux == null ? '' : `<tr><td>${esc(nom)}</td><td class="n">${t.n}</td>${barre(t.taux)}<td class="n">${pc(t.taux)}</td></tr>`;
  const bloc = (titre, ent, obj) => obj && Object.keys(obj).length ? tableau(titre, `<th>${ent}</th><th class="n">Textes</th><th colspan="2">Taux</th>`, Object.entries(obj).map(([k, v]) => ligne(k, v)).join('')) : '';
  el.innerHTML = `<div class="scroll"><table><tbody>
      ${ligne('Textes d\'IA repérés (score ≥ 60 %)', T.ia_reperee)}
      ${ligne('Textes d\'IA repérés ou jugés incertains (≥ 40 %)', T.ia_reperee_ou_incertain)}
      ${ligne('Textes humains accusés à tort (≥ 60 %)', T.humains_accuses)}
      ${ligne('Textes humains jugés incertains (40 à 60 %)', T.humains_incertains)}
    </tbody></table></div>
    ${bloc('IA repérée, par modèle', 'Modèle', T.ia_reperee_par_generateur)}
    ${bloc('IA repérée, selon la consigne', 'Consigne', T.ia_reperee_par_consigne)}
    ${bloc('Fausses alertes, par genre de texte humain', 'Genre', T.fausses_alertes_par_genre)}
    ${bloc('Fausses alertes, selon la longueur', 'Longueur', T.fausses_alertes_par_longueur)}
    ${bloc('Textes mixtes (moitié humain, moitié IA)', 'Résultat', T.textes_mixtes)}
    <p class="small">Mesuré le ${esc(T.date)} sur ${T.n_humains} textes humains et ${T.n_ia} textes d'IA que le détecteur n'avait jamais vus${T.auroc != null ? ` (AUROC ${fmt(T.auroc, 3)})` : ''}. Modèles absents de l'entraînement : ${esc((T.generateurs || []).join(', '))}.</p>`;
}

loadCalibration().then(calState);
renderPerf();
renderTest();
