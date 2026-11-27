// Page Enseignants : copies fantômes, test de résistance de la consigne, comparaison des copies, calibrage par classe.
import { words } from './heuristics.js';
import { extractText, ACCEPT } from './docs.js';
import { $, esc, store, initReglagesIA, zoneDepot, analyserSurPageTexte } from './commun.js';
import { appelIA, json, nettoyer } from './ia.js';
import { recouvrement, matriceTfidf, surligner, separerTextes } from './comparer.js';
import { analyserTexte, loadCalibration, classifierId } from './moteur.js';
import { refClasse, enregistrerClasse, oublierClasse } from './classe.js';
import { transcrire, estImage } from './manuscrit.js';

const ia = initReglagesIA();
const besoinCle = el => { if (ia.pret() && $('provider').value !== 'relais') return false; el.textContent = 'Ajoutez votre clé API (Claude, GPT ou Mistral) dans les réglages de l\'IA.'; ia.ouvrir(); return true; };

/* ---------- 1. Le sujet et ses copies fantômes ---------- */
const sujet = $('ensSujet'), niveau = $('ensNiveau');
sujet.value = store.get('detia-ens-sujet') || '';
niveau.value = store.get('detia-ens-niveau') || niveau.value;
sujet.addEventListener('input', () => store.set('detia-ens-sujet', sujet.value));
niveau.addEventListener('change', () => store.set('detia-ens-niveau', niveau.value));

let fantomes = (() => { try { return JSON.parse(store.get('detia-ens-fantomes')) || []; } catch { return []; } })();
const PROFILS = [
  { nom: 'Élève appliqué', consigne: 'Tu es un élève appliqué qui veut une très bonne note.' },
  { nom: 'Élève pressé', consigne: 'Tu es un élève qui veut finir vite : tu écris simplement, avec un vocabulaire courant et un plan basique.' },
  { nom: 'Version « humanisée »', consigne: 'Tu es un élève qui ne veut pas être repéré par un détecteur d\'IA : écris de façon naturelle, varie la longueur des phrases, évite les formules toutes faites, les listes et les conclusions qui résument.' }
];

function dessinerFantomes() {
  $('ensFantomes').innerHTML = fantomes.map((f, i) => `<details class="fantome"><summary><b>${esc(f.nom)}</b><span class="small">${words(f.texte).length} mots</span></summary><div class="annot">${esc(f.texte)}</div></details>`).join('');
  $('ensFantomesInfo').textContent = fantomes.length ? `${fantomes.length} copie(s) fantôme(s) prête(s) pour la comparaison.` : '';
  $('ensComparer').disabled = !fantomes.length || !copies.length;
}

$('ensGenerer').addEventListener('click', async () => {
  const etat = $('ensFantomesInfo');
  if (words(sujet.value).length < 3) { etat.textContent = 'Écrivez d\'abord la consigne du devoir.'; sujet.focus(); return; }
  if (besoinCle(etat)) return;
  ia.save();
  $('ensGenerer').disabled = true;
  const nouv = [];
  try {
    for (let i = 0; i < PROFILS.length; i++) {
      etat.textContent = `Rédaction de la copie fantôme ${i + 1} sur ${PROFILS.length}…`;
      const t = await appelIA(ia.cfg(), {
        system: `${PROFILS[i].consigne} Tu es en ${niveau.value}. Tu rédiges le devoir demandé, en français, avec la longueur attendue à ce niveau.`,
        user: `Consigne du devoir :\n${sujet.value}\n\nRédige le devoir complet. Donne uniquement le texte du devoir, sans titre ajouté ni commentaire.`
      });
      nouv.push({ nom: PROFILS[i].nom, texte: nettoyer(t) });
    }
    fantomes = nouv;
    store.set('detia-ens-fantomes', JSON.stringify(fantomes));
  } catch (e) { etat.textContent = `Erreur : ${e.message}`; }
  $('ensGenerer').disabled = false;
  dessinerFantomes();
});

/* ---------- 2. Test de résistance de la consigne ---------- */
$('ensResistance').addEventListener('click', async () => {
  const out = $('ensResOut');
  if (words(sujet.value).length < 3) { out.textContent = 'Écrivez d\'abord la consigne du devoir.'; sujet.focus(); return; }
  if (besoinCle(out)) return;
  ia.save();
  $('ensResistance').disabled = true;
  out.innerHTML = '<p class="small">Analyse de la consigne…</p>';
  try {
    const exemple = fantomes[0] ? `\n\nVoici ce qu'une IA a produit pour cette consigne :\n<copie>\n${fantomes[0].texte.slice(0, 6000)}\n</copie>` : '';
    const r = json(await appelIA(ia.cfg(), {
      system: 'Tu es conseiller pédagogique. Tu évalues à quel point un devoir peut être entièrement délégué à une IA générative, et tu proposes des consignes qui obligent l\'élève à produire un travail personnel, sans les rendre plus difficiles pour un élève honnête.',
      user: `Niveau : ${niveau.value}\nConsigne :\n${sujet.value}${exemple}\n\nRéponds uniquement avec un objet JSON : {"facilite_ia": <0 à 100, 100 = une IA fait tout le devoir sans effort>, "note_estimee": "<note probable sur 20 d'une copie entièrement écrite par IA>", "pourquoi": "<deux phrases>", "points_faibles": ["<ce qui rend la consigne facile à déléguer>"], "variantes": [{"consigne": "<consigne réécrite>", "pourquoi": "<ce qui la rend difficile à déléguer>"}]} avec 3 variantes.`
    }));
    const f = Math.max(0, Math.min(100, Math.round(Number(r.facilite_ia) || 0)));
    const cls = f >= 70 ? 'ia' : f >= 40 ? 'doute' : 'humain';
    out.innerHTML = `<div class="resist ${cls}"><div class="resistScore"><span>${f}<small>/100</small></span><b>Facilité à déléguer à une IA</b></div>
      <p>${esc(r.pourquoi || '')}${r.note_estimee ? ` Note probable d'une copie écrite par IA : <b>${esc(r.note_estimee)}</b>.` : ''}</p></div>
      ${(r.points_faibles || []).length ? `<h3 class="sous">Ce qui la rend facile à déléguer</h3><ul>${r.points_faibles.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : ''}
      ${(r.variantes || []).length ? `<h3 class="sous">Consignes plus résistantes</h3>${r.variantes.map(v => `<div class="variante"><p><b>${esc(v.consigne)}</b></p><p class="small">${esc(v.pourquoi || '')}</p><button class="petit" data-consigne="${esc(v.consigne)}">Utiliser cette consigne</button></div>`).join('')}` : ''}`;
  } catch (e) { out.innerHTML = `<p class="small">Erreur : ${esc(e.message)}</p>`; }
  $('ensResistance').disabled = false;
});
$('ensResOut').addEventListener('click', e => {
  const c = e.target.dataset?.consigne;
  if (!c) return;
  sujet.value = c; store.set('detia-ens-sujet', c);
  fantomes = []; store.set('detia-ens-fantomes', null); dessinerFantomes();
  sujet.scrollIntoView({ behavior: 'smooth', block: 'center' });
});

/* ---------- 3. Comparer les copies aux fantômes ---------- */
let copies = [];
let nCopie = 0;
$('ensFile').accept = ACCEPT + ',image/*';
async function ajouterCopies(files) {
  const etat = $('ensCopiesEtat');
  for (const f of files) {
    try {
      etat.textContent = `Lecture de « ${f.name} »…`;
      let texte, note = '';
      if (estImage(f)) {
        if (besoinCle(etat)) return;
        const t = await transcrire(ia.cfg(), f);
        texte = t.texte; note = t.resume;
      } else texte = (await extractText(f)).trim();
      if (words(texte).length) copies.push({ id: ++nCopie, nom: f.name, texte, note });
    } catch (e) { etat.textContent = `« ${f.name} » : ${e.message}`; }
  }
  etat.textContent = `${copies.length} copie(s) chargée(s).`;
  dessinerCopies();
}
$('ensFile').addEventListener('change', e => { ajouterCopies([...e.target.files]); e.target.value = ''; });
zoneDepot($('ensDrop'), ajouterCopies);
$('ensPaste').addEventListener('click', () => {
  separerTextes($('ensCopiesTxt').value).forEach(t => copies.push({ id: ++nCopie, nom: `Copie ${nCopie}`, texte: t, note: '' }));
  $('ensCopiesTxt').value = '';
  $('ensCopiesEtat').textContent = `${copies.length} copie(s) chargée(s).`;
  dessinerCopies();
});
$('ensVider').addEventListener('click', () => { copies = []; $('ensCopiesEtat').textContent = ''; dessinerCopies(); });

const p100 = x => Math.round(x * 100) + ' %';
function dessinerCopies() {
  $('ensComparer').disabled = !fantomes.length || !copies.length;
  $('ensTable').classList.toggle('hidden', !copies.length);
  $('ensRows').innerHTML = copies.map(c => {
    const r = c.res;
    return `<tr><td class="nom">${esc(c.nom)}${c.note ? `<div class="small">${esc(c.note)}</div>` : ''}</td><td class="n">${words(c.texte).length}</td>
      <td>${r ? `<span class="pastille ${r.alerte ? 'ia' : r.sim >= 0.35 ? 'doute' : 'humain'}">${p100(r.sim)}</span>` : ''}</td>
      <td class="n">${r ? p100(r.expr) : ''}</td>
      <td>${r ? esc(r.proche) : ''}</td>
      <td>${r ? (r.alerte ? '<b>Plus proche des fantômes que des autres copies</b>' : Number.isFinite(r.baseline) ? 'Dans la norme de la classe' : '—') : ''}</td>
      <td class="n">${r?.score != null ? r.score + ' %' : ''}</td>
      <td class="n">${r ? `<button class="petit" data-voir="${c.id}">Comparer</button>` : ''}</td></tr>`;
  }).join('');
}

$('ensComparer').addEventListener('click', async () => {
  const etat = $('ensCopiesEtat');
  const tous = [...fantomes.map(f => f.texte), ...copies.map(c => c.texte)];
  const M = matriceTfidf(tous), F = fantomes.length;
  // Ressemblance habituelle entre deux copies de la classe (médiane), pour situer chaque copie.
  copies.forEach((c, i) => {
    const autres = copies.map((_, j) => j).filter(j => j !== i).map(j => M[F + i][F + j]).sort((a, b) => a - b);
    const baseline = autres.length >= 2 ? autres[Math.floor(autres.length / 2)] : NaN;
    let best = -1, k = 0;
    for (let g = 0; g < F; g++) if (M[F + i][g] > best) { best = M[F + i][g]; k = g; }
    const expr = Math.max(...fantomes.map(f => recouvrement(c.texte, f.texte, 3)));
    const alerte = best >= 0.3 && (Number.isFinite(baseline) ? best > baseline + 0.12 : best >= 0.5) || expr >= 0.15;
    c.res = { sim: best, proche: fantomes[k].nom, k, expr, baseline, alerte, score: null };
  });
  dessinerCopies();
  if ($('ensDetecteur').checked) {
    await loadCalibration();
    for (const c of copies) {
      etat.textContent = `Détecteur : « ${c.nom} »…`;
      const r = await analyserTexte(c.texte, { useClf: true, useLm: false });
      c.res.score = r.final.score;
      dessinerCopies();
    }
  }
  etat.textContent = 'Comparaison terminée.';
});

$('ensRows').addEventListener('click', e => {
  const c = copies.find(x => x.id === Number(e.target.dataset?.voir));
  if (!c) return;
  const f = fantomes[c.res.k];
  $('ensDuo').innerHTML = `<h3 class="sous">« ${esc(c.nom)} » face à la copie fantôme « ${esc(f.nom)} »</h3>
    <p class="small">Surlignées : les suites de trois mots présentes dans les deux textes.</p>
    <div class="duo"><div><b>Copie</b><div class="annot">${surligner(c.texte, f.texte)}</div></div><div><b>Copie fantôme</b><div class="annot">${surligner(f.texte, c.texte)}</div></div></div>
    <div class="bar"><button class="petit" id="ensAnalyser">Analyser cette copie en détail</button></div>`;
  $('ensAnalyser').onclick = () => analyserSurPageTexte(c.texte, true);
  $('ensDuo').scrollIntoView({ behavior: 'smooth', block: 'start' });
});

/* ---------- 4. Calibrage par classe ---------- */
function etatClasse() {
  const r = refClasse();
  $('clsEtat').innerHTML = r
    ? `<b>${esc(r.nom)}</b> : ${r.scores.length} textes de référence (${r.date}). Scores : médiane ${[...r.scores].sort((a, b) => a - b)[Math.floor(r.scores.length / 2)]} %, maximum ${Math.max(...r.scores)} %, ${r.scores.filter(s => s >= 60).length} au-dessus de 60 %. Les pages Texte et Plusieurs documents situent désormais chaque texte par rapport à ces références.`
    : 'Aucun calibrage enregistré.';
  $('clsOublier').disabled = !r;
}
$('clsFile').accept = ACCEPT;
let refs = [];
$('clsFile').addEventListener('change', async e => {
  for (const f of e.target.files) { try { const t = (await extractText(f)).trim(); if (words(t).length >= 50) refs.push(t); } catch { /* fichier illisible */ } }
  e.target.value = '';
  $('clsCompte').textContent = `${refs.length} texte(s) de référence prêt(s).`;
});
$('clsGo').addEventListener('click', async () => {
  const tous = [...refs, ...separerTextes($('clsTxt').value).filter(t => words(t).length >= 50)];
  if (tous.length < 5) { $('clsCompte').textContent = 'Il faut au moins 5 textes de 50 mots ou plus (10 à 20 c\'est mieux).'; return; }
  await loadCalibration();
  $('clsGo').disabled = true;
  const scores = [];
  for (let i = 0; i < tous.length; i++) {
    $('clsCompte').textContent = `Analyse du texte ${i + 1} sur ${tous.length}…`;
    const r = await analyserTexte(tous[i], { useClf: !!classifierId(), useLm: false });
    scores.push(r.final.score);
  }
  enregistrerClasse(scores, $('clsNom').value.trim());
  refs = []; $('clsTxt').value = '';
  $('clsCompte').textContent = 'Calibrage enregistré sur cet appareil.';
  $('clsGo').disabled = false;
  etatClasse();
});
$('clsOublier').addEventListener('click', () => { oublierClasse(); etatClasse(); });

dessinerFantomes();
dessinerCopies();
etatClasse();
