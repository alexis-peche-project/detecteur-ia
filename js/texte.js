// Page Texte : analyse d'un texte ou d'un document.
import { words } from './heuristics.js';
import { verdict } from './scoring.js';
import { extractText, ACCEPT } from './docs.js';
import { MELD_URL } from './config.js';
import { $, esc, fmt, pct, store, download, initReglagesIA, mascotte, humeurDuScore, zoneDepot } from './commun.js';
import { analyserTexte, loadCalibration, classifierId, cancelWorkers, sig, CAL } from './moteur.js';

const ia = initReglagesIA();
const humeur = mascotte();
let last = null;
let abort = null;

/* ---------- Saisie et documents ---------- */
const txt = $('txt');
function updCount() {
  const n = words(txt.value).length;
  $('count').textContent = n.toLocaleString('fr-FR') + (n > 1 ? ' mots' : ' mot');
  if (txt.value.length < 200000) store.set('detia-draft', txt.value);
}
txt.addEventListener('input', updCount);

$('file').accept = ACCEPT;
async function loadFile(file) {
  const info = $('fileInfo');
  info.classList.remove('err');
  info.textContent = `Lecture de « ${file.name} »…`;
  try {
    const text = await extractText(file, (i, n) => { info.textContent = `Lecture de « ${file.name} » : page ${i} sur ${n}…`; });
    txt.value = text.trim();
    updCount();
    info.textContent = `« ${file.name} » importé : ${words(text).length.toLocaleString('fr-FR')} mots. Vous pouvez corriger le texte extrait avant l'analyse.`;
  } catch (err) {
    info.classList.add('err');
    info.textContent = err.message;
  }
}
$('file').addEventListener('change', e => { const f = e.target.files[0]; if (f) loadFile(f); e.target.value = ''; });
zoneDepot($('drop'), files => {
  loadFile(files[0]);
  if (files.length > 1) $('fileInfo').innerHTML = `Plusieurs fichiers déposés : seul le premier est lu ici. Pour les analyser tous, utilisez la page <a href="lot.html">Plusieurs documents</a>.`;
});

/* ---------- Options ---------- */
function updClfOption() {
  const ok = !!classifierId();
  $('useClf').disabled = !ok;
  $('optClf').classList.toggle('dis', !ok);
  if (!ok) {
    $('useClf').checked = false;
    $('clfNote').textContent = 'Pas encore disponible : le modèle doit d\'abord être entraîné (dossier entrainement/ du projet).' + (MELD_URL ? ' Les textes en anglais sont tout de même analysés par MELD.' : '');
  }
}
$('useJudge').checked = store.get('detia-judge') === '1';
$('useLm').checked = store.get('detia-lm') !== '0';
$('useJudge').addEventListener('change', e => { store.set('detia-judge', e.target.checked ? '1' : '0'); if (e.target.checked && !ia.pret()) ia.ouvrir(); });
$('useLm').addEventListener('change', e => store.set('detia-lm', e.target.checked ? '1' : '0'));

function stage(id, { on = true, p = 0, msg = '', state = '' } = {}) {
  const el = $(id);
  el.classList.toggle('off', !on);
  el.classList.toggle('done', state === 'done');
  el.classList.toggle('err', state === 'err');
  el.querySelector('.pbar i').style.width = Math.round(p * 100) + '%';
  el.querySelector('em').textContent = msg;
}

/* ---------- Analyse ---------- */
async function analyse() {
  const text = txt.value;
  if (!words(text).length) { txt.focus(); return; }
  ia.save();
  abort = new AbortController();
  humeur('chomp');
  $('go').disabled = true;
  $('status').classList.remove('hidden');
  $('cancel').classList.remove('hidden');
  const res = await analyserTexte(text, {
    useClf: $('useClf').checked, useLm: $('useLm').checked, useJudge: $('useJudge').checked,
    full: $('scope').value === 'full', cfg: ia.cfg(), signal: abort.signal,
    onStage: stage, onUpdate: r => { last = r; render(r); }
  });
  $('go').disabled = false;
  $('cancel').classList.add('hidden');
  humeur(abort.signal.aborted || !res.final ? 'idle' : humeurDuScore(res.final.score));
}
$('cancel').addEventListener('click', () => { abort?.abort(); cancelWorkers(); });
$('go').addEventListener('click', analyse);

/* ---------- Affichage ---------- */
function card(title, score, body, off) {
  return `<div class="card${off ? ' off' : ''}"><div class="top"><b>${esc(title)}</b><span>${off ? esc(off) : score + ' %'}</span></div>${off ? '' : `<div class="meter"><i style="width:${score}%"></i></div>`}${body}</div>`;
}

function render(res) {
  $('out').classList.remove('hidden');
  const E = res.final;
  $('pct').innerHTML = `${E.score}<small>%</small>`;
  $('verdict').textContent = verdict(E.score);
  const seuilTxt = res.heur.strong ? ' · formule d\'assistant ou artefact de chatbot trouvé' : E.calibrated ? (E.calibrated.flagged1 ? ' · au-delà du seuil à 1 % de faux positifs' : E.calibrated.flagged5 ? ' · au-delà du seuil à 5 % de faux positifs seulement' : ' · sous le seuil de détection') : '';
  $('conf').textContent = 'Confiance : ' + E.confidence + seuilTxt;
  $('gauge').style.width = E.score + '%';
  const H = res.heur;
  const notes = [];
  if (H.n < 80) notes.push('Texte court : résultat peu fiable.');
  if (H.strong) notes.push('Formule d\'assistant ou artefact de chatbot trouvé : signe très fort.');
  notes.push(`${H.n.toLocaleString('fr-FR')} mots, ${res.sentences.length.toLocaleString('fr-FR')} phrases.`);
  $('note').textContent = notes.join(' ');

  const L = res.lm, J = res.judge, C = res.clf, M = res.meld;
  const clfOff = !res.use.clf ? (classifierId() ? 'Désactivé' : 'Pas encore entraîné') : res.errors.clf ? 'Erreur' : !C ? 'En cours…' : null;
  const lmOff = !res.use.lm && !L ? 'Désactivé' : res.errors.lm ? 'Erreur' : !L ? 'En cours…' : null;
  const jOff = !res.use.judge && !J ? 'Désactivée' : res.errors.judge ? 'Erreur' : !J ? 'En cours…' : null;
  const clfP = C ? Math.round(sig(C.logit) * 100) : 0;
  const s1 = CAL?.classifieur?.seuil_1pc;
  $('cards').innerHTML =
    card('Détecteur entraîné', clfP, C ? `<div class="kv">
      <div>Score brut (logit)<b>${fmt(C.logit)}</b></div>
      ${Number.isFinite(s1) ? `<div>Seuil à 1 % de faux positifs<b>${fmt(s1)}</b></div><div>Verdict à ce seuil<b>${C.logit > s1 ? 'IA' : 'Humain'}</b></div>` : ''}
      <div>Fenêtres analysées<b>${C.windowsAnalysed} / ${C.windowsTotal}</b></div></div>` : res.errors.clf ? `<p class="small">${esc(res.errors.clf)}</p>` : '', clfOff) +
    (res.use.meld || M ? card('MELD (anglais)', M ? Math.round(M.p_ai * 100) : 0, M ? `<div class="kv">
      <div>Verdict au seuil de 1 % de faux positifs<b>${M.signale ? 'IA' : 'Humain'}</b></div>
      <div>Famille de modèle probable<b>${esc(M.famille || '—')}</b></div>
      <div>Type d'usage probable<b>${esc(M.tache || '—')}</b></div></div>` : res.errors.meld ? `<p class="small">${esc(res.errors.meld)}</p>` : '', M ? null : res.errors.meld ? 'Erreur' : 'En cours…') : '') +
    card('Modèle de langage', L ? Math.round(L.prob * 100) : 0, L ? `<div class="kv">
      <div>Courbure (Fast-DetectGPT)<b>${fmt(L.D)}</b></div>
      <div>Perplexité<b>${fmt(L.ppl, 1)}</b></div>
      <div>Burstiness<b>${fmt(L.burstiness)}</b></div>
      <div>Mots les plus probables<b>${pct(L.top1)}</b></div>
      <div>Dans le top 10<b>${pct(L.top10)}</b></div>
      <div>Tokens analysés<b>${L.tokens.toLocaleString('fr-FR')}</b></div></div>` : res.errors.lm ? `<p class="small">${esc(res.errors.lm)}</p>` : '', lmOff) +
    card('Vérification par IA', J ? J.probabilite_ia : 0, J ? `<p class="small">${esc(J.verdict)} Confiance du vérificateur : ${esc(J.confiance)}.</p>` : res.errors.judge ? `<p class="small">${esc(res.errors.judge)}</p>` : '', jOff) +
    card('Signes Wikipédia et style', H.score, `<div class="kv">
      <div>Signes pour 100 mots<b>${fmt(H.per100, 1)}</b></div>
      <div>Variation des phrases<b>${fmt(H.sCv)}</b></div>
      <div>Régularité<b>${Math.round(H.burst * 100)} / 100</b></div>
      <div>Mise en forme<b>${Math.round(H.fmt * 100)} / 100</b></div></div>`);

  // Texte annoté : phrases colorées selon le détecteur (ou le modèle de langage), signes soulignés.
  const marks = H.marks.slice().sort((a, b) => a[0] - b[0] || b[1] - a[1]);
  const t = res.text;
  let html = '', pos = 0, mi = 0;
  const inner = (s, e) => {
    let h = '', p = s;
    while (mi < marks.length && marks[mi][0] < e) {
      const [ms, me, c] = marks[mi++];
      if (ms < p || me > e) continue;
      h += esc(t.slice(p, ms)) + `<mark class="c-${c.id}" title="${esc(c.name)}">${esc(t.slice(ms, me))}</mark>`;
      p = me;
    }
    return h + esc(t.slice(p, e));
  };
  res.sentences.forEach((s, i) => {
    html += inner(pos, s.start);
    const zc = C?.perSentence?.[i];
    const p = Number.isFinite(zc) ? sig(zc) : L?.perSentence[i];
    const cls = p == null ? '' : p >= 0.8 ? ' hi' : p >= 0.62 ? ' mid' : '';
    const title = p == null ? '' : ` title="Probabilité IA de la phrase : ${Math.round(p * 100)} %"`;
    html += `<span class="s${cls}"${title}>${inner(s.start, s.end)}</span>`;
    pos = s.end;
  });
  html += inner(pos, t.length);
  $('annot').innerHTML = html;

  $('judgeBox').innerHTML = J ? `<div class="judge"><h2>Avis du vérificateur IA</h2><p>${esc(J.verdict)}</p>
    ${J.indices_ia.length ? '<h3>Indices d\'IA cités</h3>' + J.indices_ia.map(x => `<q>${esc(x.extrait)}</q><p>${esc(x.raison)}</p>`).join('') : ''}
    ${J.indices_humains.length ? '<h3>Indices humains cités</h3>' + J.indices_humains.map(x => `<q>${esc(x.extrait)}</q><p>${esc(x.raison)}</p>`).join('') : ''}</div>` : '';

  const fs = H.found.slice().sort((a, b) => b.ex.length * b.c.wt - a.ex.length * a.c.wt);
  $('signs').innerHTML = fs.length ? fs.map(f => {
    const u = [...new Set(f.ex.map(x => x.toLowerCase()))].slice(0, 6);
    return `<li><div class="t">${esc(f.c.name)}<span>× ${f.ex.length}</span></div><p>${esc(f.c.why)}</p><p>${u.map(x => '<code>' + esc(x) + '</code>').join(' ')}</p></li>`;
  }).join('') : '<li class="empty">Aucun signe typique repéré.</li>';
}

/* ---------- Rapport ---------- */
function report(res) {
  const E = res.final, H = res.heur, L = res.lm, J = res.judge;
  const lines = [
    `Détecteur de texte IA : ${E.score} % (${verdict(E.score)}), confiance ${E.confidence}`,
    `${H.n} mots, ${res.sentences.length} phrases`, '',
    `Signes Wikipédia et style : ${H.score} %`,
    ...H.found.map(f => `  - ${f.c.name} : ${f.ex.length}`)
  ];
  if (res.clf) lines.push('', `Détecteur entraîné : ${Math.round(sig(res.clf.logit) * 100)} % (logit ${fmt(res.clf.logit)})`);
  if (res.meld) lines.push('', `MELD : ${Math.round(res.meld.p_ai * 100)} % (${res.meld.signale ? 'IA' : 'humain'} au seuil de 1 % de faux positifs)`);
  if (L) lines.push('', `Modèle de langage : ${Math.round(L.prob * 100)} %`, `  courbure ${fmt(L.D)}, perplexité ${fmt(L.ppl, 1)}, burstiness ${fmt(L.burstiness)}, top 10 ${pct(L.top10)}, ${L.tokens} tokens`);
  if (J) lines.push('', `Vérification par IA : ${J.probabilite_ia} % (confiance ${J.confiance})`, `  ${J.verdict}`, ...J.indices_ia.map(x => `  - IA : « ${x.extrait} » ${x.raison}`), ...J.indices_humains.map(x => `  - Humain : « ${x.extrait} » ${x.raison}`));
  return lines.join('\n');
}
$('copy').addEventListener('click', async () => {
  if (!last) return;
  try { await navigator.clipboard.writeText(report(last)); $('copied').textContent = 'Rapport copié.'; }
  catch { $('copied').textContent = 'Copie refusée par le navigateur.'; }
});
$('dlTxt').addEventListener('click', () => last && download('rapport-detecteur-ia.txt', report(last), 'text/plain;charset=utf-8'));
$('dlJson').addEventListener('click', () => {
  if (!last) return;
  const { final, heur, lm, judge: j, lmMeta } = last;
  const data = { score: final.score, verdict: verdict(final.score), confiance: final.confidence, mots: heur.n,
    signes: { score: heur.score, parCategorie: heur.found.map(f => ({ categorie: f.c.name, nombre: f.ex.length, exemples: [...new Set(f.ex)].slice(0, 10) })) },
    modeleDeLangage: lm && { probabilite: lm.prob, courbure: lm.D, perplexite: lm.ppl, burstiness: lm.burstiness, top1: lm.top1, top10: lm.top10, tokens: lm.tokens, blocs: lmMeta && `${lmMeta.chunksAnalysed}/${lmMeta.chunksTotal}` },
    verificationIA: j, detecteurEntraine: last.clf && { logit: last.clf.logit, probabilite: sig(last.clf.logit) }, meld: last.meld };
  download('rapport-detecteur-ia.json', JSON.stringify(data, null, 2), 'application/json');
});
$('toRewrite').addEventListener('click', () => { store.set('detia-draft', txt.value); location.href = 'reecrire.html'; });

/* ---------- Exemples ---------- */
const EX_AI = `Le télétravail : une transformation profonde du monde professionnel

Le télétravail s'est imposé comme un véritable pilier de l'organisation moderne des entreprises. Il ne s'agit pas seulement d'un changement de lieu de travail, c'est une redéfinition complète de notre rapport à l'emploi. Cette évolution souligne l'importance cruciale de la flexibilité dans un paysage professionnel en constante évolution.

Il est important de noter que le télétravail offre de nombreux avantages. Il favorise l'autonomie, la productivité et le bien-être des salariés. En outre, il permet de réduire les déplacements, contribuant ainsi à la préservation de l'environnement. Les outils numériques jouent un rôle clé dans cette transition, facilitant la communication et la collaboration entre les équipes.

Cependant, le télétravail présente également certains défis. L'isolement social, la frontière floue entre vie privée et vie professionnelle et la difficulté à maintenir la cohésion d'équipe sont des enjeux essentiels. Les entreprises doivent donc mettre en place des stratégies adaptées, reflétant leur engagement envers le bien-être de leurs collaborateurs.

En conclusion, le télétravail incarne une évolution incontournable du monde du travail. Il témoigne de la capacité des organisations à s'adapter et à innover. J'espère que cette analyse vous sera utile !`;
const EX_HU = `Trois ans de télétravail, et je ne sais toujours pas si j'aime ça.

Le premier mois, c'était génial. Pas de RER, café à volonté, réunion en chaussons. Puis l'hiver est arrivé. Je me suis rendu compte que je n'avais parlé à personne de toute une semaine, à part la boulangère, et encore, deux mots.

Mon chef, lui, adore. Il dit qu'on n'a jamais été aussi productifs. Peut-être. Moi je trouve surtout qu'on fait plus de visios qu'avant, et qu'elles durent plus longtemps parce que personne n'ose couper la parole à travers un écran.

Bref. Aujourd'hui je bosse deux jours au bureau, trois à la maison. Pas parfait, mais ça tient.`;
$('exAi').addEventListener('click', () => { txt.value = EX_AI; updCount(); analyse(); });
$('exHu').addEventListener('click', () => { txt.value = EX_HU; updCount(); analyse(); });
$('clear').addEventListener('click', () => { txt.value = ''; updCount(); $('out').classList.add('hidden'); $('status').classList.add('hidden'); $('fileInfo').textContent = ''; humeur('idle'); txt.focus(); });

/* ---------- Démarrage ---------- */
const draft = store.get('detia-draft');
if (draft) txt.value = draft;
updCount();
loadCalibration().then(() => {
  updClfOption();
  // Texte envoyé par une autre page (Réécrire, Plusieurs documents) : analyse immédiate.
  const auto = Number(store.get('detia-auto'));
  store.set('detia-auto', null);
  if (auto && Date.now() - auto < 60000) analyse();
});
