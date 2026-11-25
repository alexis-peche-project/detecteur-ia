import { analyseHeuristics, splitSentences, words } from './heuristics.js';
import { lmSummary, ensemble, verdict, calibrate } from './scoring.js';
import { judge, PROVIDERS } from './llm-judge.js';
import { extractText, ACCEPT } from './docs.js';
import { RELAY_URL, WEIGHTS, DEFAULT_CALIBRATION } from './config.js';

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (x, d = 2) => Number(x).toFixed(d).replace('.', ',');
const pct = x => Math.round(x * 100) + ' %';
const store = {
  get(k, area = localStorage) { try { return area.getItem(k); } catch { return null; } },
  set(k, v, area = localStorage) { try { v == null ? area.removeItem(k) : area.setItem(k, v); } catch { /* stockage indisponible */ } }
};

let calib = (() => { try { return JSON.parse(store.get('detia-calib')) || DEFAULT_CALIBRATION; } catch { return DEFAULT_CALIBRATION; } })();
let last = null;
let abort = null;

/* ---------- Modèle de langage (Web Worker) ---------- */
let worker = null;
const dl = {};
function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./lm-worker.js', import.meta.url), { type: 'module' });
  return worker;
}
function runLm(sentences, maxChunks, onEvent) {
  return new Promise((resolve, reject) => {
    const w = getWorker();
    w.onmessage = e => {
      const m = e.data;
      if (m.type === 'result') resolve(m);
      else if (m.type === 'error') reject(new Error(m.message));
      else if (m.type === 'cancelled') reject(new Error('Analyse annulée.'));
      else onEvent?.(m);
    };
    w.onerror = e => reject(new Error(e.message || 'Le modèle de langage n\'a pas pu démarrer.'));
    w.postMessage({ type: 'analyse', sentences, maxChunks });
  });
}

/* ---------- Interface : saisie et documents ---------- */
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
const drop = $('drop');
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { if (e.dataTransfer?.types?.includes('Files')) { e.preventDefault(); drop.classList.add('over'); } }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, () => drop.classList.remove('over')));
drop.addEventListener('drop', e => { const f = e.dataTransfer?.files?.[0]; if (f) { e.preventDefault(); loadFile(f); } });

/* ---------- Réglages de la vérification par IA ---------- */
const provSel = $('provider');
provSel.innerHTML = Object.entries(PROVIDERS)
  .filter(([k]) => k !== 'relais' || RELAY_URL)
  .map(([k, p]) => `<option value="${k}">${esc(p.label)}</option>`).join('');
function loadProvider() {
  const p = provSel.value;
  $('model').value = store.get('detia-model-' + p) || PROVIDERS[p].model;
  $('apiKey').placeholder = PROVIDERS[p].keyHint;
  $('apiKey').value = store.get('detia-key-' + p) || store.get('detia-key-' + p, sessionStorage) || '';
  $('remember').checked = !!store.get('detia-key-' + p);
  const relay = p === 'relais';
  $('keyRow').classList.toggle('hidden', relay);
  $('rememberRow').classList.toggle('hidden', relay);
  $('model').disabled = relay;
}
function saveProvider() {
  const p = provSel.value;
  store.set('detia-provider', p);
  store.set('detia-model-' + p, $('model').value.trim() || null);
  const key = $('apiKey').value.trim() || null;
  store.set('detia-key-' + p, $('remember').checked ? key : null);
  store.set('detia-key-' + p, key, sessionStorage);
}
provSel.value = store.get('detia-provider') && [...provSel.options].some(o => o.value === store.get('detia-provider')) ? store.get('detia-provider') : (RELAY_URL ? 'relais' : 'anthropic');
loadProvider();
provSel.addEventListener('change', () => { store.set('detia-provider', provSel.value); loadProvider(); });
['model', 'apiKey'].forEach(id => $(id).addEventListener('change', saveProvider));
$('remember').addEventListener('change', saveProvider);
$('useJudge').checked = store.get('detia-judge') === '1';
$('useLm').checked = store.get('detia-lm') !== '0';
$('useJudge').addEventListener('change', e => { store.set('detia-judge', e.target.checked ? '1' : '0'); if (e.target.checked && provSel.value !== 'relais' && !$('apiKey').value) $('apiBox').open = true; });
$('useLm').addEventListener('change', e => store.set('detia-lm', e.target.checked ? '1' : '0'));

/* ---------- Progression ---------- */
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
  const useLm = $('useLm').checked, useJudge = $('useJudge').checked, full = $('scope').value === 'full';
  saveProvider();
  abort = new AbortController();
  $('go').disabled = true;
  $('status').classList.remove('hidden');
  $('cancel').classList.remove('hidden');

  const sentences = splitSentences(text);
  const res = { text, sentences, heur: analyseHeuristics(text), lm: null, lmMeta: null, judge: null, errors: {} };
  last = res;
  stage('stHeur', { p: 1, msg: 'Terminé', state: 'done' });
  stage('stLm', { on: useLm, msg: useLm ? 'En attente' : 'Désactivé' });
  stage('stJudge', { on: useJudge, msg: useJudge ? 'En attente' : 'Désactivée' });
  render(res);

  const tasks = [];
  if (useLm) {
    tasks.push(runLm(sentences.map(s => s.text), full ? 0 : 12, m => {
      if (m.type === 'download') {
        dl[m.file] = m;
        const all = Object.values(dl), L = all.reduce((a, x) => a + x.loaded, 0), T = all.reduce((a, x) => a + x.total, 0);
        stage('stLm', { p: L / T, msg: `Téléchargement du modèle : ${Math.round(L / 1e6)} / ${Math.round(T / 1e6)} Mo` });
      } else if (m.type === 'ready') {
        stage('stLm', { p: 0, msg: `Modèle prêt (${m.backend.device === 'webgpu' ? 'carte graphique' : 'processeur'})` });
      } else if (m.type === 'progress') {
        const left = m.elapsed / m.done * (m.total - m.done) / 1000;
        stage('stLm', { p: m.done / m.total, msg: `Bloc ${m.done} sur ${m.total}${m.done < m.total ? ` · environ ${Math.ceil(left)} s restantes` : ''}` });
      }
    }).then(r => {
      res.lmMeta = r;
      res.lm = lmSummary(r.stats, calib);
      stage('stLm', { p: 1, msg: `${r.chunksAnalysed} bloc(s) sur ${r.chunksTotal} analysé(s)`, state: 'done' });
      render(res);
    }).catch(err => {
      res.errors.lm = err.message;
      stage('stLm', { p: 0, msg: err.message, state: 'err' });
      render(res);
    }));
  }
  if (useJudge) {
    const cfg = { provider: provSel.value, apiKey: $('apiKey').value.trim(), model: $('model').value.trim(), relayUrl: RELAY_URL };
    stage('stJudge', { p: 0.05, msg: 'Envoi au vérificateur…' });
    tasks.push(judge(cfg, text, {
      maxCalls: full ? 8 : 2, signal: abort.signal,
      onProgress: (i, n) => stage('stJudge', { p: Math.max(0.05, i / n), msg: `Appel ${Math.min(i + 1, n)} sur ${n}` })
    }).then(j => {
      res.judge = j;
      stage('stJudge', { p: 1, msg: `${j.calls} appel(s) à ${PROVIDERS[cfg.provider].label}`, state: 'done' });
      render(res);
    }).catch(err => {
      res.errors.judge = err.name === 'AbortError' ? 'Analyse annulée.' : err.message;
      stage('stJudge', { p: 0, msg: res.errors.judge, state: 'err' });
      render(res);
    }));
  }
  await Promise.allSettled(tasks);
  $('go').disabled = false;
  $('cancel').classList.add('hidden');
}

$('cancel').addEventListener('click', () => { abort?.abort(); worker?.postMessage({ type: 'cancel' }); });
$('go').addEventListener('click', analyse);

/* ---------- Affichage ---------- */
function combine(res) {
  const parts = [{ p: res.heur.score / 100, w: WEIGHTS.heuristics, k: 'heur' }];
  if (res.lm) parts.push({ p: res.lm.prob, w: WEIGHTS.lm, k: 'lm' });
  if (res.judge) parts.push({ p: res.judge.probabilite_ia / 100, w: WEIGHTS.judge, k: 'judge' });
  return ensemble(parts, { strongSign: res.heur.strong, words: res.heur.n });
}

function card(title, score, body, off) {
  return `<div class="card${off ? ' off' : ''}"><div class="top"><b>${esc(title)}</b><span>${off ? esc(off) : score + ' %'}</span></div>${off ? '' : `<div class="meter"><i style="width:${score}%"></i></div>`}${body}</div>`;
}

function render(res) {
  $('out').classList.remove('hidden');
  const E = combine(res);
  res.final = E;
  $('pct').textContent = E.score + ' %';
  $('verdict').textContent = verdict(E.score);
  $('conf').textContent = 'Confiance : ' + E.confidence;
  $('gauge').style.width = E.score + '%';
  const H = res.heur;
  const notes = [];
  if (H.n < 80) notes.push('Texte court : résultat peu fiable.');
  if (H.strong) notes.push('Formule d\'assistant ou artefact de chatbot trouvé : signe très fort.');
  notes.push(`${H.n.toLocaleString('fr-FR')} mots, ${res.sentences.length.toLocaleString('fr-FR')} phrases.`);
  $('note').textContent = notes.join(' ');

  const L = res.lm, J = res.judge;
  const lmOff = !$('useLm').checked && !L ? 'Désactivé' : res.errors.lm ? 'Erreur' : !L ? 'En cours…' : null;
  const jOff = !$('useJudge').checked && !J ? 'Désactivée' : res.errors.judge ? 'Erreur' : !J ? 'En cours…' : null;
  $('cards').innerHTML =
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

  // Texte annoté : phrases colorées selon le modèle de langage, signes soulignés.
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
    const p = L?.perSentence[i];
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
  if (L) lines.push('', `Modèle de langage : ${Math.round(L.prob * 100)} %`, `  courbure ${fmt(L.D)}, perplexité ${fmt(L.ppl, 1)}, burstiness ${fmt(L.burstiness)}, top 10 ${pct(L.top10)}, ${L.tokens} tokens`);
  if (J) lines.push('', `Vérification par IA : ${J.probabilite_ia} % (confiance ${J.confiance})`, `  ${J.verdict}`, ...J.indices_ia.map(x => `  - IA : « ${x.extrait} » ${x.raison}`), ...J.indices_humains.map(x => `  - Humain : « ${x.extrait} » ${x.raison}`));
  return lines.join('\n');
}
function download(name, content, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([content], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
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
    verificationIA: j };
  download('rapport-detecteur-ia.json', JSON.stringify(data, null, 2), 'application/json');
});

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
$('clear').addEventListener('click', () => { txt.value = ''; updCount(); $('out').classList.add('hidden'); $('status').classList.add('hidden'); $('fileInfo').textContent = ''; txt.focus(); });

/* ---------- Calibration ---------- */
function calState() {
  const custom = calib.center !== DEFAULT_CALIBRATION.center || calib.scale !== DEFAULT_CALIBRATION.scale;
  $('calState').textContent = `Seuil actuel : ${fmt(calib.center, 3)} (${custom ? 'personnalisé' : 'par défaut'})`;
}
calState();
$('calReset').addEventListener('click', () => { calib = { ...DEFAULT_CALIBRATION }; store.set('detia-calib', null); $('calOut').innerHTML = ''; calState(); });
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
    calib = { center: c.center, scale: c.scale };
    store.set('detia-calib', JSON.stringify(calib));
    $('calOut').innerHTML = `<p>Nouveau seuil : <b>${fmt(c.center, 3)}</b>. ${Math.round(c.accuracy * 100)} % de vos textes sont bien classés avec ce seuil.</p>
      <table><thead><tr><th>Origine</th><th>Début du texte</th><th class="n">Mots</th><th class="n">Courbure</th></tr></thead><tbody>
      ${rows.map(r => `<tr><td>${r.kind}</td><td>${esc(r.start)}…</td><td class="n">${r.words}</td><td class="n">${fmt(r.D, 3)}</td></tr>`).join('')}</tbody></table>`;
  } catch (err) {
    $('calOut').innerHTML = `<p class="small">${esc(err.message)}</p>`;
  }
  $('calGo').disabled = false;
  calState();
});

/* ---------- Démarrage ---------- */
const draft = store.get('detia-draft');
if (draft) txt.value = draft;
updCount();
