// Moteur d'analyse de texte, sans interface : utilisé par la page Texte et par la page Plusieurs documents.
import { analyseHeuristics, splitSentences, words } from './heuristics.js';
import { lmSummary, ensemble, calibratedEnsemble, detectLang } from './scoring.js';
import { judge } from './llm-judge.js';
import { MELD_URL, CLASSIFIER_ID, WEIGHTS, DEFAULT_CALIBRATION } from './config.js';
import { store } from './commun.js';

/* ---------- Réglages issus du banc d'essai ---------- */
export let CAL = null;
export let calib = (() => { try { return JSON.parse(store.get('detia-calib')) || DEFAULT_CALIBRATION; } catch { return DEFAULT_CALIBRATION; } })();
export function setCalib(c) { calib = c; store.set('detia-calib', c === DEFAULT_CALIBRATION ? null : JSON.stringify(c)); }
let calPromise = null;
export function loadCalibration() {
  if (!calPromise) calPromise = (async () => {
    try {
      const r = await fetch('js/calibration.json', { cache: 'no-cache' });
      if (r.ok) CAL = await r.json();
    } catch { /* pas encore de banc d'essai */ }
    if (CAL?.lm && !store.get('detia-calib')) calib = CAL.lm;
    return CAL;
  })();
  return calPromise;
}
export const classifierId = () => CLASSIFIER_ID || (CAL?.classifieur?.id && !/^[./{]/.test(CAL.classifieur.id) ? CAL.classifieur.id : '');

/* ---------- Workers : modèle de langage et détecteur entraîné ---------- */
const workers = {};
function getWorker(kind) {
  if (!workers[kind]) {
    const file = kind === 'lm' ? './lm-worker.js' : './classifier-worker.js';
    workers[kind] = new Worker(new URL(file, import.meta.url), { type: 'module' });
  }
  return workers[kind];
}
export function runWorker(kind, msg, onEvent) {
  return new Promise((resolve, reject) => {
    const w = getWorker(kind);
    w.onmessage = e => {
      const m = e.data;
      if (m.type === 'result') resolve(m);
      else if (m.type === 'error') reject(new Error(m.message));
      else if (m.type === 'cancelled') reject(new Error('Analyse annulée.'));
      else onEvent?.(m);
    };
    w.onerror = e => reject(new Error(e.message || 'Le modèle n\'a pas pu démarrer.'));
    w.postMessage({ type: 'analyse', ...msg });
  });
}
export const runLm = (sentences, maxChunks, onEvent) => runWorker('lm', { sentences, maxChunks }, onEvent);
export function cancelWorkers() { Object.values(workers).forEach(w => w.postMessage({ type: 'cancel' })); }

const dl = { lm: {}, clf: {} };
export function dlMsg(kind, m) {
  dl[kind][m.file] = m;
  const all = Object.values(dl[kind]), L = all.reduce((a, x) => a + x.loaded, 0), T = all.reduce((a, x) => a + x.total, 0);
  return { p: T ? L / T : 0, msg: `Téléchargement du modèle : ${Math.round(L / 1e6)} / ${Math.round(T / 1e6)} Mo` };
}

async function runMeld(text, signal) {
  const r = await fetch(MELD_URL.replace(/\/$/, '') + '/score', {
    method: 'POST', signal, headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text })
  });
  const b = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(b.detail || `Serveur MELD : erreur ${r.status}`);
  return b;
}

/* ---------- Combinaison des méthodes ---------- */
export const sig = z => 1 / (1 + Math.exp(-z));
function echelle(p, s5, s1) {
  if (!(s5 > 0 && s1 > s5 && s1 < 1)) return p;
  if (p <= s5) return 0.4 * p / s5;
  if (p <= s1) return 0.4 + 0.2 * (p - s5) / (s1 - s5);
  return 0.6 + 0.4 * (p - s1) / (1 - s1);
}

export function combine(res) {
  const H = res.heur;
  const f = { heur: H.score / 100 };
  if (res.clf) f.clf = res.clf.logit;
  if (res.lm) f.lm = res.lm.D;
  if (res.judge) f.juge = res.judge.probabilite_ia / 100;
  let E = null;
  let cal = calibratedEnsemble(CAL, f);
  if (!cal && Number.isFinite(f.juge)) {
    // Vérification par IA non mesurée par le banc d'essai : combinaison réglée sans elle, puis mélange simple.
    const { juge, ...sans } = f;
    const c2 = calibratedEnsemble(CAL, sans);
    if (c2) cal = { ...c2, p: (1 - WEIGHTS.judge) * c2.p + WEIGHTS.judge * juge };
  }
  if (cal) {
    // Les seuils mesurés au banc d'essai deviennent les limites des verdicts :
    // au-delà du seuil à 1 % de faux positifs → 60 % et plus (« probablement IA »),
    // entre les seuils à 5 % et 1 % → 40 à 60 % (« incertain »), en dessous → moins de 40 %.
    let p = echelle(cal.p, cal.seuil5, cal.seuil1);
    if (H.strong) p = Math.max(p, 0.85);
    E = { p, score: Math.round(p * 100), calibrated: cal };
  } else {
    const parts = [{ p: f.heur, w: WEIGHTS.heuristics }];
    if (res.clf) parts.push({ p: sig(res.clf.logit), w: WEIGHTS.clf });
    if (res.lm) parts.push({ p: res.lm.prob, w: WEIGHTS.lm });
    if (res.judge) parts.push({ p: f.juge, w: WEIGHTS.judge });
    E = ensemble(parts, { strongSign: H.strong, words: H.n });
  }
  if (res.meld) {
    const p = WEIGHTS.meld * res.meld.p_ai + (1 - WEIGHTS.meld) * E.p;
    E = { ...E, p, score: Math.round(p * 100), meldFlag: res.meld.signale };
  }
  // Confiance : longueur du texte, accord entre méthodes, position par rapport au seuil à 1 % de faux positifs.
  const probs = [f.heur, res.clf && sig(res.clf.logit), res.lm?.prob, res.judge && f.juge, res.meld?.p_ai].filter(x => Number.isFinite(x));
  const spread = probs.length > 1 ? Math.max(...probs) - Math.min(...probs) : 0;
  let confidence = 'moyenne';
  if (H.n < 80 || spread > 0.5) confidence = 'faible';
  else if (H.strong || ((res.clf || res.meld) && H.n >= 200 && Math.abs(E.p - 0.5) > 0.3 && spread < 0.4)) confidence = 'élevée';
  E.confidence = confidence;
  return E;
}

/* ---------- Analyse complète d'un texte ----------
   options : useClf, useLm, useJudge, full, cfg (fournisseur d'IA), signal
   onStage(id, état) : progression de chaque méthode (stClf, stMeld, stHeur, stLm, stJudge)
   onUpdate(res) : appelé à chaque résultat partiel ; res.final contient le score combiné */
export async function analyserTexte(text, { useClf = true, useLm = false, useJudge = false, full = false, cfg = null, signal, onStage = () => {}, onUpdate = () => {} } = {}) {
  await loadCalibration();
  useClf = useClf && !!classifierId();
  const lang = detectLang(text);
  const useMeld = !!MELD_URL && lang === 'en' && words(text).length >= 30;
  const sentences = splitSentences(text);
  const res = { text, sentences, lang, heur: analyseHeuristics(text), lm: null, lmMeta: null, judge: null, clf: null, meld: null, errors: {},
    use: { clf: useClf, meld: useMeld, lm: useLm, judge: useJudge } };
  const maj = () => { res.final = combine(res); onUpdate(res); };
  onStage('stHeur', { p: 1, msg: 'Terminé', state: 'done' });
  onStage('stClf', { on: useClf, msg: useClf ? 'En attente' : (classifierId() ? 'Désactivé' : 'Pas encore entraîné') });
  onStage('stMeld', { on: useMeld, msg: useMeld ? 'En attente' : (MELD_URL ? 'Texte non anglais' : 'Serveur non configuré') });
  onStage('stLm', { on: useLm, msg: useLm ? 'En attente' : 'Désactivé' });
  onStage('stJudge', { on: useJudge, msg: useJudge ? 'En attente' : 'Désactivée' });
  maj();

  const tasks = [];
  if (useClf) {
    tasks.push(runWorker('clf', { id: classifierId(), sentences: sentences.map(s => s.text), maxWindows: full ? 0 : 16 }, m => {
      if (m.type === 'download') onStage('stClf', dlMsg('clf', m));
      else if (m.type === 'ready') onStage('stClf', { p: 0, msg: 'Modèle prêt' });
      else if (m.type === 'progress') onStage('stClf', { p: m.done / m.total, msg: `Fenêtre ${m.done} sur ${m.total}` });
    }).then(r => {
      res.clf = r;
      onStage('stClf', { p: 1, msg: `${r.windowsAnalysed} fenêtre(s) sur ${r.windowsTotal}`, state: 'done' });
    }).catch(err => {
      res.errors.clf = err.message;
      onStage('stClf', { p: 0, msg: err.message, state: 'err' });
    }).finally(maj));
  }
  if (useMeld) {
    onStage('stMeld', { p: 0.1, msg: 'Envoi au serveur MELD…' });
    tasks.push(runMeld(text, signal).then(r => {
      res.meld = r;
      onStage('stMeld', { p: 1, msg: `${r.tokens} tokens analysés`, state: 'done' });
    }).catch(err => {
      res.errors.meld = err.name === 'AbortError' ? 'Analyse annulée.' : err.message;
      onStage('stMeld', { p: 0, msg: res.errors.meld, state: 'err' });
    }).finally(maj));
  }
  if (useLm) {
    tasks.push(runLm(sentences.map(s => s.text), full ? 0 : 12, m => {
      if (m.type === 'download') onStage('stLm', dlMsg('lm', m));
      else if (m.type === 'ready') onStage('stLm', { p: 0, msg: `Modèle prêt (${m.backend.device === 'webgpu' ? 'carte graphique' : 'processeur'})` });
      else if (m.type === 'progress') {
        const left = m.elapsed / m.done * (m.total - m.done) / 1000;
        onStage('stLm', { p: m.done / m.total, msg: `Bloc ${m.done} sur ${m.total}${m.done < m.total ? ` · environ ${Math.ceil(left)} s restantes` : ''}` });
      }
    }).then(r => {
      res.lmMeta = r;
      res.lm = lmSummary(r.stats, calib);
      onStage('stLm', { p: 1, msg: `${r.chunksAnalysed} bloc(s) sur ${r.chunksTotal} analysé(s)`, state: 'done' });
    }).catch(err => {
      res.errors.lm = err.message;
      onStage('stLm', { p: 0, msg: err.message, state: 'err' });
    }).finally(maj));
  }
  if (useJudge && cfg) {
    onStage('stJudge', { p: 0.05, msg: 'Envoi au vérificateur…' });
    tasks.push(judge(cfg, text, {
      maxCalls: full ? 8 : 2, signal,
      onProgress: (i, n) => onStage('stJudge', { p: Math.max(0.05, i / n), msg: `Appel ${Math.min(i + 1, n)} sur ${n}` })
    }).then(j => {
      res.judge = j;
      onStage('stJudge', { p: 1, msg: `${j.calls} appel(s)`, state: 'done' });
    }).catch(err => {
      res.errors.judge = err.name === 'AbortError' ? 'Analyse annulée.' : err.message;
      onStage('stJudge', { p: 0, msg: res.errors.judge, state: 'err' });
    }).finally(maj));
  }
  await Promise.allSettled(tasks);
  return res;
}
