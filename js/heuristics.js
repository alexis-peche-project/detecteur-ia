// Indicateurs stylistiques : signes du guide Wikipédia « Signs of AI writing »
// (WikiProject AI Cleanup), régularité des phrases, uniformité et mise en forme.

const L = '\\p{L}';
const w = list => new RegExp('(?<!' + L + ')(?:' + list.join('|') + ')(?!' + L + ')', 'giu');

export const CATS = [
  { id: 'vocab', name: 'Vocabulaire typique des IA', wt: 1,
    why: 'Mots surreprésentés dans les textes de chatbots (liste Wikipédia, adaptée au français).',
    re: w(["cruciale?s?", "essentiel(?:le)?s?", "incontournables?", "fascinante?s?", "tapisserie", "témoigne(?:nt)? d[e']", "témoignage d[e']", "pivot", "fondamentale?s?", "vibrante?s?", "nichée?s?", "souligne(?:nt|r)?", "met(?:tent|tre)? en lumière", "met(?:tent|tre)? en (?:avant|exergue)", "favorise(?:nt|r)?", "paysage", "écosystème", "synergies?", "levier", "approfondir", "plonger dans", "plongeons", "naviguer", "subtile?s?", "nuancée?s?", "holistiques?", "robustes?", "transformat(?:eur|rice)s?", "révolutionnaires?", "innovante?s?", "indéniablement", "en constante évolution", "à l'ère (?:du numérique|de)", "riche patrimoine", "véritable", "multiples? facettes", "incarne(?:nt)?", "myriade", "delve", "delves", "delving", "tapestry", "testament", "pivotal", "underscores?", "showcas(?:e|es|ing)", "vibrant", "nestled", "boasts?", "enduring", "landscape", "crucial", "intricate", "foster(?:s|ing)?", "garner(?:ed|s)?", "interplay", "realm", "multifaceted", "seamless(?:ly)?", "leverag(?:e|es|ing)", "meticulous(?:ly)?", "additionally", "moreover", "furthermore", "notably", "navigat(?:e|ing) the"]) },
  { id: 'trans', name: 'Transitions mécaniques', wt: 0.6,
    why: 'Connecteurs placés en tête de phrase de façon systématique.',
    re: /(?:^|[.!?]\s+)(?:De plus|En outre|Par ailleurs|Cependant|Ainsi|Néanmoins|Toutefois|D'autre part|En effet|Enfin|Moreover|Furthermore|Additionally|However),/gmu },
  { id: 'neg', name: 'Parallélisme négatif', wt: 2.5,
    why: 'Structure « ce n\'est pas seulement X, c\'est Y » utilisée pour paraître profond.',
    re: /(?:ce n'est pas (?:seulement|simplement|juste|uniquement)|il ne s'agit pas (?:seulement|simplement|uniquement|juste) d|non seulement|pas seulement .{1,50}?mais (?:aussi|également)|not (?:just|only|merely) .{1,50}?(?:but|it'?s|it is)|it'?s not .{1,40}?, it'?s)/giu },
  { id: 'edito', name: 'Remarques éditoriales', wt: 2,
    why: 'Formules de mise en valeur qui n\'apportent aucune information.',
    re: /(?:il est (?:important|essentiel|crucial|intéressant) de (?:noter|souligner|rappeler|comprendre)|il convient de (?:noter|souligner|rappeler)|il est à noter|notons que|force est de constater|il est indéniable|it'?s (?:important|worth|crucial) (?:to note|noting)|it is (?:important|worth) (?:to note|noting))/giu },
  { id: 'ana', name: 'Étiquettes analytiques vides', wt: 1.5,
    why: 'Participes en fin de phrase qui ajoutent une importance vague (« soulignant son influence durable »).',
    re: /(?:,\s*(?:soulignant|reflétant|témoignant|illustrant|contribuant|marquant|renforçant|symbolisant|consolidant|incarnant)|jou(?:e|ent|er|ant) un rôle (?:clé|crucial|essentiel|majeur|central|déterminant|primordial)|se dresse comme|s'impose comme|constitue un véritable|,\s*(?:highlighting|underscoring|reflecting|showcasing|emphasizing|symbolizing|contributing to)|plays? a (?:crucial|pivotal|key|vital|significant) role|stands as|serves as a)/giu },
  { id: 'concl', name: 'Conclusions et résumés répétés', wt: 1.5,
    why: 'Paragraphes qui reformulent ce qui vient d\'être dit.',
    re: /(?:en conclusion|en résumé|en somme|pour conclure|en définitive|dans l'ensemble|pour résumer|in conclusion|in summary|to sum up|overall,|ultimately,)/giu },
  { id: 'collab', name: 'Formules d\'assistant oubliées', wt: 5,
    why: 'Phrases de chatbot laissées dans le texte. Signe très fort.',
    re: /(?:j'espère que (?:cela|ceci|ça|cette|ce|ces|vous)|n'hésitez pas à|faites-moi savoir|dites-moi si|excellente question|bien sûr\s*!|certainement\s*!|je serais ravie? de|voici (?:un|une|le|la|les) (?:exemple|résumé|version|texte|liste|proposition)|en tant qu'(?:ia|intelligence artificielle|modèle de langage|assistant)|selon mes dernières (?:connaissances|informations|mises à jour)|à la date de ma dernière mise à jour|i hope this helps|let me know if|feel free to|great question|certainly!|as an ai|as of my last (?:knowledge )?update)/giu },
  { id: 'artefact', name: 'Artefacts techniques de chatbot', wt: 8,
    why: 'Restes de code de citation ou de liens générés (oaicite, utm_source=chatgpt.com…).',
    re: /(?:oaicite|contentReference|turn\d+search\d+|utm_source=(?:chatgpt|openai|copilot|perplexity)[\w.]*)/giu },
  { id: 'triplet', name: 'Énumérations en triplets', wt: 0.8,
    why: 'Listes de trois éléments (« l\'autonomie, la productivité et le bien-être »).',
    re: /(?<!\p{L})(?:l'|la |le |les |des |du |de |d')?\p{L}{3,}(?:[ -]\p{L}{2,})?, (?:l'|la |le |les |des |du |de |d')?\p{L}{3,}(?:[ -]\p{L}{2,})?,? (?:et |and |ou |or )(?:l'|la |le |les |des |du |de |d')?\p{L}{3,}/gu }
];

export const FMT = [
  { id: 'dash', name: 'Tirets cadratins (—)', re: /—/g, wt: 1.2, why: 'Les chatbots abusent du tiret long, rare en français courant.' },
  { id: 'bold', name: 'Gras Markdown (**…**)', re: /\*\*[^*\n]{1,80}\*\*/g, wt: 1.5, why: 'Mise en gras excessive héritée du Markdown.' },
  { id: 'head', name: 'Titres Markdown (#)', re: /^#{1,4} .+$/gm, wt: 1.5, why: 'Titres au format Markdown collés tels quels.' },
  { id: 'list', name: 'Listes à puces', re: /^\s*(?:[-•*]|\d+[.)])\s+\S/gm, wt: 0.4, why: 'Structure en listes systématique.', noMark: true },
  { id: 'emoji', name: 'Émojis décoratifs', re: /\p{Extended_Pictographic}/gu, wt: 1, why: 'Émojis en début de titres ou de puces.' }
];

export const words = t => t.match(/[\p{L}\p{N}'’-]+/gu) || [];
export const clamp = x => Math.max(0, Math.min(1, x));
export const cv = a => {
  if (a.length < 2) return 0;
  const m = a.reduce((x, y) => x + y, 0) / a.length;
  const v = a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length;
  return m ? Math.sqrt(v) / m : 0;
};

// Découpe en phrases avec leurs positions dans le texte.
export function splitSentences(text) {
  const out = [];
  const re = /[^.!?…\n]+(?:[.!?…]+["»”')\]]*|\n|$)/gu;
  let m;
  while ((m = re.exec(text))) {
    if (!m[0]) { re.lastIndex++; continue; }
    const raw = m[0];
    const lead = raw.length - raw.trimStart().length;
    const s = raw.trim();
    if (s) out.push({ text: s, start: m.index + lead, end: m.index + lead + s.length });
  }
  return out;
}

export function analyseHeuristics(t) {
  const W = words(t), n = W.length;
  const sents = splitSentences(t).filter(s => words(s.text).length >= 3);
  const sl = sents.map(s => words(s.text).length);
  const paras = t.split(/\n\s*\n/).map(p => words(p).length).filter(x => x >= 15);
  const sCv = cv(sl), pCv = paras.length >= 3 ? cv(paras) : null;
  const burst = clamp((0.62 - sCv) / 0.36);
  const freq = {};
  sents.forEach(s => { const f = (words(s.text)[0] || '').toLowerCase(); freq[f] = (freq[f] || 0) + 1; });
  const topOpen = sents.length ? Math.max(...Object.values(freq)) / sents.length : 0;
  const uni = clamp(pCv === null ? topOpen * 1.5 : ((0.55 - pCv) / 0.4) * 0.6 + topOpen * 1.5 * 0.4);
  const marks = [], found = [];
  let signW = 0, fmtW = 0;
  for (const c of CATS) {
    const ex = [];
    for (const m of t.matchAll(c.re)) {
      const s = m[0].replace(/^[.!?\s]+/, '');
      const start = m.index + (m[0].length - s.length);
      ex.push(s); marks.push([start, start + s.length, c]);
    }
    if (ex.length) { found.push({ c, ex }); signW += ex.length * c.wt; }
  }
  for (const f of FMT) {
    const ex = [...t.matchAll(f.re)];
    if (ex.length) {
      found.push({ c: f, ex: ex.map(m => m[0].trim().slice(0, 40)) });
      fmtW += ex.length * f.wt;
      if (!f.noMark) ex.forEach(m => marks.push([m.index, m.index + m[0].length, f]));
    }
  }
  const per100 = n ? signW / n * 100 : 0;
  const signs = clamp(1 - Math.exp(-per100 / 2.2));
  const fmt = clamp(1 - Math.exp(-(n ? fmtW / n * 100 : 0) / 1.5));
  const strong = found.some(f => f.c.id === 'collab' || f.c.id === 'artefact');
  const score = 0.5 * signs + 0.3 * burst + 0.12 * uni + 0.08 * fmt;
  return { n, sents: sents.length, sCv, pCv, burst, uni, signs, fmt, per100, found, marks, strong, score: Math.round(score * 100) };
}
