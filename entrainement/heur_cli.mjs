// Applique exactement les mêmes règles que le site (js/heuristics.js) à des textes lus sur l'entrée standard.
// Entrée : une ligne JSON {"text": "..."} par texte. Sortie : une ligne JSON {"score", "strong"} par texte.
import { analyseHeuristics } from '../js/heuristics.js';
import readline from 'node:readline';

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
  if (!line.trim()) continue;
  const { text } = JSON.parse(line);
  const h = analyseHeuristics(text);
  process.stdout.write(JSON.stringify({ score: h.score / 100, strong: h.strong }) + '\n');
}
