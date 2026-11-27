// Copies manuscrites : une IA transcrit la photo, et relève les ratures et corrections.
// Une longue rédaction sans aucune rature peut indiquer un texte recopié.
import { appelIA, json, imageVersBase64 } from './ia.js';

const CONSIGNE = `Tu transcris des copies manuscrites d'élèves.
Recopie exactement le texte écrit, ligne à ligne en paragraphes, sans corriger l'orthographe ni la grammaire.
Ne transcris pas les mots barrés.
Compte les ratures (mots barrés ou raturés) et les corrections (ajouts au-dessus de la ligne, surcharges, flèches de déplacement).
Réponds uniquement avec un objet JSON : {"texte": "<transcription>", "ratures": <entier>, "corrections": <entier>, "lisibilite": "bonne" | "moyenne" | "faible", "remarques": "<une phrase sur l'aspect de la copie : régularité, hésitations, mise en page>"}`;

export const estImage = f => f.type.startsWith('image/') || /\.(jpe?g|png|webp|heic|heif)$/i.test(f.name);

export async function transcrire(cfg, file, signal) {
  const img = await imageVersBase64(file);
  const r = json(await appelIA(cfg, { system: CONSIGNE, user: 'Transcris cette copie.', images: [img], signal }));
  const texte = String(r.texte || '').trim();
  const mots = (texte.match(/\S+/g) || []).length;
  const ratures = Number(r.ratures) || 0, corrections = Number(r.corrections) || 0;
  const alerte = mots >= 150 && ratures + corrections === 0;
  return {
    texte, mots, ratures, corrections, lisibilite: r.lisibilite || 'moyenne', remarques: String(r.remarques || ''),
    resume: `Copie manuscrite transcrite (${mots} mots, ${ratures} rature(s), ${corrections} correction(s), lisibilité ${r.lisibilite || 'moyenne'}).` +
      (alerte ? ' Aucune rature ni correction sur une longue rédaction : cela arrive quand un texte est recopié, mais aussi avec un brouillon soigneusement mis au propre.' : '') +
      (r.remarques ? ' ' + r.remarques : '')
  };
}
