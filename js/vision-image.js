// Analyse visuelle d'une image par un grand modèle (Claude, GPT, Mistral) : il cherche les détails
// typiques des images générées (feuillages, textures, lumière, textes, mains…) et les situe dans l'image.
// Indice seulement : ces modèles se trompent aussi, surtout sur les générateurs récents.
import { appelIA, json, imageVersBase64 } from './ia.js';

const CONSIGNE = `Tu es expert en criminalistique d'images. Tu examines une image pour savoir si elle a été générée ou retouchée par une IA (Midjourney, DALL-E, GPT Image, Imagen, Gemini, Flux, Stable Diffusion…) ou si c'est une vraie photographie ou une œuvre faite à la main.

Examine en détail, zone par zone :
- végétation : feuilles et branches qui fusionnent, se dédoublent ou se dissolvent, motifs trop répétitifs, herbe en « tapis » uniforme ;
- textures : surfaces cireuses ou trop lisses, détails qui deviennent bouillie en arrière-plan, bruit absent ou trop régulier ;
- lumière : direction des ombres incohérente avec la source, reflets impossibles, halo trop parfait autour du soleil, ciel en dégradé de carte postale ;
- géométrie : horizons courbés, perspectives impossibles, objets qui se fondent les uns dans les autres, symétries suspectes ;
- éléments humains et artificiels : mains, dents, yeux, bijoux, textes et panneaux illisibles ou inventés, bâtiments et clôtures incohérents ;
- composition : rendu « trop beau », saturation et contraste typiques des générateurs, absence d'imperfections d'objectif.
Note aussi les indices de vraie photo : bruit de capteur, flou de bougé, aberrations chromatiques, défauts d'objectif cohérents, détails fins cohérents jusque dans le fond.

Sois calibré : beaucoup de vraies photos sont belles et retouchées. Si tu ne vois rien de net, dis-le. Ne décris que ce que tu vois vraiment.

Réponds uniquement avec un objet JSON :
{"probabilite_ia": <0 à 100>, "confiance": "faible" | "moyenne" | "élevée", "verdict": "<une ou deux phrases>",
 "indices_ia": [{"zone": {"x": <0-1>, "y": <0-1>, "l": <0-1>, "h": <0-1>}, "titre": "<court>", "detail": "<ce qu'on voit>"}],
 "indices_reels": [{"titre": "<court>", "detail": "<ce qu'on voit>"}]}
x et y sont le coin haut gauche de la zone, l et h sa largeur et sa hauteur, en fraction de l'image. Au plus 6 indices de chaque type.`;

const borne = v => Math.max(0, Math.min(1, Number(v) || 0));

export async function analyseVisuelle(cfg, file, signal) {
  const img = await imageVersBase64(file, 1600);
  const r = json(await appelIA(cfg, { system: CONSIGNE, user: 'Analyse cette image.', images: [img], signal, maxTokens: 6000 }));
  const p = Math.max(0, Math.min(100, Math.round(Number(r.probabilite_ia) || 0)));
  return {
    p: p / 100,
    confiance: ['faible', 'moyenne', 'élevée'].includes(r.confiance) ? r.confiance : 'moyenne',
    verdict: String(r.verdict || ''),
    indicesIA: (Array.isArray(r.indices_ia) ? r.indices_ia : []).slice(0, 6).map(i => ({
      titre: String(i.titre || ''), detail: String(i.detail || ''),
      zone: i.zone ? { x: borne(i.zone.x), y: borne(i.zone.y), l: borne(i.zone.l), h: borne(i.zone.h) } : null
    })),
    indicesReels: (Array.isArray(r.indices_reels) ? r.indices_reels : []).slice(0, 6).map(i => ({ titre: String(i.titre || ''), detail: String(i.detail || '') }))
  };
}
