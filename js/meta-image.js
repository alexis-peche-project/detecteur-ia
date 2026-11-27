// Lecture des marques d'origine laissées dans un fichier image, sans rien envoyer :
// Content Credentials (C2PA), type de source IPTC, paramètres des générateurs (PNG),
// logiciel et données d'appareil photo (EXIF).

const GENERATEURS = [
  ['OpenAI', /openai|dall[·-]?e|chatgpt|gpt-4o|gpt-image|sora/i],
  ['Adobe Firefly', /firefly/i],
  ['Google (Imagen, Gemini)', /imagen|gemini|nano banana|synthid/i],
  ['Microsoft Designer / Bing', /microsoft designer|bing image creator|image creator from designer/i],
  ['Midjourney', /midjourney/i],
  ['Stable Diffusion', /stable[ -]?diffusion|stability ?ai|sdxl|automatic1111|comfyui|invokeai|fooocus/i],
  ['Flux', /\bflux(\.1)?[ -]?(dev|pro|schnell)\b|black forest labs/i],
  ['Meta AI', /meta ai|imagine with meta/i],
  ['Ideogram', /ideogram/i],
  ['Leonardo', /leonardo\.ai/i],
  ['Runway', /runwayml|runway gen/i],
  ['NovelAI', /novelai/i],
  ['Canva (IA)', /canva magic media|dream lab/i]
];

const latin1 = (u8, a = 0, b = u8.length) => {
  let s = '';
  for (let i = a; i < b; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, Math.min(b, i + 0x8000)));
  return s;
};
const nomsGenerateurs = s => GENERATEURS.filter(([, r]) => r.test(s)).map(([n]) => n);

/* ---------- EXIF (TIFF) ---------- */
function lireTiff(u8, debut) {
  const dv = new DataView(u8.buffer, u8.byteOffset + debut);
  const le = dv.getUint16(0) === 0x4949;
  const u16 = o => dv.getUint16(o, le), u32 = o => dv.getUint32(o, le);
  const tags = {};
  const TYPES = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8 };
  const ifd = (off, garder) => {
    if (!off || off + 2 > dv.byteLength) return;
    const n = u16(off);
    for (let i = 0; i < n && off + 2 + 12 * i + 12 <= dv.byteLength; i++) {
      const e = off + 2 + 12 * i, tag = u16(e), type = u16(e + 2), cnt = u32(e + 4);
      const taille = (TYPES[type] || 1) * cnt, pos = taille > 4 ? u32(e + 8) : e + 8;
      if (pos + taille > dv.byteLength) continue;
      let v;
      if (type === 2) v = latin1(new Uint8Array(dv.buffer, dv.byteOffset + pos, taille)).replace(/\0+$/, '').trim();
      else if (type === 3) v = u16(pos);
      else if (type === 4) v = u32(pos);
      else if (type === 5) v = u32(pos) / (u32(pos + 4) || 1);
      if (garder.includes(tag)) tags[tag] = v;
      if (tag === 0x8769) ifd(v, garder);
    }
  };
  try { ifd(u32(4), [0x010F, 0x0110, 0x0131, 0x013B, 0x8769, 0x9003, 0x829A, 0x829D, 0x8827, 0x920A, 0xA434]); } catch { /* EXIF abîmé */ }
  return {
    marque: tags[0x010F], modele: tags[0x0110], logiciel: tags[0x0131], auteur: tags[0x013B],
    date: tags[0x9003], exposition: tags[0x829A], ouverture: tags[0x829D], iso: tags[0x8827], focale: tags[0x920A], objectif: tags[0xA434]
  };
}

function trouverExif(u8, s) {
  // JPEG : segment APP1 « Exif\0\0 » ; PNG : bloc eXIf ; WebP : bloc EXIF.
  let i = s.indexOf('Exif\0\0');
  if (i >= 0) return lireTiff(u8, i + 6);
  i = s.indexOf('eXIf');
  if (i >= 0) return lireTiff(u8, i + 4);
  i = s.indexOf('EXIF');
  if (i >= 0 && s.startsWith('RIFF')) return lireTiff(u8, i + 8);
  return null;
}

/* ---------- PNG : blocs de texte (paramètres de génération) ---------- */
function textesPng(u8) {
  const out = {};
  if (u8[0] !== 0x89 || u8[1] !== 0x50) return out;
  const dv = new DataView(u8.buffer, u8.byteOffset);
  let o = 8;
  while (o + 12 <= u8.length) {
    const len = dv.getUint32(o), type = latin1(u8, o + 4, o + 8);
    if (type === 'tEXt' || type === 'iTXt') {
      const data = u8.subarray(o + 8, o + 8 + len);
      const z = data.indexOf(0);
      const cle = latin1(data, 0, z);
      let val = '';
      if (type === 'tEXt') val = latin1(data, z + 1);
      else if (data[z + 1] === 0) { // iTXt non compressé : drapeau, méthode, langue\0, mot-clé traduit\0, texte
        const l = data.indexOf(0, z + 3), t = l >= 0 ? data.indexOf(0, l + 1) : -1;
        if (t >= 0) val = new TextDecoder().decode(data.subarray(t + 1));
      }
      out[cle] = val.slice(0, 4000);
    }
    if (type === 'IEND') break;
    o += 12 + len;
  }
  return out;
}

/* ---------- Analyse complète ---------- */
export async function lireMarques(file) {
  const u8 = new Uint8Array(await file.arrayBuffer());
  const s = latin1(u8);
  const constats = [];   // { force: 'ia' | 'photo' | 'info', titre, detail }
  let generateurs = new Set();

  // 1. Content Credentials (C2PA) et type de source numérique IPTC.
  const iC2pa = s.search(/c2pa|jumb/);
  const fenetre = iC2pa >= 0 ? s.slice(Math.max(0, iC2pa - 2000), iC2pa + 200000) : '';
  const iptcIA = /(?:composite(?:with)?)?trainedalgorithmicmedia|algorithmicmedia/i.exec(s);
  const captation = /digitalcapture|computationalcapture/i.test(fenetre);
  if (iptcIA) {
    const compose = /composite/i.test(iptcIA[0]);
    constats.push({ force: 'ia', titre: compose ? 'Déclarée en partie générée par IA' : 'Déclarée générée par IA',
      detail: `Le fichier contient la mention officielle IPTC « ${iptcIA[0]} »${iC2pa >= 0 ? ', dans ses Content Credentials (C2PA)' : ''}. Les générateurs d'OpenAI, Adobe, Google ou Microsoft l'ajoutent à leurs images.` });
  }
  if (iC2pa >= 0) {
    nomsGenerateurs(fenetre).forEach(g => generateurs.add(g));
    if (!iptcIA) constats.push({ force: captation ? 'photo' : 'info', titre: captation ? 'Signée comme prise par un appareil' : 'Content Credentials présents',
      detail: captation ? 'Les Content Credentials (C2PA) décrivent une prise de vue par un appareil photo ou un téléphone. La signature n\'est pas vérifiée ici : utilisez le lien de vérification.' : 'Le fichier porte des Content Credentials (C2PA) sans mention d\'IA lisible ici. Ouvrez-les avec le lien de vérification pour voir leur contenu.' });
  }

  // 2. XMP : outil de création.
  const outil = /CreatorTool(?:="|>)([^"<]{1,200})/.exec(s)?.[1];
  if (outil) nomsGenerateurs(outil).forEach(g => generateurs.add(g));

  // 3. PNG : paramètres de génération laissés par Stable Diffusion, ComfyUI, NovelAI, InvokeAI…
  const png = textesPng(u8);
  const clesGen = Object.keys(png).filter(k => /^(parameters|prompt|workflow|dream|invokeai_metadata|sd-metadata|comment|software|description)$/i.test(k));
  const params = clesGen.map(k => `${k} : ${png[k]}`).join('\n');
  if (/steps:\s*\d+|sampler|cfg scale|seed:\s*\d+|"class_type"|negative prompt|"ksampler"|novelai|invokeai/i.test(params)) {
    nomsGenerateurs(params).forEach(g => generateurs.add(g));
    if (!generateurs.size) generateurs.add('Stable Diffusion ou dérivé');
    constats.push({ force: 'ia', titre: 'Paramètres de génération trouvés', detail: `L'image contient les réglages de l'outil qui l'a produite (consigne, nombre d'étapes, graine…) : « ${params.slice(0, 220).replace(/\s+/g, ' ')}… »` });
  }

  // 4. EXIF : logiciel et appareil.
  const ex = trouverExif(u8, s);
  if (ex?.logiciel) {
    const g = nomsGenerateurs(ex.logiciel);
    g.forEach(x => generateurs.add(x));
    if (g.length) constats.push({ force: 'ia', titre: 'Logiciel de génération indiqué', detail: `Champ « Logiciel » : ${ex.logiciel}.` });
  }
  if (ex && (ex.marque || ex.modele) && (ex.exposition || ex.iso || ex.focale)) {
    const reglages = [ex.exposition && (ex.exposition < 1 ? `1/${Math.round(1 / ex.exposition)} s` : `${ex.exposition} s`), ex.ouverture && `f/${Math.round(ex.ouverture * 10) / 10}`, ex.iso && `ISO ${ex.iso}`, ex.focale && `${Math.round(ex.focale)} mm`].filter(Boolean).join(', ');
    constats.push({ force: 'photo', titre: 'Données d\'appareil photo', detail: `${[ex.marque, ex.modele].filter(Boolean).join(' ')}${reglages ? ` · ${reglages}` : ''}${ex.date ? ` · prise le ${ex.date}` : ''}. Ces données plaident pour une vraie photo, mais elles peuvent être copiées ou inventées.` });
  }
  if (generateurs.size && !constats.some(c => c.force === 'ia')) {
    constats.push({ force: 'ia', titre: 'Générateur d\'images cité dans le fichier', detail: `Mention trouvée dans les métadonnées : ${[...generateurs].join(', ')}.` });
  }
  if (!constats.length) constats.push({ force: 'info', titre: 'Aucune marque d\'origine', detail: 'Pas de Content Credentials, ni de paramètres de génération, ni de données d\'appareil. C\'est courant : les réseaux sociaux et les captures d\'écran les effacent. Cela ne dit rien de l\'origine de l\'image.' });

  return { constats, generateurs: [...generateurs], c2pa: iC2pa >= 0, exif: ex };
}
