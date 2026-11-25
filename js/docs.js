// Extraction du texte des documents importés, entièrement dans le navigateur.
const PDFJS = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@6.5.438/build/pdf.min.mjs';
const PDFJS_WORKER = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@6.5.438/build/pdf.worker.min.mjs';
const MAMMOTH = 'https://cdn.jsdelivr.net/npm/mammoth@1.13.0/mammoth.browser.min.js';
const JSZIP = 'https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js';

export const ACCEPT = '.txt,.md,.markdown,.pdf,.docx,.odt,.rtf,.html,.htm,.csv,.json,.tex';

const loaded = {};
function loadScript(src) {
  return loaded[src] ||= new Promise((ok, ko) => {
    const s = document.createElement('script');
    s.src = src; s.onload = ok; s.onerror = () => ko(new Error('Impossible de charger ' + src));
    document.head.appendChild(s);
  });
}

async function decodeText(buf) {
  const utf8 = new TextDecoder('utf-8').decode(buf);
  if ((utf8.match(/�/g) || []).length > 3) return new TextDecoder('windows-1252').decode(buf);
  return utf8.replace(/^﻿/, '');
}

async function readPdf(buf, onProgress) {
  const pdfjs = await import(PDFJS);
  pdfjs.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
  const doc = await pdfjs.getDocument({ data: new Uint8Array(buf) }).promise;
  const pages = [];
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i);
    const tc = await page.getTextContent();
    let line = '', out = '', lastY = null;
    for (const it of tc.items) {
      const y = it.transform?.[5];
      if (lastY !== null && Math.abs(y - lastY) > 2) { out += line.trimEnd() + '\n'; line = ''; }
      line += it.str + (it.hasEOL ? '\n' : '');
      lastY = y;
    }
    out += line;
    pages.push(out);
    onProgress?.(i, doc.numPages);
  }
  // Recolle les lignes coupées en fin de ligne (mise en page PDF), garde les paragraphes.
  const text = pages.join('\n\n')
    .replace(/-\n(?=\p{Ll})/gu, '')
    .replace(/([^\n.!?:;»”])\n(?=[\p{Ll}\d(«“])/gu, '$1 ')
    .replace(/\n{3,}/g, '\n\n');
  if (text.replace(/\s/g, '').length < 20) throw new Error('Ce PDF ne contient pas de texte sélectionnable (document scanné). Utilisez un PDF texte ou copiez le texte.');
  return text;
}

async function readDocx(buf) {
  await loadScript(MAMMOTH);
  const { value } = await window.mammoth.extractRawText({ arrayBuffer: buf });
  return value.replace(/\n{3,}/g, '\n\n');
}

async function readOdt(buf) {
  await loadScript(JSZIP);
  const zip = await window.JSZip.loadAsync(buf);
  const xml = await zip.file('content.xml').async('string');
  const doc = new DOMParser().parseFromString(xml, 'application/xml');
  const paras = [...doc.getElementsByTagNameNS('*', 'p'), ...doc.getElementsByTagNameNS('*', 'h')];
  const ordered = paras.sort((a, b) => a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1);
  return ordered.map(p => p.textContent.trim()).filter(Boolean).join('\n\n');
}

function readRtf(s) {
  return s
    .replace(/\\par[d]?/g, '\n')
    .replace(/\\'([0-9a-f]{2})/gi, (_, h) => new TextDecoder('windows-1252').decode(new Uint8Array([parseInt(h, 16)])))
    .replace(/\\u(-?\d+)\??/g, (_, n) => String.fromCharCode(n < 0 ? 65536 + +n : +n))
    .replace(/\{\\\*[^{}]*\}/g, '')
    .replace(/\\[a-z]+-?\d* ?/gi, '')
    .replace(/[{}]/g, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function readHtml(s) {
  const doc = new DOMParser().parseFromString(s, 'text/html');
  doc.querySelectorAll('script,style,noscript,nav,header,footer').forEach(n => n.remove());
  doc.querySelectorAll('p,div,li,h1,h2,h3,h4,h5,h6,br,tr').forEach(n => n.append('\n'));
  return (doc.body?.textContent || '').replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
}

export async function extractText(file, onProgress) {
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  const buf = await file.arrayBuffer();
  if (ext === 'pdf' || file.type === 'application/pdf') return readPdf(buf, onProgress);
  if (ext === 'docx') return readDocx(buf);
  if (ext === 'odt') return readOdt(buf);
  if (ext === 'doc') throw new Error('Le format .doc (Word 97-2003) n\'est pas lisible dans le navigateur. Enregistrez-le en .docx ou en PDF.');
  const text = await decodeText(buf);
  if (ext === 'rtf') return readRtf(text);
  if (ext === 'html' || ext === 'htm') return readHtml(text);
  return text;
}
