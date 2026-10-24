// Runs the VSA models off the main thread.
import { Speller, ContextModel } from './vsa-core.js';

const SPELL = { D: 1024, wPhon: 0.5, wFreq: 0.8, wLen: 0, wEd: 0.35 }; // tuned on a held-out dev split
let speller = null, ctx = null;
const post = (type, data) => self.postMessage({ type, ...data });

async function init() {
  const t0 = performance.now();
  post('progress', { stage: 'Downloading models' });
  const [wordsTxt, meta, vecBuf] = await Promise.all([
    fetch('words.txt').then(r => { if (!r.ok) throw new Error('words.txt ' + r.status); return r.text(); }),
    fetch('ctx_meta.json').then(r => { if (!r.ok) throw new Error('ctx_meta.json ' + r.status); return r.json(); }),
    fetch('ctx_vectors.bin').then(r => { if (!r.ok) throw new Error('ctx_vectors.bin ' + r.status); return r.arrayBuffer(); }),
  ]);
  const tDl = performance.now();
  post('progress', { stage: 'Encoding 80,000 words as hypervectors' });
  const entries = wordsTxt.trim().split('\n').map(l => { const [w, c] = l.split(' '); return [w, +c]; });
  speller = new Speller({ D: SPELL.D });
  speller.build(entries);
  ctx = new ContextModel(meta, new Int8Array(vecBuf));
  const t1 = performance.now();
  post('ready', {
    downloadMs: tDl - t0, buildMs: t1 - tDl, vocab: entries.length,
    spellMB: speller.bytes() / 1e6, ctxMB: vecBuf.byteLength / 1e6, sets: meta.sets.length, enabledSets: meta.enabled.length, ctxWords: meta.words.length, D: meta.D,
  });
}

const WORD = /[A-Za-z]+(?:['’][A-Za-z]+)*/g;
function check(text, ignore, scale) {
  const t0 = performance.now(); const issues = []; let nWords = 0, nUnknown = 0;
  const ign = new Set(ignore || []);
  const tokens = []; let m;
  WORD.lastIndex = 0;
  while ((m = WORD.exec(text))) tokens.push({ s: m[0], start: m.index, end: m.index + m[0].length });
  // sentence-initial detection: previous non-space char is . ! ? or start
  const sentenceStart = i => { let j = i - 1; while (j >= 0 && /[\s"“'(]/.test(text[j])) j--; return j < 0 || /[.!?\n]/.test(text[j]); };
  const bad = new Set();
  for (const tk of tokens) {
    nWords++;
    const w = tk.s, lw = w.toLowerCase();
    if (/['’]/.test(w) || ign.has(lw)) continue;
    if (speller.has(lw)) continue;
    if (w.length > 1 && w === w.toUpperCase()) continue;                 // acronyms
    if (/^[A-Z]/.test(w) && !sentenceStart(tk.start)) continue;           // probable proper noun
    nUnknown++; bad.add(tk.start);
    let sug = speller.suggest(lw, SPELL, 5);
    if (/^[A-Z]/.test(w)) sug = sug.map(s => s[0].toUpperCase() + s.slice(1));
    issues.push({ kind: 'spelling', start: tk.start, end: tk.end, word: w, suggestions: sug });
  }
  // context: per sentence, same tokenisation as training (lowercase letter runs)
  const sentRe = /[^.!?\n]+[.!?]*/g; let sm;
  while ((sm = sentRe.exec(text))) {
    const off = sm.index, seg = sm[0]; const parts = []; const re = /[A-Za-z]+/g; let pm;
    while ((pm = re.exec(seg))) parts.push({ w: pm[0].toLowerCase(), raw: pm[0], start: off + pm.index, end: off + pm.index + pm[0].length });
    if (!parts.length) continue;
    const toks = ['<s>', ...parts.map(p => p.w), '</s>'];
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i]; if (bad.has(p.start) || ign.has(p.w)) continue;
      // skip pieces of contractions like don't / it's
      if (/['’]/.test(text[p.end] || '') || /['’]/.test(text[p.start - 1] || '')) continue;
      const r = ctx.check(toks, i + 1, scale || 1); if (!r) continue;
      let sug = r.suggestion; if (/^[A-Z]/.test(p.raw)) sug = sug[0].toUpperCase() + sug.slice(1);
      issues.push({ kind: 'context', start: p.start, end: p.end, word: p.raw, suggestions: [sug], margin: r.margin });
    }
  }
  issues.sort((a, b) => a.start - b.start);
  post('result', { issues, ms: performance.now() - t0, nWords, nUnknown });
}

self.onmessage = e => {
  const { type } = e.data;
  if (type === 'init') init().catch(err => post('error', { message: String(err && err.message || err) }));
  if (type === 'check' && speller) check(e.data.text, e.data.ignore, e.data.scale);
};
