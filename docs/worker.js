// Runs the VSA models off the main thread.
import { Speller, ContextModel, unpackModel } from './vsa-core.js';
import { SentenceDecoder } from './decoder.js';
import { DECODER_WEIGHTS, DOMAIN_WEIGHTS } from './decoder-weights.js';

const SPELL = { D: 1024, wPhon: 0.5, wFreq: 0.8, wLen: 0, wEd: 0.35 }; // tuned on a held-out dev split
let speller = null, ctx = null, entries = null;
let grammar = null, profiles = null, preset = null, decoder = null, interpLoading = null;
const post = (type, data) => self.postMessage({ type, ...data });

// Models are cached in Cache Storage so a second visit loads nothing from the network.
// Bump MODEL_VERSION whenever a model file changes; old caches are deleted.
const MODEL_VERSION = 'vsa-models-2026-10-24b';
let fromCache = 0, fromNetwork = 0;
async function cachedFetch(url) {
  let cache = null;
  try {
    cache = await caches.open(MODEL_VERSION);
    const hit = await cache.match(url);
    if (hit) { fromCache++; return hit; }
  } catch (e) { cache = null; } // no Cache Storage (file://, some sandboxes): plain fetch
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + ' ' + res.status);
  fromNetwork++;
  if (cache) try { await cache.put(url, res.clone()); } catch (e) {}
  return res;
}
(async () => { try { for (const k of await caches.keys()) if (k.startsWith('vsa-models-') && k !== MODEL_VERSION) await caches.delete(k); } catch (e) {} })();

async function init() {
  const t0 = performance.now();
  post('progress', { stage: 'Downloading models' });
  const [wordsTxt, meta, vecBuf] = await Promise.all([
    cachedFetch('words.txt').then(r => r.text()),
    cachedFetch('ctx_meta.json').then(r => r.json()),
    cachedFetch('ctx_vectors.bin').then(r => r.arrayBuffer()),
  ]);
  const tDl = performance.now();
  post('progress', { stage: 'Encoding 80,000 words as hypervectors' });
  entries = wordsTxt.trim().split('\n').map(l => { const [w, c] = l.split(' '); return [w, +c]; });
  speller = new Speller({ D: SPELL.D });
  speller.build(entries);
  const ctxVec = unpackModel(vecBuf, meta);
  ctx = new ContextModel(meta, ctxVec);
  const t1 = performance.now();
  post('ready', {
    downloadMs: tDl - t0, buildMs: t1 - tDl, vocab: entries.length, cached: fromCache > 0 && fromNetwork === 0,
    memMB: (speller.bytes() + ctxVec.byteLength) / 1e6, downloadMB: (wordsTxt.length + vecBuf.byteLength) / 1e6, sets: meta.sets.length, enabledSets: meta.enabled.length, ctxWords: meta.words.length, D: meta.D,
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

// ---------------------------------------------------------- sentence interpretation
// Loaded on first use: grammar (0.9 MB), word-context profiles (15.4 MB), command preset.
async function interpInit() {
  if (!interpLoading) interpLoading = (async () => {
    const t0 = performance.now();
    const [g, pm, pb, pr] = await Promise.all([
      cachedFetch('grammar.json').then(r => r.json()), cachedFetch('profiles_meta.json').then(r => r.json()),
      cachedFetch('profiles.bin').then(r => r.arrayBuffer()),
      cachedFetch('presets/agent-commands.json').then(r => r.json()),
    ]);
    grammar = g; preset = pr;
    profiles = { meta: pm, int8: unpackModel(pb, pm), words: entries.slice(0, pm.N).map(e => e[0]) };
    return { loadMs: performance.now() - t0, profilesMB: pb.byteLength / 1e6 };
  })();
  return interpLoading;
}
const general = () => new Map(entries);
function parseDictionary(text) {
  const g = general(), out = [];
  for (const line of String(text).split('\n')) {
    const m = line.trim().match(/^([A-Za-z]+)\s+(\d+)$/);
    if (m) { out.push([m[1].toLowerCase(), +m[2]]); continue; }
    for (const w of line.split(/[^A-Za-z]+/)) if (w) out.push([w.toLowerCase(), g.get(w.toLowerCase()) || 1000]);
  }
  return out;
}
async function interpConfigure({ source, dictionary, examples }) {
  const info = await interpInit(); const t0 = performance.now();
  let dict, ex = [], weights = DECODER_WEIGHTS;
  if (source === 'commands') { dict = preset.dictionary; ex = preset.examples; weights = { ...DECODER_WEIGHTS, ...DOMAIN_WEIGHTS }; }
  else if (source === 'custom') {
    dict = parseDictionary(dictionary || ''); ex = String(examples || '').split('\n').map(s => s.trim()).filter(Boolean);
    if (dict.length < 2) throw new Error('Add at least two words to the dictionary.');
    weights = ex.length ? { ...DECODER_WEIGHTS, ...DOMAIN_WEIGHTS } : DECODER_WEIGHTS;
  } else dict = entries;
  decoder = new SentenceDecoder({ dictionary: dict, examples: ex, grammar, profiles, confusion: ctx, weights, speller: source === 'general' ? speller : null });
  post('interp-ready', { source, words: decoder.dict.size, examples: ex.length, buildMs: performance.now() - t0, ...info,
    sampleWords: source === 'commands' ? preset.dictionary.slice(0, 400).map(d => d[0]) : null, sampleExamples: source === 'commands' ? preset.examples.slice(0, 12) : null });
}
function interpret(text, id) {
  const t0 = performance.now(); const r = decoder.decode(text, { nbest: 6 });
  post('interp-result', { id, ...r, ms: performance.now() - t0 });
}

self.onmessage = e => {
  const { type } = e.data;
  if (type === 'interp-config') interpConfigure(e.data).catch(err => post('interp-error', { message: String(err && err.message || err) }));
  if (type === 'interp' && decoder) interpret(e.data.text, e.data.id);
  if (type === 'init') init().catch(err => post('error', { message: String(err && err.message || err) }));
  if (type === 'check' && speller) check(e.data.text, e.data.ignore, e.data.scale);
};
