// Sentence-level evaluation of the decoder on held-out big.txt sentences with injected errors
// (non-word typos, real-word confusions, grammar: agreement / verb form / a-an), against
// simpler systems. Weights can be tuned on the dev split (never on test).
// Usage: node src/eval_decoder.mjs [--tune N] [--n 1000] [--harper]
import fs from 'fs';
import { SentenceDecoder, DEFAULT_WEIGHTS, morphVariants } from '../docs/decoder.js';
import { ContextModel, Speller, mulberry32, unpackModel } from '../docs/vsa-core.js';

const arg = (k, d) => { const i = process.argv.indexOf(k); return i < 0 ? d : (process.argv[i + 1] && !process.argv[i + 1].startsWith('--') ? process.argv[i + 1] : true); };
const TUNE = +arg('--tune', 0), NTEST = +arg('--n', 1000), HARPER = !!arg('--harper', false), QUICK = !!arg('--quick', false);
const rng = mulberry32(2024); const rf = () => rng() / 4294967296; const pick = a => a[Math.floor(rf() * a.length)];

// ---- models
const dict = fs.readFileSync('docs/words.txt', 'utf8').trim().split('\n').map(l => { const [w, c] = l.split(' '); return [w, +c]; });
const dictSet = new Set(dict.map(d => d[0]));
const grammar = JSON.parse(fs.readFileSync('docs/grammar.json'));
const pm = JSON.parse(fs.readFileSync('docs/profiles_meta.json')), pb = fs.readFileSync('docs/profiles.bin');
const cm = JSON.parse(fs.readFileSync('docs/ctx_meta.json')), cb = fs.readFileSync('docs/ctx_vectors.bin');
const profiles = { meta: pm, int8: unpackModel(pb, pm), words: fs.readFileSync('docs/words.txt', 'utf8').trim().split('\n').slice(0, pm.N).map(l => l.split(' ')[0]) };
const confusion = new ContextModel(cm, unpackModel(cb, cm));
const speller = new Speller({ D: 1024 }); speller.build(dict);
const make = (weights, opts = {}) => new SentenceDecoder({ dictionary: dict, grammar, profiles: opts.noProfiles ? null : profiles, confusion: opts.noConfusion ? null : confusion, weights, speller });

// ---- held-out sentences (same split as the other context experiments)
const r1 = mulberry32(1); const rf1 = () => r1() / 4294967296;
const raw = fs.readFileSync('data/big.txt', 'utf8').replace(/\s+/g, ' ');
let sents = raw.split(/(?<=[.!?])\s+(?=[A-Z"])/).filter(s => s.length > 20 && s.length < 400);
for (let i = sents.length - 1; i > 0; i--) { const j = Math.floor(rf1() * (i + 1)); [sents[i], sents[j]] = [sents[j], sents[i]]; }
const nTr = Math.floor(sents.length * 0.85), nDev = Math.floor(sents.length * 0.05);
const pool = s => s.filter(x => { const w = x.match(/[A-Za-z]+/g) || []; return w.length >= 6 && w.length <= 22 && !/['’\d_]/.test(x) && w.every(t => dictSet.has(t.toLowerCase())); });
const devS = pool(sents.slice(nTr, nTr + nDev)), testS = pool(sents.slice(nTr + nDev));

// ---- error injection
const missp = JSON.parse(fs.readFileSync('data/misspellings_by_word.json'));
const KEYS = 'qwertyuiop asdfghjkl zxcvbnm'; const near = c => { const rows = KEYS.split(' '); for (const r of rows) { const i = r.indexOf(c); if (i >= 0) return (r[i - 1] || '') + (r[i + 1] || ''); } return ''; };
function typo(w) {
  if (missp[w] && rf() < 0.6) return pick(missp[w]);
  for (let tries = 0; tries < 20; tries++) {
    const i = Math.floor(rf() * w.length), op = rf(); let x;
    if (op < 0.3) x = w.slice(0, i) + w.slice(i + 1);
    else if (op < 0.55 && i < w.length - 1) x = w.slice(0, i) + w[i + 1] + w[i] + w.slice(i + 2);
    else if (op < 0.8) { const n = near(w[i]) || 'e'; x = w.slice(0, i) + n[Math.floor(rf() * n.length)] + w.slice(i + 1); }
    else x = w.slice(0, i) + w[i] + w.slice(i);
    if (x && !dictSet.has(x)) return x;
  }
  return null;
}
const GRAM = [['is', 'are'], ['was', 'were'], ['has', 'have'], ['does', 'do'], ['a', 'an'], ['this', 'these'], ['that', 'those']];
const gmap = new Map(); GRAM.forEach(([a, b]) => { gmap.set(a, b); gmap.set(b, a); });
function grammarError(w) {
  if (gmap.has(w)) return gmap.get(w);
  const lex = grammar.lex[w]; if (!lex) return null; const tags = lex.map(([t]) => grammar.tags[t]);
  if (tags[0] === 'vbz' && w.endsWith('s') && dictSet.has(w.slice(0, -1))) return w.slice(0, -1);
  if (tags[0] === 'vb' && dictSet.has(w + 's')) return w + 's';
  if (tags[0] === 'nns' && w.endsWith('s') && dictSet.has(w.slice(0, -1))) return w.slice(0, -1);
  return null;
}
function corrupt(s) {
  const words = s.match(/[A-Za-z]+/g); const nErr = rf() < 0.2 ? 0 : rf() < 0.6 ? 1 : 2;
  const errs = []; const out = s; let text = s; const used = new Set();
  for (let e = 0; e < nErr; e++) {
    const type = rf() < 0.45 ? 'typo' : rf() < 0.55 ? 'realword' : 'grammar';
    for (let tries = 0; tries < 30; tries++) {
      const i = Math.floor(rf() * words.length); if (used.has(i)) continue; const w = words[i].toLowerCase(); let x = null;
      if (type === 'typo' && w.length >= 4) x = typo(w);
      else if (type === 'realword') { const k = confusion.setOf.get(w); if (k !== undefined) x = pick(confusion.sets[k].filter(c => c !== w)); }
      else if (type === 'grammar') x = grammarError(w);
      if (!x) continue;
      used.add(i); errs.push({ i, type, from: w, to: x }); break;
    }
  }
  // rebuild text replacing word i
  let k = -1; text = s.replace(/[A-Za-z]+/g, m => { k++; const e = errs.find(z => z.i === k); if (!e) return m; return /^[A-Z]/.test(m) ? e.to[0].toUpperCase() + e.to.slice(1) : e.to; });
  return { gold: words.map(w => w.toLowerCase()), text, errs };
}
const mk = (S, n) => { const out = []; for (const s of S) { if (out.length >= n) break; out.push(corrupt(s)); } return out; };
const DEV = mk(devS, 250), TEST = mk(testS, NTEST);

// ---- scoring
function score(items, outWords) {
  let sent = 0, errN = 0, errFix = 0, words = 0, falseCh = 0; const byType = {};
  items.forEach((it, k) => {
    const o = outWords[k]; if (o.length === it.gold.length && o.every((w, i) => w === it.gold[i])) sent++;
    const errIdx = new Set(it.errs.map(e => e.i));
    for (const e of it.errs) { errN++; const ok = o[e.i] === it.gold[e.i]; if (ok) errFix++; byType[e.type] = byType[e.type] || [0, 0]; byType[e.type][0]++; if (ok) byType[e.type][1]++; }
    it.gold.forEach((g, i) => { if (errIdx.has(i)) return; words++; if (o[i] !== g) falseCh++; });
  });
  const bt = {}; for (const t in byType) bt[t] = +(byType[t][1] / byType[t][0]).toFixed(3);
  return { sentence: +(sent / items.length).toFixed(4), fixed: +(errFix / errN).toFixed(4), falseChangesPer100: +(100 * falseCh / words).toFixed(2), byType: bt, n: items.length, errors: errN };
}
const wordsOf = t => (t.match(/[A-Za-z]+/g) || []).map(w => w.toLowerCase());
const runDecoder = (dec, items) => items.map(it => dec.decode(it.text).best.words);

// ---- baselines
const identity = items => items.map(it => wordsOf(it.text));
const spellingOnly = items => items.map(it => wordsOf(it.text).map(w => dictSet.has(w) ? w : speller.suggest(w, { wPhon: 0.5, wFreq: 0.8, wLen: 0, wEd: 0.35 }, 1)[0]));
const proofreader = items => items.map(it => { const w = spellingOnly([it])[0]; const toks = ['<s>', ...w, '</s>']; return w.map((x, i) => { const r = confusion.check(toks, i + 1, 1); return r ? r.suggestion : x; }); });

// ---- tuning (random search around the defaults, on dev)
let W = { ...DEFAULT_WEIGHTS };
if (fs.existsSync('results/decoder_weights.json')) W = { ...W, ...JSON.parse(fs.readFileSync('results/decoder_weights.json')) };
if (TUNE) {
  const obj = r => r.fixed - 0.05 * r.falseChangesPer100 + 0.5 * r.sentence;
  let best = { W, r: score(DEV, runDecoder(make(W), DEV)) }; best.o = obj(best.r);
  console.log('start', JSON.stringify(best.r));
  const keys = ['freq', 'grammar', 'profile', 'confusion', 'sim', 'edit', 'keepUnknown', 'nearWord', 'confusionSwap', 'morph', 'article'];
  for (let it = 0; it < TUNE; it++) {
    const cand = { ...best.W }; const nk = 1 + Math.floor(rf() * 3);
    for (let q = 0; q < nk; q++) { const k = pick(keys); cand[k] = +(cand[k] * (rf() < 0.5 ? 0.6 + rf() * 0.3 : 1.2 + rf() * 0.6)).toFixed(3); }
    const r = score(DEV, runDecoder(make(cand), DEV)), o = obj(r);
    if (o > best.o) { best = { W: cand, r, o }; console.log(`iter ${it} improved`, JSON.stringify(r)); }
  }
  W = best.W; fs.mkdirSync('results', { recursive: true }); fs.writeFileSync('results/decoder_weights.json', JSON.stringify(W, null, 1));
  console.log('tuned weights', JSON.stringify(W));
}

// ---- test
const res = { nTest: TEST.length, weights: W, systems: {} };
const add = (name, out) => { res.systems[name] = score(TEST, out); console.log(name.padEnd(28), JSON.stringify(res.systems[name])); };
add('no correction', identity(TEST));
add('spelling only (VSA top-1)', spellingOnly(TEST));
add('proofreader page (careful)', proofreader(TEST));
let t0 = performance.now();
add('decoder (full)', runDecoder(make(W), TEST)); res.msPerSentence = +((performance.now() - t0) / TEST.length).toFixed(1);
if (!QUICK) add('decoder, no grammar', runDecoder(make({ ...W, grammar: 0, article: 0 }), TEST));
if (!QUICK) add('decoder, no context profiles', runDecoder(make(W, { noProfiles: true }), TEST));
if (!QUICK) add('decoder, no confusion classes', runDecoder(make(W, { noConfusion: true }), TEST));
if (!QUICK) add('decoder, grammar + spelling only', runDecoder(make({ ...W, profile: 0, confusion: 0 }, { noProfiles: true, noConfusion: true }), TEST));
if (HARPER) {
  const { LocalLinter } = await import('harper.js'); const { binary } = await import('harper.js/binary');
  const L = new LocalLinter({ binary }); await L.setup();
  const out = [];
  for (const it of TEST) {
    let text = it.text;
    const lints = (await L.lint(text)).filter(l => l.suggestions().length).sort((a, b) => b.span().start - a.span().start);
    let lastStart = 1e9;
    for (const l of lints) { const sp = l.span(); if (sp.end > lastStart) continue; const s0 = l.suggestions()[0]; text = text.slice(0, sp.start) + s0.get_replacement_text() + text.slice(sp.end); lastStart = sp.start; }
    const w = wordsOf(text); out.push(w.length === it.gold.length ? w : wordsOf(it.text).map((x, i) => w[i] ?? x));
  }
  add('Harper (first suggestions)', out);
}
res.examples = TEST.slice(0, 12).map(it => ({ input: it.text, decoded: make(W).decode(it.text).best.text, errs: it.errs }));
if (!QUICK) fs.writeFileSync('results/decoder_general.json', JSON.stringify(res, null, 1));
console.log(`${res.msPerSentence} ms/sentence`);
