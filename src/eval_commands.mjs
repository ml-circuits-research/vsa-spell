// Agent-command domain: decoder restricted to the command vocabulary (153 words) and
// learning context from 150 example commands, vs the general 80k-word decoder.
// Test commands are disjoint from the examples. Usage: node src/eval_commands.mjs
import fs from 'fs';
import { SentenceDecoder, DEFAULT_WEIGHTS } from '../docs/decoder.js';
import { ContextModel, Speller, mulberry32 } from '../docs/vsa-core.js';

const rng = mulberry32(99); const rf = () => rng() / 4294967296; const pick = a => a[Math.floor(rf() * a.length)];
const general = fs.readFileSync('docs/words.txt', 'utf8').trim().split('\n').map(l => { const [w, c] = l.split(' '); return [w, +c]; });
const genSet = new Set(general.map(g => g[0]));
const grammar = JSON.parse(fs.readFileSync('docs/grammar.json'));
const pm = JSON.parse(fs.readFileSync('docs/profiles_meta.json')), pb = fs.readFileSync('docs/profiles.bin');
const cm = JSON.parse(fs.readFileSync('docs/ctx_meta.json')), cb = fs.readFileSync('docs/ctx_vectors.bin');
const profiles = { meta: pm, int8: new Int8Array(pb.buffer, pb.byteOffset, pb.length) };
const confusion = new ContextModel(cm, new Int8Array(cb.buffer, cb.byteOffset, cb.length));
const preset = JSON.parse(fs.readFileSync('docs/presets/agent-commands.json'));
const { dev, test } = JSON.parse(fs.readFileSync('data/commands_eval.json'));
const domSet = new Set(preset.dictionary.map(d => d[0]));
let W = { ...DEFAULT_WEIGHTS, ...(fs.existsSync('results/decoder_weights.json') ? JSON.parse(fs.readFileSync('results/decoder_weights.json')) : {}) };

// typos: codespell misspellings or keyboard edits, always producing a non-word for the domain
const missp = JSON.parse(fs.readFileSync('data/misspellings_by_word.json'));
const KEYS = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']; const near = c => { for (const r of KEYS) { const i = r.indexOf(c); if (i >= 0) return (r[i - 1] || '') + (r[i + 1] || ''); } return 'e'; };
function typo(w) {
  if (missp[w] && rf() < 0.5) return pick(missp[w]);
  for (let t = 0; t < 20; t++) {
    const i = Math.floor(rf() * w.length), op = rf(); let x;
    if (op < 0.3) x = w.slice(0, i) + w.slice(i + 1); else if (op < 0.55 && i < w.length - 1) x = w.slice(0, i) + w[i + 1] + w[i] + w.slice(i + 2);
    else if (op < 0.85) { const n = near(w[i]); x = w.slice(0, i) + n[Math.floor(rf() * n.length)] + w.slice(i + 1); } else x = w.slice(0, i) + w[i] + w.slice(i);
    if (x && x !== w && !domSet.has(x)) return x;
  }
  return null;
}
function corrupt(c) {
  const words = c.split(' '); const n = rf() < 0.15 ? 0 : rf() < 0.55 ? 1 : 2; const errs = []; const used = new Set();
  for (let e = 0; e < n; e++) for (let t = 0; t < 30; t++) { const i = Math.floor(rf() * words.length); if (used.has(i) || words[i].length < 3) continue; const x = typo(words[i]); if (!x) continue; used.add(i); errs.push({ i, from: words[i], to: x }); break; }
  const out = words.slice(); errs.forEach(e => { out[e.i] = e.to; });
  return { gold: words, text: out.join(' '), errs };
}
const DEV = dev.map(corrupt), TEST = test.map(corrupt);
const wordsOf = t => (t.match(/[A-Za-z]+/g) || []).map(w => w.toLowerCase());
function score(items, outs) {
  let sent = 0, eN = 0, eF = 0, words = 0, fc = 0;
  items.forEach((it, k) => { const o = outs[k]; if (o.join(' ') === it.gold.join(' ')) sent++; const ei = new Set(it.errs.map(e => e.i)); for (const e of it.errs) { eN++; if (o[e.i] === it.gold[e.i]) eF++; } it.gold.forEach((g, i) => { if (!ei.has(i)) { words++; if (o[i] !== g) fc++; } }); });
  return { sentence: +(sent / items.length).toFixed(4), fixed: +(eF / eN).toFixed(4), falseChangesPer100: +(100 * fc / words).toFixed(2), n: items.length, errors: eN };
}
const genSpeller = new Speller({ D: 1024 }); genSpeller.build(general);
const domSpeller = new Speller({ D: 1024 }); domSpeller.build(preset.dictionary.slice().sort((a, b) => b[1] - a[1]));
const spellOnly = (sp, set) => items => items.map(it => wordsOf(it.text).map(w => set.has(w) ? w : sp.suggest(w, { wPhon: 0.5, wFreq: 0.8, wLen: 0, wEd: 0.35 }, 1)[0]));
const dec = (opts, w = W) => new SentenceDecoder({ grammar, profiles, confusion, weights: w, ...opts });
const run = (d, items) => items.map(it => d.decode(it.text).best.words);

// small tune of the domain weight and the cost of keeping unknown tokens (dev only)
let best = null;
for (const domain of [1, 3, 6]) for (const keepUnknown of [W.keepUnknown, W.keepUnknown - 4]) {
  const w = { ...W, domain, keepUnknown }; const r = score(DEV, run(dec({ dictionary: preset.dictionary, examples: preset.examples }, w), DEV));
  if (!best || r.sentence > best.r.sentence) best = { w, r };
}
console.log('dev best', JSON.stringify({ domain: best.w.domain, keepUnknown: best.w.keepUnknown }), JSON.stringify(best.r));
const res = { nTest: TEST.length, domainWeights: { domain: best.w.domain, keepUnknown: best.w.keepUnknown }, systems: {} };
const add = (k, o) => { res.systems[k] = score(TEST, o); console.log(k.padEnd(40), JSON.stringify(res.systems[k])); };
add('no correction', TEST.map(it => wordsOf(it.text)));
add('spelling only, general 80k dictionary', spellOnly(genSpeller, genSet)(TEST));
add('spelling only, command dictionary', spellOnly(domSpeller, domSet)(TEST));
add('decoder, general 80k dictionary', run(dec({ dictionary: general, speller: genSpeller }), TEST));
add('decoder, command dictionary', run(dec({ dictionary: preset.dictionary }, best.w), TEST));
let t0 = performance.now();
const full = dec({ dictionary: preset.dictionary, examples: preset.examples }, best.w);
add('decoder, command dictionary + examples', run(full, TEST)); res.msPerSentence = +((performance.now() - t0) / TEST.length).toFixed(1);
res.examples = TEST.filter(it => it.errs.length).slice(0, 10).map(it => ({ input: it.text, decoded: full.decode(it.text).best.text, gold: it.gold.join(' ') }));
fs.writeFileSync('results/decoder_commands.json', JSON.stringify(res, null, 1));
console.log(res.msPerSentence, 'ms/sentence'); console.log(res.examples.slice(0, 6));
