// Real-word errors, round 3: ~320 confusion sets, sparse random-indexing VSA
// with HDC retraining, int8-quantised class vectors, per-set enabling from dev.
// Usage: node src/context3.mjs <ext|ext+big> [--export]
import fs from 'fs';

const SRC = process.argv[2] || 'ext', EXPORT = process.argv.includes('--export');
const SETS = JSON.parse(fs.readFileSync('data/confusions.json')).map(s => ({ ...s, words: s.words }));
const conf = new Map(); SETS.forEach((s, k) => s.words.forEach(w => conf.set(w, k)));
const candsOf = w => SETS[conf.get(w)].words;
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const tok = s => s.toLowerCase().match(/[a-z]+/g) || [];

// identical big.txt split as rounds 1–2
const rnd = mulberry32(1);
const raw = fs.readFileSync('data/big.txt', 'utf8').replace(/\s+/g, ' ');
let sents = raw.split(/(?<=[.!?])\s+(?=[A-Z"])/).filter(s => s.length > 20 && s.length < 400);
for (let i = sents.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [sents[i], sents[j]] = [sents[j], sents[i]]; }
const nTr = Math.floor(sents.length * 0.85), nDev = Math.floor(sents.length * 0.05);
const bigTrain = sents.slice(0, nTr), dev = sents.slice(nTr, nTr + nDev), test = sents.slice(nTr + nDev);
let train = fs.readFileSync('data/train_corpus.txt', 'utf8').split('\n');
if (SRC.includes('big')) train = train.concat(bigTrain);
const trainTok = train.map(s => ['<s>', ...tok(s), '</s>']);
console.log(`[${SRC}] sets=${SETS.length} words=${trainTok.reduce((a, t) => a + t.length - 2, 0)}`);

// n-gram baseline
const uni = new Map(), bi = new Map(), tri = new Map();
const inc = (m, k) => m.set(k, (m.get(k) || 0) + 1);
for (const t of trainTok) for (let i = 0; i < t.length; i++) { inc(uni, t[i]); if (i) inc(bi, t[i - 1] + ' ' + t[i]); if (i > 1) inc(tri, t[i - 2] + ' ' + t[i - 1] + ' ' + t[i]); }
let TOTAL = 0; for (const v of uni.values()) TOTAL += v;
const cnt = (m, k) => m.get(k) || 0;
function S(w, h1, h2) {
  if (h2 !== undefined) { const c3 = cnt(tri, `${h2} ${h1} ${w}`), h = cnt(bi, `${h2} ${h1}`); if (c3 && h) return c3 / h; }
  if (h1 !== undefined) { const c2 = cnt(bi, `${h1} ${w}`), h = cnt(uni, h1); if (c2 && h) return 0.4 * c2 / h; }
  return 0.16 * (cnt(uni, w) + 0.5) / TOTAL;
}
function scoreNgram(t, i, c) { const u = t.slice(); u[i] = c; let s = 0; for (let j = i; j <= Math.min(i + 2, u.length - 1); j++) s += Math.log(S(u[j], u[j - 1], u[j - 2])); return s; }

// ---- VSA (identical encoding to what ships in the browser)
const K = 8, WMAX = 4;
function hash32(s, seed) { let h = 2166136261 ^ seed; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
// word ids; index vectors are generated from hash(word) so nothing per word needs shipping
const vid = new Map(); const vw = [];
const idOf = w => { let k = vid.get(w); if (k === undefined) { k = vw.length; vid.set(w, k); vw.push(w); } return k; };
const ivTables = new Map(); // D -> {T: Int32Array, n: rows filled}
function ivTable(D) {
  let e = ivTables.get(D); if (!e) { e = { T: new Int32Array(Math.max(1024, vw.length * 2) * 2 * K), n: 0 }; ivTables.set(D, e); }
  if (e.n < vw.length) {
    if (e.T.length < vw.length * 2 * K) { const T2 = new Int32Array(vw.length * 4 * K); T2.set(e.T); e.T = T2; }
    for (let k = e.n; k < vw.length; k++) { const r = mulberry32(hash32(vw[k], 12345)); for (let j = 0; j < K; j++) { e.T[k * 2 * K + 2 * j] = Math.floor(r() * D); e.T[k * 2 * K + 2 * j + 1] = r() < 0.5 ? -1 : 1; } }
    e.n = vw.length;
  }
  return e.T;
}
const POS = []; for (let p = -WMAX; p <= WMAX; p++) if (p) POS.push(p);
function ctxIds(t, i) { const c = new Int32Array(POS.length); POS.forEach((p, k) => { const j = i + p; c[k] = idOf(j < 0 ? '<s>' : j >= t.length ? '</s>' : t[j]); }); return c; }
function ctxOf(t, i, cfg) { return ctxIds(t, i); }
function dot(P, c, cfg) {
  const T = ivTable(cfg.D), D = cfg.D; let s = 0;
  for (let k = 0; k < POS.length; k++) { const p = POS[k]; if (Math.abs(p) > cfg.W) continue; const w = 1 / Math.abs(p), sh = ((p + 16) * 7919) % D, b = c[k] * 2 * K;
    for (let j = 0; j < K; j++) s += w * T[b + 2 * j + 1] * P[(T[b + 2 * j] + sh) % D]; }
  return s;
}
function addTo(P, c, cfg, a) {
  const T = ivTable(cfg.D), D = cfg.D;
  for (let k = 0; k < POS.length; k++) { const p = POS[k]; if (Math.abs(p) > cfg.W) continue; const w = 1 / Math.abs(p), sh = ((p + 16) * 7919) % D, b = c[k] * 2 * K;
    for (let j = 0; j < K; j++) P[(T[b + 2 * j] + sh) % D] += a * w * T[b + 2 * j + 1]; }
}
const qnormW = W => { let s = 0; for (const p of POS) if (Math.abs(p) <= W) s += K / (p * p); return Math.sqrt(s); };
const vnorm = P => { let s = 0; for (const x of P) s += x * x; return Math.sqrt(s) || 1; };

// training contexts (cap per word keeps the giant classes from dominating time)
const CAP = 12000;
const perW = new Map(); const tInst = [];
for (const t of trainTok) for (let i = 1; i < t.length - 1; i++) if (conf.has(t[i])) {
  const w = t[i], n = perW.get(w) || 0; if (n < CAP) { tInst.push([w, ctxIds(t, i)]); perW.set(w, n + 1); }
}
console.log(`training contexts ${tInst.length}`);

function trainVSA(cfg) {
  ivTable(cfg.D);
  const P = new Map(); conf.forEach((_, w) => P.set(w, new Float32Array(cfg.D)));
  for (const [w, c] of tInst) addTo(P.get(w), c, cfg, 1);
  for (const v of P.values()) { const n = vnorm(v); for (let d = 0; d < v.length; d++) v[d] /= n; }
  const q = qnormW(cfg.W);
  for (let e = 0; e < cfg.epochs; e++) {
    for (const [w, c] of tInst) {
      const cands = candsOf(w);
      let best = -1e9, pred = w; for (const x of cands) { const s = dot(P.get(x), c, cfg) / q; if (s > best) { best = s; pred = x; } }
      if (pred !== w) { addTo(P.get(w), c, cfg, cfg.lr / q); addTo(P.get(pred), c, cfg, -cfg.lr / q); }
    }
  }
  return P;
}
// int8 quantisation with one scale per class vector (this is what ships)
function quantize(P) {
  const Q = new Map();
  for (const [w, v] of P) { let m = 0; for (const x of v) m = Math.max(m, Math.abs(x)); const s = m / 127 || 1; const q = new Int8Array(v.length); for (let d = 0; d < v.length; d++) q[d] = Math.round(v[d] / s); Q.set(w, { q, s }); }
  return Q;
}
const dequant = Q => new Map([...Q].map(([w, { q, s }]) => [w, Float32Array.from(q, x => x * s)]));
const logPrior = c => Math.log(cnt(uni, c) + 1);
const scoreVSA = (P, cfg, wPrior) => x => { const c = ctxIds(x.t, x.i), q = qnormW(cfg.W); ivTable(cfg.D); return x.cands.map(w => dot(P.get(w), c, cfg) / q + wPrior * logPrior(w) / 20); };

function instances(set, cap) {
  const per = new Map(); const out = [];
  for (const s of set) { const t = ['<s>', ...tok(s), '</s>'];
    for (let i = 1; i < t.length - 1; i++) if (conf.has(t[i])) { const k = t[i]; if ((per.get(k) || 0) >= cap) continue; per.set(k, (per.get(k) || 0) + 1); out.push({ t, i, w: k, cands: candsOf(k), set: conf.get(k) }); } }
  return out;
}
const devI = instances(dev, 25), testI = instances(test, 30);
console.log(`instances dev=${devI.length} test=${testI.length}`);
const argmax = a => a.indexOf(Math.max(...a));
const accOn = (I, fn) => I.filter(x => x.cands[argmax(fn(x))] === x.w).length / I.length;

// ---- tune (dev only)
let best = null; const t0 = Date.now();
for (const D of [8192, 16384]) for (const epochs of [12, 20]) {
  const cfg = { D, W: 4, epochs, lr: 1 };
  const P = trainVSA(cfg);
  for (const wPrior of [0, 0.5, 1, 2]) {
    const a = accOn(devI, scoreVSA(P, cfg, wPrior));
    if (!best || a > best.acc) best = { acc: a, cfg, wPrior, P };
  }
  console.log(`  D=${D} epochs=${epochs} best-so-far dev=${best.acc.toFixed(4)} (${((Date.now() - t0) / 1000) | 0}s)`);
}
const { cfg, wPrior } = best;
const Q = quantize(best.P), PQ = dequant(Q);
const vsaF = scoreVSA(best.P, cfg, wPrior), vsaQ = scoreVSA(PQ, cfg, wPrior);
const ngF = x => x.cands.map(c => scoreNgram(x.t, x.i, c));
let bestA = { a: 1, acc: -1 };
for (const a of [1, 3, 10, 30]) { const acc = accOn(devI, x => { const v = vsaQ(x), g = ngF(x); return v.map((q, k) => a * q + g[k]); }); if (acc > bestA.acc) bestA = { a, acc }; }
const ensF = x => { const v = vsaQ(x), g = ngF(x); return v.map((q, k) => bestA.a * q + g[k]); };
const priorF = x => x.cands.map(c => cnt(uni, c));

// detection: clean margin & corrupted margin for each instance (corrupt with every other member)
function detectStats(fn, I) {
  return I.map(x => {
    const sc = fn(x); const clean = Math.max(...sc) - sc[x.cands.indexOf(x.w)];
    const wrong = x.cands[(x.cands.indexOf(x.w) + 1) % x.cands.length]; const tw = x.t.slice(); tw[x.i] = wrong;
    const sc2 = fn({ ...x, t: tw }); const corrupt = Math.max(...sc2) - sc2[x.cands.indexOf(wrong)];
    return { set: x.set, clean, corrupt, fixOK: x.cands[argmax(sc2)] === x.w };
  });
}
// per-set enabling: keep a set only if, on dev, it fixes more than it falsely flags
function calibrate(devS, fa = 0.02) {
  const cl = devS.map(r => r.clean).sort((a, b) => b - a); const tau = cl[Math.floor(cl.length * fa)] + 1e-9;
  const per = new Map(); for (const r of devS) { const p = per.get(r.set) || { n: 0, fix: 0, fa: 0 }; p.n++; if (r.corrupt > tau && r.fixOK) p.fix++; if (r.clean > tau) p.fa++; per.set(r.set, p); }
  const enabled = new Set([...per].filter(([, p]) => p.fix / p.n >= 0.3 && p.fa / p.n <= 0.1).map(([k]) => k));
  return { tau, enabled };
}
const res = { src: SRC, cfg, wPrior, ensembleA: bestA.a, sets: SETS.length, nTest: testI.length, choice: {}, detect: {}, bySource: {} };
res.choice = { prior: accOn(testI, priorF), ngram: accOn(testI, ngF), vsa_f32: accOn(testI, vsaF), vsa_int8: accOn(testI, vsaQ), 'vsa+ngram': accOn(testI, ensF) };
for (const [name, fn] of [['vsa_int8', vsaQ], ['ngram', ngF], ['vsa+ngram', ensF]]) {
  const { tau, enabled } = calibrate(detectStats(fn, devI));
  const ts = detectStats(fn, testI);
  const on = r => enabled.has(r.set);
  res.detect[name] = {
    allSets: { fixed: ts.filter(r => r.corrupt > tau && r.fixOK).length / ts.length, falseAlarm: ts.filter(r => r.clean > tau).length / ts.length },
    enabledSets: { n: enabled.size, fixed: ts.filter(r => on(r) && r.corrupt > tau && r.fixOK).length / ts.length, falseAlarm: ts.filter(r => on(r) && r.clean > tau).length / ts.length },
  };
  if (name === 'vsa_int8') { res.vsaTau = tau; res.vsaEnabled = [...enabled]; }
}
for (const src of ['curated', 'typo', 'homophone']) {
  const I = testI.filter(x => SETS[x.set].src === src); if (!I.length) continue;
  res.bySource[src] = { n: I.length, ngram: accOn(I, ngF), vsa: accOn(I, vsaQ), ens: accOn(I, ensF) };
}
res.mem = { vsaInt8MB: conf.size * cfg.D / 1e6, ngramEntries: uni.size + bi.size + tri.size };
console.log(JSON.stringify(res, (k, v) => k === 'vsaEnabled' ? `[${v.length} sets]` : typeof v === 'number' ? +v.toFixed(4) : v, 1));
fs.mkdirSync('results', { recursive: true });
fs.writeFileSync(`results/context_${SRC}.json`, JSON.stringify(res, null, 1));

if (EXPORT) { // model for the browser: header JSON + one int8 blob
  const words = [...Q.keys()];
  const meta = { D: cfg.D, W: cfg.W, K, seed: 12345, shiftMul: 7919, shiftOff: 16, wPrior, tau: res.vsaTau,
    sets: SETS.map(s => s.words), enabled: res.vsaEnabled, words, scales: words.map(w => Q.get(w).s), logPrior: words.map(w => +logPrior(w).toFixed(4)) };
  const blob = new Int8Array(words.length * cfg.D); words.forEach((w, k) => blob.set(Q.get(w).q, k * cfg.D));
  fs.mkdirSync('docs', { recursive: true });
  fs.writeFileSync('data/ctx_meta_int8.json', JSON.stringify(meta));
  fs.writeFileSync('data/ctx_int8.bin', Buffer.from(blob.buffer)); // packed for shipping by src/pack_models.mjs
  // golden cases to check that the browser reproduces Node scores exactly
  const gold = testI.slice(0, 200).map(x => ({ toks: x.t, i: x.i, scores: vsaQ(x).map(v => +v.toFixed(5)) }));
  fs.writeFileSync('data/ctx_gold.json', JSON.stringify(gold));
  console.log(`exported ${words.length} vectors, ${(blob.byteLength / 1e6).toFixed(2)} MB`);
}
