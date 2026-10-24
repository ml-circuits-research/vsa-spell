// General word-context profiles for the sentence decoder: one hypervector per
// frequent word, the bundle (sum) of its position-permuted sparse context vectors
// over the training corpus. Unlike the confusion-set classes these exist for every
// frequent word, so the decoder can compare any candidates at a position.
// Usage: node src/train_profiles.mjs <N words> <D> [--eval] [--export]
import fs from 'fs';
import { mulberry32, hash32 } from '../docs/vsa-core.js';

const N = +(process.argv[2] || 30000), D = +(process.argv[3] || 1024);
const EVAL = process.argv.includes('--eval'), EXPORT = process.argv.includes('--export');
const K = 8, W = 4, SEED = 12345;
const tok = s => s.toLowerCase().match(/[a-z]+/g) || [];

// vocabulary: top-N of the 80k list (same ranks the speller uses)
const freqList = fs.readFileSync('data/en-80k.txt', 'utf8').trim().split('\n').map(l => l.split(' ')[0]);
const vocab = freqList.slice(0, N); const vid = new Map(vocab.map((w, i) => [w, i]));

// index vectors from hash (identical to ContextModel.indexVec)
const ivc = new Map();
function iv(w) {
  let v = ivc.get(w); if (v) return v;
  const r = mulberry32(hash32(w, SEED)); v = new Int32Array(2 * K);
  for (let j = 0; j < K; j++) { v[2 * j] = Math.floor(r() / 4294967296 * D); v[2 * j + 1] = r() / 4294967296 < 0.5 ? -1 : 1; }
  if (ivc.size > 300000) ivc.clear(); ivc.set(w, v); return v;
}
const POS = []; for (let p = -W; p <= W; p++) if (p) POS.push(p);
const SH = POS.map(p => ((p + 16) * 7919) % D), PW = POS.map(p => 1 / Math.abs(p));
const at = (t, j) => j < 0 ? '<s>' : j >= t.length ? '</s>' : t[j];

const P = new Float32Array(N * D); const cnt = new Float64Array(N);
let lines = fs.readFileSync('data/train_corpus.txt', 'utf8').split('\n');
let t0 = Date.now();
for (const s of lines) {
  const t = tok(s);
  for (let i = 0; i < t.length; i++) {
    const r = vid.get(t[i]); if (r === undefined) continue; cnt[r]++;
    const base = r * D;
    for (let k = 0; k < POS.length; k++) {
      const v = iv(at(t, i + POS[k])), sh = SH[k], w = PW[k];
      for (let j = 0; j < K; j++) P[base + (v[2 * j] + sh) % D] += w * v[2 * j + 1];
    }
  }
}
// centring: subtract each word's expected share of the average context, so a profile keeps
// what is specific to the word (a PMI-like correction in hypervector space)
const CENTER = +(process.argv.find(a => a.startsWith('--center='))?.split('=')[1] ?? 0);
{
  const tot = cnt.reduce((a, b) => a + b, 0), m = new Float64Array(D);
  for (let r = 0; r < N; r++) for (let d = 0; d < D; d++) m[d] += P[r * D + d];
  for (let d = 0; d < D; d++) m[d] /= tot;
  for (let r = 0; r < N; r++) for (let d = 0; d < D; d++) P[r * D + d] -= CENTER * cnt[r] * m[d];
}
console.log(`N=${N} D=${D} trained in ${((Date.now() - t0) / 1000).toFixed(1)}s; words seen ${cnt.filter(c => c > 0).length}`);

// int8 quantisation, one scale per row (rows normalised first)
const Q = new Int8Array(N * D), scale = new Float32Array(N);
for (let r = 0; r < N; r++) {
  let n2 = 0, mx = 0; for (let d = 0; d < D; d++) { const x = P[r * D + d]; n2 += x * x; }
  const n = Math.sqrt(n2) || 1; for (let d = 0; d < D; d++) mx = Math.max(mx, Math.abs(P[r * D + d] / n));
  const s = mx / 127 || 1; scale[r] = s / n; // stored value * scale[r]... see below
  for (let d = 0; d < D; d++) Q[r * D + d] = Math.round(P[r * D + d] / n / s);
  scale[r] = s; // dequantised row = Q * s (unit norm)
}

if (EVAL) {
  // choice accuracy on the round-3 real-word test instances (same split as context3.mjs)
  const SETS = JSON.parse(fs.readFileSync('data/confusions.json'));
  const conf = new Map(); SETS.forEach((s, k) => s.words.forEach(w => conf.set(w, k)));
  const rnd = mulberry32(1); const rf = () => rnd() / 4294967296;
  const raw = fs.readFileSync('data/big.txt', 'utf8').replace(/\s+/g, ' ');
  let sents = raw.split(/(?<=[.!?])\s+(?=[A-Z"])/).filter(s => s.length > 20 && s.length < 400);
  for (let i = sents.length - 1; i > 0; i--) { const j = Math.floor(rf() * (i + 1)); [sents[i], sents[j]] = [sents[j], sents[i]]; }
  const nTr = Math.floor(sents.length * 0.85), nDev = Math.floor(sents.length * 0.05);
  const test = sents.slice(nTr + nDev);
  const per = new Map(); let ok = 0, okPrior = 0, n = 0, covered = 0;
  const logf = w => { const r = vid.get(w); return r === undefined ? 0 : Math.log(cnt[r] + 1); };
  for (const s of test) {
    const t = tok(s);
    for (let i = 0; i < t.length; i++) if (conf.has(t[i])) {
      const k = t[i]; if ((per.get(k) || 0) >= 30) continue; per.set(k, (per.get(k) || 0) + 1);
      const cands = SETS[conf.get(k)].words; n++;
      if (cands.every(c => vid.has(c))) covered++;
      const sc = cands.map(c => {
        const r = vid.get(c); if (r === undefined) return -1e9;
        let s = 0;
        for (let q = 0; q < POS.length; q++) { const v = iv(at(t, i + POS[q])); for (let j = 0; j < K; j++) s += PW[q] * v[2 * j + 1] * Q[r * D + (v[2 * j] + SH[q]) % D] * scale[r]; }
        return s;
      });
      const best = (arr) => cands[arr.indexOf(Math.max(...arr))];
      // small prior weight; tuned value printed for a few settings
      for (const wp of [0]) if (best(sc) === k) ok++;
      if (best(cands.map(logf)) === k) okPrior++;
    }
  }
  console.log(`choice accuracy (profiles, no prior) ${(ok / n).toFixed(4)}  prior-only ${(okPrior / n).toFixed(4)}  n=${n} covered=${(covered / n).toFixed(3)}`);
}
if (EXPORT) {
  fs.writeFileSync('docs/profiles.bin', Buffer.from(Q.buffer));
  fs.writeFileSync('docs/profiles_meta.json', JSON.stringify({ N, D, K, W, seed: SEED, shiftMul: 7919, shiftOff: 16, words: vocab, scales: Array.from(scale, x => +x.toPrecision(6)), logCount: Array.from(cnt, c => +Math.log(c + 1).toFixed(3)) }));
  console.log(`exported docs/profiles.bin ${(Q.byteLength / 1e6).toFixed(1)} MB`);
}
