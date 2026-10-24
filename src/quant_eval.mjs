// How much precision do the shipped hypervectors need? Re-quantises the confusion classes
// and the word-context profiles (int8 -> int4 / 2-bit / 1-bit sign) and measures choice
// accuracy on the round-3 real-word test (9,198 instances, same split as context3.mjs).
import fs from 'fs';
import { ContextModel, mulberry32, hash32 } from '../docs/vsa-core.js';
const ONLY_PROFILES = process.argv.includes('--profiles-only');

const tok = s => s.toLowerCase().match(/[a-z]+/g) || [];
const SETS = JSON.parse(fs.readFileSync('data/confusions.json'));
const conf = new Map(); SETS.forEach((s, k) => s.words.forEach(w => conf.set(w, k)));
const r1 = mulberry32(1); const rf = () => r1() / 4294967296;
const raw = fs.readFileSync('data/big.txt', 'utf8').replace(/\s+/g, ' ');
let sents = raw.split(/(?<=[.!?])\s+(?=[A-Z"])/).filter(s => s.length > 20 && s.length < 400);
for (let i = sents.length - 1; i > 0; i--) { const j = Math.floor(rf() * (i + 1)); [sents[i], sents[j]] = [sents[j], sents[i]]; }
const nTr = Math.floor(sents.length * 0.85), nDev = Math.floor(sents.length * 0.05);
const per = new Map(); const I = [];
for (const s of sents.slice(nTr + nDev)) { const t = ['<s>', ...tok(s), '</s>']; for (let i = 1; i < t.length - 1; i++) if (conf.has(t[i])) { const k = t[i]; if ((per.get(k) || 0) >= 30) continue; per.set(k, (per.get(k) || 0) + 1); I.push({ t, i, w: k, cands: SETS[conf.get(k)].words }); } }

// re-quantise rows of an int8 matrix (row-major, D columns) to `bits` (1 = sign)
function requant(q, scales, D, bits) {
  const n = scales.length, out = new Int8Array(q.length), sc = new Float32Array(n);
  for (let r = 0; r < n; r++) {
    let mx = 0, ma = 0; for (let d = 0; d < D; d++) { const x = Math.abs(q[r * D + d]); mx = Math.max(mx, x); ma += x; }
    if (bits === 2) { // non-uniform 4 levels (Lloyd–Max for a Gaussian): ±0.453σ, ±1.51σ stored as ±1, ±3
      let v2 = 0; for (let d = 0; d < D; d++) v2 += q[r * D + d] ** 2; const sd = Math.sqrt(v2 / D) || 1, t = 0.9816 * sd;
      for (let d = 0; d < D; d++) { const x = q[r * D + d]; out[r * D + d] = x >= 0 ? (x > t ? 3 : 1) : (x < -t ? -3 : -1); }
      sc[r] = scales[r] * sd * 0.5025; continue;
    }
    if (bits === 1) { const m = ma / D || 1; for (let d = 0; d < D; d++) out[r * D + d] = q[r * D + d] >= 0 ? 1 : -1; sc[r] = scales[r] * m; continue; }
    const L = (1 << (bits - 1)) - 1, s = mx / L || 1; // symmetric levels -L..L
    for (let d = 0; d < D; d++) out[r * D + d] = Math.round(q[r * D + d] / s);
    sc[r] = scales[r] * s;
  }
  return { q: out, scales: Array.from(sc) };
}

// 1) confusion classes
// int8 classes as written by `context3.mjs --export` (before packing)
const cm = JSON.parse(fs.readFileSync('data/ctx_meta_int8.json')), cb = fs.readFileSync('data/ctx_int8.bin');
const c8 = new Int8Array(cb.buffer, cb.byteOffset, cb.length);
const accC = model => I.filter(x => { const s = model.scores(x.t, x.i); return x.cands[s.indexOf(Math.max(...s))] === x.w; }).length / I.length;
const res = { classes: {}, profiles: {} };
if (!ONLY_PROFILES) for (const bits of [8, 4, 2, 1]) {
  const v = bits === 8 ? { q: c8, scales: cm.scales } : requant(c8, cm.scales, cm.D, bits);
  const model = new ContextModel({ ...cm, scales: v.scales }, v.q);
  const acc = accC(model);
  // detection at the shipped threshold (Careful): corrupted instance fixed / clean instance flagged
  let fixed = 0, fa = 0;
  for (const x of I) {
    if (model.check(x.t, x.i)) fa++;
    const wrong = x.cands[(x.cands.indexOf(x.w) + 1) % x.cands.length], t2 = x.t.slice(); t2[x.i] = wrong;
    const r = model.check(t2, x.i); if (r && r.suggestion === x.w) fixed++;
  }
  res.classes[`int${bits}`] = { acc: +acc.toFixed(4), fixed: +(fixed / I.length).toFixed(4), falseMarks: +(fa / I.length).toFixed(4), MB: +(c8.length * bits / 8 / 1e6).toFixed(2) };
  console.log(`classes D=${cm.D} ${bits}-bit: ${(acc * 100).toFixed(2)}%  fixed ${(100 * fixed / I.length).toFixed(1)}%  false ${(100 * fa / I.length).toFixed(2)}%  ${(c8.length * bits / 8 / 1e6).toFixed(2)} MB`);
}
// 2) word-context profiles (choice accuracy using profiles alone)
// profiles must be exported at 8 bits for this part: node src/train_profiles.mjs 30000 1024 --export --bits=8
const pm = JSON.parse(fs.readFileSync('docs/profiles_meta.json')), pb = fs.readFileSync('docs/profiles.bin');
if ((pm.bits || 8) !== 8) { console.log('profiles: shipped file is packed; re-export at --bits=8 to rerun this part'); process.exit(0); }
pm.words = pm.words || fs.readFileSync('docs/words.txt', 'utf8').trim().split('\n').slice(0, pm.N).map(l => l.split(' ')[0]);
const p8 = new Int8Array(pb.buffer, pb.byteOffset, pb.length);
const row = new Map(pm.words.map((w, k) => [w, k])); const POS = []; for (let p = -4; p <= 4; p++) if (p) POS.push(p);
const ivc = new Map(); const iv = w => { let v = ivc.get(w); if (v) return v; const r = mulberry32(hash32(w, pm.seed)); v = new Int32Array(16); for (let j = 0; j < 8; j++) { v[2 * j] = Math.floor(r() / 4294967296 * pm.D); v[2 * j + 1] = r() / 4294967296 < 0.5 ? -1 : 1; } ivc.set(w, v); return v; };
const accP = (q, scales) => I.filter(x => {
  const s = x.cands.map(c => { const r = row.get(c); if (r === undefined) return -1e9; let v = 0; for (const p of POS) { const j = x.i + p, w = x.t[j] ?? (j < 0 ? '<s>' : '</s>'), ix = iv(w), sh = ((p + 16) * 7919) % pm.D; for (let k = 0; k < 8; k++) v += ix[2 * k + 1] * q[r * pm.D + (ix[2 * k] + sh) % pm.D] / Math.abs(p); } return v * scales[r]; });
  return x.cands[s.indexOf(Math.max(...s))] === x.w;
}).length / I.length;
for (const bits of [8, 4, 2, 1]) {
  const v = bits === 8 ? { q: p8, scales: pm.scales } : requant(p8, pm.scales, pm.D, bits);
  const acc = accP(v.q, v.scales);
  res.profiles[`int${bits}`] = { acc: +acc.toFixed(4), MB: +(p8.length * bits / 8 / 1e6).toFixed(2) };
  console.log(`profiles N=${pm.N} D=${pm.D} ${bits}-bit: ${(acc * 100).toFixed(2)}%  ${(p8.length * bits / 8 / 1e6).toFixed(2)} MB`);
}
if (!ONLY_PROFILES) fs.writeFileSync('results/quantization.json', JSON.stringify(res, null, 1));
