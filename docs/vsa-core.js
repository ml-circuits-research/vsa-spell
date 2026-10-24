// VSA proofreading core — runs unchanged in the browser and in Node.
// 1) Spelling: words as bundles of position-bound character n-grams
//    (binary hypervectors, XOR binding, word-rotation permutation, majority
//    bundling via bit-sliced counters), plus a phonetic (Double Metaphone)
//    hypervector; retrieval by Hamming distance; light rerank.
// 2) Context (real-word errors): sparse random-indexing context vectors
//    (index vectors generated from a hash, so nothing per context word is
//    stored), position by permutation, int8 class vectors per confusion word.
import { doubleMetaphone } from './double-metaphone.js';

export function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0); };
}
export function hash32(s, seed) { let h = 2166136261 ^ seed; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); } return h >>> 0; }
const popcnt = x => { x = x - ((x >>> 1) & 0x55555555); x = (x & 0x33333333) + ((x >>> 2) & 0x33333333); return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24; };

export function osa(a, b) {
  const m = a.length, n = b.length; if (!m) return n; if (!n) return m;
  let p2 = new Array(n + 1), p1 = new Array(n + 1), c = new Array(n + 1);
  for (let j = 0; j <= n; j++) p1[j] = j;
  for (let i = 1; i <= m; i++) {
    c[0] = i;
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let v = Math.min(p1[j] + 1, c[j - 1] + 1, p1[j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) v = Math.min(v, p2[j - 2] + 1);
      c[j] = v;
    }
    [p2, p1, c] = [p1, c, p2];
  }
  return p1[n];
}

// ------------------------------------------------------------------ spelling
export class Speller {
  constructor({ D = 1024, seed = 7 } = {}) {
    this.D = D; this.W = D >>> 5; this.seed = seed; this.sym = new Map(); this.rot = new Map();
    this.B = 7; // bit planes for counters (up to 127 n-grams per word)
    this.planes = Array.from({ length: this.B }, () => new Uint32Array(this.W));
    this.tmp = new Uint32Array(this.W);
  }
  // one hypervector per symbol, seeded by the symbol itself (order independent)
  base(s) {
    let v = this.sym.get(s); if (v) return v;
    const r = mulberry32(hash32(s, this.seed)); v = new Uint32Array(this.W); for (let i = 0; i < this.W; i++) v[i] = r();
    this.sym.set(s, v); return v;
  }
  rho(s, k) {
    const key = s + '|' + k; let v = this.rot.get(key); if (v) return v;
    const b = this.base(s); v = new Uint32Array(this.W); for (let i = 0; i < this.W; i++) v[(i + k) % this.W] = b[i];
    this.rot.set(key, v); return v;
  }
  // bundle of n-grams of '#str#', each n-gram = ρ^{n-1}(c0) ⊕ … ⊕ c_{n-1}
  encode(str, ns, tag, out) {
    const s = '#' + str + '#', W = this.W, P = this.planes, tmp = this.tmp, B = this.B;
    for (const p of P) p.fill(0);
    let n = 0;
    for (const g of ns) for (let i = 0; i + g <= s.length; i++) {
      tmp.fill(0);
      for (let j = 0; j < g; j++) { const v = this.rho(tag + s[i + j], g - 1 - j); for (let w = 0; w < W; w++) tmp[w] ^= v[w]; }
      for (let w = 0; w < W; w++) { let carry = tmp[w]; for (let b = 0; b < B && carry; b++) { const t = P[b][w] & carry; P[b][w] ^= carry; carry = t; } }
      n++;
    }
    // majority: count > n/2 → 1, count == n/2 → tie vector
    const m = n >>> 1, tie = this.base('__tie__'), even = (n & 1) === 0;
    for (let w = 0; w < W; w++) {
      let gt = 0, eq = 0xffffffff;
      for (let b = B - 1; b >= 0; b--) { const c = P[b][w]; if ((m >>> b) & 1) eq &= c; else { gt |= eq & c; eq &= ~c; } }
      out[w] = (gt | (even ? eq & tie[w] : 0)) >>> 0;
    }
    return out;
  }
  phoneKey(w) { const a = doubleMetaphone(w)[0]; return a ? a.toLowerCase() : w; }
  build(entries) { // [[word, count], …] sorted by count desc
    const N = entries.length, W = this.W; this.N = N; this.words = new Array(N); this.index = new Map();
    this.logf = new Float32Array(N); this.O = new Uint32Array(N * W); this.P = new Uint32Array(N * W);
    const maxLog = Math.log(entries[0][1]); const o = new Uint32Array(W);
    for (let i = 0; i < N; i++) {
      const [w, c] = entries[i]; this.words[i] = w; this.index.set(w, i); this.logf[i] = Math.log(c) / maxLog;
      this.O.set(this.encode(w, [2, 3], '', o), i * W);
      this.P.set(this.encode(this.phoneKey(w), [2, 3], 'P', o), i * W);
    }
    this._so = new Float32Array(N); this._sp = new Float32Array(N);
  }
  has(w) { return this.index.has(w); }
  sims(q, M, out) {
    const W = this.W, D = this.D;
    for (let i = 0, off = 0; i < this.N; i++, off += W) { let h = 0; for (let w = 0; w < W; w++) h += popcnt(q[w] ^ M[off + w]); out[i] = 1 - (2 * h) / D; }
  }
  top(arr, K) {
    const idx = new Int32Array(K).fill(-1), val = new Float32Array(K).fill(-9);
    for (let i = 0; i < this.N; i++) { const s = arr[i]; if (s > val[K - 1]) { let j = K - 1; while (j > 0 && val[j - 1] < s) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; } val[j] = s; idx[j] = i; } }
    return idx;
  }
  // g: {wPhon, wFreq, wLen, wEd} tuned offline on a dev split
  suggest(bad, g, k = 5) {
    const W = this.W, o = new Uint32Array(W), p = new Uint32Array(W);
    this.encode(bad, [2, 3], '', o); this.encode(this.phoneKey(bad), [2, 3], 'P', p);
    this.sims(o, this.O, this._so); this.sims(p, this.P, this._sp);
    const pool = new Set([...this.top(this._so, 60), ...this.top(this._sp, 60)]);
    for (let i = 0; i < Math.min(200, this.N); i++) pool.add(i);
    pool.delete(-1);
    return [...pool].map(i => {
      const w = this.words[i];
      let s = this._so[i] + g.wPhon * this._sp[i] + g.wFreq * this.logf[i] - g.wLen * Math.abs(w.length - bad.length);
      if (g.wEd) s -= g.wEd * osa(bad, w);
      return [s, w];
    }).sort((a, b) => b[0] - a[0]).slice(0, k).map(x => x[1]);
  }
  bytes() { return this.O.byteLength + this.P.byteLength; }
}

// ------------------------------------------------------------------- context
export class ContextModel {
  constructor(meta, int8) {
    Object.assign(this, meta); // D, W, K, seed, shiftMul, shiftOff, wPrior, tau, sets, enabled, words, scales, logPrior
    this.vec = int8; this.row = new Map(this.words.map((w, k) => [w, k]));
    this.setOf = new Map(); this.sets.forEach((s, k) => s.forEach(w => this.setOf.set(w, k)));
    this.on = new Set(this.enabled); this.iv = new Map();
    this.POS = []; for (let p = -4; p <= 4; p++) if (p) this.POS.push(p);
    let q = 0; for (const p of this.POS) if (Math.abs(p) <= this.W) q += this.K / (p * p); this.qn = Math.sqrt(q);
  }
  indexVec(w) {
    let v = this.iv.get(w); if (v) return v;
    const r = mulberry32(hash32(w, this.seed)); v = new Int32Array(2 * this.K);
    for (let j = 0; j < this.K; j++) { v[2 * j] = Math.floor(r() / 4294967296 * this.D); v[2 * j + 1] = r() / 4294967296 < 0.5 ? -1 : 1; }
    if (this.iv.size > 50000) this.iv.clear(); this.iv.set(w, v); return v;
  }
  // toks: ['<s>', ...lowercase words, '</s>'], i: position of the word
  scores(toks, i) {
    const cands = this.sets[this.setOf.get(toks[i])], D = this.D;
    return cands.map(c => {
      const r = this.row.get(c), base = r * D, sc = this.scales[r]; let s = 0;
      for (const p of this.POS) {
        if (Math.abs(p) > this.W) continue; const j = i + p;
        const v = this.indexVec(j < 0 ? '<s>' : j >= toks.length ? '</s>' : toks[j]);
        const sh = ((p + this.shiftOff) * this.shiftMul) % D, w = 1 / Math.abs(p);
        for (let k = 0; k < this.K; k++) s += w * v[2 * k + 1] * this.vec[base + (v[2 * k] + sh) % D] * sc;
      }
      return s / this.qn + this.wPrior * this.logPrior[r] / 20;
    });
  }
  // returns null or {suggestion, margin}
  check(toks, i, scale = 1) {
    const w = toks[i], k = this.setOf.get(w); if (k === undefined || !this.on.has(k)) return null;
    const cands = this.sets[k], sc = this.scores(toks, i), cur = sc[cands.indexOf(w)];
    let best = 0; for (let j = 1; j < sc.length; j++) if (sc[j] > sc[best]) best = j;
    const margin = sc[best] - cur;
    return margin > this.tau * scale ? { suggestion: cands[best], margin, alternatives: cands.filter(c => c !== w) } : null;
  }
}

// ------------------------------------------------------ packed hypervector matrices
// Rows of class / profile hypervectors are shipped at 8, 4 or 2 bits per element:
//   8: int8, value × scale
//   4: two's-complement nibbles in −7..7 (uniform, max-scaled), two per byte
//   2: codes 0..3 → −3, −1, +1, +3 (non-uniform, thresholds at ±0.98σ; clips outliers), four per byte
// unpackRows returns an Int8Array with the integer levels; multiply by the row's scale.
export function unpackRows(bytes, count, bits) {
  const u = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes.buffer || bytes, bytes.byteOffset || 0, bytes.byteLength);
  if (bits === 8) return new Int8Array(u.buffer, u.byteOffset, count);
  const out = new Int8Array(count);
  if (bits === 4) for (let i = 0; i < count; i++) { const b = u[i >> 1], n = (i & 1) ? b >> 4 : b & 15; out[i] = n > 7 ? n - 16 : n; }
  else if (bits === 2) { const L = [-3, -1, 1, 3]; for (let i = 0; i < count; i++) out[i] = L[(u[i >> 2] >> ((i & 3) * 2)) & 3]; }
  else throw new Error('unsupported bits: ' + bits);
  return out;
}
// quantise row-major float (or int8 × scale) rows; returns packed bytes and per-row scales
export function quantizeRows(get, rows, D, bits) {
  const scales = new Float32Array(rows), n = rows * D;
  const packed = new Uint8Array(bits === 8 ? n : Math.ceil(n * bits / 8));
  const q8 = bits === 8 ? new Int8Array(packed.buffer) : null;
  for (let r = 0; r < rows; r++) {
    let mx = 0, v2 = 0; for (let d = 0; d < D; d++) { const x = get(r, d); mx = Math.max(mx, Math.abs(x)); v2 += x * x; }
    if (bits === 2) {
      const sd = Math.sqrt(v2 / D) || 1, t = 0.9816 * sd; scales[r] = sd * 0.5025;
      for (let d = 0; d < D; d++) { const x = get(r, d), c = x >= 0 ? (x > t ? 3 : 2) : (x < -t ? 0 : 1), i = r * D + d; packed[i >> 2] |= c << ((i & 3) * 2); }
    } else {
      const L = bits === 8 ? 127 : 7, s = mx / L || 1; scales[r] = s;
      for (let d = 0; d < D; d++) { const v = Math.round(get(r, d) / s), i = r * D + d; if (bits === 8) q8[i] = v; else packed[i >> 1] |= (v & 15) << ((i & 1) * 4); }
    }
  }
  return { packed, scales };
}
// unpack a model file given its meta (meta.bits defaults to 8; rows = meta.words?.length ?? meta.N)
export function unpackModel(bytes, meta) {
  const rows = meta.words ? meta.words.length : meta.N;
  return unpackRows(bytes, rows * meta.D, meta.bits || 8);
}
