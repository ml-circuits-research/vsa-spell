// Statistical grammar for the sentence decoder, learned from the tagged Brown corpus:
//  - a tag set (Brown tags without title/headline/negation markers; it keeps agreement
//    distinctions such as pps "he" vs ppss "they", vbz "goes" vs vb "go", bez "is" vs ber "are"),
//  - tag bigram transitions log P(t_i | t_{i-1}) with sentence boundaries,
//  - a lexicon log P(tag | word) and a suffix model for words Brown never saw.
// Output: docs/grammar.json (small enough for the browser).
import fs from 'fs';
import path from 'path';

const DIR = 'corpus/brown';
const norm = t => {
  t = t.toLowerCase().replace(/^fw-/, '').replace(/-(tl|hl|nc)$/g, '').replace(/-(tl|hl|nc)$/g, '');
  return t;
};
const sentences = [];
for (const f of fs.readdirSync(DIR).filter(f => /^c[a-r]\d\d$/.test(f))) {
  for (const line of fs.readFileSync(path.join(DIR, f), 'utf8').split('\n')) {
    const toks = line.trim().split(/\s+/).filter(Boolean); if (!toks.length) continue;
    const out = [];
    for (const tk of toks) {
      const k = tk.lastIndexOf('/'); if (k < 0) continue;
      const word = tk.slice(0, k).toLowerCase(); let tag = norm(tk.slice(k + 1));
      const pieces = word.match(/[a-z]+/g); if (!pieces) continue;          // punctuation, numbers
      const [t1, t2] = tag.split('+');
      const neg = /\*$/.test(t1);
      const base = t1.replace(/\*$/, '');
      if (!base || /^[^a-z]/.test(base)) continue;
      out.push([pieces[0], base]);
      // clitic pieces from our tokenizer ("don't" -> don, t ; "i'm" -> i, m)
      for (let j = 1; j < pieces.length; j++) out.push([pieces[j], j === 1 && t2 ? t2.replace(/\*$/, '') : neg ? 'neg' : 'cl']);
    }
    if (out.length) sentences.push(out);
  }
}
// tag inventory (drop very rare tags into their closest frequent prefix)
const tagCount = new Map();
for (const s of sentences) for (const [, t] of s) tagCount.set(t, (tagCount.get(t) || 0) + 1);
const MINT = 30;
const keep = [...tagCount].filter(([, c]) => c >= MINT).map(([t]) => t);
const remap = t => keep.includes(t) ? t : (keep.find(k => t.startsWith(k)) || 'nn');
for (const s of sentences) for (const x of s) x[1] = remap(x[1]);
const tags = ['<s>', ...keep.sort()];
const T = tags.length, ti = new Map(tags.map((t, i) => [t, i]));

// transitions with add-k smoothing; tag index 0 = <s> doubles as </s>
const tr = Array.from({ length: T }, () => new Float64Array(T).fill(0.1));
const tc = new Float64Array(T);
const lex = new Map(); // word -> Map(tag -> count)
for (const s of sentences) {
  let prev = 0;
  for (const [w, t] of s) {
    const k = ti.get(t); tr[prev][k]++; tc[k]++; prev = k;
    let m = lex.get(w); if (!m) { m = new Map(); lex.set(w, m); } m.set(k, (m.get(k) || 0) + 1);
  }
  tr[prev][0]++;
}
const trans = tr.map(row => { const s = row.reduce((a, b) => a + b, 0); return Array.from(row, v => +Math.log(v / s).toFixed(3)); });
const total = tc.reduce((a, b) => a + b, 0);
const tagPrior = Array.from(tc, v => +Math.log((v + 1) / (total + T)).toFixed(3));

// suffix model from rare words (open-class behaviour), suffix lengths 1..4
const suf = new Map();
for (const [w, m] of lex) {
  const n = [...m.values()].reduce((a, b) => a + b, 0); if (n > 3) continue;
  for (let L = 1; L <= Math.min(4, w.length - 1); L++) {
    const sx = w.slice(-L); let sm = suf.get(sx); if (!sm) { sm = new Map(); suf.set(sx, sm); }
    for (const [k, c] of m) sm.set(k, (sm.get(k) || 0) + c);
  }
}
const toDist = (m, minTotal = 1, topK = 6) => {
  const n = [...m.values()].reduce((a, b) => a + b, 0); if (n < minTotal) return null;
  return [...m].sort((a, b) => b[1] - a[1]).slice(0, topK).map(([k, c]) => [k, +Math.log((c + 0.1) / (n + 0.1 * m.size)).toFixed(3)]);
};
// keep lexicon entries for words in the 80k dictionary (what the decoder can output)
const dict = new Set(fs.readFileSync('data/en-80k.txt', 'utf8').split('\n').map(l => l.split(' ')[0]));
const lexOut = {}; let nl = 0;
for (const [w, m] of lex) if (dict.has(w) || ['t', 'm', 's', 'd', 'll', 've', 're'].includes(w)) { const d = toDist(m); if (d) { lexOut[w] = d; nl++; } }
const sufOut = {};
for (const [sx, m] of suf) { const d = toDist(m, 20); if (d) sufOut[sx] = d; }
const out = { tags, trans, tagPrior, lex: lexOut, suf: sufOut, nSentences: sentences.length };
fs.writeFileSync('docs/grammar.json', JSON.stringify(out));
console.log(`sentences ${sentences.length}, tags ${T}, lexicon ${nl} words, suffixes ${Object.keys(sufOut).length}, ${(fs.statSync('docs/grammar.json').size / 1e6).toFixed(2)} MB`);
console.log(tags.join(' '));
