// Sentence decoder: given a (possibly misspelled) sentence, return the most probable
// intended sentence over a configurable dictionary, plus alternatives.
//
//   score(w1..wn) = Σ channel(token_i → w_i)            how far the typed token is from w_i (VSA spelling)
//                 + λf Σ log-frequency(w_i)              word prior (general list or your dictionary counts)
//                 + λg · grammar(w1..wn)                 POS tag-bigram HMM, Viterbi (Brown corpus)
//                 + λp Σ profile(w_i | neighbours)       VSA word-context profiles (general text)
//                 + λd Σ domainProfile(w_i | neighbours) VSA profiles learned from your example sentences
//                 + λc Σ confusion(w_i | neighbours)     trained confusion-set classes (form/from …)
//
// Search: iterated conditional modes (exact grammar update with forward/backward Viterbi
// tables, local update for the context terms), then single-change variants for the n-best.
// Runs unchanged in the browser and in Node.
import { Speller, osa, mulberry32, hash32 } from './vsa-core.js';

export const DEFAULT_WEIGHTS = {
  freq: 0.6, grammar: 0.35, article: 4, profile: 2.0, domain: 3.0, confusion: 1.5,
  sim: 6.0, edit: 1.4, keepUnknown: -9, nearWord: -3.0, confusionSwap: -1.2, morph: -2.6, kNonword: 6, kNear: 3,
};

// ---------------------------------------------------------------- morphology
const IRREG = [
  ['be', 'is', 'are', 'am', 'was', 'were', 'been', 'being'], ['have', 'has', 'had', 'having'], ['do', 'does', 'did', 'done', 'doing'],
  ['go', 'goes', 'went', 'gone', 'going'], ['say', 'says', 'said'], ['make', 'makes', 'made'], ['take', 'takes', 'took', 'taken'],
  ['come', 'comes', 'came'], ['see', 'sees', 'saw', 'seen'], ['get', 'gets', 'got', 'gotten'], ['give', 'gives', 'gave', 'given'],
  ['know', 'knows', 'knew', 'known'], ['think', 'thinks', 'thought'], ['run', 'runs', 'ran'], ['write', 'writes', 'wrote', 'written'],
  ['a', 'an'], ['this', 'these'], ['that', 'those'], ['man', 'men'], ['woman', 'women'], ['child', 'children'], ['person', 'people'],
  ['much', 'many'], ['less', 'fewer'], ['its', 'it'], ['there', 'their'], ['your', 'you'],
];
const IRMAP = new Map(); IRREG.forEach(g => g.forEach(w => IRMAP.set(w, g)));
export function morphVariants(w) {
  const out = new Set(IRMAP.get(w) || []);
  const add = x => { if (x && x !== w) out.add(x); };
  add(w + 's'); add(w + 'es'); add(w + 'ed'); add(w + 'd'); add(w + 'ing');
  if (w.endsWith('s')) add(w.slice(0, -1)); if (w.endsWith('es')) add(w.slice(0, -2));
  if (w.endsWith('ies')) add(w.slice(0, -3) + 'y'); if (w.endsWith('y')) add(w.slice(0, -1) + 'ies');
  if (w.endsWith('ed')) { add(w.slice(0, -2)); add(w.slice(0, -1)); }
  if (w.endsWith('ing')) { add(w.slice(0, -3)); add(w.slice(0, -3) + 'e'); }
  if (w.endsWith('e')) add(w.slice(0, -1) + 'ing');
  out.delete(w); return [...out];
}

const ALPHA = 'abcdefghijklmnopqrstuvwxyz';
export function edits1(w) {
  const out = new Set();
  for (let i = 0; i <= w.length; i++) {
    if (i < w.length) out.add(w.slice(0, i) + w.slice(i + 1));
    if (i < w.length - 1) out.add(w.slice(0, i) + w[i + 1] + w[i] + w.slice(i + 2));
    for (const a of ALPHA) { out.add(w.slice(0, i) + a + w.slice(i)); if (i < w.length) out.add(w.slice(0, i) + a + w.slice(i + 1)); }
  }
  out.delete(w); return [...out];
}
// "a" vs "an" by the next word's first sound (spelling approximation with common exceptions)
const AN_CONS = /^(one|once|uni(?!n)|use|usu|ure|uro|eu|ewe|u[bcfgklmprstv][aeiou])/; // vowel letter, consonant sound
const AN_VOW = /^(hour|honest|honou?r|heir)/;                                        // consonant letter, vowel sound
export function articleOK(article, next) {
  if (!next) return true; const v = (/^[aeiou]/.test(next) && !AN_CONS.test(next)) || AN_VOW.test(next);
  return article === 'an' ? v : !v;
}

// --------------------------------------------------------------- the decoder
export class SentenceDecoder {
  // dictionary: [[word, count], …] (count optional); grammar: parsed grammar.json;
  // profiles: { meta, int8 } (optional); confusion: ContextModel (optional); examples: [sentence, …]
  constructor({ dictionary, grammar, profiles = null, confusion = null, examples = [], weights = {}, D = 1024, speller = null }) {
    this.w = { ...DEFAULT_WEIGHTS, ...weights };
    const counts = new Map();
    for (const e of dictionary) { const [w, c] = Array.isArray(e) ? e : [e, 0]; const lw = String(w).toLowerCase(); if (/^[a-z]+$/.test(lw)) counts.set(lw, Math.max(counts.get(lw) || 0, +c || 0)); }
    // examples add domain counts so words used in your examples rank higher
    const exTok = examples.map(s => (String(s).toLowerCase().match(/[a-z]+/g) || []));
    for (const t of exTok) for (const x of t) if (counts.has(x)) counts.set(x, counts.get(x) + 1);
    const minC = 1;
    const entries = [...counts].map(([w, c]) => [w, Math.max(c, minC)]).sort((a, b) => b[1] - a[1]);
    this.dict = new Set(entries.map(e => e[0]));
    const maxLog = Math.log(entries[0][1] + 1);
    this.logf = new Map(entries.map(([w, c]) => [w, Math.log(c + 1) / maxLog]));
    // spelling memory over exactly this dictionary (reuse a prebuilt speller if it matches)
    if (speller && speller.N === entries.length) this.speller = speller;
    else { this.speller = new Speller({ D }); this.speller.build(entries); }
    this.g = grammar; this.T = grammar.tags.length;
    this.trans = grammar.trans.map(r => Float64Array.from(r));
    this.lexCache = new Map();
    this.prof = profiles ? this._loadProfiles(profiles) : null;
    this.conf = confusion;
    this.domain = exTok.length ? this._trainDomain(exTok) : null;
  }

  // ----- grammar: emissions log P(t|w) - log P(t)
  emit(w, cap) {
    const key = w + (cap ? '^' : ''); let e = this.lexCache.get(key); if (e) return e;
    const T = this.T, g = this.g; e = new Float64Array(T).fill(-12);
    let dist = g.lex[w];
    if (!dist) { for (let L = Math.min(4, w.length - 1); L >= 1 && !dist; L--) dist = g.suf[w.slice(-L)]; }
    if (dist) for (const [t, lp] of dist) e[t] = lp - g.tagPrior[t];
    if (cap && !g.lex[w]) { const np = g.tags.indexOf('np'); e[np] = Math.max(e[np], -0.5 - g.tagPrior[np]); }
    if (this.lexCache.size > 20000) this.lexCache.clear(); this.lexCache.set(key, e); return e;
  }
  // Viterbi forward (fw[i][t] best score of prefix ending at i with tag t) and backward
  _viterbiTables(words, caps) {
    const n = words.length, T = this.T, tr = this.trans;
    const fw = Array.from({ length: n }, () => new Float64Array(T)), bw = Array.from({ length: n }, () => new Float64Array(T));
    for (let i = 0; i < n; i++) {
      const e = this.emit(words[i], caps[i]);
      for (let t = 1; t < T; t++) {
        let best = -1e18;
        if (i === 0) best = tr[0][t]; else { const p = fw[i - 1]; for (let s = 1; s < T; s++) { const v = p[s] + tr[s][t]; if (v > best) best = v; } }
        fw[i][t] = best + e[t];
      }
    }
    for (let i = n - 1; i >= 0; i--) {
      for (let t = 1; t < T; t++) {
        let best = -1e18;
        if (i === n - 1) best = tr[t][0]; else { const e = this.emit(words[i + 1], caps[i + 1]), nx = bw[i + 1]; for (let s = 1; s < T; s++) { const v = tr[t][s] + e[s] + nx[s]; if (v > best) best = v; } }
        bw[i][t] = best;
      }
    }
    return { fw, bw };
  }
  grammarScore(words, caps) { const { fw } = this._viterbiTables(words, caps); const n = words.length; let b = -1e18; for (let t = 1; t < this.T; t++) b = Math.max(b, fw[n - 1][t] + this.trans[t][0]); return b; }
  // grammar score of the sentence with position i replaced by c, from the tables of the current assignment
  _grammarAt(tabs, words, caps, i, c) {
    const T = this.T, tr = this.trans, e = this.emit(c, caps[i]); let best = -1e18;
    for (let t = 1; t < T; t++) {
      let left = -1e18;
      if (i === 0) left = tr[0][t]; else { const p = tabs.fw[i - 1]; for (let s = 1; s < T; s++) { const v = p[s] + tr[s][t]; if (v > left) left = v; } }
      const v = left + e[t] + tabs.bw[i][t]; if (v > best) best = v;
    }
    return best;
  }

  // ----- VSA context profiles (general text)
  _loadProfiles({ meta, int8, words }) {
    const row = new Map((meta.words || words).map((w, k) => [w, k])); const POS = []; for (let p = -meta.W; p <= meta.W; p++) if (p) POS.push(p);
    return { ...meta, row, Q: int8, POS, SH: POS.map(p => ((p + meta.shiftOff) * meta.shiftMul) % meta.D), PW: POS.map(p => 1 / Math.abs(p)), iv: new Map() };
  }
  _iv(P, w) {
    let v = P.iv.get(w); if (v) return v;
    const r = mulberry32(hash32(w, P.seed)); v = new Int32Array(2 * P.K);
    for (let j = 0; j < P.K; j++) { v[2 * j] = Math.floor(r() / 4294967296 * P.D); v[2 * j + 1] = r() / 4294967296 < 0.5 ? -1 : 1; }
    if (P.iv.size > 50000) P.iv.clear(); P.iv.set(w, v); return v;
  }
  // cosine-like score of word words[i] against its neighbours (0 if the word has no profile)
  _profileScore(P, words, i, rowOf, getRow) {
    const r = rowOf(words[i]); if (r === undefined) return 0;
    let s = 0, qn = 0;
    for (let k = 0; k < P.POS.length; k++) {
      const j = i + P.POS[k]; const c = j < 0 ? '<s>' : j >= words.length ? '</s>' : words[j];
      const v = this._iv(P, c), sh = P.SH[k], w = P.PW[k];
      for (let q = 0; q < P.K; q++) s += w * v[2 * q + 1] * getRow(r, (v[2 * q] + sh) % P.D);
      qn += w * w * P.K;
    }
    return s / Math.sqrt(qn);
  }
  _prof(words, i) { const P = this.prof; if (!P) return 0; return this._profileScore(P, words, i, w => P.row.get(w), (r, d) => P.Q[r * P.D + d] * P.scales[r]); }
  _trainDomain(exTok) {
    const P = { D: 512, K: 8, W: 4, seed: 12345, shiftMul: 7919, shiftOff: 16 }; const POS = []; for (let p = -P.W; p <= P.W; p++) if (p) POS.push(p);
    Object.assign(P, { POS, SH: POS.map(p => ((p + 16) * 7919) % P.D), PW: POS.map(p => 1 / Math.abs(p)), iv: new Map(), rows: new Map() });
    for (const t of exTok) for (let i = 0; i < t.length; i++) {
      if (!this.dict.has(t[i])) continue; let row = P.rows.get(t[i]); if (!row) { row = new Float32Array(P.D); P.rows.set(t[i], row); }
      for (let k = 0; k < POS.length; k++) { const j = i + POS[k]; const c = j < 0 ? '<s>' : j >= t.length ? '</s>' : t[j]; const v = this._iv(P, c); for (let q = 0; q < P.K; q++) row[(v[2 * q] + P.SH[k]) % P.D] += P.PW[k] * v[2 * q + 1]; }
    }
    for (const row of P.rows.values()) { let n = 0; for (const x of row) n += x * x; n = Math.sqrt(n) || 1; for (let d = 0; d < row.length; d++) row[d] /= n; }
    P.list = [...P.rows.keys()]; return P;
  }
  _dom(words, i) { const P = this.domain; if (!P) return 0; return this._profileScore(P, words, i, w => (P.rows.has(w) ? w : undefined), (r, d) => P.rows.get(r)[d]); }
  // confusion classes: score relative to the mean of the set (0 when not in an active set)
  _conf(words, i) {
    const C = this.conf; if (!C) return 0; const w = words[i]; const k = C.setOf.get(w); if (k === undefined || !C.on.has(k)) return 0;
    const set = C.sets[k]; if (!set.every(x => this.dict.has(x))) return 0;
    const toks = ['<s>', ...words, '</s>']; const sc = C.scores(toks, i + 1); const mean = sc.reduce((a, b) => a + b, 0) / sc.length;
    return sc[set.indexOf(w)] - mean;
  }
  // context terms that change when position i changes: positions within the window
  _art(words, j) { const w = words[j]; return (w === 'a' || w === 'an') && !articleOK(w, words[j + 1]) ? -1 : 0; }
  _localContext(words, i) {
    const W = this.w; let s = 0;
    for (let j = Math.max(0, i - 1); j <= i; j++) s += W.article * this._art(words, j);
    for (let j = Math.max(0, i - 4); j <= Math.min(words.length - 1, i + 4); j++)
      s += W.profile * this._prof(words, j) + W.domain * this._dom(words, j) + W.confusion * this._conf(words, j);
    return s;
  }

  // ----- candidates for one token
  candidates(tok, isCap, midSentence) {
    const W = this.w, x = tok.toLowerCase(), out = new Map();
    const put = (w, ch, why) => { if (!this.dict.has(w)) return; const o = out.get(w); if (!o || o.ch < ch) out.set(w, { w, ch, why }); };
    const sp = this.speller;
    const simScore = (c) => { // channel from VSA similarity + edit distance (same features as the speller)
      const i = sp.index.get(c); const ed = osa(x, c);
      return { ch: W.sim * (sp._so[i] + 0.5 * sp._sp[i] - 1.5) - W.edit * ed, ed };
    };
    if (this.dict.has(x)) {
      put(x, 0, 'as typed');
      if (this.conf) { const k = this.conf.setOf.get(x); if (k !== undefined) for (const c of this.conf.sets[k]) if (c !== x) put(c, W.confusionSwap, 'often confused'); }
      for (const m of morphVariants(x)) put(m, W.morph, 'other form');
      // dictionary words one edit away (real-word typos such as stat→start), most frequent first
      const near = edits1(x).filter(c => this.dict.has(c)).sort((a, b) => (this.logf.get(b) || 0) - (this.logf.get(a) || 0)).slice(0, W.kNear);
      for (const c of near) put(c, W.nearWord - W.edit, 'one letter off');
    } else {
      const sug = sp.suggest(x, { wPhon: 0.5, wFreq: 0.8, wLen: 0, wEd: 0.35 }, W.kNonword);
      for (const c of sug) { const { ch } = simScore(c); put(c, ch, 'spelling'); }
      // an unknown word can also be a name or an id: keep it, at a cost
      out.set(x, { w: x, ch: W.keepUnknown + (isCap && midSentence ? 6 : 0), why: 'keep as typed', unknown: true });
    }
    return [...out.values()];
  }

  // ----- decode a sentence
  decode(text, { nbest = 5, sweeps = 3 } = {}) {
    const W = this.w;
    const re = /[A-Za-z]+/g; const toks = []; let m;
    while ((m = re.exec(text))) {
      const apo = /['’]/.test(text[m.index - 1] || '') || /['’]/.test(text[m.index + m[0].length] || '');
      toks.push({ raw: m[0], start: m.index, end: m.index + m[0].length, fixed: apo });
    }
    if (!toks.length) return { text, best: { text, words: [], score: 0 }, alternatives: [], positions: [] };
    const n = toks.length;
    const isSentStart = k => { let j = toks[k].start - 1; while (j >= 0 && /[\s"“'(]/.test(text[j])) j--; return j < 0 || /[.!?\n:]/.test(text[j]); };
    const caps = toks.map(t => /^[A-Z]/.test(t.raw));
    const cands = toks.map((t, k) => t.fixed ? [{ w: t.raw.toLowerCase(), ch: 0, why: 'part of a contraction', unknown: !this.dict.has(t.raw.toLowerCase()) }]
      : this.candidates(t.raw, caps[k], !isSentStart(k)));
    const capsG = toks.map((t, k) => caps[k] && !isSentStart(k));
    const fq = c => (c.unknown ? -0.2 : W.freq * (this.logf.get(c.w) || 0));
    // initial assignment: best channel + frequency per position
    let words = cands.map(cs => cs.reduce((a, b) => (b.ch + fq(b) > a.ch + fq(a) ? b : a)).w);
    const chOf = cands.map(cs => new Map(cs.map(c => [c.w, c])));
    const unaryOf = (i, w) => { const c = chOf[i].get(w); return c.ch + fq(c); };
    const total = ws => {
      let s = W.grammar * this.grammarScore(ws, capsG);
      for (let i = 0; i < n; i++) s += W.article * this._art(ws, i) + unaryOf(i, ws[i]) + W.profile * this._prof(ws, i) + W.domain * this._dom(ws, i) + W.confusion * this._conf(ws, i);
      return s;
    };
    const seen = new Map(); const remember = ws => { const k = ws.join(' '); if (!seen.has(k)) seen.set(k, { words: ws.slice(), score: total(ws) }); return seen.get(k); };
    // ICM sweeps
    for (let sweep = 0; sweep < sweeps; sweep++) {
      let changed = false;
      for (let i = 0; i < n; i++) {
        if (cands[i].length < 2) continue;
        const tabs = this._viterbiTables(words, capsG);
        let bestW = words[i], bestS = -1e18;
        for (const c of cands[i]) {
          const trial = words.slice(); trial[i] = c.w;
          const s = unaryOf(i, c.w) + W.grammar * this._grammarAt(tabs, words, capsG, i, c.w) + this._localContext(trial, i);
          if (s > bestS) { bestS = s; bestW = c.w; }
        }
        if (bestW !== words[i]) { words[i] = bestW; changed = true; }
      }
      remember(words);
      if (!changed) break;
    }
    const best = remember(words);
    // n-best: single changes of the best assignment
    for (let i = 0; i < n; i++) for (const c of cands[i]) { if (c.w === words[i]) continue; const ws = words.slice(); ws[i] = c.w; remember(ws); }
    const ranked = [...seen.values()].sort((a, b) => b.score - a.score).slice(0, nbest);
    const render = ws => {
      let out = '', pos = 0;
      ws.forEach((w, i) => {
        const t = toks[i]; let r = w;
        if (t.raw === t.raw.toUpperCase() && t.raw.length > 1) r = w.toUpperCase(); else if (caps[i]) r = w[0].toUpperCase() + w.slice(1);
        if (chOf[i].get(w)?.unknown) r = t.raw;
        out += text.slice(pos, t.start) + r; pos = t.end;
      });
      return out + text.slice(pos);
    };
    const positions = toks.map((t, i) => ({
      raw: t.raw, start: t.start, end: t.end, chosen: best.words[i], changed: best.words[i] !== t.raw.toLowerCase(),
      why: chOf[i].get(best.words[i]).why,
      candidates: cands[i].map(c => ({ w: c.w, channel: +c.ch.toFixed(2), why: c.why })).sort((a, b) => b.channel - a.channel).slice(0, 6),
    }));
    return {
      text, best: { text: render(best.words), words: best.words, score: best.score },
      alternatives: ranked.map(r => ({ text: render(r.words), words: r.words, score: +r.score.toFixed(2), delta: +(r.score - best.score).toFixed(2) })),
      positions,
    };
  }
}
