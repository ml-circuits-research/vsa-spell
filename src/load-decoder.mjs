// Node helper: load the shipped models from docs/ and build a SentenceDecoder.
//
//   import { loadDecoder } from './src/load-decoder.mjs';
//   const dec = await loadDecoder({ dictionary: ['open', 'the', 'file', …], examples: ['open the file', …] });
//   dec.decode('opne teh fiel').best.text   // → 'open the file'
//
// Options:
//   dictionary  'general' (default, 80k words) | array of words | array of [word, count]
//   examples    array of example sentences in your domain (optional)
//   weights     overrides for the scoring weights (see docs/DECODER.md)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SentenceDecoder } from '../docs/decoder.js';
import { ContextModel, Speller } from '../docs/vsa-core.js';
import { DECODER_WEIGHTS, DOMAIN_WEIGHTS } from '../docs/decoder-weights.js';

const DOCS = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'docs');
const bin = f => { const b = fs.readFileSync(path.join(DOCS, f)); return new Int8Array(b.buffer, b.byteOffset, b.length); };
const json = f => JSON.parse(fs.readFileSync(path.join(DOCS, f), 'utf8'));
let shared = null;
function models() {
  if (!shared) {
    const general = fs.readFileSync(path.join(DOCS, 'words.txt'), 'utf8').trim().split('\n').map(l => { const [w, c] = l.split(' '); return [w, +c]; });
    shared = {
      general, freq: new Map(general), grammar: json('grammar.json'),
      profiles: { meta: json('profiles_meta.json'), int8: bin('profiles.bin') },
      confusion: new ContextModel(json('ctx_meta.json'), bin('ctx_vectors.bin')),
    };
  }
  return shared;
}

export async function loadDecoder({ dictionary = 'general', examples = [], weights = {} } = {}) {
  const m = models();
  let dict, speller = null;
  if (dictionary === 'general') { dict = m.general; if (!m.speller) { m.speller = new Speller({ D: 1024 }); m.speller.build(m.general); } speller = m.speller; }
  else dict = dictionary.map(e => Array.isArray(e) ? e : [String(e), m.freq.get(String(e).toLowerCase()) || 1000]);
  const base = dictionary === 'general' || !examples.length ? DECODER_WEIGHTS : { ...DECODER_WEIGHTS, ...DOMAIN_WEIGHTS };
  return new SentenceDecoder({ dictionary: dict, examples, grammar: m.grammar, profiles: m.profiles, confusion: m.confusion, speller, weights: { ...base, ...weights } });
}
