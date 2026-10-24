// Golden context scores from Node for the browser parity test (tests/test_demo.mjs).
import fs from 'fs';
import { ContextModel, unpackModel } from '../docs/vsa-core.js';
const meta = JSON.parse(fs.readFileSync('docs/ctx_meta.json')), m = new ContextModel(meta, unpackModel(fs.readFileSync('docs/ctx_vectors.bin'), meta));
const sents = ['I got a letter form my bank', 'he is taller then me', 'the weather was cold today', 'we will meet here on friday', 'they went there by train'];
const gold = [];
for (const s of sents) { const toks = ['<s>', ...s.split(' '), '</s>']; for (let i = 1; i < toks.length - 1; i++) if (m.setOf.has(toks[i])) gold.push({ toks, i, scores: m.scores(toks, i).map(v => +v.toFixed(5)) }); }
fs.mkdirSync('data', { recursive: true }); fs.writeFileSync('data/ctx_gold.json', JSON.stringify(gold)); console.log('gold cases', gold.length);
