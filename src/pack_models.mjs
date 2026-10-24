// Pack the confusion-set classes for shipping: int8 (from context3 --export, kept in data/)
// re-quantised to 4 bits. Measured on 9,198 test instances: right word 90.48% -> 90.06%,
// errors fixed at the Careful threshold 62.0% -> 61.7% (src/quant_eval.mjs).
import fs from 'fs';
import { quantizeRows } from '../docs/vsa-core.js';
const BITS = +(process.argv[2] || 4);
const meta = JSON.parse(fs.readFileSync('data/ctx_meta_int8.json')), b = fs.readFileSync('data/ctx_int8.bin');
const q = new Int8Array(b.buffer, b.byteOffset, b.length), D = meta.D, rows = meta.words.length;
const { packed, scales } = quantizeRows((r, d) => q[r * D + d] * meta.scales[r], rows, D, BITS);
fs.writeFileSync('docs/ctx_vectors.bin', Buffer.from(packed.buffer, packed.byteOffset, packed.byteLength));
fs.writeFileSync('docs/ctx_meta.json', JSON.stringify({ ...meta, bits: BITS, scales: Array.from(scales, x => +x.toPrecision(6)) }));
console.log(`classes: ${rows} x ${D} at ${BITS} bits -> ${(packed.byteLength / 1e6).toFixed(2)} MB`);
