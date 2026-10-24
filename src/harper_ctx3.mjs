// Harper on the exact round-3 test instances (320 sets, same split, same corruption rule).
import fs from 'fs';
import { LocalLinter } from 'harper.js';
import { binary } from 'harper.js/binary';
const SETS = JSON.parse(fs.readFileSync('data/confusions.json'));
const conf = new Map(); SETS.forEach((s, k) => s.words.forEach(w => conf.set(w, k)));
function mulberry32(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const tok = s => s.toLowerCase().match(/[a-z]+/g) || [];
const rnd = mulberry32(1);
const raw = fs.readFileSync('data/big.txt', 'utf8').replace(/\s+/g, ' ');
let sents = raw.split(/(?<=[.!?])\s+(?=[A-Z"])/).filter(s => s.length > 20 && s.length < 400);
for (let i = sents.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [sents[i], sents[j]] = [sents[j], sents[i]]; }
const nTr = Math.floor(sents.length * 0.85), nDev = Math.floor(sents.length * 0.05);
const test = sents.slice(nTr + nDev);
const per = new Map(); const I = [];
for (const s of test) { const t = ['<s>', ...tok(s), '</s>'];
  for (let i = 1; i < t.length - 1; i++) if (conf.has(t[i])) { const k = t[i]; if ((per.get(k) || 0) >= 30) continue; per.set(k, (per.get(k) || 0) + 1); I.push({ s, i, w: k, cands: SETS[conf.get(k)].words }); } }
const linter = new LocalLinter({ binary }); await linter.setup();
const spanOf = (s, i) => { const re = /[A-Za-z]+/g; let m, k = 0; while ((m = re.exec(s))) { k++; if (k === i) return [m.index, m.index + m[0].length]; } return null; };
const hit = (L, a, b) => L.filter(l => { const sp = l.span(); return sp.start < b && sp.end > a && l.lint_kind() !== 'Readability' && sp.end - sp.start < 40; });
let n = 0, fix = 0, fa = 0;
for (const x of I) {
  const sp = spanOf(x.s, x.i); if (!sp) continue; n++;
  const orig = x.s.slice(sp[0], sp[1]); const wrong = x.cands[(x.cands.indexOf(x.w) + 1) % x.cands.length];
  const wc = orig[0] === orig[0].toUpperCase() ? wrong[0].toUpperCase() + wrong.slice(1) : wrong;
  const bad = x.s.slice(0, sp[0]) + wc + x.s.slice(sp[1]);
  const L1 = hit(await linter.lint(bad), sp[0], sp[0] + wc.length);
  if (L1.some(l => l.suggestions().some(s => s.get_replacement_text().toLowerCase() === x.w))) fix++;
  if (hit(await linter.lint(x.s), sp[0], sp[1]).length) fa++;
}
const r = { n, fixed: fix / n, falseAlarm: fa / n };
console.log(JSON.stringify(r)); fs.writeFileSync('results/harper_ctx3.json', JSON.stringify(r));
