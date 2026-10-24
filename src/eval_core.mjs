// Tune (dev) + evaluate (test) the shipped browser Speller, with the exact code that runs in the page.
import fs from 'fs';
import { Speller, osa } from '../web/vsa-core.js';
const D = +(process.argv[2] || 1024);
const entries = fs.readFileSync('data/en-80k.txt', 'utf8').trim().split('\n').map(l => { const [w, c] = l.split(' '); return [w, +c]; });
let t = performance.now(); const sp = new Speller({ D }); sp.build(entries);
const buildS = (performance.now() - t) / 1000;
console.log(`D=${D} build ${buildS.toFixed(2)}s mem ${(sp.bytes() / 1e6).toFixed(1)}MB`);
const dev = JSON.parse(fs.readFileSync('data/dev_nonword.json')), test = JSON.parse(fs.readFileSync('data/test_nonword.json'));
const grid = [];
for (const wPhon of [0.15, 0.25, 0.5]) for (const wFreq of [0.3, 0.4, 0.6, 0.8]) for (const wLen of [0, 0.05]) for (const wEd of [0, 0.2, 0.35])
  grid.push({ g: { wPhon, wFreq, wLen, wEd }, t1: 0 });
const W = sp.W, o = new Uint32Array(W), p = new Uint32Array(W);
for (const [bad, good] of dev) {
  sp.encode(bad, [2, 3], '', o); sp.encode(sp.phoneKey(bad), [2, 3], 'P', p);
  sp.sims(o, sp.O, sp._so); sp.sims(p, sp.P, sp._sp);
  const pool = new Set([...sp.top(sp._so, 60), ...sp.top(sp._sp, 60)]); for (let i = 0; i < 200; i++) pool.add(i); pool.delete(-1);
  const c = [...pool].map(i => ({ w: sp.words[i], so: sp._so[i], spp: sp._sp[i], f: sp.logf[i], dl: Math.abs(sp.words[i].length - bad.length), ed: osa(bad, sp.words[i]) }));
  for (const x of grid) { let best = -1e9, bw = null; for (const k of c) { const s = k.so + x.g.wPhon * k.spp + x.g.wFreq * k.f - x.g.wLen * k.dl - x.g.wEd * k.ed; if (s > best) { best = s; bw = k.w; } } if (bw === good) x.t1++; }
}
const pure = grid.filter(x => !x.g.wEd).sort((a, b) => b.t1 - a.t1)[0], hyb = grid.filter(x => x.g.wEd).sort((a, b) => b.t1 - a.t1)[0];
const out = { D, buildS, memMB: sp.bytes() / 1e6 };
for (const [name, x] of [['pure', pure], ['hybrid', hyb]]) {
  let t1 = 0, t5 = 0; t = performance.now();
  for (const [bad, good] of test) { const s = sp.suggest(bad, x.g); if (s[0] === good) t1++; if (s.includes(good)) t5++; }
  out[name] = { params: x.g, dev: x.t1 / dev.length, top1: t1 / test.length, top5: t5 / test.length, msPerWord: (performance.now() - t) / test.length };
  console.log(name, JSON.stringify(out[name]));
}
fs.writeFileSync(`data/res_core_${D}.json`, JSON.stringify(out, null, 1));
fs.writeFileSync(`results/spell_params_${D}.json`, JSON.stringify({ D, ...out.hybrid.params }));
