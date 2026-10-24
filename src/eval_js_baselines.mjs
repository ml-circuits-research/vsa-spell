// Hunspell (nspell = same engine family as browser spellcheck) and Harper.
import fs from 'fs';
import nspell from 'nspell';
import dictionary from 'dictionary-en';
import { LocalLinter } from 'harper.js';
import { binary } from 'harper.js/binary';

const test = JSON.parse(fs.readFileSync('data/test_nonword.json'));
const which = process.argv[2] || 'both';
const LIMIT = +(process.argv[3] || test.length);
const sub = test.slice(0, LIMIT);
const res = {};

function score(name, getSug, ms) {
  let t1 = 0, t5 = 0; const fails = []; const t = performance.now();
  for (const [bad, good] of sub) {
    const s = getSug(bad);
    if (s[0] === good) t1++; else if (fails.length < 15) fails.push(`${bad}->${s[0]} (want ${good})`);
    if (s.slice(0, 5).includes(good)) t5++;
  }
  return { top1: t1 / sub.length, top5: t5 / sub.length, msPerWord: (performance.now() - t) / sub.length, n: sub.length, fails };
}

if (which !== 'harper') {
  const h = nspell(dictionary);
  res['Hunspell (nspell)'] = score('hunspell', w => h.suggest(w));
  console.log('Hunspell', JSON.stringify(res['Hunspell (nspell)']));
}
if (which !== 'hunspell') {
  const linter = new LocalLinter({ binary });
  await linter.setup();
  const sugs = [];
  let t = performance.now();
  for (const [bad] of sub) {
    const lints = await linter.lint(bad);
    const sp = lints.find(l => /spell/i.test(l.lint_kind())) || lints[0];
    sugs.push(sp ? sp.suggestions().map(s => s.get_replacement_text()) : []);
  }
  const ms = (performance.now() - t) / sub.length;
  let i = 0; const r = score('harper', () => sugs[i++]); r.msPerWord = ms;
  res['Harper'] = r;
  console.log('Harper', JSON.stringify(r));
}
fs.writeFileSync(`results/res_js_${which}.json`, JSON.stringify(res, null, 1));
