// Build confusion sets: a curated list of commonly confused English words
// plus automatically mined pairs (frequent edit-distance-1 typos and
// homophones with identical Double Metaphone codes).
import fs from 'fs';
import { doubleMetaphone } from 'double-metaphone';
import { osa } from '../docs/vsa-core.js';

const CURATED = `
their there|then than|to too two|affect effect|accept except|lose loose loss|form from|quite quiet quit|weather whether
principal principle|advice advise|piece peace|passed past|lead led|brake break|hear here|know no now|buy by bye
later latter|through threw thorough though|were where wear|council counsel|complement compliment|sight site cite|dose does
adapt adopt|aisle isle|allusion illusion|altar alter|ate eight|bare bear|board bored|bought brought|breath breathe
capital capitol|cell sell|cent scent sent|choose chose|cloths clothes|coarse course|conscience conscious|dairy diary
dear deer|desert dessert|die dye|discreet discrete|dual duel|elicit illicit|emigrate immigrate|ensure insure|fair fare
farther further|flour flower|for four|grate great|hair hare|heal heel|hole whole|hour our|knew new|knight night
lay lie|lightening lightning|made maid|mail male|meat meet|miner minor|moral morale|one won|pair pear|pale pail
patience patients|pause paws|peak peek|personal personnel|plain plane|pole poll|pray prey|precede proceed|presence presents
rain reign rein|raise rays|right write rite|road rode|role roll|sale sail|scene seen|sea see|seam seem|sew so sow
sole soul|some sum|son sun|stair stare|stationary stationery|steal steel|tail tale|waist waste|wait weight|wander wonder
way weigh|weak week|which witch|wood would|of off|angel angle|casual causal|trail trial|filed field|marital martial
sacred scared|tough though|thought though|united untied|bean been|cloth clothe|desert desserts|lessen lesson|medal meddle
mourning morning|naval navel|past passed|plaque plague|quote quota|rational rationale|vain vein vane|wail whale|wary weary
access excess|addition edition|advance advanced|affected effected|all awl|allowed aloud|bazaar bizarre|beside besides
born borne|bridal bridle|cereal serial|chord cord|complacent complaisant|confidant confident|credible creditable|decent descent
device devise|dew due do|dinner diner|elude allude|eminent imminent|envelop envelope|everyone every|expand expend|faze phase
feat feet|flair flare|forth fourth|foul fowl|gorilla guerrilla|hanger hangar|hoard horde|idle idol|incite insight
its it|jewel joule|loath loathe|manner manor|metal medal|mind mined|morale moral|overdo overdue|pedal peddle|pier peer
pore pour poor|pray prey|prophecy prophesy|rap wrap|real reel|reality realty|residence residents|ring wring|sweet suite
taught taut|team teem|tide tied|vary very|ware wear|wave waive|whine wine|whose who|yoke yolk|your you
`.trim().split(/[|\n]/).map(s => s.trim().split(/\s+/)).filter(s => s.length > 1);

// corpus counts (training corpus only — never test text)
const counts = new Map();
for (const l of fs.readFileSync('data/train_corpus.txt', 'utf8').split('\n'))
  for (const w of (l.toLowerCase().match(/[a-z]+/g) || [])) counts.set(w, (counts.get(w) || 0) + 1);
const c = w => counts.get(w) || 0;
const MIN = 150; // need enough training occurrences per word

// dedupe curated (a word may belong to only one set: keep first)
const used = new Set(); const sets = [];
for (const s of CURATED) {
  const f = [...new Set(s)].filter(w => !used.has(w) && c(w) >= 30);
  if (f.length >= 2) { f.forEach(w => used.add(w)); sets.push({ words: f, src: 'curated' }); }
}
const nCur = sets.length;

// auto-mined: frequent words, edit-distance-1 (subst/insert/delete/transpose) or same metaphone
const rank = new Map(fs.readFileSync('data/en-80k.txt', 'utf8').trim().split('\n').map((l, i) => [l.split(' ')[0], i]));
const vocab = [...counts].filter(([w, n]) => n >= MIN && w.length >= 3 && w.length <= 12 && (rank.get(w) ?? 1e9) < 30000).map(([w]) => w);
const vs = new Set(vocab);
const STOP = new Set('the and you she her his him our out not but for are was has had can all one any its'.split(' '));
const cand = [];
const alpha = 'abcdefghijklmnopqrstuvwxyz';
for (const w of vocab) {
  const ed1 = new Set();
  for (let i = 0; i <= w.length; i++) {
    if (i < w.length) ed1.add(w.slice(0, i) + w.slice(i + 1));                       // delete
    if (i < w.length - 1) ed1.add(w.slice(0, i) + w[i + 1] + w[i] + w.slice(i + 2));  // transpose
    for (const a of alpha) { ed1.add(w.slice(0, i) + a + w.slice(i)); if (i < w.length) ed1.add(w.slice(0, i) + a + w.slice(i + 1)); }
  }
  for (const v of ed1) if (v !== w && vs.has(v) && w < v) cand.push([w, v, 'typo']);
}
const meta = new Map();
for (const w of vocab) { const k = doubleMetaphone(w)[0]; if (!meta.has(k)) meta.set(k, []); meta.get(k).push(w); }
for (const ws of meta.values()) for (let i = 0; i < ws.length; i++) for (let j = i + 1; j < ws.length; j++) {
  const [a, b] = [ws[i], ws[j]].sort();
  if (Math.abs(a.length - b.length) <= 1 && a[0] === b[0] && osa(a, b) <= 2) cand.push([a, b, 'homophone']);
}
// skip morphological variants (plural/past/-ly…) — those are grammar, not confusions
const morph = (a, b) => { const [s, l] = a.length < b.length ? [a, b] : [b, a]; return l.startsWith(s) && /^(s|es|ed|d|ly|er|ing|y|e|n)$/.test(l.slice(s.length)); };
const seen = new Set();
const scored = cand.filter(([a, b]) => !morph(a, b) && !STOP.has(a) && !STOP.has(b) && !used.has(a) && !used.has(b))
  .filter(([a, b]) => { const k = a + ' ' + b; if (seen.has(k)) return false; seen.add(k); return true; })
  .map(([a, b, src]) => ({ words: [a, b], src, score: Math.min(c(a), c(b)) }))
  .sort((x, y) => y.score - x.score);
const TARGET = +(process.argv[2] || 320);
for (const s of scored) {
  if (sets.length >= TARGET) break;
  if (s.words.some(w => used.has(w))) continue;
  s.words.forEach(w => used.add(w)); sets.push({ words: s.words, src: s.src });
}
fs.writeFileSync('data/confusions.json', JSON.stringify(sets));
const by = sets.reduce((a, s) => (a[s.src] = (a[s.src] || 0) + 1, a), {});
console.log(`sets=${sets.length} curated=${nCur}`, by, 'words', sets.reduce((a, s) => a + s.words.length, 0));
console.log('auto sample:', sets.slice(nCur, nCur + 40).map(s => s.words.join('/')).join(', '));
