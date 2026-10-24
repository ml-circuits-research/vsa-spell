// Agent-command domain: a small command grammar used to (1) build the "Agent commands"
// preset for the demo (dictionary + example commands) and (2) evaluate the decoder with
// a restricted dictionary. Example commands and test commands are disjoint.
import fs from 'fs';
import { mulberry32 } from '../docs/vsa-core.js';

const r = mulberry32(77); const rf = () => r() / 4294967296; const pick = a => a[Math.floor(rf() * a.length)];
const L = {
  NAME: ['reports', 'invoices', 'budget', 'photos', 'drafts', 'notes', 'travel', 'project', 'backup', 'archive', 'contracts', 'receipts', 'slides', 'music', 'taxes'],
  OBJ: ['folder', 'file', 'document', 'spreadsheet', 'note', 'playlist', 'presentation'],
  LOC: ['documents', 'desktop', 'downloads', 'the shared drive', 'the cloud', 'my home folder'],
  PERSON: ['john', 'maria', 'the team', 'my manager', 'alex', 'sara', 'the client', 'everyone'],
  TOPIC: ['the budget', 'the project', 'the release', 'the invoice', 'the meeting', 'the contract', 'the schedule'],
  DAY: ['today', 'tomorrow', 'on monday', 'on tuesday', 'on wednesday', 'on thursday', 'on friday', 'next week', 'this evening'],
  TIME: ['noon', 'nine am', 'ten am', 'eleven am', 'two pm', 'three pm', 'four pm', 'five pm', 'six pm'],
  TASK: ['call mom', 'buy milk', 'pay the rent', 'water the plants', 'send the report', 'book a table', 'renew the license', 'check the server'],
  NUM: ['five', 'ten', 'fifteen', 'twenty', 'thirty', 'forty five'],
  DEVICE: ['lights', 'heating', 'fan', 'air conditioning', 'television'],
  ROOM: ['kitchen', 'bedroom', 'living room', 'office', 'bathroom'],
  CITY: ['paris', 'london', 'zurich', 'berlin', 'vienna', 'madrid'],
};
const T = [
  () => `${pick(['create', 'make'])} a new ${pick(L.OBJ)} named ${pick(L.NAME)}${rf() < 0.5 ? ' in ' + pick(L.LOC) : ''}`,
  () => `${pick(['delete', 'remove'])} the ${pick(L.OBJ)} named ${pick(L.NAME)}${rf() < 0.4 ? ' from ' + pick(L.LOC) : ''}`,
  () => `${pick(['open', 'show'])} ${pick(['the', 'my'])} ${pick(L.NAME)} ${pick(L.OBJ)}`,
  () => `${pick(['move', 'copy'])} the ${pick(['file', 'document', 'folder'])} ${pick(L.NAME)} to ${pick(L.LOC)}`,
  () => `rename the ${pick(L.OBJ)} ${pick(L.NAME)} to ${pick(L.NAME)}`,
  () => `${pick(['send', 'forward'])} an email to ${pick(L.PERSON)} about ${pick(L.TOPIC)}`,
  () => `schedule a meeting with ${pick(L.PERSON)} ${pick(L.DAY)} at ${pick(L.TIME)}`,
  () => `remind me to ${pick(L.TASK)} ${pick(L.DAY)}${rf() < 0.5 ? ' at ' + pick(L.TIME) : ''}`,
  () => rf() < 0.5 ? `set a timer for ${pick(L.NUM)} minutes` : `set an alarm for ${pick(L.TIME)} ${pick(L.DAY)}`,
  () => `turn ${pick(['on', 'off'])} the ${pick(L.DEVICE)} in the ${pick(L.ROOM)}`,
  () => `play ${pick(['some music', 'the playlist ' + pick(L.NAME), 'my favorite songs'])}`,
  () => `what is the weather ${pick([pick(L.DAY), 'in ' + pick(L.CITY)])}`,
  () => `${pick(['search', 'find'])} ${pick(['files', 'emails', 'messages', 'notes'])} about ${pick(L.TOPIC)}`,
  () => `list all ${pick(['files', 'meetings', 'reminders', 'emails'])} ${pick(L.DAY)}`,
  () => `cancel the meeting with ${pick(L.PERSON)} ${pick(L.DAY)}`,
  () => `share the ${pick(L.OBJ)} ${pick(L.NAME)} with ${pick(L.PERSON)}`,
];
const all = new Set(); while (all.size < 1400) all.add(pick(T)());
const cmds = [...all];
const examples = cmds.slice(0, 150), dev = cmds.slice(150, 400), test = cmds.slice(400, 1000);
// dictionary: every word the grammar can produce (and nothing else)
const vocab = new Set(); for (const g of Object.values(L)) for (const p of g) for (const w of p.split(' ')) vocab.add(w);
for (const t of T) for (let i = 0; i < 300; i++) for (const w of t().split(' ')) vocab.add(w);
const general = new Map(fs.readFileSync('docs/words.txt', 'utf8').trim().split('\n').map(l => { const [w, c] = l.split(' '); return [w, +c]; }));
const dictionary = [...vocab].sort().map(w => [w, general.get(w) || 1000]);
fs.mkdirSync('docs/presets', { recursive: true });
fs.writeFileSync('docs/presets/agent-commands.json', JSON.stringify({ name: 'Agent commands', dictionary, examples }, null, 0));
fs.writeFileSync('data/commands_eval.json', JSON.stringify({ dev, test }));
console.log(`vocab ${dictionary.length}, examples ${examples.length}, dev ${dev.length}, test ${test.length}`);
console.log(examples.slice(0, 5));
