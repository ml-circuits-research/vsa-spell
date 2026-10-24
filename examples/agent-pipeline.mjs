// Put the decoder in front of a command parser: typos are fixed against the agent's
// own vocabulary before the parser sees the text.  Run: node examples/agent-pipeline.mjs
import fs from 'node:fs';
import { loadDecoder } from '../src/load-decoder.mjs';

const preset = JSON.parse(fs.readFileSync(new URL('../docs/presets/agent-commands.json', import.meta.url)));
const decoder = await loadDecoder({ dictionary: preset.dictionary, examples: preset.examples });

// a toy semantic parser that only understands well-formed commands
function parse(text) {
  let m;
  if ((m = text.match(/^(?:create|make) a new (\w+) named (\w+)(?: in (.+))?$/))) return { intent: 'create', type: m[1], name: m[2], location: m[3] ?? null };
  if ((m = text.match(/^remind me to (.+?) (today|tomorrow|on \w+|next week|this evening)(?: at (.+))?$/))) return { intent: 'reminder', task: m[1], day: m[2], time: m[3] ?? null };
  if ((m = text.match(/^turn (on|off) the (.+) in the (.+)$/))) return { intent: 'device', state: m[1], device: m[2], room: m[3] };
  return { intent: 'unknown' };
}

for (const input of ['creat a new fodler nmaed reports in documnets', 'remnd me to wtaer the plnats tomorow at five pm', 'trun of the lihgts in the kitchn']) {
  const r = decoder.decode(input);
  console.log(`typed:   ${input}\nfixed:   ${r.best.text}\nparsed:  ${JSON.stringify(parse(r.best.text))}\nwithout: ${JSON.stringify(parse(input))}\n`);
}
