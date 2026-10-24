# Sentence decoder

`docs/decoder.js` turns a typed sentence into its most probable intended sentence over a
dictionary you choose, and returns alternatives and a per-word explanation. It is meant to sit in
front of a semantic parser, an intent classifier or an agent: the parser only ever sees words it
knows.

## Use it

### In Node

```js
import { loadDecoder } from './src/load-decoder.mjs';

// general English (80,000 words)
const en = await loadDecoder();
en.decode('This are a sentence with a error in it').best.text;  // 'This is a sentence with an error in it'
en.decode('I recieved you letter form the comittee').best.text;  // 'I received your letter from the committee'

// your own vocabulary, plus example sentences that teach word order
const dec = await loadDecoder({
  dictionary: ['add', 'remove', 'show', 'milk', 'eggs', 'to', 'from', 'the', 'my', 'list', 'shopping'],
  examples: ['add milk to the shopping list', 'remove eggs from the list', 'show my list'],
});
const r = dec.decode('ad milk to the shoping lsit');
r.best.text            // 'add milk to the shopping list'
r.alternatives         // [{ text, words, score, delta }, …]  best first, delta = score − best
r.positions            // per word: { raw, chosen, changed, why, candidates: [{ w, channel, why }] }
```

`dictionary` accepts `'general'`, an array of words, or an array of `[word, count]` pairs. Counts
weight words; without them, a word's frequency comes from the general list (or 1,000 if it is not
there). Every word in `examples` that is in the dictionary also gets its count raised.

`npm run example:agent` runs `examples/agent-pipeline.mjs`, where a toy regex parser fails on all
three raw commands and parses all three after the decoder.

### In the browser

Open the page's **Interpret a sentence** tab, or import the module directly:

```js
import { SentenceDecoder } from './decoder.js';
import { ContextModel, unpackModel } from './vsa-core.js';
import { DECODER_WEIGHTS, DOMAIN_WEIGHTS } from './decoder-weights.js';

const dec = new SentenceDecoder({
  dictionary,                 // [[word, count], …]
  examples,                   // ['open the file', …]  (optional)
  grammar,                    // await (await fetch('grammar.json')).json()
  profiles: { meta, int8: unpackModel(profilesBin, meta), words },  // words = first meta.N lines of words.txt (optional)
  confusion,                  // new ContextModel(ctxMeta, unpackModel(ctxBin, ctxMeta))   (optional)
  weights: { ...DECODER_WEIGHTS, ...(examples.length ? DOMAIN_WEIGHTS : {}) },
});
```

`docs/worker.js` shows the full loading sequence.

## How it scores a sentence

For typed tokens `x1 … xn` and a candidate sentence `w1 … wn`:

```
score = Σ channel(xi → wi)                  spelling: VSA similarity (letters + Double Metaphone) and edit distance
      + freq     · Σ log-frequency(wi)
      + grammar  · Viterbi log-probability   part-of-speech tag bigrams and P(tag | word), Brown corpus
      + article  · [a/an agrees with the next word]
      + profile  · Σ context(wi | wi±1..4)   general word-context hypervectors (30,000 words, D = 1,024, 2 bits)
      + domain   · Σ context(wi | wi±1..4)   hypervectors learned from your example sentences
      + confusion· Σ class(wi | wi±1..4)     the trained confusion-set classes (form/from …)
```

**Candidates for each token.** A dictionary word keeps itself at no cost and adds its confusion-set
partners, other forms (`go/goes/went/gone`, `-s`, `-ed`, `-ing`, `a/an`, `this/these` …) and
dictionary words one edit away. A word not in the dictionary gets the 6 nearest words from the
spelling memory, and the option to keep it as typed (a name, an id) at a cost; a capitalised
unknown word in mid-sentence is cheaper to keep.

**Search.** Start from the best candidate per position, then sweep left to right up to three times,
choosing at each position the candidate with the best total score (iterated conditional modes).
The grammar term is exact for every trial: forward and backward Viterbi tables of the current
sentence give the best tag path through any replacement. Context terms are recomputed only inside
the ±4-word window they depend on. Alternatives are single-word changes of the best sentence,
rescored in full.

**Weights** live in `docs/decoder-weights.js`, written by `src/write_weights.mjs` after tuning on
dev sentences (`npm run tune:decoder`). Pass `weights` to override any of them:

| Weight | Default | Effect |
|---|---:|---|
| `sim`, `edit` | 2.04, 1.36 | how strongly spelling distance counts |
| `keepUnknown` | −9 | cost of leaving an unknown word as typed |
| `nearWord` | −5.65 | cost of changing a real word into a word one edit away |
| `confusionSwap`, `morph` | −0.90, −3.13 | cost of switching to a confusable word or another form |
| `freq`, `grammar`, `article` | 0.6, 0.26, 7.65 | priors |
| `profile`, `domain`, `confusion` | 5.5, 3, 0.56 | context terms |

Raising the three costs makes the decoder more conservative (fewer false changes, fewer fixes).

## Results

Results with the shipped (4-bit / 2-bit) models and weights retuned for them.

General English, 1,000 held-out sentences from `big.txt` (80% with one or two injected errors:
typos, confused words, agreement and article errors):

| System | Sentences right | Errors fixed | Typos | Confused words | Grammar | False changes |
|---|---:|---:|---:|---:|---:|---:|
| No correction | 23.0% | — | — | — | — | — |
| Spelling only (VSA top-1) | 43.0% | 35.1% | 76.2% | 0% | 0% | 0 |
| Proofread mode (careful) | 49.3% | 54.1% | 76.4% | 60.5% | 0% | 1.16 / 100 words |
| Harper, first suggestions | 43.2% | 36.5% | 61.4% | 7.2% | 26.2% | 4.03 / 100 words |
| **Decoder (all terms)** | **64.7%** | **68.1%** | **85.5%** | **76.3%** | 21.7% | 0.64 / 100 words |
| Decoder without grammar | 58.0% | 60.7% | 84.8% | 68.3% | 1.2% | 0.50 |
| Decoder without context profiles | 61.1% | 62.9% | 82.6% | 66.8% | 17.6% | 0.59 |
| Decoder without confusion classes | 47.5% | 42.2% | 82.4% | 2.7% | 14.8% | 0.06 |

Agent commands, own dictionary (153 words) and 150 example commands; 600 test commands, disjoint
from the examples, with one or two typos each:

| System | Commands right | Typos fixed | False changes |
|---|---:|---:|---:|
| Spelling only, general dictionary | 56.8% | 60.3% | 0 |
| Decoder, general dictionary | 63.3% | 69.8% | 0.94 / 100 words |
| Spelling only, command dictionary | 92.8% | 94.2% | 0 |
| Decoder, command dictionary | 96.5% | 97.2% | 0 |
| **Decoder, command dictionary + examples** | **99.7%** | **99.7%** | **0** |

Decoding takes about 60 ms per sentence with the general dictionary and 5 ms with the command
dictionary (Node, one core).

## Is the grammar enough?

No, not for grammar errors in open text. The part-of-speech layer is what lets the decoder fix any
agreement errors at all (1% → 22% of them), and it adds 7 points of whole-sentence accuracy, but
about four out of five grammar errors still pass. Reasons:

- Tag bigrams see one neighbour. *The list of items are long* needs the subject three words back.
- Brown's tags mark agreement for pronouns and *be/have/do*, but many errors (*he go* → *goes* or
  *went*?) have several grammatical fixes, and the decoder cannot know the intended tense.
- The test injects grammar errors that a human finds obvious and that often leave a locally
  plausible tag sequence (*this are* → *these are* is grammatical, so it counts as a miss when
  the original was *this is*).

What would help, in order of expected gain: tag trigrams (the tables already exist in Brown,
78³ entries), rules for subject–verb agreement across noun phrases, Harper as an extra term
instead of a separate layer, and more training text for the context hypervectors. For short
commands in a fixed vocabulary, the example sentences matter far more than the grammar.

## Caveats

- The command test sentences come from the same template grammar as the examples (different
  combinations, same patterns). Real users will phrase things in ways the examples do not cover;
  add those phrasings to the examples.
- Tokens are letter runs; numbers, punctuation and contractions pass through unchanged.
- One word is replaced by one word: split or merged words (*log in* / *login*) are not handled.
