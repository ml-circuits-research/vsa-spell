# Benchmarks

All numbers come from the scripts in `src/` and the JSON files in `results/`. Hyperparameters
were tuned on dev splits only; test sets were used once per configuration reported.

## Misspelled (non-word) errors

**Data.** `codespell`'s dictionary of real misspellings, filtered to lowercase alphabetic pairs
whose correction is among the 50,000 most frequent words and whose misspelling is not a word.
48,396 eligible pairs; shuffled with seed 42; 3,000 test, 1,000 dev (`src/make_testset.py`).

| System | Top-1 | Top-5 | ms / word | Memory |
|---|---:|---:|---:|---:|
| VSA + edit-distance rerank (shipped, D = 1,024) | **92.6%** | **98.6%** | 16 | 20.5 MB |
| VSA pure (D = 1,024) | 89.7% | 97.9% | 14 | 20.5 MB |
| SymSpell, max edit distance 3 | 90.2% | 97.4% | 6.6 | — |
| SymSpell, max edit distance 2 | 89.1% | 95.6% | 0.5 | — |
| Harper 2.13.0 | 85.2% | 93.1% | 6.9 | — |
| Hunspell via nspell 2.1.5 + dictionary-en 4.0.0 | 78.9% | 86.5% | 16.4 | — |

Caveats:
- VSA and SymSpell use the same 80,000-word frequency list, which also defined which test
  corrections were eligible. Harper and Hunspell use their own dictionaries; a few of their
  misses are words missing from those dictionaries.
- nspell is a JavaScript port of Hunspell; native Hunspell (used by browsers) suggests better.
- SymSpell was timed in Python, the others in Node on one core. VSA does a full scan; SymSpell
  is an index lookup and is far faster for distance ≤ 2.
- D = 2,048 gave the same accuracy (92.0%) at twice the memory.

## Real-word errors

**Data.** Norvig's `big.txt`, split into sentences and shuffled (seed 1): 85% train, 5% dev,
10% test. Instances are occurrences of confusion-set words, capped per word (25 dev, 30 test):
6,013 dev and 9,198 test instances over 320 sets. A *corrupted* instance swaps in the next
member of the set.

Metrics:
- **Right word**: the system's top choice among the set equals the original word.
- **Fixed**: on the corrupted sentence, the system flags the word and proposes the original.
- **False marks**: on the original sentence, the system flags the (correct) word.

Fixed and false marks are measured on enabled sets only, with τ chosen on dev for about 2%
false marks.

### Round 3: 320 sets (current)

Trained on 6.8M words of NLTK corpora, no `big.txt` text (out of domain):

| System | Right word | Fixed | False marks |
|---|---:|---:|---:|
| Most frequent member | 59.8% | — | — |
| Word n-grams, stupid backoff (6.0M entries) | 88.3% | 66.0% | 1.8% |
| VSA, int8 (D = 16,384, 10.6 MB) | 88.0% | 51.4% | 1.2% |
| VSA + n-grams | 89.4% | 69.8% | 2.0% |
| Harper 2.13.0 | — | 2.0% | 0.5% |

By set origin (right word): curated 89.0 / 88.6 (n-gram / VSA), typo pairs 88.2 / 87.7,
homophones 87.5 / 87.7.

Shipped model (adds the `big.txt` training split, 7.7M words; same test sentences, now in
domain): n-gram 92.1%, VSA 90.5%, VSA + n-gram 92.7%; VSA fixes 62.0% at 1.6% false marks.

### Round 2: 25 classic sets, learning curve

Same test sentences, 1,269 instances.

| Training words | n-gram | VSA | VSA + n-gram |
|---|---:|---:|---:|
| 1.0M (out of domain) | 86.8% | 88.1% | 88.7% |
| 2.5M (out of domain) | 89.4% | 89.9% | 92.0% |
| 6.8M (out of domain) | 91.1% | 92.1% | 94.0% |
| 7.7M (incl. big.txt train) | 93.6% | 93.5% | 95.9% |

With few sets and little data VSA generalised slightly better than n-grams. With 320 sets the
two are level. One point is about 13 sentences in round 2 and about 92 in round 3.

### Round 1: what changed along the way

- A dense random-indexing centroid model trained on 0.8M in-domain words: 87.6% (n-gram 91.4%).
- Adding retraining, a ±4 window and larger D raised the same setting to 91.5%.
- Harper on the 25 classic sets fixed 17.0% with 0.4% false marks; most of that was `to/too`.

## Sentence interpretation

See [DECODER.md](DECODER.md) for the method and the full tables. Summary:

- General English, 1,000 held-out `big.txt` sentences (20% clean, 48% one error, 32% two errors;
  errors are 45% typos, 30% confused words, 25% grammar): the decoder gets 64.3% of sentences
  exactly right (proofread mode 49.3%, Harper 43.2%), fixes 68.1% of errors and changes 0.64 of
  every 100 correct words. Removing the grammar layer costs 7 points of sentence accuracy.
- Agent commands with their own 153-word dictionary and 150 example commands: 99.7% of 600 test
  commands right with no false changes, against 92.8% for spelling alone over the same dictionary.

Weights were tuned by random search on 250 dev sentences (`results/decoder_weights.json`).

## Model size

The shipped hypervectors were measured at lower precision on the 9,198 real-word test instances
(`src/quant_eval.mjs`, `results/quantization.json`).

Confusion-set classes (D = 16,384, 649 words), shipped model:

| Bits | Size | Right word | Fixed (Careful) | False marks |
|---:|---:|---:|---:|---:|
| 8 | 10.6 MB | 90.5% | 62.0% | 1.57% |
| **4 (shipped)** | **5.3 MB** | **90.1%** | **61.7%** | **1.53%** |
| 2 | 2.7 MB | 89.2% | 43.0% | 1.12% |
| 1 | 1.3 MB | 87.4% | 13.8% | 0.42% |

At 2 and 1 bits the scores shrink, so the Careful threshold becomes too strict; recalibrating it
would recover part of the detection rate.

Word-context profiles (30,000 words), right word using the profiles alone:

| D | 8 bits | 4 bits | 2 bits | 1 bit |
|---:|---:|---:|---:|---:|
| 512 | 77.6% (15.4 MB) | 76.7% (7.7) | 78.7% (3.8) | 73.6% (1.9) |
| **1,024** | 78.2% (30.7) | 77.3% (15.4) | **80.8% (7.7, shipped)** | 74.4% (3.8) |
| 2,048 | 80.0% (61.4) | 78.6% (30.7) | 83.0% (15.4) | 73.7% (7.7) |

The 2-bit code uses four non-uniform levels (±1, ±3 at ±0.98σ). It beats 8 bits because it
clips the few large components that frequent context words create, a saturation effect similar to
PMI weighting. Uniform 4-bit and 1-bit codes do not have it.

Download, before → after: proofreading 12 → 6 MB, interpretation 17 → 8 MB.

## Browser

Chromium (headless, one run): model download and build 1.9–2.4 s, checking a 65-word text
about 100 ms. Browser and Node context scores differ by at most 7·10⁻⁶ on 200 test cases.
