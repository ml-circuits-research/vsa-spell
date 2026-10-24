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
  exactly right (proofread mode 49.8%, Harper 43.2%), fixes 66.9% of errors and changes 0.58 of
  every 100 correct words. Removing the grammar layer costs 7 points of sentence accuracy.
- Agent commands with their own 153-word dictionary and 150 example commands: 99.3% of 600 test
  commands right with no false changes, against 92.8% for spelling alone over the same dictionary.

Weights were tuned by random search on 250 dev sentences (`results/decoder_weights.json`).

## Browser

Chromium (headless, one run): model download and build 1.9–2.4 s, checking a 65-word text
about 100 ms. Browser and Node context scores differ by at most 7·10⁻⁶ on 200 test cases.
