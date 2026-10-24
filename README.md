# Hypervector Proofreader

English spelling and real-word error checking ("form" vs "from", "loose" vs "lose") built on
**vector symbolic architectures** (VSA, hyperdimensional computing). It runs entirely in the
browser, with no language model and no server. An optional grammar layer comes from
[Harper](https://github.com/Automattic/harper).

![Screenshot](docs/screenshot.png)

## Run it on your computer

Requirements: Node.js 18 or newer. The trained models are committed, so nothing needs training.

```bash
git clone https://github.com/ml-circuits-research/vsa-spell.git
cd vsa-spell
npm run serve          # no install needed for the demo
```

Open <http://127.0.0.1:8080/>. The page loads about 12 MB of models, builds 80,000 word
hypervectors in about 2 seconds and checks as you type.

A local web server is required: browsers will not start the module Web Worker from a `file://` URL.
`python3 -m http.server 8080 -d web` works as well.

### Add the Harper grammar layer

Tick **Add grammar rules (Harper)** in the page. By default the page loads Harper 2.10.0 from
jsDelivr (about 27 MB). To run fully offline instead:

```bash
npm install
npm run vendor:harper  # copies Harper into web/vendor/harper.js
npm run serve
```

The page then loads Harper from `web/vendor/` with no network access. Details, versions and
troubleshooting: [docs/HARPER.md](docs/HARPER.md).

## What is in the page

| Layer | Method | Size in memory |
|---|---|---|
| Spelling | 80,000 words as 1,024-bit hypervectors (character bigrams + trigrams, XOR binding, rotation for position, majority bundling) plus a Double Metaphone hypervector; Hamming search, rerank by frequency and edit distance | 20.5 MB |
| Word choice | 320 confusion sets; one int8 class hypervector (D = 16,384) per word, learned by sparse random indexing over ±4 words of context with HDC retraining | 10.6 MB |
| Grammar (optional) | Harper's rule-based linter (its spelling rules are hidden) | about 27 MB |

How it works in detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Results

Misspelled words: 3,000 real misspellings from the codespell list, test set never used for tuning.

| System | Top-1 | Top-5 | Per word |
|---|---:|---:|---:|
| **VSA (this repo)** | **92.6%** | **98.6%** | 16 ms |
| SymSpell, edit distance 3 | 90.2% | 97.4% | 7 ms |
| SymSpell, edit distance 2 | 89.1% | 95.6% | 0.5 ms |
| Harper | 85.2% | 93.1% | 7 ms |
| Hunspell (nspell) | 78.9% | 86.5% | 16 ms |

Real-word errors: 9,198 held-out sentences, 320 confusion sets, models trained on 6.8M words of
other text (out of domain).

| System | Right word | Fixed | False marks |
|---|---:|---:|---:|
| **VSA (this repo)** | 88.0% | 51.4% | 1.2% |
| Word n-grams (stupid backoff) | 88.3% | 66.0% | 1.8% |
| VSA + n-grams | 89.4% | 69.8% | 2.0% |
| Harper | — | 2.0% | 0.5% |

The VSA context model is on par with word n-grams at picking the right word and uses 10.6 MB
instead of about 6 million n-gram entries; n-grams are still better at flagging errors at a low
false-alarm rate. Full numbers, earlier rounds and caveats: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Repository layout

```
web/        the demo: index.html, worker.js, vsa-core.js (shared with Node), models
src/        corpus building, confusion-set mining, training and benchmarks
scripts/    static server, Harper vendoring, data download
results/    JSON outputs of the benchmark runs
docs/       architecture, benchmarks, Harper, reproduction
tests/      headless browser test (Playwright)
data/       confusions.json (other data files are downloaded, see docs/REPRODUCE.md)
```

## Reproduce or retrain

See [docs/REPRODUCE.md](docs/REPRODUCE.md). In short: `npm install`, `pip install -r requirements.txt`,
`npm run data:fetch`, `npm run data:build`, then the `bench:*` and `train:context` scripts.

## Licenses

Code in this repository: MIT. `web/double-metaphone.js` is Titus Wormer's double-metaphone (MIT).
`web/words.txt` is SymSpell's English frequency dictionary (MIT). Harper is Apache-2.0 and is not
committed; it is loaded from npm or the CDN.
