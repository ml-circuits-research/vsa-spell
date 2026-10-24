# Reproduce the results

Requirements: Node.js 18+, Python 3.10+, `curl`, `unzip`, about 16 GB of RAM for context training.

```bash
npm install
pip install -r requirements.txt
npm run data:fetch      # en-80k.txt, big.txt, NLTK corpora → data/ and corpus/
npm run data:build      # training corpus, misspelling test/dev sets, confusion sets
```

`data:build` overwrites `data/confusions.json`. It is deterministic, so it should reproduce the
committed file; if a corpus changed upstream, keep the committed file with `git checkout data/confusions.json`.

## Spelling

```bash
npm run bench:spell       # VSA: tunes on dev, evaluates on test   → results/res_core_1024.json
npm run bench:symspell    # SymSpell ed2 and ed3                    → results/res_symspell.json
npm run bench:baselines   # Harper and Hunspell (nspell)            → results/res_js_*.json
```

## Real-word errors

```bash
npm run bench:context        # out-of-domain eval, about 15 min      → results/context_ext.json
npm run bench:harper-context # Harper on the same test instances     → results/harper_ctx3.json
npm run train:context        # trains the shipped model and writes docs/ctx_meta.json, docs/ctx_vectors.bin
```

`train:context` also writes `data/ctx_gold.json`, which the browser test uses to compare browser
and Node scores.

## Browser test

```bash
npx playwright install chromium   # once
npm run serve &                   # in another terminal
npm run test:browser              # screenshots in tests/out/
```
