# Architecture

Everything that runs in the page lives in `docs/vsa-core.js`. Node imports the same file for
tuning and benchmarks, so the browser runs exactly the code that was measured. A headless test
checks that browser and Node context scores agree (maximum difference 7·10⁻⁶ over 200 cases).

## 1. Spelling: binary hypervectors

**Representation.** Each symbol (letter, `#` boundary, phonetic letter) gets a random
1,024-bit hypervector seeded by a hash of the symbol. A word `w` is encoded from `#w#`:

- every n-gram (n = 2 and 3) `c0 c1 … c(n-1)` becomes `ρ^(n-1)(c0) ⊕ … ⊕ ρ^0(c(n-1))`, where
  `⊕` is XOR (binding) and `ρ` rotates the vector by one 32-bit word (a permutation that marks
  position inside the n-gram);
- all n-grams are bundled by bitwise majority; ties use a fixed random vector.

A second hypervector encodes the word's primary Double Metaphone code the same way, so words
that sound alike are close even when they look different (`ressapee` → `recipe`).

**Bundling in the browser.** Majority is computed with bit-sliced counters: seven bit planes
hold a per-bit count, each n-gram is added with a ripple-carry of 32-bit words, and the
majority is a bitwise comparison of the planes against `n/2`. Building all 80,000 words takes
about 2 seconds in Chrome (the naive per-bit version took about 15 seconds).

**Lookup.** An unknown word is encoded and compared with all 80,000 word vectors by Hamming
distance (XOR + popcount), for both the spelling and the phonetic memory. The candidate pool is
the top 60 of each plus the 200 most frequent words. Candidates are scored as

```
score = sim_spelling + 0.5·sim_phonetic + 0.8·log_freq − 0.35·edit_distance
```

with the weights tuned on a 1,000-word dev set (`results/spell_params_1024.json`). Without the
edit-distance term ("pure VSA") top-1 is 89.7%; with it, 92.6%.

**Heuristics in the page.** Words in the dictionary, acronyms, words with apostrophes and
capitalised words not at the start of a sentence (probable names) are not checked.

## 2. Word choice: random-indexing context vectors

**Confusion sets.** `data/confusions.json` holds 320 sets (649 words): 123 curated sets of
commonly confused words and 197 mined automatically from the training corpus (edit-distance-1
pairs such as `thing/think`, and homophones with the same Double Metaphone code). See
`src/make_confusions.mjs`.

**Index vectors.** Each context word has a sparse ternary index vector of dimension D = 16,384
with 8 non-zero ±1 entries, generated from a hash of the word. Nothing per context word is
stored, so the vocabulary is unbounded.

**Context of an occurrence.** The 4 words on each side are bundled; each is permuted by a
position-specific cyclic shift (`((p + 16) · 7919) mod D` for offset p) and weighted by `1/|p|`.

**Class vectors.** For each confusion word, the contexts of its training occurrences are summed
and normalised (bundling). Then 20 epochs of HDC retraining: when an occurrence is closer to
another member of its set, its context is added to the right class and subtracted from the
wrong one (a perceptron update in hypervector space). Class vectors are quantised to int8 with
one scale per vector (0.02 points), then packed to 4 bits for shipping (0.4 points; see BENCHMARKS.md).

**Decision.** For a word in an enabled set, each member is scored by
`cos(context, class) + log(count)/20` (prior weight 1.0, tuned on dev). The word is flagged when another member beats it
by more than τ = 1.16, the margin that kept false marks near 2% on the dev set. The page's
*Balanced* and *Eager* modes use τ/2 and τ/4. Sets that, on the dev set, fixed under 30% of
errors or falsely marked over 10% of correct uses are disabled; 253 of 320 are active in the
shipped model.

**Training data.** The shipped model was trained on 7.7M words: Brown, Reuters, Gutenberg
(NLTK selection), movie reviews, ABC, webtext, Europarl (English), State of the Union,
inaugural addresses, tweets, plus the training part of `big.txt`. The benchmark numbers use
a model trained without `big.txt`, so they are out of domain.

## 3. Page structure

- `docs/index.html`: editor (textarea with a highlight backdrop), margin notes, benchmark tables.
- `docs/worker.js`: loads `words.txt`, `ctx_meta.json`, `ctx_vectors.bin`, builds the speller
  and answers `check` messages off the main thread.
- Harper (optional) runs on the main thread; see [HARPER.md](HARPER.md).
- Drafts, ignored words and the strictness setting are kept in `localStorage`.

## 4. Sentence interpretation

`docs/decoder.js` combines the speller, the confusion classes, a part-of-speech grammar
(`docs/grammar.json`, from the tagged Brown corpus) and word-context hypervectors for 30,000 words
(`docs/profiles.bin`) to choose the most probable sentence over any dictionary. It is described
in [DECODER.md](DECODER.md).
