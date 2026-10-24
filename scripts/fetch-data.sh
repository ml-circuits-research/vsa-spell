#!/usr/bin/env bash
# Download everything needed to retrain and re-run the benchmarks.
# The demo in docs/ does NOT need this: its models are committed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data corpus

echo "== word frequencies (SymSpell, 80k words) and Norvig's big.txt"
curl -sSfL -o data/en-80k.txt https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell.FrequencyDictionary/en-80k.txt
curl -sSfL -o data/big.txt https://raw.githubusercontent.com/norvig/pytudes/main/data/text/big.txt

echo "== NLTK corpora for the context model"
for c in brown reuters gutenberg movie_reviews abc webtext europarl_raw state_union inaugural twitter_samples; do
  curl -sSfL -o "corpus/$c.zip" "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/$c.zip"
  (cd corpus && unzip -q -o "$c.zip")
  echo "   $c"
done
echo "Done. Next: npm run data:build"
