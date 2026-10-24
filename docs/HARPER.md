# Harper grammar layer

[Harper](https://github.com/Automattic/harper) is a rule-based English grammar checker written in
Rust and compiled to WebAssembly. The demo uses it only for grammar: Harper's own spelling marks
are hidden so the VSA speller decides spelling, and VSA marks win wherever both mark the same
words.

## Option 1: CDN (default, no setup)

Tick **Add grammar rules (Harper)** in the page. It imports `harper.js@2.10.0` from jsDelivr,
using the build with the WebAssembly binary inlined (`binaryInlined.js`, about 27 MB). The browser
caches it after the first load.

## Option 2: local copy (offline)

```bash
npm install            # installs harper.js 2.13.0 from package.json
npm run vendor:harper  # copies node_modules/harper.js/dist to docs/vendor/harper.js (about 40 MB: the JS API and the two .wasm builds)
npm run serve
```

When `docs/vendor/harper.js/index.js` exists, the page loads Harper from there and fetches the
`.wasm` file directly instead of the inlined build. `docs/vendor/` is in `.gitignore`.

To use another Harper version, change it in `package.json`, run `npm install` and
`npm run vendor:harper` again. For the CDN path, change `HARPER_CDN` near the end of
`docs/index.html`.

## Using Harper from Node

The benchmarks call Harper from Node the same way:

```js
import { LocalLinter } from 'harper.js';
import { binary } from 'harper.js/binary';

const linter = new LocalLinter({ binary });
await linter.setup();
for (const lint of await linter.lint('There going to the store.')) {
  console.log(lint.lint_kind(), lint.message(), lint.suggestions().map(s => s.get_replacement_text()));
}
```

`src/eval_js_baselines.mjs` (spelling) and `src/harper_ctx3.mjs` (real-word errors) are the
benchmark scripts.

## Troubleshooting

- **"Harper could not load here"**: the CDN is blocked (offline, a strict network or a page CSP).
  Use option 2. The VSA checks keep working without Harper.
- **Nothing happens after ticking the box**: open the browser console; the first load of the
  27 MB inlined build can take a while on a slow connection.
- **Module or MIME errors**: serve the page with `npm run serve` (it sends `application/wasm`
  for `.wasm`). Some minimal servers send the wrong type, which breaks streaming compilation.

## What Harper adds

On the benchmarks Harper is precise but conservative: on real-word errors it fixed 2.0% of
corrupted sentences with 0.5% false marks (mostly `to`/`too`). It is most useful for things the
VSA layers do not model at all: agreement, articles, repeated words, capitalisation and
punctuation rules.
