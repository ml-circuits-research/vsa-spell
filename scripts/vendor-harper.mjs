// Copy harper.js (installed by `npm install`) into web/vendor so the page can
// load Harper offline, without the jsDelivr CDN. See docs/HARPER.md.
import fs from 'node:fs';
import path from 'node:path';

const src = path.resolve('node_modules/harper.js/dist');
const dst = path.resolve('web/vendor/harper.js');
if (!fs.existsSync(src)) { console.error('harper.js is not installed. Run `npm install` first.'); process.exit(1); }
fs.rmSync(dst, { recursive: true, force: true });
fs.mkdirSync(dst, { recursive: true });
let bytes = 0;
// the page needs the JS API, the binary loaders and the .wasm files; skip types and the inlined builds
const KEEP = f => ['index.js', 'binary.js', 'slimBinary.js', 'harper_wasm_bg.wasm', 'harper_wasm_slim_bg.wasm'].includes(f) || /^BinaryModule-.*\.js$/.test(f);
for (const f of fs.readdirSync(src)) {
  if (!KEEP(f)) continue;
  fs.copyFileSync(path.join(src, f), path.join(dst, f)); bytes += fs.statSync(path.join(dst, f)).size;
}
const v = JSON.parse(fs.readFileSync('node_modules/harper.js/package.json')).version;
console.log(`Copied harper.js ${v} to web/vendor/harper.js (${(bytes / 1e6).toFixed(1)} MB)`);
