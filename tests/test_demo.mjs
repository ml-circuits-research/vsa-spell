// Headless check of the demo: models load in a real browser, marks appear,
// browser context scores match Node's golden scores (if data/ctx_gold.json exists
// from `npm run train:context`), screenshots in tests/out/.
import { chromium } from 'playwright';
import fs from 'fs';

const URL = process.env.URL || 'http://127.0.0.1:8080/index.html'; // start it with: npm run serve
const GOLD = fs.existsSync('data/ctx_gold.json') ? JSON.parse(fs.readFileSync('data/ctx_gold.json')) : null;
const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
const out = {};
for (const [name, vp, scheme] of [['desktop', { width: 1280, height: 1000 }, 'light'], ['phone', { width: 400, height: 860 }, 'dark']]) {
  const page = await browser.newPage({ viewport: vp, colorScheme: scheme });
  const errs = []; page.on('pageerror', e => errs.push(String(e))); page.on('response', r => { if (r.status() >= 400) errs.push(r.status() + ' ' + r.url()); }); page.on('requestfailed', r => errs.push('failed ' + r.url()));
  await page.goto(URL);
  await page.waitForFunction(() => document.querySelector('#st-model').classList.contains('ok') || document.querySelector('#st-model').classList.contains('err'), null, { timeout: 60000 });
  await page.waitForSelector('.issue', { timeout: 20000 }).catch(() => {});
  await page.waitForTimeout(500);
  out[name] = {
    status: await page.textContent('#st-model'), size: await page.textContent('#st-size'), time: await page.textContent('#st-time'), foot: await page.textContent('#foot-left'),
    issues: await page.$$eval('.issue', els => els.map(e => e.querySelector('.kind').textContent + ': ' + e.querySelector('.word').textContent.trim() + ' -> ' + [...e.querySelectorAll('.sug')].map(s => s.textContent).join('|'))),
    hscroll: await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth), errs,
  };
  fs.mkdirSync('tests/out', { recursive: true });
  await page.screenshot({ path: `tests/out/${name}.png`, fullPage: true });
  if (name === 'desktop') {
    // parity: run the browser ContextModel on Node's golden cases
    if (GOLD) out.parity = await page.evaluate(async (gold) => {
      const { ContextModel } = await import('./vsa-core.js');
      const [meta, buf] = await Promise.all([fetch('ctx_meta.json').then(r => r.json()), fetch('ctx_vectors.bin').then(r => r.arrayBuffer())]);
      const m = new ContextModel(meta, new Int8Array(buf)); let maxDiff = 0;
      for (const g of gold) { const s = m.scores(g.toks, g.i); s.forEach((v, k) => { maxDiff = Math.max(maxDiff, Math.abs(v - g.scores[k])); }); }
      return { cases: gold.length, maxDiff };
    }, GOLD);
    // apply first suggestion of first issue, check text changed
    const before = await page.inputValue('#text');
    await page.click('.issue .sug'); await page.waitForTimeout(800);
    out.applyWorks = before !== await page.inputValue('#text');
    // Harper layer (local vendor copy or CDN)
    await page.fill('#text', 'This are a sentence with a error in it. He go to school every days.');
    await page.check('#harper-on');
    await page.waitForFunction(() => /loaded|could not/i.test(document.querySelector('#harper-status').textContent), null, { timeout: 120000 });
    await page.waitForTimeout(1500);
    out.harper = { status: await page.textContent('#harper-status'), grammarMarks: await page.$$eval('.issue.grammar', els => els.map(e => e.querySelector('.word').textContent.trim() + ' -> ' + [...e.querySelectorAll('.sug')].map(s => s.textContent).join('|'))) };
  }
  await page.close();
}
await browser.close();
console.log(JSON.stringify(out, null, 1));
