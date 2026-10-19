// DOM fixtures only: these tests do not claim real-model browser predictions.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync('ticketnlp/static/app.js', 'utf8');
const report = JSON.parse(fs.readFileSync('reports/full/final_comparison.json', 'utf8'));
const tick = () => new Promise(resolve => setImmediate(resolve));

function element() {
  const classes = new Set();
  return {
    value: '', hidden: false, disabled: false, textContent: '', dataset: {}, attributes: {}, listeners: {},
    classList: { toggle(name, enabled) { enabled ? classes.add(name) : classes.delete(name); }, remove(name) { classes.delete(name); }, contains(name) { return classes.has(name); } },
    addEventListener(name, fn) { this.listeners[name] = fn; },
    setAttribute(name, value) { this.attributes[name] = value; },
    removeAttribute(name) { delete this.attributes[name]; },
    focus() { this.focused = true; },
    set innerHTML(_) { throw new Error('Unsafe HTML assignment'); },
  };
}
function fixture(predict) {
  const elements = new Map();
  const get = id => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); };
  const panel = element(), spinner = element(), buttons = [element()];
  buttons[0].dataset.example = 'A card query';
  let requests = 0;
  vm.runInNewContext(script, {
    document: { getElementById: get, querySelector: selector => selector === '.result-panel' ? panel : spinner, querySelectorAll: () => buttons },
    fetch: async (url, options) => {
      if (url === '/health') return { ok: true };
      if (url === '/evaluation') return { ok: true, json: async () => report };
      requests++;
      return predict(JSON.parse(options.body));
    },
    AbortController, AbortSignal, setTimeout, clearTimeout, Number,
  });
  const edit = text => { get('query').value = text; get('query').listeners.input(); };
  const submit = () => get('query-form').listeners.submit({ preventDefault() {} });
  return { get, panel, spinner, buttons, edit, submit, requests: () => requests };
}
const response = (changes = {}) => ({ ok: true, json: async () => ({ label: 'card_arrival', score: 0.8, model: 'distilbert', mode: 'full', truncated: false, ...changes }) });

test('empty and excessive inputs are rejected before requesting a prediction', async () => {
  const f = fixture(() => response());
  f.edit('   '); await f.submit();
  assert.match(f.get('query-error').textContent, /Write a banking query/);
  assert.equal(f.get('query').attributes['aria-invalid'], 'true');
  f.edit('x'.repeat(4001)); await f.submit();
  assert.match(f.get('query-error').textContent, /4,000/);
  assert.equal(f.requests(), 0);
});

test('pending input edits cannot associate the old result with the new query', async () => {
  let finish;
  const f = fixture(() => new Promise(resolve => { finish = resolve; }));
  f.edit('Original query');
  const pending = f.submit();
  assert.equal(f.get('classify-button').disabled, true);
  assert.equal(f.spinner.hidden, false);
  assert.equal(f.panel.attributes['aria-busy'], 'true');
  await f.submit();
  assert.equal(f.requests(), 1);
  f.edit('A different query');
  finish(response()); await pending;
  assert.equal(f.get('submitted-query').textContent, 'Original query');
  assert.match(f.get('result-state').textContent, /input changed/);
  assert.equal(f.panel.classList.contains('outdated'), true);
  assert.equal(f.get('classify-button').disabled, false);
  f.edit('Original query');
  assert.equal(f.panel.classList.contains('outdated'), false);
});

test('unavailable API never fabricates predictions and supports retry', async () => {
  let fail = true;
  const f = fixture(() => { if (fail) throw new Error('offline'); return response(); });
  f.edit('My query'); await f.submit();
  assert.equal(f.get('result-content').hidden, true);
  assert.match(f.get('query-error').textContent, /isn’t available/);
  assert.equal(f.get('query').value, 'My query');
  fail = false; await f.submit();
  assert.equal(f.get('result-content').hidden, false);
  assert.equal(f.get('query-error').hidden, true);
});

test('smoke mode, truncation and user/output text remain explicit and safe', async () => {
  const malicious = '<img src=x onerror=alert(1)>';
  const f = fixture(() => response({ label: malicious, mode: 'smoke', truncated: true }));
  f.edit(malicious); await f.submit();
  assert.equal(f.get('canonical-label').textContent, malicious);
  assert.equal(f.get('submitted-query').textContent, malicious);
  assert.match(f.get('mode-notice').textContent, /Smoke-test checkpoint/);
  assert.equal(f.get('truncation-notice').hidden, false);
  assert.equal(f.get('model-score').textContent, '80.0%');
});

test('422 and malformed responses show errors rather than predictions', async () => {
  const rejected = fixture(() => ({ ok: false, status: 422 }));
  rejected.edit('query'); await rejected.submit();
  assert.match(rejected.get('query-error').textContent, /Check the query/);
  const malformed = fixture(() => response({ score: 1.5 }));
  malformed.edit('query'); await malformed.submit();
  assert.equal(malformed.get('result-content').hidden, true);
  assert.match(malformed.get('query-error').textContent, /isn’t available/);
});

test('evaluation is rendered from the actual committed report', async () => {
  const f = fixture(() => response()); await tick();
  assert.equal(f.get('baseline-accuracy').textContent, (report.baseline.accuracy * 100).toFixed(2) + '%');
  assert.equal(f.get('distilbert-f1').textContent, report.distilbert.macro_f1.toFixed(4));
  assert.match(f.get('evaluation-context').textContent, /3,020/);
});
