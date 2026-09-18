const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { test } = require('node:test');

function page() {
  const strokes = [];
  const context = {
    setTransform() {}, clearRect() { strokes.length = 0; }, beginPath() {},
    moveTo() {}, lineTo() {}, closePath() {}, rect() {}, fillText() {},
    stroke() { strokes.push(this.strokeStyle); },
  };
  const nodes = new Map();
  const get = (id) => {
    if (!nodes.has(id)) nodes.set(id, {
      checked: false, value: '', addEventListener() {},
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 720, height: 586 }),
      getContext: () => context,
    });
    return nodes.get(id);
  };
  const sandbox = vm.createContext({
    document: { querySelector: get }, console,
    window: { devicePixelRatio: 1, addEventListener() {}, setTimeout() {} },
    fetch: () => new Promise(() => {}),
  });
  vm.runInContext(fs.readFileSync('src/labelbench/static/app.js', 'utf8'), sandbox);
  return { sandbox, strokes, get };
}

test('actual UI draw function renders every provider and obeys visibility/color controls', () => {
  const { sandbox, strokes, get } = page();
  const providers = ['ppocr', 'ppocr6', 'mask2former', 'sam2', 'yolo26', 'rfdetr_historical', 'docufcn', 'eynollah_textline', 'rtmdet_lines'];
  const run = { image_size: [1440, 1172], providers: Object.fromEntries(providers.map(name => [name, {
    annotations: [
      { id: name + '-1', label: 'text', bbox_xywh: [20, 20, 100, 20], polygon: [[20,20], [120,20], [120,40], [20,40]] },
      { id: name + '-2', label: 'text', bbox_xywh: [20, 80, 100, 20], polygon: null },
    ],
  }])) };
  // Exercise the shipped JavaScript, not a separate renderer implementation.
  sandbox.fixture = run;
  vm.runInContext(`state.result = fixture;
    Object.entries(fixture.providers).forEach(([name, item]) => ensureFilter(name, item.annotations));
    drawAnnotations();`, sandbox);
  const count = Object.values(run.providers).reduce((n, item) => n + item.annotations.length, 0);
  assert.equal(strokes.length, count);
  vm.runInContext(`Object.values(state.filters).forEach(f => f.visible = false); drawAnnotations();`, sandbox);
  assert.equal(strokes.length, 0);
  vm.runInContext(`state.filters.ppocr.visible = true; state.filters.ppocr.colors.text = '#123456'; drawAnnotations();`, sandbox);
  assert.equal(strokes.length, run.providers.ppocr.annotations.length);
  assert.ok(strokes.every(color => color === '#123456'));
  get('#llm-apply').checked = true;
  vm.runInContext(`state.llm.refinedAnnotations = [fixture.providers.ppocr.annotations[0]]; drawAnnotations();`, sandbox);
  assert.equal(strokes.length, 1);
});
