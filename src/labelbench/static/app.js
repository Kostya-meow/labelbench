const state = { health: null, result: null, filters: {} };
const elements = {
  image: document.querySelector('#image-select'), providers: document.querySelector('#providers'),
  run: document.querySelector('#run'), refresh: document.querySelector('#refresh'), status: document.querySelector('#run-status'),
  source: document.querySelector('#source'), canvas: document.querySelector('#canvas-wrap'), overlay: document.querySelector('#overlay'),
  title: document.querySelector('#image-title'), consensus: document.querySelector('#consensus'),
  strip: document.querySelector('#result-strip'), export: document.querySelector('#export'), device: document.querySelector('#device'), template: document.querySelector('#provider-template'),
};
const colors = {
  ppocr: '#f36f38', mask2former: '#29b6a6', sam2: '#9b73e8', yolo26: '#e2b93b',
  rfdetr_historical: '#dc5a8a', docufcn: '#5378d8', eynollah_textline: '#c45a35',
};
const classColors = ['#f36f38', '#29b6a6', '#9b73e8', '#e2b93b', '#dc5a8a', '#5378d8'];
const providerTitles = {
  ppocr: 'PP-OCRv5 Server', mask2former: 'Mask2Former', sam2: 'SAM 2.1', yolo26: 'YOLO26-seg',
  rfdetr_historical: 'RF-DETR Historical Textline', docufcn: 'Doc-UFCN Generic Historical Line', eynollah_textline: 'Eynollah Textline',
};

async function json(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Ошибка API');
  return data;
}
function escapeHtml(value) { const node = document.createElement('span'); node.textContent = value; return node.innerHTML; }
async function loadHealth() {
  state.health = await json('/api/health');
  elements.device.textContent = `device: ${state.health.device}`;
  elements.providers.replaceChildren();
  Object.entries(state.health.providers).forEach(([name, info]) => {
    const row = elements.template.content.firstElementChild.cloneNode(true);
    const input = row.querySelector('input');
    input.value = name; input.checked = info.available; input.disabled = !info.available;
    row.querySelector('.provider-name').textContent = name;
    row.querySelector('.provider-status').textContent = info.detail;
    if (!info.available) row.classList.add('off');
    elements.providers.append(row);
  });
  updateRunEnabled();
}
async function loadImages() {
  const { images } = await json('/api/images');
  elements.image.replaceChildren();
  if (!images.length) elements.image.add(new Option('Нет доступных изображений', ''));
  images.forEach((name) => elements.image.add(new Option(name, name)));
  updateRunEnabled();
}
function updateRunEnabled() {
  elements.run.disabled = !elements.image.value || !document.querySelector('.provider-row input:checked');
}
function selectProviders() { return [...document.querySelectorAll('.provider-row input:checked')].map((input) => input.value); }
function imageUrl(name) { return `/files/images/${name.split('/').map(encodeURIComponent).join('/')}`; }
function safeColor(value, fallback) { return /^#[0-9a-f]{6}$/i.test(value) ? value : fallback; }
function ensureFilter(provider, annotations) {
  const previous = state.filters[provider] || { visible: true, labels: {}, colors: {} };
  const labels = [...new Set(annotations.map((annotation) => annotation.label))];
  const next = { visible: previous.visible !== false, labels: {}, colors: { ...previous.colors } };
  labels.forEach((label, index) => {
    next.labels[label] = previous.labels[label] !== false;
    next.colors[label] = safeColor(previous.colors[label], index ? classColors[index % classColors.length] : colors[provider] || '#ffffff');
  });
  state.filters[provider] = next;
}
function visibleAnnotations(provider, item) {
  const filter = state.filters[provider];
  if (!filter || filter.visible === false) return [];
  return item.annotations.filter((annotation) => filter.labels[annotation.label] !== false);
}
function renderResultStrip() {
  if (!state.result) return;
  const cards = Object.entries(state.result.providers).map(([name, item]) => {
    ensureFilter(name, item.annotations);
    const filter = state.filters[name];
    const labels = [...new Set(item.annotations.map((annotation) => annotation.label))];
    const controls = labels.map((label) => `<label class="class-control"><input type="checkbox" data-class-provider="${escapeHtml(name)}" data-class-label="${escapeHtml(label)}"${filter.labels[label] ? ' checked' : ''}><input type="color" value="${filter.colors[label]}" data-color-provider="${escapeHtml(name)}" data-color-label="${escapeHtml(label)}" title="Цвет ${escapeHtml(label)}"><span>${escapeHtml(label)}</span></label>`).join('');
    return `<article class="result-card" style="--card-color:${colors[name] || '#16221d'}"><div class="result-card-head"><strong>${escapeHtml(providerTitles[name] || name)}</strong><label class="visibility-control"><input type="checkbox" data-visible-provider="${escapeHtml(name)}"${filter.visible ? ' checked' : ''}> показывать</label></div><span>${visibleAnnotations(name, item).length}/${item.annotations.length} оставлено<br>${item.elapsed_seconds.toFixed(2)} sec</span><div class="class-controls">${controls || '<small>Нет сегментов</small>'}</div></article>`;
  });
  elements.strip.innerHTML = cards.join('');
}
function showResult(result) {
  state.result = result;
  elements.title.textContent = result.image_name;
  elements.consensus.textContent = `${Math.round(result.consensus_score * 100)}%`;
  elements.source.src = imageUrl(result.image_name);
  elements.source.onload = () => { elements.canvas.classList.add('loaded'); drawAnnotations(); };
  Object.entries(result.providers).forEach(([name, item]) => ensureFilter(name, item.annotations));
  renderResultStrip();
  elements.export.disabled = false;
}
function drawAnnotations() {
  const result = state.result;
  if (!result) return;
  const imageRect = elements.source.getBoundingClientRect();
  const wrapRect = elements.canvas.getBoundingClientRect();
  const offsetX = imageRect.left - wrapRect.left;
  const offsetY = imageRect.top - wrapRect.top;
  const sx = imageRect.width / result.image_size[0];
  const sy = imageRect.height / result.image_size[1];
  const shapes = Object.entries(result.providers).flatMap(([provider, item]) => visibleAnnotations(provider, item).map((annotation) => {
    const color = state.filters[provider]?.colors[annotation.label] || colors[provider] || '#fff';
    const [x, y, width, height] = annotation.bbox_xywh;
    if (annotation.polygon) {
      const points = annotation.polygon.map(([px, py]) => `${offsetX + px * sx},${offsetY + py * sy}`).join(' ');
      return `<polygon points="${points}" fill="none" stroke="${color}" stroke-width="2.5"/><text x="${offsetX + x * sx + 3}" y="${offsetY + y * sy - 5}" fill="${color}">${escapeHtml(annotation.text || annotation.label)}</text>`;
    }
    return `<rect x="${offsetX + x * sx}" y="${offsetY + y * sy}" width="${width * sx}" height="${height * sy}" fill="${color}" fill-opacity=".08" stroke="${color}" stroke-width="2"/><text x="${offsetX + x * sx + 3}" y="${offsetY + y * sy - 5}" fill="${color}">${escapeHtml(annotation.label)}</text>`;
  }));
  elements.overlay.innerHTML = shapes.join('');
}
function exportFiltered() {
  if (!state.result) return;
  const providers = Object.fromEntries(Object.entries(state.result.providers).map(([name, item]) => [name, {
    ...item,
    annotations: visibleAnnotations(name, item),
  }]));
  const payload = { ...state.result, providers, filters: state.filters };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `${state.result.image_name.replace(/[^a-z0-9._-]+/gi, '_')}_${state.result.run_id}_filtered.json`;
  link.click();
  URL.revokeObjectURL(link.href);
  elements.status.textContent = 'Оставленная разметка скачана в JSON.';
}
async function run() {
  elements.run.disabled = true;
  elements.status.textContent = 'Модели выполняют inference — первый запуск может скачать веса…';
  try {
    const result = await json('/api/runs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_name: elements.image.value, providers: selectProviders() }),
    });
    showResult(result);
    elements.status.textContent = `Готово: run ${result.run_id}`;
  } catch (error) { elements.status.textContent = `Ошибка: ${error.message}`; }
  finally { updateRunEnabled(); }
}
function showError(error) { elements.status.textContent = `Ошибка: ${error.message}`; }
elements.run.addEventListener('click', run);
elements.refresh.addEventListener('click', () => loadImages().catch(showError));
elements.image.addEventListener('change', updateRunEnabled);
elements.providers.addEventListener('change', updateRunEnabled);
elements.strip.addEventListener('change', (event) => {
  const target = event.target;
  const provider = target.dataset.visibleProvider || target.dataset.classProvider || target.dataset.colorProvider;
  if (!provider || !state.filters[provider]) return;
  if (target.dataset.visibleProvider) state.filters[provider].visible = target.checked;
  if (target.dataset.classProvider) state.filters[provider].labels[target.dataset.classLabel] = target.checked;
  if (target.dataset.colorProvider) state.filters[provider].colors[target.dataset.colorLabel] = safeColor(target.value, colors[provider] || '#ffffff');
  renderResultStrip();
  drawAnnotations();
});
elements.export.addEventListener('click', exportFiltered);
window.addEventListener('resize', drawAnnotations);
Promise.all([loadHealth(), loadImages()]).catch(showError);
