const state = {
  health: null, result: null, filters: {}, runBusy: false,
  llm: { busy: false, resultModel: '', models: [], messages: [], refinedAnnotations: [], reviewAnnotations: [], reviewFilters: {}, compactReview: null, notes: '', model: '' },
};
const elements = {
  image: document.querySelector('#image-select'), providers: document.querySelector('#providers'),
  run: document.querySelector('#run'), refresh: document.querySelector('#refresh'), status: document.querySelector('#run-status'),
  source: document.querySelector('#source'), canvas: document.querySelector('#canvas-wrap'), overlay: document.querySelector('#overlay'),
  title: document.querySelector('#image-title'), consensus: document.querySelector('#consensus'),
  strip: document.querySelector('#result-strip'), export: document.querySelector('#export'), device: document.querySelector('#device'), template: document.querySelector('#provider-template'),
  llmModel: document.querySelector('#llm-model'), llmRefresh: document.querySelector('#llm-refresh'),
  llmUrl: document.querySelector('#llm-url'),
  llmPrompt: document.querySelector('#llm-prompt'), llmSend: document.querySelector('#llm-send'),
  llmStatus: document.querySelector('#llm-status'), llmResponse: document.querySelector('#llm-response'),
  llmApply: document.querySelector('#llm-apply'), llmExport: document.querySelector('#llm-export'),
  llmPreview: document.querySelector('#llm-preview'), llmPreviewCount: document.querySelector('#llm-preview-count'),
  llmSource: document.querySelector('#llm-source'), llmCanvas: document.querySelector('#llm-canvas-wrap'),
  llmOverlay: document.querySelector('#llm-overlay'),
  llmResultMessage: document.querySelector('#llm-result-message'),
};
const colors = {
  yolo26_onnx: '#aa790b',
  ppocr6: '#16a085',
  ppocr6_small: '#287fca', ppocr6_tiny: '#b068b0',
  ppocr: '#f36f38', mask2former: '#29b6a6', sam2: '#9b73e8', yolo26: '#e2b93b',
  rfdetr_historical: '#dc5a8a', docufcn: '#5378d8', eynollah_textline: '#c45a35', rtmdet_lines: '#008c7a',
};
const classColors = ['#f36f38', '#29b6a6', '#9b73e8', '#e2b93b', '#dc5a8a', '#5378d8'];
const reviewStyles = {
  kept: ['#159447', 'Верно'], removed: ['#de3c3c', 'Удалено'], uncertain: ['#ce9500', 'Сомнительно'],
  modified: ['#1675db', 'Исправлено'], merged: ['#884bd8', 'Объединено'], added: ['#009d99', 'Добавлено'],
};
const providerTitles = {
  yolo26_onnx: 'YOLO26-seg · ONNX GPU',
  ppocr6: 'PP-OCRv6 Medium Det',
  ppocr6_small: 'PP-OCRv6 Small Det', ppocr6_tiny: 'PP-OCRv6 Tiny Det',
  ppocr: 'PP-OCRv5 Server', mask2former: 'Mask2Former', sam2: 'SAM 2.1', yolo26: 'YOLO26-seg',
  rfdetr_historical: 'RF-DETR Historical Textline', docufcn: 'Doc-UFCN Generic Historical Line', eynollah_textline: 'Eynollah Textline', rtmdet_lines: 'Riksarkivet RTMDet Lines',
};

async function json(url, options) {
  const response = await fetch(url, options);
  const body = await response.text();
  let data;
  try { data = body ? JSON.parse(body) : {}; } catch { data = {}; }
  if (!response.ok) throw new Error(data.detail || body || `Ошибка API (${response.status})`);
  return data;
}
function escapeHtml(value) { const node = document.createElement('span'); node.textContent = value; return node.innerHTML; }
async function loadHealth() {
  state.health = await json('/api/health');
  elements.device.textContent = `device: ${state.health.device}`;
  elements.providers.replaceChildren();
  Object.entries(state.health.providers).forEach(([name, info]) => {
    if (name.endsWith('_onnx') && state.health.providers[name.slice(0, -5)]) return;
    const row = elements.template.content.firstElementChild.cloneNode(true);
    const input = row.querySelector('input');
    input.value = name; input.checked = info.available && ['ppocr6_tiny', 'rfdetr_historical', 'docufcn'].includes(name); input.disabled = !info.available;
    row.querySelector('.provider-name').textContent = providerTitles[name] || name;
    row.querySelector('.provider-status').textContent = info.detail;
    if (!info.available) row.classList.add('off');
    elements.providers.append(row);
    const onnxName = `${name}_onnx`;
    const onnxInfo = state.health.providers[onnxName];
    if (onnxInfo) {
      providerTitles[onnxName] = `${providerTitles[name] || name} · ONNX`;
      colors[onnxName] = colors[name];
      const backend = document.createElement('select');
      backend.className = 'backend-select';
      backend.setAttribute('aria-label', `Backend ${name}`);
      backend.add(new Option('PyTorch', name));
      const option = new Option(onnxInfo.available ? 'ONNX · GPU' : 'ONNX · нужен экспорт', onnxName);
      option.disabled = !onnxInfo.available;
      backend.add(option);
      backend.addEventListener('change', () => {
        input.value = backend.value;
        settings.querySelector('input').dataset.threshold = backend.value;
        row.querySelector('.provider-status').textContent = state.health.providers[backend.value].detail;
        updateRunEnabled();
      });
      elements.providers.append(backend);
    }
    const settings = document.createElement('label');
    settings.className = 'threshold-control';
    settings.innerHTML = `Confidence <input type="number" data-threshold="${name}" min="0" max="1" step="0.05" placeholder="auto" aria-label="Confidence ${name}">`;
    const hint = document.createElement('small');
    hint.textContent = ['docufcn', 'eynollah_textline', 'rtmdet_lines'].includes(name) ? 'Фильтр найденных объектов' : 'Порог модели';
    settings.append(hint); elements.providers.append(settings);
  });
  updateRunEnabled();
}
async function loadImages() {
  const dataset = document.querySelector('#dataset-select')?.value;
  const data = await json(dataset ? `/api/datasets/${dataset}` : '/api/images');
  const images = dataset ? data.images.map(x => x.name) : data.images;
  elements.image.replaceChildren();
  if (!images.length) elements.image.add(new Option('Нет доступных изображений', ''));
  images.forEach((name) => elements.image.add(new Option(name, name)));
  updateRunEnabled();
}
function updateRunEnabled() {
  elements.run.disabled = state.runBusy || !elements.image.value || !document.querySelector('.provider-row input:checked');
}
function selectProviders() { return [...document.querySelectorAll('.provider-row input:checked')].map((input) => input.value); }
function imageUrl(name) {
  const dataset = state.result?.dataset_id;
  return dataset ? `/api/datasets/${dataset}/image?name=${encodeURIComponent(name)}` : `/files/images/${name.split('/').map(encodeURIComponent).join('/')}`;
}
function providerOptions() {
  const options = {};
  selectProviders().forEach(name => {
    const input = document.querySelector(`[data-threshold="${name}"]`);
    if (input && input.value !== '') {
      if (!input.checkValidity()) throw new Error(`Confidence ${name}: допустимо 0–1`);
      options[name] = { confidence: Number(input.value) };
    }
  });
  return options;
}
function safeColor(value, fallback) { return /^#[0-9a-f]{6}$/i.test(value) ? value : fallback; }
function ensureFilter(provider, annotations) {
  const previous = state.filters[provider] || { visible: true, labels: {}, colors: {} };
  const labels = [...new Set(annotations.map((annotation) => annotation.label))];
  const next = { visible: previous.visible !== false, labels: {}, colors: { ...previous.colors } };
  labels.forEach((label, index) => {
    next.labels[label] = previous.labels[label] !== false;
    next.colors[label] = safeColor(previous.colors[label], index ? classColors[index % classColors.length] : colors[provider] || '#0d7f80');
  });
  state.filters[provider] = next;
}
function visibleAnnotations(provider, item) {
  const filter = state.filters[provider];
  if (!filter || filter.visible === false) return [];
  return item.annotations.filter((annotation) => filter.labels[annotation.label] !== false);
}
function resultTiming(item) {
  if (!item.cached) return item.elapsed_seconds.toFixed(2) + ' sec';
  return item.cache_origin === 'legacy' ? 'Сохранено (старый запуск)' : 'Из кэша';
}
function renderResultStrip() {
  if (!state.result) return;
  const cards = Object.entries(state.result.providers).map(([name, item]) => {
    ensureFilter(name, item.annotations);
    const filter = state.filters[name];
    const labels = [...new Set(item.annotations.map((annotation) => annotation.label))];
    const controls = labels.map((label) => `<label class="class-control"><input type="checkbox" data-class-provider="${escapeHtml(name)}" data-class-label="${escapeHtml(label)}"${filter.labels[label] ? ' checked' : ''}><input type="color" value="${filter.colors[label]}" data-color-provider="${escapeHtml(name)}" data-color-label="${escapeHtml(label)}" title="Цвет ${escapeHtml(label)}"><span>${escapeHtml(label)}</span></label>`).join('');
    const recognized = visibleAnnotations(name, item).map((annotation) => annotation.text?.trim()).filter(Boolean);
    const textPreview = recognized.length ? `<div class="recognized-text"><b>Распознано:</b> ${escapeHtml(recognized.join(' · ').slice(0, 1200))}${recognized.join(' · ').length > 1200 ? '…' : ''}</div>` : '';
    return `<article class="result-card" style="--card-color:${colors[name] || '#16221d'}"><div class="result-card-head"><strong>${escapeHtml(providerTitles[name] || name)}</strong><label class="visibility-control"><input type="checkbox" data-visible-provider="${escapeHtml(name)}"${filter.visible ? ' checked' : ''}> показывать</label></div><span>${visibleAnnotations(name, item).length}/${item.annotations.length} оставлено<br>${resultTiming(item)}</span><div class="class-controls">${controls || '<small>Нет сегментов</small>'}</div>${textPreview}</article>`;
  });
  elements.strip.innerHTML = cards.join('');
}
function showResult(result) {
  state.result = result;
  elements.title.textContent = result.image_name;
  elements.consensus.textContent = `${Math.round(result.consensus_score * 100)}%`;
  elements.source.onload = () => {
    elements.canvas.classList.add('loaded');
    requestAnimationFrame(drawAnnotations);
  };
  elements.source.src = imageUrl(result.image_name);
  if (elements.source.complete) {
    elements.canvas.classList.add('loaded');
    requestAnimationFrame(drawAnnotations);
  }
  Object.entries(result.providers).forEach(([name, item]) => ensureFilter(name, item.annotations));
  state.llm.messages = [];
  state.llm.refinedAnnotations = [];
  state.llm.reviewAnnotations = [];
  state.llm.compactReview = null;
  state.llm.notes = '';
  elements.llmPreview.hidden = true;
  elements.llmResultMessage.textContent = '';
  elements.llmApply.checked = false;
  elements.llmApply.disabled = true;
  elements.llmExport.disabled = true;
  elements.llmResponse.textContent = 'Ответ VLM появится здесь.';
  renderResultStrip();
  elements.export.disabled = false;
  updateLlmEnabled();
  requestAnimationFrame(drawAnnotations);
  window.setTimeout(drawAnnotations, 120);
  window.dispatchEvent(new CustomEvent('labelbench-result', {detail: result}));
}
function drawAnnotation(context, annotation, color, imageRect, wrapRect, result) {
  const offsetX = imageRect.left - wrapRect.left;
  const offsetY = imageRect.top - wrapRect.top;
  const sx = imageRect.width / result.image_size[0];
  const sy = imageRect.height / result.image_size[1];
  const [x, y, width, height] = annotation.bbox_xywh;
  const caption = annotation.text?.trim() || annotation.label;
  context.beginPath();
  if (annotation.polygon) {
    annotation.polygon.forEach(([px, py], index) => {
      const pointX = offsetX + px * sx;
      const pointY = offsetY + py * sy;
      if (index === 0) context.moveTo(pointX, pointY);
      else context.lineTo(pointX, pointY);
    });
    context.closePath();
  } else {
    context.rect(offsetX + x * sx, offsetY + y * sy, width * sx, height * sy);
  }
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.stroke();
  context.fillStyle = color;
  context.font = '10px DM Mono, monospace';
  if (document.querySelector('#show-captions')?.checked !== false) context.fillText(caption, offsetX + x * sx + 3, Math.max(10, offsetY + y * sy - 5));
}
function drawAnnotations() {
  const result = state.result;
  if (!result) return;
  const canvas = elements.overlay;
  const wrapRect = elements.canvas.getBoundingClientRect();
  const imageRect = elements.source.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(wrapRect.width * pixelRatio));
  canvas.height = Math.max(1, Math.round(wrapRect.height * pixelRatio));
  const context = canvas.getContext('2d');
  if (!context) return;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, wrapRect.width, wrapRect.height);
  if (elements.llmApply.checked && !elements.llmApply.disabled) {
    state.llm.refinedAnnotations.forEach((annotation) => drawAnnotation(context, annotation, '#ef7d32', imageRect, wrapRect, result));
    return;
  }
  Object.entries(result.providers).forEach(([provider, item]) => {
    visibleAnnotations(provider, item).forEach((annotation) => {
      drawAnnotation(context, annotation, state.filters[provider]?.colors[annotation.label] || colors[provider] || '#fff', imageRect, wrapRect, result);
    });
  });
  window.dispatchEvent(new Event('labelbench-drawn'));
}
function drawLlmPreview() {
  if (!state.result || elements.llmPreview.hidden || !elements.llmSource.naturalWidth) return;
  const wrapRect = elements.llmCanvas.getBoundingClientRect();
  const imageRect = elements.llmSource.getBoundingClientRect();
  const canvas = elements.llmOverlay;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(wrapRect.width * ratio));
  canvas.height = Math.max(1, Math.round(wrapRect.height * ratio));
  const context = canvas.getContext('2d');
  if (!context) return;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  state.llm.reviewAnnotations.forEach(annotation => {
    const status = annotation.attributes?.review_status || 'kept';
    if (state.llm.reviewFilters[status] === false) return;
    context.setLineDash(status === 'removed' ? [6, 4] : []);
    drawAnnotation(context, annotation, reviewStyles[status]?.[0] || '#ef7d32', imageRect, wrapRect, state.result);
  });
}
function showLlmPreview(parsed) {
  elements.llmPreview.hidden = !parsed;
  if (!parsed) return;
  const counts = Object.entries(reviewStyles).map(([status, style]) => `${style[1]}: ${state.llm.reviewAnnotations.filter(a => (a.attributes?.review_status || 'kept') === status).length}`);
  elements.llmPreviewCount.textContent = counts.join(' · ');
  elements.llmSource.onload = drawLlmPreview;
  elements.llmSource.src = imageUrl(state.result.image_name);
  requestAnimationFrame(drawLlmPreview);
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
function updateLlmEnabled() {
  const hasProviders = state.result && Object.keys(state.result.providers).some(name => state.filters[name]?.visible !== false);
  const options = llmRequestOptions();
  elements.llmSend.disabled = state.llm.busy || !hasProviders || !options.model || !elements.llmPrompt.value.trim()
    || (options.backend === 'routerai' && !options.api_key)
    || !Number.isInteger(options.max_output_tokens) || options.max_output_tokens < 512 || options.max_output_tokens > 32768;
}
async function loadLlmModels() {
  elements.llmStatus.textContent = 'Проверяю LM Studio...';
  try {
    const previousModel = state.llm.model;
    const data = await json('/api/llm/models');
    state.llm.localUrl = data.base_url;
    state.llm.models = data.models || [];
    elements.llmModel.replaceChildren();
    if (!state.llm.models.length) {
      elements.llmModel.add(new Option('Модель не найдена', ''));
      state.llm.model = '';
      elements.llmStatus.textContent = data.detail || 'Запусти модель в LM Studio и обнови список.';
    } else {
      state.llm.model = state.llm.models.includes(previousModel) ? previousModel : (state.llm.models.find(model => /qwen3-vl-4b/i.test(model)) || state.llm.models.find(model => /qwen.*vl/i.test(model)) || state.llm.models[0]);
      state.llm.models.forEach((model) => elements.llmModel.add(new Option(model, model)));
      elements.llmModel.value = state.llm.model;
      elements.llmStatus.textContent = `LM Studio API: ${state.llm.models.length} моделей в списке`;
    }
  } catch (error) { elements.llmStatus.textContent = `LM Studio: ${error.message}`; }
  updateLlmConnection();
}
async function sendToLlm() {
  if (!state.result || state.llm.busy) return;
  state.llm.busy = true;
  const sourceRun = state.result;
  const options = llmRequestOptions();
  const journal = startTaskConsole('llm');
  const controller = new AbortController();
  const job = { identifier: journal.identifier, controller, stopping: false };
  state.llm.activeJob = job;
  document.querySelector('#llm-cancel').disabled = false;
  const started = Date.now();
  elements.llmSend.disabled = true;
  elements.llmApply.checked = false;
  elements.llmApply.disabled = true;
  elements.llmExport.disabled = true;
  showLlmPreview(false);
  drawAnnotations();
  const showWaiting = () => { if (!job.stopping) elements.llmStatus.textContent = `VLM проверяет полигоны ${options.request_mode === 'all' ? 'все сразу' : 'по частям'}: ${Math.floor((Date.now() - started) / 1000)} сек. Ожидаю ответ…`; };
  showWaiting();
  const timer = setInterval(showWaiting, 1000);
  const prompt = elements.llmPrompt.value.trim();
  try {
    const data = await json('/api/llm/review', {
      signal: controller.signal,
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        run_id: state.result.run_id, ...options, prompt, progress_id: journal.identifier,
        providers: Object.keys(state.result.providers).filter(name => state.filters[name]?.visible !== false),
        annotation_ids: Object.entries(state.result.providers).flatMap(([name,item]) => visibleAnnotations(name,item).map(a => a.id)),
      }),
    });
    clearInterval(timer);
    if (job.stopping) throw new DOMException('Cancelled', 'AbortError');
    if (state.result !== sourceRun) {
      elements.llmStatus.textContent = 'Открыт другой результат. Запустите проверку VLM для него.';
      return;
    }
    state.llm.resultModel = data.model;
    state.llm.resultBackend = data.backend || options.backend;
    state.llm.resultMode = data.request_mode || options.request_mode;
    state.llm.refinedAnnotations = data.refined_annotations || [];
    state.llm.reviewAnnotations = data.review_annotations || state.llm.refinedAnnotations;
    state.llm.compactReview = data.compact_review;
    state.llm.notes = data.notes || '';
    elements.llmResponse.textContent = data.parsed ? data.content : 'Ответ модели не получен. Исходная разметка не изменена.';
    elements.llmStatus.textContent = data.parsed ? `${data.complete === false ? 'Частичный ответ. ' : ''}Принято: ${state.llm.refinedAnnotations.length}; частей: ${data.batch_count || 1}${data.usage?.total_tokens ? '; токенов: ' + data.usage.total_tokens : ''}` : data.notes || 'VLM ответил, но JSON не распознан.';
    elements.llmResultMessage.textContent = data.parsed
      ? `${data.retried ? 'Ответ получен после автоматического повтора. ' : ''}${data.notes || ''}${data.invalid_decisions ? ' Некорректных решений пропущено: ' + data.invalid_decisions : ''}`
      : `Итоговое изображение не построено: ${data.notes || 'незавершённый JSON'}. Исходная разметка сохранена.`;
    if (data.evaluation) elements.llmResultMessage.textContent += ` GT @ IoU 0.5: F1 ${data.evaluation.f1.toFixed(3)}, precision ${data.evaluation.precision.toFixed(3)}, recall ${data.evaluation.recall.toFixed(3)}.`;
    elements.llmApply.disabled = !data.parsed;
    elements.llmExport.disabled = !data.parsed;
    if (!data.parsed) elements.llmApply.checked = false;
    showLlmPreview(data.parsed);
    drawAnnotations();
  } catch (error) {
    elements.llmStatus.textContent = error.name === 'AbortError' || job.stopping
      ? 'VLM остановлена. Исходная разметка сохранена.' : `Ошибка VLM: ${error.message}`;
    elements.llmResponse.textContent = error.name === 'AbortError' || job.stopping ? 'Запрос отменён пользователем.' : 'Ответ модели не получен.';
  }
  finally {
    clearInterval(timer);
    document.querySelector('#llm-cancel').disabled = true;
    await journal.finish(elements.llmStatus.textContent);
    state.llm.activeJob = null;
    state.llm.busy = false;
    updateLlmEnabled();
  }
}
function exportLlmResult() {
  if (!state.result || elements.llmExport.disabled) return;
  const payload = {
    schema_version: '1.0', source: state.llm.resultBackend, model: state.llm.resultModel,
    request_mode: state.llm.resultMode,
    image_name: state.result.image_name, image_size: state.result.image_size,
    run_id: state.result.run_id, review: state.llm.compactReview, notes: state.llm.notes,
    annotations: state.llm.refinedAnnotations,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob);
  link.download = `${state.result.image_name.replace(/[^a-z0-9._-]+/gi, '_')}_${state.result.run_id}_vlm.json`;
  link.click(); URL.revokeObjectURL(link.href);
  elements.llmStatus.textContent = 'Улучшенная разметка скачана в JSON.';
}
async function run() {
  if (state.runBusy) return;
  state.runBusy = true;
  const journal = startTaskConsole('run');
  elements.run.disabled = true;
  elements.status.textContent = 'Модели выполняют inference — первый запуск может скачать веса…';
  try {
    const result = await json('/api/runs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_name: elements.image.value, providers: selectProviders(), options: providerOptions(), dataset_id: document.querySelector('#dataset-select')?.value || null, force: document.querySelector('#force-rerun').checked, progress_id: journal.identifier }),
    });
    showResult(result);
    elements.status.textContent = `Готово: run ${result.run_id}`;
  } catch (error) { elements.status.textContent = `Ошибка: ${error.message}`; }
  finally { await journal.finish(elements.status.textContent); state.runBusy = false; updateRunEnabled(); }
}
function showError(error) { elements.status.textContent = `Ошибка: ${error.message}`; }
elements.run.addEventListener('click', run);
elements.refresh.addEventListener('click', () => loadImages().catch(showError));
elements.image.addEventListener('change', updateRunEnabled);
elements.providers.addEventListener('change', updateRunEnabled);
elements.llmModel.addEventListener('change', () => { state.llm.model = elements.llmModel.value; updateLlmEnabled(); });
elements.llmPrompt.addEventListener('input', updateLlmEnabled);
elements.llmRefresh.addEventListener('click', () => loadLlmModels());
elements.llmSend.addEventListener('click', sendToLlm);
elements.llmApply.addEventListener('change', drawAnnotations);
elements.llmExport.addEventListener('click', exportLlmResult);
elements.llmPreview.addEventListener('change', event => {
  const status = event.target.dataset.reviewStatus;
  if (!status) return;
  state.llm.reviewFilters[status] = event.target.checked;
  drawLlmPreview();
});
elements.strip.addEventListener('change', (event) => {
  const target = event.target;
  const provider = target.dataset.visibleProvider || target.dataset.classProvider || target.dataset.colorProvider;
  if (!provider || !state.filters[provider]) return;
  if (target.dataset.visibleProvider) state.filters[provider].visible = target.checked;
  if (target.dataset.classProvider) state.filters[provider].labels[target.dataset.classLabel] = target.checked;
  if (target.dataset.colorProvider) state.filters[provider].colors[target.dataset.colorLabel] = safeColor(target.value, colors[provider] || '#ffffff');
  renderResultStrip();
  drawAnnotations();
  updateLlmEnabled();
});
elements.export.addEventListener('click', exportFiltered);
window.addEventListener('resize', drawAnnotations);
window.addEventListener('resize', drawLlmPreview);
initializeLlmControls();
Promise.all([loadHealth(), loadImages(), loadLlmModels()]).catch(showError);
