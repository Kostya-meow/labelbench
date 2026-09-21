/* Dataset selection, GT overlays and persisted experiments. No API secrets stored here. */
(() => {
  const $ = id => document.getElementById(id);
  const experiment = {id:null, current:null, timer:null, truth:[], truthRun:null};
  const post = (url, body) => json(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const message = error => { $('experiment-status').textContent = error.message; };

  function tab(name) {
    document.querySelector('.viewer').hidden = name !== 'viewer';
    $('experiments-panel').hidden = name !== 'experiments';
    $('robustness-panel').hidden = name !== 'robustness';
    document.querySelectorAll('[data-tab]').forEach(button => button.setAttribute('aria-selected', String(button.dataset.tab === name)));
    requestAnimationFrame(drawAnnotations);
  }
  document.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => tab(button.dataset.tab)));

  async function datasets(selected = '') {
    const data = await json('/api/datasets');
    $('dataset-select').replaceChildren(new Option('Рабочие изображения', ''));
    data.datasets.forEach(item => $('dataset-select').add(new Option(`${item.name} · ${item.image_count} фото`, item.id)));
    $('dataset-select').value = selected;
    if (!$('dataset-root').value) $('dataset-root').value = data.suggested_root;
  }
  $('dataset-select').addEventListener('change', () => loadImages().catch(showError));
  $('dataset-import').addEventListener('click', async () => {
    $('dataset-import').disabled = true;
    $('dataset-status').textContent = 'Читаю GT и проверяю изображения…';
    try {
      const data = await post('/api/datasets', {root:$('dataset-root').value});
      await datasets(data.id); await loadImages();
      $('dataset-status').textContent = `${data.images.length} фото · ${data.polygons} объектов · отсутствуют: ${data.missing_images.length} · некорректных полигонов: ${data.invalid_polygons}`;
    } catch(error) { $('dataset-status').textContent = error.message; }
    finally { $('dataset-import').disabled = false; }
  });

  function table(rows) {
    return `<table class="metrics-table"><thead><tr><th>Метод</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP / FP / FN</th><th>IoU совпавших</th></tr></thead><tbody>${Object.entries(rows).map(([name,m]) => `<tr><td>${escapeHtml(providerTitles[name] || name)}</td><td>${m.precision.toFixed(3)}</td><td>${m.recall.toFixed(3)}</td><td>${m.f1.toFixed(3)}</td><td>${m.tp} / ${m.fp} / ${m.fn}</td><td>${m.mean_matched_iou == null ? '—' : m.mean_matched_iou.toFixed(3)}</td></tr>`).join('')}</tbody></table>`;
  }
  window.addEventListener('labelbench-result', async event => {
    const run = event.detail;
    experiment.truth = []; experiment.truthRun = null;
    $('evaluate-page').disabled = !run.dataset_id;
    $('page-metrics').replaceChildren();
    if (!run.dataset_id) return;
    try {
      const data = await json(`/api/datasets/${run.dataset_id}/truth?name=${encodeURIComponent(run.image_name)}`);
      if (state.result !== run) return;
      experiment.truth = data.annotations; experiment.truthRun = run;
      drawAnnotations();
    } catch(error) { $('page-metrics').textContent = `GT: ${error.message}`; }
  });
  window.addEventListener('labelbench-drawn', () => {
    if (!$('show-gt').checked || experiment.truthRun !== state.result) return;
    const ctx = elements.overlay.getContext('2d');
    ctx.save(); ctx.setLineDash([5,3]);
    experiment.truth.forEach(a => drawAnnotation(ctx,a,'#df29ba',elements.source.getBoundingClientRect(),elements.canvas.getBoundingClientRect(),state.result));
    ctx.restore();
  });
  $('show-gt').addEventListener('change', drawAnnotations);
  $('show-captions').addEventListener('change', () => {drawAnnotations(); drawLlmPreview();});
  $('fullscreen').addEventListener('click', () => elements.canvas.requestFullscreen().catch(showError));
  document.addEventListener('fullscreenchange', () => requestAnimationFrame(drawAnnotations));
  $('evaluate-page').addEventListener('click', async () => {
    const run = state.result;
    try {
      const scores = await post(`/api/runs/${run.run_id}/evaluate`, {iou:Number($('experiment-iou').value),text_only:$('experiment-text').checked});
      if (state.result === run) $('page-metrics').innerHTML = table(scores);
    } catch(error) { $('page-metrics').textContent = error.message; }
  });

  function render(item) {
    experiment.current = item; experiment.id = item.id;
    $('experiment-status').textContent = `${item.config.name} · ${item.status} · ${item.completed}/${item.images.length} · ${item.elapsed.toFixed(1)} с · ${item.message}`;
    $('experiment-progress').max = item.images.length; $('experiment-progress').value = item.completed;
    $('experiment-log').textContent = [...item.events.map(x => `[${x.seconds.toFixed(1)}с] ${x.message}`), ...item.errors.map(x => `${x.image}: ${x.error}`)].join('\n');
    $('experiment-summary').innerHTML = table(item.summary);
    const running = ['queued','running'].includes(item.status);
    $('experiment-cancel').disabled = !running; $('experiment-start').disabled = running;
    $('experiment-export').disabled = false;
    $('experiment-open').disabled = item.completed === 0;
    const currentPage = $('experiment-page').value;
    $('experiment-page').replaceChildren();
    item.images.forEach((name,index) => $('experiment-page').add(new Option(name, String(index))));
    if (currentPage) $('experiment-page').value = currentPage;
    const currentMethod = $('experiment-method').value;
    $('experiment-method').replaceChildren();
    Object.keys(item.summary).forEach(name => $('experiment-method').add(new Option(providerTitles[name] || name,name)));
    if (Object.keys(item.summary).includes(currentMethod)) $('experiment-method').value = currentMethod;
    clearTimeout(experiment.timer);
    if (running) experiment.timer = setTimeout(() => load(item.id).catch(error => {
      message(error); experiment.timer = setTimeout(() => load(item.id).catch(message),3000);
    }),1000);
  }
  async function load(id) { render(await json(`/api/experiments/${id}`)); }
  async function history() {
    const {experiments} = await json('/api/experiments');
    $('experiment-history').replaceChildren();
    experiments.forEach(item => {
      const button = document.createElement('button'); button.className='history-row';
      button.textContent = `${item.config.name} · ${item.config.split} · ${item.status} · ${item.completed}/${item.images.length} · ${new Date(item.created_at).toLocaleString()}`;
      button.addEventListener('click', () => load(item.id).catch(message));
      $('experiment-history').append(button);
    });
  }
  $('experiment-form').addEventListener('submit', async event => {
    event.preventDefault();
    if (!$('dataset-select').value) return message(new Error('Подключите и выберите датасет с эталоном слева.'));
    if (!selectProviders().length) return message(new Error('Выберите хотя бы одну модель слева.'));
    $('experiment-start').disabled = true;
    try {
      const data = await post('/api/experiments', {
        name:$('experiment-name').value, dataset_id:$('dataset-select').value, providers:selectProviders(), options:providerOptions(),
        split:$('experiment-split').value, seed:Number($('experiment-seed').value), limit:Number($('experiment-limit').value),
        match_iou:Number($('experiment-iou').value), fusion_iou:Number($('fusion-iou').value), min_votes:Number($('fusion-votes').value),
        methods:[...document.querySelectorAll('[name="method"]:checked')].map(x=>x.value), text_only:$('experiment-text').checked,
        force:$('force-rerun').checked,
      });
      render(data); await history();
    } catch(error) {message(error); $('experiment-start').disabled=false;}
  });
  $('experiment-cancel').addEventListener('click', async () => {
    try {await post(`/api/experiments/${experiment.id}/cancel`, {}); $('experiment-status').textContent='Остановка после текущего inference; новые модели не запускаются.';}
    catch(error) {message(error);}
  });
  $('experiment-open').addEventListener('click', async () => {
    try {
      const data = await json(`/api/experiments/${experiment.id}/pages/${$('experiment-page').value}`);
      const method = $('experiment-method').value;
      const run = await post(`/api/experiments/${experiment.id}/pages/${$('experiment-page').value}/open?method=${encodeURIComponent(method)}`, {});
      showResult(run);
      $('page-metrics').innerHTML = table({[method]:data.metrics[method]});
      tab('viewer');
    } catch(error) {message(error);}
  });
  $('experiment-export').addEventListener('click', () => {
    const link=document.createElement('a'); link.href=URL.createObjectURL(new Blob([JSON.stringify(experiment.current,null,2)],{type:'application/json'}));
    link.download=`experiment-${experiment.id}.json`; link.click(); URL.revokeObjectURL(link.href);
  });
  $('external-import').addEventListener('change', async event => {
    try {
      const file = event.target.files[0]; if (!file) return;
      if (!$('dataset-select').value) throw new Error('Выберите датасет с GT.');
      const data = JSON.parse(await file.text());
      const result = await post('/api/evaluations', {dataset_id:$('dataset-select').value,
        image_name:data.image_name || elements.image.value, method:data.fusion_method || data.method || file.name,
        annotations:data.annotations || data.refined_annotations, iou:Number($('experiment-iou').value)});
      showResult(result.run); $('page-metrics').innerHTML=table(result.metrics); tab('viewer');
    } catch(error) { message(error); }
    finally { event.target.value=''; }
  });
  datasets().catch(showError); history().catch(message);
})();
