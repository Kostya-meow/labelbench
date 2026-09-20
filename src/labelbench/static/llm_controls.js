function llmRequestOptions() {
  const backend = document.querySelector('#llm-backend').value;
  return {
    backend,
    request_mode: document.querySelector('#llm-mode').value,
    model: backend === 'routerai' ? document.querySelector('#routerai-model').value.trim() : state.llm.model,
    api_key: backend === 'routerai' ? document.querySelector('#routerai-key').value.trim() : '',
    max_output_tokens: Number(document.querySelector('#llm-max-tokens').value),
  };
}

function updateLlmConnection() {
  const options = llmRequestOptions();
  const remote = options.backend === 'routerai';
  document.querySelector('#routerai-fields').hidden = !remote;
  elements.llmModel.hidden = remote;
  elements.llmRefresh.hidden = remote;
  elements.llmUrl.value = remote ? 'https://routerai.ru/api/v1' : (state.llm.localUrl || 'http://localhost:1234/v1');
  const privacy = remote ? 'Изображение и разметка отправятся в RouterAI. Ключ хранится только в этой вкладке. ' : 'Запрос обрабатывает локальный сервер. ';
  document.querySelector('#llm-connection-hint').textContent = privacy + (options.request_mode === 'all'
    ? 'Все выбранные группы отправятся вместе, без автоматического разделения при переполнении контекста.'
    : 'Группы отправляются по частям. Кандидаты одной группы остаются вместе.');
  updateLlmEnabled();
}

function initializeLlmControls() {
  document.querySelector('#llm-cancel').addEventListener('click', cancelVlm);
  ['#llm-backend', '#llm-mode'].forEach(selector => document.querySelector(selector).addEventListener('change', updateLlmConnection));
  ['#routerai-model', '#routerai-key', '#llm-max-tokens'].forEach(selector => document.querySelector(selector).addEventListener('input', updateLlmEnabled));
  updateLlmConnection();
}

async function cancelVlm() {
  const job = state.llm.activeJob;
  if (!job || job.stopping) return;
  job.stopping = true;
  document.querySelector('#llm-cancel').disabled = true;
  elements.llmStatus.textContent = 'Останавливаю запрос VLM…';
  try {
    let response;
    for (let attempt = 0; attempt < 15; attempt++) {
      response = await fetch(`/api/llm/cancel/${job.identifier}`, {method: 'POST', signal: AbortSignal.timeout(5000)});
      if (response.status !== 404) break;
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    if (!response.ok) throw new Error(`Сервер не подтвердил отмену (${response.status})`);
    job.controller.abort();
  } catch (error) {
    if (state.llm.activeJob !== job) return;
    job.stopping = false;
    document.querySelector('#llm-cancel').disabled = false;
    elements.llmStatus.textContent = `Не удалось отменить: ${error.message}. Попробуйте ещё раз.`;
  }
}
