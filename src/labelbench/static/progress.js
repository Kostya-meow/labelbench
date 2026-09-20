function startTaskConsole(kind) {
  const panel = document.querySelector(`#${kind}-console`);
  const timerLabel = document.querySelector(`#${kind}-elapsed`);
  const identifier = crypto.randomUUID();
  const started = Date.now();
  let active = true, polling = false, lastEvents = [], finalMessage = '';
  const render = () => {
    timerLabel.textContent = `${((Date.now() - started) / 1000).toFixed(1)} сек.`;
    const lines = lastEvents.map(event => `[${event.seconds.toFixed(1)}с] ${event.message}`);
    if (!lines.length) lines.push('[0.0с] Отправляю запрос серверу…');
    if (finalMessage) lines.push(finalMessage);
    panel.textContent = lines.join('\n');
    panel.scrollTop = panel.scrollHeight;
  };
  const poll = async () => {
    if (polling) return;
    polling = true;
    try {
      const response = await fetch(`/api/progress/${identifier}`, {cache: 'no-store', signal: AbortSignal.timeout(4000)});
      if (response.ok) lastEvents = (await response.json()).events || [];
    } catch { /* Keep the existing journal when the connection is temporarily lost. */ }
    finally { polling = false; if (active) render(); }
  };
  render();
  const clock = setInterval(render, 250);
  const updates = setInterval(poll, 800);
  return {
    identifier,
    async finish(message) {
      clearInterval(updates);
      clearInterval(clock);
      await poll();
      finalMessage = `[${((Date.now() - started) / 1000).toFixed(1)}с] ${message}`;
      render();
      active = false;
    },
  };
}
