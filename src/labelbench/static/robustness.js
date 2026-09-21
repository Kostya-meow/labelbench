/* Durable background experiments: this script only observes, never owns the job. */
(() => {
  const $ = id => document.getElementById(id);
  let identifier = localStorage.getItem('labelbench-robust-run'), timer, snapshot, pageData;
  const request = (url, data = {}) => json(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  const message = error => { $('robust-status').textContent = error.message; };
  const elapsed = seconds => seconds == null ? 'рассчитывается' : `${Math.floor(seconds/3600)}ч ${Math.floor(seconds%3600/60)}м ${Math.floor(seconds%60)}с`;
  const fixed = value => value == null ? '—' : Number(value).toFixed(3);
  const root = () => `/api/robustness/${identifier}`;

  async function history() {
    const data = await json('/api/robustness');
    $('robust-history').replaceChildren();
    data.runs.forEach(run => $('robust-history').add(new Option(`${run.name} · ${run.status} · ${run.pages_done} фото`, run.id)));
    if (!data.runs.some(r => r.id === identifier)) identifier = data.runs.find(r => ['running','queued','cancelling'].includes(r.status))?.id || data.runs[0]?.id;
    if (identifier) { $('robust-history').value = identifier; await poll(); }
  }
  function charts(report) {
    const rows = Object.entries(report.methods || {});
    $('robust-chart').innerHTML = rows.map(([name,row]) => `<div class="robust-bar"><span>${escapeHtml(name)}</span><meter min="0" max="1" value="${row.f1}"></meter><b>${fixed(row.f1)}</b></div>`).join('');
    const fields = ['precision','recall','f1','f1_75','boundary_f1','region_iou','region_dice'];
    $('robust-metrics').innerHTML = `<table class="metrics-table"><thead><tr><th>Метод</th>${fields.map(f=>`<th>${f}</th>`).join('')}<th>TP / FP / FN</th><th>Страниц</th></tr></thead><tbody>${rows.map(([name,r])=>`<tr><td>${escapeHtml(name)}</td>${fields.map(f=>`<td>${fixed(r[f])}</td>`).join('')}<td>${r.tp} / ${r.fp} / ${r.fn}</td><td>${r.pages}</td></tr>`).join('')}</tbody></table>`;
    const field=$('robust-heat-field').value, source=(field==='f1'?report.views:report.stability)||{};
    const keys = Object.keys(source);
    const providers = [...new Set(keys.map(k=>k.split('|')[0]))], views = [...new Set(keys.map(k=>k.split('|')[1]))];
    $('robust-heatmap').innerHTML = `<table class="metrics-table"><thead><tr><th>Модель / ${escapeHtml(field)}</th>${views.map(v=>`<th>${escapeHtml(v)}</th>`).join('')}</tr></thead><tbody>${providers.map(p=>`<tr><td>${escapeHtml(p)}</td>${views.map(v=>{const value=source[`${p}|${v}`]?.[field];return `<td style="background:rgba(38,132,99,${field!=='latency'&&value!=null ? value*.6 : 0})">${fixed(value)}</td>`;}).join('')}</tr>`).join('')}</tbody></table>`;
  }
  async function poll() {
    clearTimeout(timer);
    if (!identifier) return;
    localStorage.setItem('labelbench-robust-run', identifier);
    try {
      const requestedId=identifier;
      const reply = await json(root());
      if(requestedId!==identifier)return;
      snapshot = reply;
      const item = snapshot, active = ['running','queued','cancelling'].includes(item.status);
      $('robust-status').textContent = `${item.config.name} · ${item.status} · ${item.message}\n${item.config.providers.join(', ')} · ${item.config.views.length} видов · max ${item.config.max_side}px`;
      const finished = item.tasks_done+item.task_errors;
      $('robust-progress').max = item.tasks_total; $('robust-progress').value = finished;
      $('robust-counters').innerHTML = [['Страницы',`${item.pages_done} / ${item.manifest.images.length}`],['Предсказания',`${finished} / ${item.tasks_total}`],['Прошло',elapsed(item.elapsed)],['Осталось ≈',active?elapsed(item.eta_seconds):'—'],['Ошибки',item.task_errors]].map(([title,value])=>`<div><small>${title}</small><strong>${value}</strong></div>`).join('');
      $('robust-log').textContent = item.events.map(e=>`[${new Date(e.stamp*1000).toLocaleTimeString()}] ${e.message}`).join('\n');
      $('robust-log').scrollTop = $('robust-log').scrollHeight;
      $('robust-cancel').disabled = !active; $('robust-resume').disabled = active || item.status==='done'; $('robust-start').disabled = active;
      charts(item.report || {});
      const previous = $('robust-page').value;
      $('robust-page').replaceChildren();
      (item.completed_images || []).forEach(name=>$('robust-page').add(new Option(name,name)));
      if ((item.completed_images || []).includes(previous)) $('robust-page').value=previous;
      const previousMethod = $('robust-method').value;
      $('robust-method').replaceChildren();
      Object.keys(item.report?.methods || {}).forEach(name=>$('robust-method').add(new Option(name,name)));
      if (previousMethod && item.report?.methods?.[previousMethod]) $('robust-method').value=previousMethod;
      else if (item.report?.methods?.joint_stable) $('robust-method').value='joint_stable';
      if(!$('robust-method').options.length)['joint_stable','joint_medoid','nms',...item.config.providers.map(p=>'model:'+p)].forEach(name=>$('robust-method').add(new Option(name,name)));
      $('robust-artifacts').innerHTML = ['protocol.json',...(!active && item.report?.statistics ? ['analysis.json','page-metrics.csv','analysis-report.md','figures/method-f1.png','figures/corruption-f1.png','figures/trust-risk.png'] : [])].map(name=>`<a class="export-button" href="${root()}/artifact?name=${encodeURIComponent(name)}">${escapeHtml(name)}</a>`).join('');
      if(!pageData && $('robust-page').value)await openPage();
    } catch(error) { message(error); }
    timer = setTimeout(poll, 3000);
  }
  function draw() {
    if (!pageData || !$('robust-image').naturalWidth) return;
    const image=$('robust-image'), canvas=$('robust-overlay'), rect=image.getBoundingClientRect(), host=$('robust-view').getBoundingClientRect();
    canvas.width=Math.round(host.width*devicePixelRatio); canvas.height=Math.round(host.height*devicePixelRatio);
    const context=canvas.getContext('2d'); context.scale(devicePixelRatio,devicePixelRatio);
    const paint=(rows,color,dashed)=>{context.strokeStyle=color;context.lineWidth=1.8;context.setLineDash(dashed?[5,3]:[]); rows.forEach(a=>{if(!a.polygon?.length)return;context.beginPath();a.polygon.forEach(([x,y],i)=>{const px=rect.left-host.left+x/pageData.size[0]*rect.width,py=rect.top-host.top+y/pageData.size[1]*rect.height;i?context.lineTo(px,py):context.moveTo(px,py);});context.closePath();context.stroke();});};
    if($('robust-gt').checked)paint(pageData.truth,'#df29ba',true);
    if($('robust-pred').checked)paint(pageData.annotations,'#16ad69',false);
  }
  async function openPage() {
    if (!identifier || !$('robust-page').value) return;
    const requestedId=identifier, requestedPage=$('robust-page').value, requestedMethod=$('robust-method').value;
    const data=await json(`${root()}/page?image=${encodeURIComponent(requestedPage)}&method=${encodeURIComponent(requestedMethod)}`);
    if(identifier!==requestedId || $('robust-page').value!==requestedPage || $('robust-method').value!==requestedMethod)return;
    if(!data.ready){$('robust-page-status').textContent=data.message;return;}
    pageData=data;
    $('robust-page-status').textContent=`${data.image} · выбрано ${data.annotations.length} · GT ${data.has_gt?data.truth.length:'неполный, не оценивается'} · F1 ${fixed(data.metrics?.f1)}`;
    $('robust-image').onload=draw;
    $('robust-image').src=`${root()}/image?name=${encodeURIComponent(data.image)}`;
    $('robust-evidence').textContent=data.evidence.map(g=>`${g.group}: ${g.status}; семейств ${g.families}; устойчивость ${fixed(g.stability)}; согласие ${fixed(g.agreement)}; доверие ${fixed(g.trust)}; выбран ${g.selected_source}`).join('\n');
    requestAnimationFrame(draw);
  }
  $('robust-history').addEventListener('change',()=>{identifier=$('robust-history').value;pageData=null;poll();});
  $('robust-refresh').addEventListener('click',()=>history().catch(message));
  $('robust-open').addEventListener('click',()=>openPage().catch(message));
  $('robust-method').addEventListener('change',()=>openPage().catch(message));
  $('robust-heat-field').addEventListener('change',()=>charts(snapshot?.report||{}));
  $('robust-cancel').addEventListener('click',async()=>{try{await request(root()+'/cancel');await poll();}catch(e){message(e);}});
  $('robust-resume').addEventListener('click',async()=>{try{await request(root()+'/resume');await poll();}catch(e){message(e);}});
  ['robust-gt','robust-pred'].forEach(id=>$(id).addEventListener('change',draw));
  window.addEventListener('resize',draw);
  $('robust-form').addEventListener('submit',async event=>{
    event.preventDefault();
    try {
      const dataset=$('dataset-select').value;
      if(!dataset)throw Error('Выберите датасет с эталоном слева');
      const providers=selectProviders(),limit=Number($('robust-limit').value);
      if(!providers.length)throw Error('Выберите модели слева');
      const meta=await json('/api/datasets/'+dataset),options=providerOptions();
      const config={name:$('robust-name').value,dataset_id:dataset,providers,max_side:Number($('robust-size').value),include_unlabelled:$('robust-unlabelled').checked,confidence:Object.fromEntries(Object.entries(options).map(([p,o])=>[p,o.confidence]))};
      if(limit)config.images=meta.images.slice(0,limit).map(x=>x.name);
      $('robust-start').disabled=true;$('robust-status').textContent='Фиксирую протокол и версии изображений…';
      identifier=(await request('/api/robustness',config)).id;await history();
    } catch(error){message(error);$('robust-start').disabled=false;}
  });
  if(new URLSearchParams(location.search).has('robustness')) {
    identifier=new URLSearchParams(location.search).get('robustness')||identifier;
    document.querySelector('[data-tab="robustness"]').click();
  }
  history().catch(message);
})();
