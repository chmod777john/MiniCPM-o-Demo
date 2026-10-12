// 控制面板前端：每 3 秒拉一次 /control/api/state 重画；401 时显示登录框。
// 展示原则：先说人话的状态（空闲 / 服务中 / 加载中…），技术细节（job id、Pod 地址、完整路径）弱化或悬停可见。

const $ = (sel) => document.querySelector(sel);
const ACTIVE = new Set(['submitting', 'queued', 'loading', 'ready', 'draining', 'stopping']);
const STARTING = ['submitting', 'queued', 'loading'];
const openKeys = new Set();          // 重画时保持展开状态的 <details data-key>
let lastState = null;
let toastTimer = null;

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function duration(seconds) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分 ${s % 60} 秒`;
  const h = Math.floor(s / 3600);
  if (h < 48) return `${h} 小时 ${Math.floor((s % 3600) / 60)} 分`;
  return `${Math.floor(h / 24)} 天 ${h % 24} 小时`;
}

function since(ts) {
  return duration(lastState.now - ts);
}

function clock(ts) {
  return new Date(ts * 1000).toLocaleTimeString('zh-CN', { hour12: false });
}

// ---------------------------------------------------------------- 状态翻译

/** 一个 Worker 的人话状态：{cls, label, detail}。cls 决定颜色。 */
function describe(worker) {
  const gw = worker.gateway || {};
  switch (worker.phase) {
    case 'submitting': return { cls: 's-starting', label: '提交中', detail: `正在向集群提交 job · ${since(worker.phase_since)}` };
    case 'queued': return { cls: 's-starting', label: '排队等资源', detail: `资源池暂时没有空卡 · 已等 ${since(worker.phase_since)}` };
    case 'loading': return { cls: 's-starting', label: '加载模型', detail: `job 已运行，正在加载 · 已 ${since(worker.phase_since)}` };
    case 'ready':
      if (worker.move_to) return { cls: 's-releasing', label: '移动中', detail: `等当前会话结束后换到 ${worker.move_to}` };
      if (gw.busy) return { cls: 's-busy', label: '服务中', detail: `本次会话已 ${gw.session_started_at ? since(gw.session_started_at) : '—'}` };
      if (gw.status && gw.status !== 'idle') return { cls: 's-failed', label: '连不上', detail: `健康检查：${gw.status}（Worker job 仍在运行）` };
      return { cls: 's-idle', label: '空闲', detail: `可接会话 · 已可用 ${since(worker.phase_since)}` };
    case 'draining':
      return { cls: 's-releasing', label: '驱逐中', detail: gw.busy ? `不再接新会话 · 当前会话已 ${since(gw.session_started_at)}` : '不再接新会话，马上停止' };
    case 'stopping': return { cls: 's-releasing', label: '停止中', detail: `正在停 job · ${since(worker.phase_since)}` };
    case 'stopped': {
      const how = worker.release_mode === 'force' ? '强制关闭' : worker.release_mode === 'drain' ? '驱逐后停止' : '已停止';
      return { cls: 's-stopped', label: '已停止', detail: `${how} · ${since(worker.phase_since)}前` };
    }
    case 'failed': return { cls: 's-failed', label: '失败', detail: `${since(worker.phase_since)}前` };
    default: return { cls: 's-stopped', label: worker.phase, detail: '' };
  }
}

function steps(worker) {
  const names = ['提交', '排队', '加载', '可用'];
  const current = STARTING.indexOf(worker.phase);
  return `<div class="steps">${names.map((name, i) => {
    const state = i < current ? 'done' : i === current ? 'now' : '';
    const bar = i < names.length - 1 ? `<span class="bar ${i < current ? 'done' : ''}"></span>` : '';
    return `<span class="step ${state}"><span class="dot">${i < current ? '✓' : ''}</span>${name}</span>${bar}`;
  }).join('')}</div>`;
}

function ttsTag(protocol) {
  return protocol ? `<span class="tag ${esc(protocol)}" title="checkpoint 训练侧的 tts_pad 口径">tts_pad ${esc(protocol)}</span>` : '';
}

function pathView(path) {
  // 外层 rtl 让溢出时裁掉开头、保留结尾（目录名最有区分度）；<bdi> 保证路径本身仍按从左到右显示
  return `<div class="path" title="${esc(path)}"><span><bdi>${esc(path)}</bdi></span>
    <button class="small" data-copy="${esc(path)}" title="复制完整路径">复制</button></div>`;
}

// ---------------------------------------------------------------- Worker 卡片

function workerCard(worker, endpoint, endpoints, mainTemplate) {
  const id = esc(worker.worker_id);
  const s = describe(worker);
  const active = ACTIVE.has(worker.phase);
  const starting = STARTING.includes(worker.phase);
  const releasing = worker.phase === 'draining' || worker.phase === 'stopping';
  const otherTemplate = worker.template.name !== mainTemplate
    ? `<div class="w-line">模型 <b>${esc(worker.template.name)}</b> ${ttsTag(worker.template.tts_pad_protocol)}</div>` : '';

  let actions = '';
  if (!active) {
    actions = `<button class="small" data-scale="${id}" title="用同一模板在本端点再起一个">用同一模板再起一个</button>
      <button class="small" data-forget="${id}">删除记录</button>`;
  } else if (releasing) {
    actions = worker.phase === 'draining'
      ? `<button class="small danger" data-force="${id}" title="不等会话结束，立刻停">立刻强制关闭</button>` : '';
  } else {
    const others = endpoints.filter((e) => e.name !== endpoint.name);
    const moveItems = !starting && others.length
      ? `<span class="menu-label">移动到其他端点（等会话结束后切换，job 不重启）</span>${others.map((e) => `<button data-move="${id}" data-to="${esc(e.name)}">→ ${esc(e.name)}</button>`).join('')}`
      : '<span class="menu-label">没有其他端点可移动</span>';
    actions = `
      <button class="small primary" data-scale="${id}" title="用同一模板在本端点再起一个副本">＋ 扩容</button>
      <button class="small" data-drain="${id}" title="${starting ? '取消启动' : '不再派新会话，手上的会话结束后停 job'}">${starting ? '取消启动' : '驱逐'}</button>
      ${starting ? '' : `<button class="small danger" data-force="${id}" title="立刻打断会话并停 job">强制关闭</button>`}
      <details class="more" data-key="more-${id}"><summary>更多 ▾</summary><div class="menu">${moveItems}</div></details>`;
  }

  const history = worker.history.map((h) => `<li>${clock(h.at)} ${esc(describe({ ...worker, phase: h.phase, gateway: null, move_to: null }).label)}${h.note ? ` — ${esc(h.note)}` : ''}</li>`).join('');
  return `<div class="worker ${s.cls}">
    <div class="w-top"><span class="w-id">${id}</span><span class="state ${s.cls}">${esc(s.label)}</span></div>
    <div class="w-line">${esc(s.detail)}</div>
    ${starting ? steps(worker) : ''}
    ${otherTemplate}
    ${worker.error ? `<div class="w-err">${esc(worker.error)}</div>` : ''}
    <div class="w-sub">
      ${worker.job_id ? `<span title="cctl job">job ${esc(worker.job_id)}${worker.job_status ? ` · ${esc(worker.job_status)}` : ''}</span>` : ''}
      ${worker.host ? `<span title="Worker 在集群内的地址">${esc(worker.host)}:${esc(worker.port)}</span>` : ''}
    </div>
    <div class="w-actions">${actions}</div>
    <details class="history" data-key="hist-${id}"><summary>生命周期记录（${worker.history.length}）</summary><ol>${history}</ol></details>
  </div>`;
}

// ---------------------------------------------------------------- 端点卡片

function mainTemplateOf(workers) {
  const counts = new Map();
  for (const w of workers) counts.set(w.template.name, (counts.get(w.template.name) || 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || null;
}

function endpointCard(endpoint, state) {
  const name = esc(endpoint.name);
  const wsUrl = `${state.public_base.replace(/^http/, 'ws')}/e/${endpoint.name}/v1/realtime?mode=audio`;
  const infoUrl = `${state.public_base}/e/${endpoint.name}/v1/info`;
  const active = endpoint.workers.filter((w) => ACTIVE.has(w.phase));
  const finished = endpoint.workers.filter((w) => !ACTIVE.has(w.phase));
  const mainTemplate = mainTemplateOf(active.length ? active : endpoint.workers);
  const tpl = state.templates.find((t) => t.name === mainTemplate)
    || endpoint.workers.find((w) => w.template.name === mainTemplate)?.template;
  const busy = active.filter((w) => w.phase === 'ready' && w.gateway?.busy).length;
  const startingCount = active.filter((w) => STARTING.includes(w.phase)).length;
  const counts = [
    `<span class="tag ok">可用 ${endpoint.available}</span>`,
    busy ? `<span class="tag" style="background:var(--purple-bg);color:var(--purple)">服务中 ${busy}</span>` : '',
    startingCount ? `<span class="tag" style="background:var(--blue-bg);color:var(--blue)">启动中 ${startingCount}</span>` : '',
    `<span class="tag ${endpoint.queue_length ? 'warn' : 'model'}">排队 ${endpoint.queue_length}</span>`,
  ].join('');
  const options = state.templates.map((t) => `<option ${t.name === mainTemplate ? 'selected' : ''}>${esc(t.name)}</option>`).join('');

  return `<div class="endpoint">
    <div class="ep-head">
      <div>
        <div class="ep-title"><span class="name">${name}</span><span class="muted">${esc(endpoint.description)}</span></div>
        <div class="ep-meta">
          ${tpl ? `<span class="tag model" title="${esc(tpl.model_dir)}">模型 ${esc(tpl.name)}</span>${ttsTag(tpl.tts_pad_protocol)}` : '<span class="muted">还没有模型</span>'}
          ${counts}
        </div>
      </div>
      <button class="small danger" data-del-endpoint="${name}" ${active.length ? 'disabled title="端点下还有在跑的 Worker，先释放它们"' : ''}>删除端点</button>
    </div>
    <div class="ep-url">
      <span class="label">接入地址</span><code>${esc(wsUrl)}</code>
      <button class="small" data-copy="${esc(wsUrl)}">复制</button>
      <a href="${esc(infoUrl)}" target="_blank">查看端点信息 ↗</a>
    </div>
    <div class="ep-body">
      ${active.length ? `<div class="workers">${active.map((w) => workerCard(w, endpoint, state.endpoints, mainTemplate)).join('')}</div>`
        : '<div class="empty">这个端点还没有 Worker；在下面选模板添加一个，2～4 分钟后可用。</div>'}
      <form class="add-worker" data-add-worker="${name}">
        <select name="template" required>${options || '<option value="">（先新建模板）</option>'}</select>
        <button type="submit" ${state.templates.length ? '' : 'disabled'}>＋ 添加 Worker</button>
        <span class="hint">每个 Worker 占 2 张 A100</span>
      </form>
      ${finished.length ? `<details class="finished" data-key="fin-${name}"><summary>已结束的 Worker（${finished.length}）</summary>
        <div class="workers">${finished.map((w) => workerCard(w, endpoint, state.endpoints, mainTemplate)).join('')}</div></details>` : ''}
    </div>
  </div>`;
}

// ---------------------------------------------------------------- 汇总 / 模板

function renderSummary(state) {
  const workers = state.endpoints.flatMap((e) => e.workers).filter((w) => ACTIVE.has(w.phase));
  const idle = workers.filter((w) => w.phase === 'ready' && !w.gateway?.busy && !w.move_to).length;
  const busy = workers.filter((w) => w.phase === 'ready' && w.gateway?.busy).length;
  const starting = workers.filter((w) => STARTING.includes(w.phase)).length;
  const gpus = workers.reduce((n, w) => n + (w.template.gpus || 2), 0);
  const queue = state.endpoints.reduce((n, e) => n + e.queue_length, 0);
  const stat = (value, label) => `<div class="stat"><b>${value}</b><span>${label}</span></div>`;
  $('#summary').innerHTML = [
    stat(state.endpoints.length, '端点'),
    stat(idle, '空闲 Worker'),
    stat(busy, '服务中'),
    stat(starting, '启动中'),
    stat(queue, '排队的会话'),
    stat(gpus, '占用 A100（deploy 池）'),
  ].join('');
}

function renderTemplates(state) {
  const usage = new Map();
  for (const e of state.endpoints) {
    for (const w of e.workers) {
      if (!ACTIVE.has(w.phase)) continue;
      const key = w.template.name;
      usage.set(key, [...(usage.get(key) || []), e.name]);
    }
  }
  $('#templates').innerHTML = state.templates.length ? `<div class="tpl-list">${state.templates.map((t) => {
    const used = usage.get(t.name) || [];
    const usedText = used.length
      ? `使用中：${[...new Set(used)].map((ep) => `${esc(ep)} ×${used.filter((x) => x === ep).length}`).join('，')}`
      : '未使用';
    return `<div class="tpl">
      <div class="tpl-top"><span class="tpl-name">${esc(t.name)}</span>${ttsTag(t.tts_pad_protocol)}</div>
      ${pathView(t.model_dir)}
      <div class="muted">预算 听 ${t.non_spoken_budget_while_listening} / 说 ${t.non_spoken_budget_while_speaking} · Unit ${t.unit_sec} 秒 · ${esc(t.non_spoken_scheduling)} · ${usedText}</div>
      ${t.note ? `<div class="w-line">${esc(t.note)}</div>` : ''}
      <div class="w-actions"><button class="small" data-edit-template="${esc(t.name)}">编辑</button>
        <button class="small danger" data-del-template="${esc(t.name)}">删除</button></div>
    </div>`;
  }).join('')}</div>` : '<div class="empty">还没有模板。先新建一个（checkpoint 目录 + tts_pad 口径），再到端点里添加 Worker。</div>';
}

function render(state) {
  lastState = state;
  // 菜单展开 / 下拉选择进行中时不重画，免得操作被打断
  if (document.querySelector('details.more[open]')) return;
  if (document.activeElement?.tagName === 'SELECT' && document.activeElement.closest('#endpoints')) return;
  renderSummary(state);
  $('#endpoints').innerHTML = state.endpoints.length
    ? state.endpoints.map((e) => endpointCard(e, state)).join('')
    : '<div class="empty">还没有端点。端点 = 一个对外接入地址；新建后再往里添加 Worker。</div>';
  renderTemplates(state);
  for (const el of document.querySelectorAll('details[data-key]')) {
    if (openKeys.has(el.dataset.key)) el.open = true;
  }
}

// ---------------------------------------------------------------- 交互

async function api(path, init = {}) {
  const response = await fetch(`/control/api/${path}`, {
    ...init, headers: { 'Content-Type': 'application/json', ...(init.headers || {}) }, credentials: 'same-origin',
  });
  const body = await response.json().catch(() => ({}));
  if (response.status === 401) { showLogin(); throw new Error(body.detail || '请先登录'); }
  if (!response.ok) {
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join('；') : body.detail;
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return body;
}

function toast(text, isError = false) {
  const el = $('#toast');
  el.textContent = text;
  el.className = `toast${isError ? ' error' : ''}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, isError ? 6000 : 2500);
}

async function act(path, init, confirmText, doneText) {
  if (confirmText && !window.confirm(confirmText)) return false;
  try {
    await api(path, init);
    if (doneText) toast(doneText);
    document.querySelectorAll('details.more[open]').forEach((d) => { d.open = false; openKeys.delete(d.dataset.key); });
    await refresh();
    return true;
  } catch (error) {
    toast(error.message, true);
    return false;
  }
}

function showLogin(message = '') {
  $('#main').hidden = true;
  $('#logout').hidden = true;
  $('#login').hidden = false;
  $('#login-error').textContent = message;
}

async function refresh() {
  try {
    const state = await api('state');
    $('#login').hidden = true;
    $('#main').hidden = false;
    $('#logout').hidden = false;
    render(state);
  } catch (error) {
    if ($('#login').hidden) toast(error.message, true);
  }
}

function fillTemplateForm(name) {
  const template = lastState.templates.find((t) => t.name === name);
  if (!template) return;
  const form = $('#template-form');
  for (const [key, value] of Object.entries(template)) {
    if (form.elements[key]) form.elements[key].value = value;
  }
  $('#template-editor').open = true;
  openKeys.add('template-editor');
  form.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

document.addEventListener('toggle', (event) => {
  const key = event.target?.dataset?.key;
  if (!key) return;
  if (event.target.open) openKeys.add(key); else openKeys.delete(key);
}, true);

document.addEventListener('click', (event) => {
  const el = event.target.closest('button');
  if (!el) return;
  const d = el.dataset;
  if (d.copy) { navigator.clipboard?.writeText(d.copy); toast('已复制'); }
  if (d.scale) act(`workers/${d.scale}/scale`, { method: 'POST' }, null, '已提交：新 Worker 正在启动');
  if (d.drain) act(`workers/${d.drain}/drain`, { method: 'POST' }, `驱逐 ${d.drain}？\n\n不再给它派新会话；手上的会话正常结束后自动停掉 job。`, '已开始驱逐');
  if (d.force) act(`workers/${d.force}/force`, { method: 'POST' }, `强制关闭 ${d.force}？\n\n正在进行的会话会被立刻打断（客户端收到 worker_removed），然后停掉 job。`, '已强制关闭');
  if (d.move) act(`workers/${d.move}/move`, { method: 'POST', body: JSON.stringify({ endpoint: d.to }) },
    `把 ${d.move} 移到端点 ${d.to}？\n\n等它手上的会话结束后切换，job 不重启。`, '已安排移动');
  if (d.forget) act(`workers/${d.forget}`, { method: 'DELETE' }, null, '已删除记录');
  if (d.delEndpoint) act(`endpoints/${d.delEndpoint}`, { method: 'DELETE' }, `删除端点 ${d.delEndpoint}？接入地址会立即失效。`, '端点已删除');
  if (d.delTemplate) act(`templates/${d.delTemplate}`, { method: 'DELETE' }, `删除模板 ${d.delTemplate}？\n\n已起的 Worker 不受影响。`, '模板已删除');
  if (d.editTemplate) fillTemplateForm(d.editTemplate);
});

document.addEventListener('submit', (event) => {
  const form = event.target;
  if (form.dataset.addWorker) {
    event.preventDefault();
    act('workers', { method: 'POST', body: JSON.stringify({ endpoint: form.dataset.addWorker, template: form.elements.template.value }) },
      null, '已提交：新 Worker 正在启动（2～4 分钟）');
  }
});

$('#login').addEventListener('submit', async (event) => {
  event.preventDefault();
  const response = await fetch('/control/api/login', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'same-origin',
    body: JSON.stringify({ password: $('#password').value }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) { showLogin(body.detail || `HTTP ${response.status}`); return; }
  $('#password').value = '';
  await refresh();
});

$('#logout').addEventListener('click', async () => {
  await fetch('/control/api/logout', { method: 'POST', credentials: 'same-origin' });
  showLogin();
});

$('#new-endpoint').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = event.target;
  const ok = await act('endpoints', { method: 'POST', body: JSON.stringify({ name: form.elements.name.value.trim(), description: form.elements.description.value.trim() }) },
    null, '端点已创建');
  if (ok) form.reset();
});

$('#template-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = event.target;
  const data = Object.fromEntries(new FormData(form).entries());
  for (const key of ['non_spoken_budget_while_listening', 'non_spoken_budget_while_speaking']) data[key] = Number(data[key]);
  data.unit_sec = Number(data.unit_sec);
  const ok = await act(`templates/${encodeURIComponent(data.name)}`, { method: 'PUT', body: JSON.stringify(data) }, null, '模板已保存');
  if (ok) form.reset();
});

refresh();
setInterval(refresh, 3000);
