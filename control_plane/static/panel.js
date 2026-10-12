// 控制面板前端：每 3 秒拉一次 /control/api/state 重画；401 时显示登录框。

const PHASE_LABEL = {
  submitting: '提交中', queued: '排队中', loading: '加载中', ready: '可用', draining: '驱逐中',
  stopping: '停止中', stopped: '已停止', failed: '失败',
};
const ACTIVE = new Set(['submitting', 'queued', 'loading', 'ready', 'draining', 'stopping']);
const $ = (sel) => document.querySelector(sel);
let lastState = null;

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function duration(seconds) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分 ${s % 60} 秒`;
  return `${Math.floor(s / 3600)} 小时 ${Math.floor((s % 3600) / 60)} 分`;
}

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

function notice(text) { $('#notice').textContent = text || ''; }

async function act(path, init, confirmText) {
  if (confirmText && !window.confirm(confirmText)) return;
  try {
    await api(path, init);
    notice('');
    await refresh();
  } catch (error) {
    notice(error.message);
  }
}

function showLogin(message = '') {
  $('#main').hidden = true;
  $('#logout').hidden = true;
  $('#login').hidden = false;
  $('#login-error').textContent = message;
}

function workerPhase(worker) {
  const busy = worker.phase === 'ready' && worker.gateway?.busy;
  const key = busy ? 'busy' : worker.phase;
  const label = busy ? '服务中' : (PHASE_LABEL[worker.phase] || worker.phase);
  const since = duration(lastState.now - worker.phase_since);
  const extra = worker.move_to ? `<div class="muted">待移到 ${esc(worker.move_to)}（等会话结束）</div>` : '';
  const gw = worker.gateway && worker.phase === 'ready' && worker.gateway.status !== 'idle' && !busy
    ? `<div class="muted">Gateway：${esc(worker.gateway.status)}</div>` : '';
  return `<span class="phase phase-${key}">${label}</span> <span class="muted">${since}</span>${extra}${gw}`;
}

function workerSession(worker) {
  const gw = worker.gateway;
  if (!gw || !gw.busy || !gw.session_started_at) return '<span class="muted">—</span>';
  return `会话已 ${duration(lastState.now - gw.session_started_at)}`;
}

function workerActions(worker, endpoints) {
  const id = esc(worker.worker_id);
  const active = ACTIVE.has(worker.phase);
  const releasing = worker.phase === 'draining' || worker.phase === 'stopping';
  const others = endpoints.filter((e) => e.name !== worker.endpoint);
  const moveSelect = others.length && active && !releasing
    ? `<select data-move="${id}"><option value="">移动到…</option>${others.map((e) => `<option>${esc(e.name)}</option>`).join('')}</select>` : '';
  return `<div class="actions">
    <button class="primary" data-scale="${id}" title="用同一模板在本端点再起一个副本">扩容</button>
    ${active && !releasing ? `<button data-drain="${id}" title="不再派新会话，手上的会话结束后停 job">驱逐</button>` : ''}
    ${active && worker.phase !== 'stopping' ? `<button class="danger" data-force="${id}" title="立刻打断会话并停 job">强制关闭</button>` : ''}
    ${moveSelect}
    ${!active ? `<button data-forget="${id}">删除记录</button>` : ''}
  </div>`;
}

function workerRow(worker, endpoints) {
  const t = worker.template;
  const history = worker.history.map((h) => `<div>${new Date(h.at * 1000).toLocaleTimeString()} ${esc(PHASE_LABEL[h.phase] || h.phase)} ${esc(h.note)}</div>`).join('');
  return `<tr>
    <td><b>${esc(worker.worker_id)}</b><details class="history"><summary>生命周期</summary><div class="muted">${history}</div></details></td>
    <td>${esc(t.name)}<div class="mono muted">${esc(t.model_dir)}</div></td>
    <td>${esc(t.tts_pad_protocol)}</td>
    <td>${workerPhase(worker)}${worker.error ? `<div class="err">${esc(worker.error)}</div>` : ''}</td>
    <td>${worker.job_id ? `${worker.job_id}<div class="muted">${esc(worker.job_status || '')}</div>` : '<span class="muted">—</span>'}</td>
    <td class="mono">${worker.host ? `${esc(worker.host)}:${esc(worker.port)}` : '<span class="muted">—</span>'}</td>
    <td>${workerSession(worker)}</td>
    <td>${workerActions(worker, endpoints)}</td>
  </tr>`;
}

function endpointCard(endpoint, state) {
  const name = esc(endpoint.name);
  const wsUrl = `${state.public_base.replace(/^http/, 'ws')}/e/${endpoint.name}/v1/realtime?mode=audio`;
  const infoUrl = `${state.public_base}/e/${endpoint.name}/v1/info`;
  const options = state.templates.map((t) => `<option>${esc(t.name)}</option>`).join('');
  const workers = endpoint.workers;
  const activeCount = workers.filter((w) => ACTIVE.has(w.phase)).length;
  return `<div class="endpoint">
    <div class="endpoint-head">
      <b>${name}</b><span class="muted">${esc(endpoint.description)}</span>
      <span>Worker ${activeCount} · 可用 ${endpoint.available} · 排队 ${endpoint.queue_length}</span>
      <span class="url">${esc(wsUrl)}</span><button data-copy="${esc(wsUrl)}">复制</button>
      <a href="${esc(infoUrl)}" target="_blank" class="muted">/v1/info</a>
      <button class="danger" data-del-endpoint="${name}" ${activeCount ? 'disabled title="先释放端点下的 Worker"' : ''}>删除端点</button>
    </div>
    ${workers.length ? `<table><thead><tr><th>Worker</th><th>模板 / checkpoint</th><th>tts_pad</th><th>状态</th><th>job</th><th>地址</th><th>当前会话</th><th>操作</th></tr></thead>
      <tbody>${workers.map((w) => workerRow(w, state.endpoints)).join('')}</tbody></table>` : '<p class="muted">还没有 Worker。</p>'}
    <form class="inline" data-add-worker="${name}">
      <select name="template" required><option value="">选择模板…</option>${options}</select>
      <button type="submit">添加 Worker</button>
    </form>
  </div>`;
}

function renderTemplates(state) {
  $('#templates tbody').innerHTML = state.templates.length ? state.templates.map((t) => `<tr>
    <td><b>${esc(t.name)}</b></td><td class="mono">${esc(t.model_dir)}</td><td>${esc(t.tts_pad_protocol)}</td>
    <td>${t.non_spoken_budget_while_listening} / ${t.non_spoken_budget_while_speaking}</td><td>${t.unit_sec}s · ${esc(t.non_spoken_scheduling)}</td>
    <td>${esc(t.note)}</td>
    <td class="actions"><button data-edit-template="${esc(t.name)}">编辑</button><button class="danger" data-del-template="${esc(t.name)}">删除</button></td>
  </tr>`).join('') : '<tr><td colspan="7" class="muted">还没有模板。先在下面新建一个，再到端点里添加 Worker。</td></tr>';
}

function render(state) {
  lastState = state;
  // 正在下拉选择时不重画，免得选择被打断
  if (document.activeElement && document.activeElement.tagName === 'SELECT' && document.activeElement.closest('#endpoints')) return;
  $('#endpoints').innerHTML = state.endpoints.length
    ? state.endpoints.map((e) => endpointCard(e, state)).join('')
    : '<p class="muted">还没有端点。端点 = 一个对外 URL；新建后再往里添加 Worker。</p>';
  renderTemplates(state);
}

async function refresh() {
  try {
    const state = await api('state');
    $('#login').hidden = true;
    $('#main').hidden = false;
    $('#logout').hidden = false;
    render(state);
  } catch (error) {
    if (!$('#login').hidden) return;
    notice(error.message);
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
  form.scrollIntoView({ behavior: 'smooth' });
}

document.addEventListener('click', (event) => {
  const el = event.target.closest('button');
  if (!el) return;
  const d = el.dataset;
  if (d.copy) { navigator.clipboard?.writeText(d.copy); notice('已复制'); setTimeout(() => notice(''), 1200); }
  if (d.scale) act(`workers/${d.scale}/scale`, { method: 'POST' });
  if (d.drain) act(`workers/${d.drain}/drain`, { method: 'POST' }, `驱逐 ${d.drain}？不再派新会话，手上的会话结束后停 job。`);
  if (d.force) act(`workers/${d.force}/force`, { method: 'POST' }, `强制关闭 ${d.force}？正在进行的会话会被立刻打断。`);
  if (d.forget) act(`workers/${d.forget}`, { method: 'DELETE' });
  if (d.delEndpoint) act(`endpoints/${d.delEndpoint}`, { method: 'DELETE' }, `删除端点 ${d.delEndpoint}？`);
  if (d.delTemplate) act(`templates/${d.delTemplate}`, { method: 'DELETE' }, `删除模板 ${d.delTemplate}？已起的 Worker 不受影响。`);
  if (d.editTemplate) fillTemplateForm(d.editTemplate);
});

document.addEventListener('change', (event) => {
  const el = event.target;
  if (el.dataset.move && el.value) {
    act(`workers/${el.dataset.move}/move`, { method: 'POST', body: JSON.stringify({ endpoint: el.value }) },
      `把 ${el.dataset.move} 移到 ${el.value}？会等它手上的会话结束后切换，job 不重启。`);
  }
});

document.addEventListener('submit', (event) => {
  const form = event.target;
  if (form.dataset.addWorker) {
    event.preventDefault();
    act('workers', { method: 'POST', body: JSON.stringify({ endpoint: form.dataset.addWorker, template: form.elements.template.value }) });
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

$('#new-endpoint').addEventListener('submit', (event) => {
  event.preventDefault();
  const form = event.target;
  act('endpoints', { method: 'POST', body: JSON.stringify({ name: form.elements.name.value.trim(), description: form.elements.description.value.trim() }) })
    .then(() => form.reset());
});

$('#template-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const form = event.target;
  const data = Object.fromEntries(new FormData(form).entries());
  for (const key of ['non_spoken_budget_while_listening', 'non_spoken_budget_while_speaking']) data[key] = Number(data[key]);
  data.unit_sec = Number(data.unit_sec);
  act(`templates/${encodeURIComponent(data.name)}`, { method: 'PUT', body: JSON.stringify(data) }).then(() => form.reset());
});

refresh();
setInterval(refresh, 3000);
