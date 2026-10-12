// 公开状态页（只读，无需登录）：每 5 秒拉一次 /overview/api/state。

const $ = (sel) => document.querySelector(sel);
let lastOkAt = 0;

const LABEL = {
  idle: '空闲', busy: '使用中', queued: '启动中 · 等资源', loading: '启动中 · 加载模型',
  leaving: '即将下线', unavailable: '暂不可用',
};
const DOT = { idle: 'idle', busy: 'busy', queued: 'starting', loading: 'starting', leaving: 'leaving', unavailable: 'unavailable' };

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function duration(seconds) {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分 ${s % 60} 秒`;
  const h = Math.floor(s / 3600);
  return h < 48 ? `${h} 小时 ${Math.floor((s % 3600) / 60)} 分` : `${Math.floor(h / 24)} 天`;
}

function chip(worker, endpointModel, now) {
  const time = worker.state === 'busy' && worker.session_started_at
    ? `本次会话 ${duration(now - worker.session_started_at)}`
    : worker.state === 'idle' ? '' : `已 ${duration(now - worker.since)}`;
  const tpl = worker.model !== endpointModel ? `<span class="tpl">${esc(worker.model)}</span>` : '';
  return `<span class="chip" title="${esc(worker.worker_id)} · ${esc(worker.model)} · 默认 tts_pad ${esc(worker.tts_pad_protocol)}">
    <i class="dot ${DOT[worker.state] || 'unavailable'}"></i><b>${esc(worker.worker_id)}</b>${esc(LABEL[worker.state] || worker.state)}
    ${time ? `<span class="time">${time}</span>` : ''}${tpl}</span>`;
}

function verdict(endpoint) {
  if (endpoint.available > 0) return `<span class="verdict ok">可直接使用 · 空闲 ${endpoint.available}</span>`;
  const serving = endpoint.workers.filter((w) => w.state === 'busy').length;
  if (serving > 0) return `<span class="verdict wait">全部使用中 · 排队 ${endpoint.queue_length}</span>`;
  if (endpoint.workers.some((w) => w.state === 'queued' || w.state === 'loading')) return '<span class="verdict wait">正在启动</span>';
  return '<span class="verdict none">暂无可用 Worker</span>';
}

function render(state) {
  const now = state.now;
  const workers = state.endpoints.flatMap((e) => e.workers);
  const n = (s) => workers.filter((w) => w.state === s).length;
  const queue = state.endpoints.reduce((sum, e) => sum + e.queue_length, 0);
  $('#totals').innerHTML = `<b>${state.endpoints.length}</b> 个端点 · <b>${workers.length}</b> 个 Worker：`
    + `空闲 <b>${n('idle')}</b> · 使用中 <b>${n('busy')}</b> · 启动中 <b>${n('queued') + n('loading')}</b> · 排队的会话 <b>${queue}</b>`;
  $('#endpoints').innerHTML = state.endpoints.length ? state.endpoints.map((e) => {
    const url = `${state.public_base.replace(/^http/, 'ws')}/e/${e.name}/v1/realtime?mode=audio`;
    return `<section class="ep">
      <div class="ep-row">
        <div><span class="ep-name">${esc(e.name)}</span><span class="ep-desc">${esc(e.description)}</span>
          <div class="ep-model">${e.model ? `模型 ${esc(e.model)} · 默认 tts_pad ${esc(e.tts_pad_protocol)}` : '暂无模型'}</div></div>
        ${verdict(e)}
      </div>
      <div class="url"><code title="${esc(url)}">${esc(url)}</code><button data-copy="${esc(url)}">复制</button>
        <button data-copy="${esc(`${state.public_base}/e/${e.name}/v1/info`)}" title="复制端点信息地址">info</button></div>
      <div class="chips">${e.workers.length ? e.workers.map((w) => chip(w, e.model, now)).join('') : '<span class="empty">当前没有 Worker</span>'}</div>
    </section>`;
  }).join('') : '<div class="error">当前没有开放的端点</div>';
}

async function refresh() {
  try {
    const response = await fetch('/overview/api/state', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
    lastOkAt = Date.now();
    $('#updated').textContent = `更新于 ${new Date().toLocaleTimeString('zh-CN', { hour12: false })} · 每 5 秒刷新`;
    $('#updated').classList.remove('stale');
  } catch (error) {
    $('#updated').textContent = lastOkAt ? `刷新失败（${error.message}），显示的是 ${Math.round((Date.now() - lastOkAt) / 1000)} 秒前的状态` : `加载失败：${error.message}`;
    $('#updated').classList.add('stale');
  }
}

document.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-copy]');
  if (!button) return;
  navigator.clipboard?.writeText(button.dataset.copy);
  const old = button.textContent;
  button.textContent = '已复制';
  setTimeout(() => { button.textContent = old; }, 1200);
});

refresh();
setInterval(refresh, 5000);
