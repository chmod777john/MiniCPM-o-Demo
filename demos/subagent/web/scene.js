// subagent 场景：把模型发出的工具调用交给后端（Claude Code 执行），把进度 / 结果回灌给模型。
//
//   delegate_task / web_search / calculate → POST tasks → 轮询 events：
//       progress → client.sendStandaloneEvent（训练形态：子任务进度）
//       result   → client.sendToolResult
//       error    → client.sendToolResult（如实告诉模型执行失败，不重试、不补救）
//   show_card → 训练数据里没有出现过；在主面板画出卡片，回一句 "result ok True"。
//
// 每个 Demo 会话一个工作区（会话开始时为空目录）；上传的文件落在这里，并以
// "[FILE_UPLOADED] 文件名" 的 standalone 通知模型（sub_agent think 数据里的训练形态）。
// 模型说了什么、想了什么由 shell 面板如实展示。

import { escapeHtml } from '../../../static/demo-gallery/core/ui-utils.js';
import {
  PRESET_OPTIONS,
  PRESET_ZH_NOTHINK,
  buildSubagentQuickSettings,
  presetLanguage,
  presetTools,
  unavailablePresets,
} from './quick-settings.js';

const POLL_MS = 800;

function newWorkspaceId() {
  const rand = (globalThis.crypto?.randomUUID?.() || `${Date.now()}${Math.random()}`).replace(/[^A-Za-z0-9]/g, '');
  return `ws_${rand.slice(0, 24)}`;
}

export function createScene({ stageEl, getClient, api }) {
  const state = {
    workspaceId: newWorkspaceId(),
    workspacePath: '',
    files: [],
    entries: [],          // 时间线
    pendingStandalone: new Map(),   // event_id → entry
    pendingTool: new Map(),         // tool_call_id → entry
    seq: 0,
    notice: '',
    defaults: null,
    presetId: PRESET_ZH_NOTHINK,
    handled: ['delegate_task', 'web_search', 'calculate'],
  };
  let built = false;

  async function call(path, init = {}) {
    const response = await fetch(api.url(path), init);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
    return body;
  }

  async function openWorkspace() {
    try {
      const body = await call(`workspaces/${state.workspaceId}`, { method: 'POST' });
      state.workspacePath = body.path;
      await refreshFiles();
    } catch (error) {
      state.notice = `工作区不可用：${error.message}`;
    }
    render();
  }

  async function refreshFiles() {
    const body = await call(`workspaces/${state.workspaceId}/files`);
    state.files = body.files || [];
  }

  function ready() {
    const client = getClient();
    return client && client.ready ? client : null;
  }

  function sendStandalone(text, label) {
    const client = ready();
    if (!client) return false;
    const id = `sa_${++state.seq}`;
    client.sendStandaloneEvent({ content: text, eventId: id });
    const entry = { kind: 'standalone', label, text, unit: null };
    state.pendingStandalone.set(id, entry);
    state.entries.push(entry);
    return true;
  }

  function sendToolResult(toolCallId, content, entry) {
    const client = ready();
    if (!client) return;
    client.sendToolResult({ toolCallId, content });
    entry.resultSent = true;
    state.pendingTool.set(toolCallId, entry);
  }

  async function uploadFile(file) {
    if (!ready()) {
      state.notice = '会话还没就绪：先点 Start mic live，等底部状态栏出现「Listening」再上传。';
      render();
      return;
    }
    try {
      const buffer = new Uint8Array(await file.arrayBuffer());
      let binary = '';
      for (let i = 0; i < buffer.length; i += 0x8000) binary += String.fromCharCode(...buffer.subarray(i, i + 0x8000));
      await call(`workspaces/${state.workspaceId}/files`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: file.name, content_base64: btoa(binary) }),
      });
      await refreshFiles();
      sendStandalone(`[FILE_UPLOADED] ${file.name}`, '文件上传通知');
      state.notice = '';
    } catch (error) {
      state.notice = `上传失败：${error.message}`;
    }
    render();
  }

  async function runTool(event, entry) {
    const call_ = event.call || {};
    try {
      const { task_id: taskId } = await call('tasks', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          workspace_id: state.workspaceId, tool_call_id: event.tool_call_id, name: call_.name,
          arguments: parseArguments(call_.arguments), language: presetLanguage(state.defaults, state.presetId),
        }),
      });
      let after = 0;
      for (;;) {
        const body = await call(`tasks/${taskId}/events?after=${after}`);
        for (const item of body.events || []) {
          after = item.seq + 1;
          if (item.kind === 'progress') {
            sendStandalone(item.text, `${call_.name} 进度`);
          } else {
            entry.status = item.kind === 'result' ? '已完成' : '执行失败';
            entry.result = item.text;
            sendToolResult(event.tool_call_id, item.text, entry);
          }
        }
        render();
        if (body.done) break;
        await new Promise((resolve) => setTimeout(resolve, POLL_MS));
      }
      await refreshFiles().catch(() => {});
    } catch (error) {
      entry.status = '执行失败';
      entry.result = `后台不可用：${error.message}`;
      sendToolResult(event.tool_call_id, entry.result, entry);
    }
    render();
  }

  function build() {
    stageEl.innerHTML = `
      <div class="sg-workspace">
        <div class="sg-ws-head">
          <span class="sg-ws-title">工作区</span>
          <code class="sg-ws-path"></code>
          <label class="sg-upload">上传文件<input type="file" hidden /></label>
        </div>
        <ul class="sg-files"></ul>
      </div>
      <div class="sg-notice" role="status"></div>
      <ol class="sg-timeline"></ol>`;
    stageEl.querySelector('.sg-upload input').addEventListener('change', (event) => {
      const file = event.target.files?.[0];
      event.target.value = '';
      if (file) uploadFile(file);
    });
    built = true;
  }

  function renderEntry(entry) {
    if (entry.kind === 'standalone') {
      const status = entry.unit === null ? '<span class="sg-wait">已发送</span>' : `<span class="sg-ok">注入 Unit ${escapeHtml(String(entry.unit))}</span>`;
      return `<li class="sg-row sg-sa"><span class="sg-tag">${escapeHtml(entry.label)}</span><span class="sg-text">${escapeHtml(entry.text)}</span>${status}</li>`;
    }
    if (entry.kind === 'card') {
      return `<li class="sg-row sg-card"><span class="sg-tag">show_card</span><pre>${escapeHtml(entry.card)}</pre></li>`;
    }
    const consumed = entry.unit === null || entry.unit === undefined ? '' : ` · 结果注入 Unit ${escapeHtml(String(entry.unit))}`;
    return `<li class="sg-row sg-tool">
      <span class="sg-tag">U${escapeHtml(String(entry.callUnit ?? '?'))} 模型 → ${escapeHtml(entry.name)}</span>
      <span class="sg-text">${escapeHtml(entry.args)}</span>
      <span class="${entry.status === '执行失败' ? 'sg-err' : entry.status === '已完成' ? 'sg-ok' : 'sg-wait'}">${escapeHtml(entry.status)}${consumed}</span>
      ${entry.result ? `<div class="sg-result">${escapeHtml(entry.result)}</div>` : ''}
    </li>`;
  }

  function render() {
    if (!built) {
      build();
      openWorkspace();      // 首次渲染即建本会话的（空）工作区
    }
    stageEl.querySelector('.sg-ws-path').textContent = state.workspacePath || '（会话开始后创建）';
    stageEl.querySelector('.sg-files').innerHTML = state.files.length
      ? state.files.map((f) => `<li>${escapeHtml(f.name)} <span>${escapeHtml(String(f.size))} B</span></li>`).join('')
      : '<li class="sg-empty">（空）Claude Code 只能读写这个目录</li>';
    const missing = state.defaults ? unavailablePresets(state.defaults) : [];
    const notices = [state.notice, ...missing.map((m) => `preset ${m.id} 不可用：${m.error}`)].filter(Boolean);
    stageEl.querySelector('.sg-notice').innerHTML = notices.map(escapeHtml).join('<br>');
    const timeline = stageEl.querySelector('.sg-timeline');
    timeline.innerHTML = state.entries.length
      ? state.entries.slice().reverse().map(renderEntry).join('')
      : '<li class="sg-empty">开麦后对模型说一件需要查、算或读文件的事；它调用工具时，由 Claude Code 在上面的工作区里执行。</li>';
  }


  return {
    panelTitle: 'Subagent · Claude Code 执行工具',
    panelHint: 'delegate_task / web_search / calculate → Claude Code',
    liveStatus: 'Listening via /v1/realtime · 工具调用由 Claude Code 执行',
    presetOptions: PRESET_OPTIONS,
    fallbackQuickSetting: PRESET_ZH_NOTHINK,

    buildQuickSettings(defaults) {
      state.defaults = defaults;
      if (Array.isArray(defaults?.subagent_handled_tools)) state.handled = defaults.subagent_handled_tools;
      return buildSubagentQuickSettings(defaults);
    },

    tools(defaults, presetId) {
      if (presetId) state.presetId = presetId;
      return presetTools(defaults, presetId);
    },

    render,

    reset() {
      // 新会话 = 新的空工作区
      state.workspaceId = newWorkspaceId();
      state.workspacePath = '';
      state.files = [];
      state.entries = [];
      state.pendingStandalone.clear();
      state.pendingTool.clear();
      state.notice = '';
      render();
      openWorkspace();
    },


    onUnitStarted(event) {
      for (const item of event.standalone_events || []) {
        const entry = state.pendingStandalone.get(item.event_id);
        if (entry) { entry.unit = event.unit_index; state.pendingStandalone.delete(item.event_id); }
      }
      for (const item of event.tool_events || []) {
        if (item.type !== 'tool_result') continue;
        const entry = state.pendingTool.get(item.tool_call_id);
        if (entry) { entry.unit = event.unit_index; state.pendingTool.delete(item.tool_call_id); }
      }
      render();
    },

    onToolCallDone(event) {
      const call_ = event.call || {};
      if (event.error || !call_.name) return;   // 解析失败的调用由服务端自动回包
      if (call_.name === 'show_card') {
        state.entries.push({ kind: 'card', card: JSON.stringify(parseArguments(call_.arguments), null, 2) });
        const entry = { kind: 'tool', name: 'show_card', args: '', status: '已完成', callUnit: event.unit_index, unit: null, result: 'result ok True' };
        state.entries.push(entry);
        sendToolResult(event.tool_call_id, 'result ok True', entry);
        render();
        return;
      }
      const entry = {
        kind: 'tool', name: call_.name, args: JSON.stringify(parseArguments(call_.arguments)),
        status: state.handled.includes(call_.name) ? '执行中（Claude Code）' : '未处理', callUnit: event.unit_index, unit: null, result: '',
      };
      state.entries.push(entry);
      render();
      if (state.handled.includes(call_.name)) runTool(event, entry);
    },
  };
}

function parseArguments(value) {
  if (!value) return {};
  if (typeof value === 'object') return value;
  try { return JSON.parse(value); } catch (_) { return {}; }
}
