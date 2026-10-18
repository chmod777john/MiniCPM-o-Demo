// UI Card 场景：左边卡片区（最新一张 + 本会话展示过的卡片），右边可折叠的消息区（默认展开）。
//
//   模型 tool_call → POST tools/execute（后端真调 SERP / Tavily / 高德）→ content 回灌给模型
//                                                                    → cards 登记进注册表（模型看不到）
//   show_card({card_id}) → 按编号从注册表取出工具拼好的卡片画出来；查不到就如实回错误
//   show_card({card})    → 画模型自己写的卡片
//
// 消息区照搬 cjp/o5-fc-dev 的 jinpeng_agent_v2 页面：模型说的话 / think / 工具调用（参数逐 token 出现，
// 返回结果默认折叠）。它与 shell 下方的 AI Speech、Think/Tool-Call 面板内容有重叠，两边都可以各自折叠。
// 工具设置（搜索后端、模拟延迟）放在消息区顶部，改了下一次工具调用就生效，不用重连。

import { escapeHtml, textFromSteps } from '../../../static/demo-gallery/core/ui-utils.js';
import { renderCard, setMapDisplayConfig } from './card-view.js';
import { CardRegistry, unknownCardIdMessage } from './card-registry.js';
import {
  PRESET_1016,
  PRESET_OPTIONS,
  buildUiCardQuickSettings,
  presetDataId,
  presetTools,
  unavailablePresets,
} from './quick-settings.js';
import { restampSegments } from './session-time.js';

const SIDE_COLLAPSED_KEY = 'uicard.sideCollapsed';
const SETTINGS_KEY = 'uicard.toolSettings';
const RESULT_PREVIEW_CHARS = 4000;
const DEFAULT_SETTINGS = {
  searchProvider: 'serp',
  serpDelaySec: 0,
  get_client_location: { enabled: false, min: 0, max: 1000 },
  amap_weather: { enabled: false, min: 0, max: 1000 },
};

function loadSettings() {
  try {
    const saved = JSON.parse(globalThis.localStorage?.getItem(SETTINGS_KEY) || '{}');
    return {
      ...DEFAULT_SETTINGS,
      ...saved,
      get_client_location: { ...DEFAULT_SETTINGS.get_client_location, ...(saved.get_client_location || {}) },
      amap_weather: { ...DEFAULT_SETTINGS.amap_weather, ...(saved.amap_weather || {}) },
    };
  } catch {
    return JSON.parse(JSON.stringify(DEFAULT_SETTINGS));
  }
}

function compactJson(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function parseArguments(raw) {
  if (!raw) return {};
  if (typeof raw === 'object') return raw;
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

export function createScene({ stageEl, getClient, api }) {
  const registry = new CardRegistry();
  const state = {
    defaults: null,
    presetId: PRESET_1016,
    sideCollapsed: globalThis.localStorage?.getItem(SIDE_COLLAPSED_KEY) === '1',
    settings: loadSettings(),
    shown: [],               // 本会话展示过的卡片 [{card, label, meta}]
    current: null,           // 当前大图显示的那一项
    toolMessages: new Map(), // tool_call_id → 消息区里的工具气泡
  };
  const el = {};

  // ------------------------------------------------------------------ 骨架

  function build() {
    stageEl.innerHTML = `<div class="uc-root">
      <section class="uc-cards">
        <div class="uc-card-slot" data-ref="cardSlot"></div>
        <div class="uc-card-meta" data-ref="cardMeta"></div>
        <div class="uc-strip" data-ref="strip"></div>
      </section>
      <aside class="uc-side" data-ref="side">
        <header class="uc-side-head">
          <span class="uc-side-title">消息区</span>
          <span class="uc-side-hint">模型说的话 / 思考 / 工具调用</span>
          <button type="button" class="uc-btn" data-action="clear">清空</button>
          <button type="button" class="uc-btn" data-action="toggle-side"></button>
        </header>
        <div class="uc-side-body">
          <details class="uc-settings" data-ref="settings"></details>
          <div class="uc-notice" data-ref="notice"></div>
          <div class="uc-messages" data-ref="messages"></div>
        </div>
      </aside>
    </div>`;
    for (const node of stageEl.querySelectorAll('[data-ref]')) el[node.dataset.ref] = node;
    el.root = stageEl.querySelector('.uc-root');
    stageEl.addEventListener('click', onClick);
    el.settings.addEventListener('change', onSettingsChange);
    clearMessages();
  }

  function ensureBuilt() {
    if (!el.messages || !stageEl.contains(el.messages)) build();
  }

  // ------------------------------------------------------------------ 渲染

  function render() {
    ensureBuilt();
    renderSide();
    renderSettings();
    renderNotice();
    renderCards();
  }

  function renderSide() {
    el.side.classList.toggle('collapsed', state.sideCollapsed);
    el.root.classList.toggle('side-collapsed', state.sideCollapsed);
    const button = el.side.querySelector('[data-action="toggle-side"]');
    button.textContent = state.sideCollapsed ? '◂ 展开' : '收起 ▸';
    button.setAttribute('aria-expanded', String(!state.sideCollapsed));
  }

  function delayRow(tool, label) {
    const value = state.settings[tool];
    return `<div class="uc-field">
        <label class="uc-check"><input type="checkbox" data-setting="${tool}.enabled" ${value.enabled ? 'checked' : ''}/>${label}延迟</label>
        <input type="number" min="0" max="300000" step="10" data-setting="${tool}.min" value="${escapeHtml(value.min)}"/>
        <span>–</span>
        <input type="number" min="0" max="300000" step="10" data-setting="${tool}.max" value="${escapeHtml(value.max)}"/>
        <span>ms</span>
      </div>`;
  }

  function renderSettings() {
    const open = el.settings.open;
    const s = state.settings;
    el.settings.innerHTML = `<summary>工具设置</summary>
      <div class="uc-field">
        <span class="uc-field-label">serp_google_search 后端</span>
        <select data-setting="searchProvider">
          <option value="serp" ${s.searchProvider === 'serp' ? 'selected' : ''}>SERP.HK (Google)</option>
          <option value="tavily" ${s.searchProvider === 'tavily' ? 'selected' : ''}>Tavily（转成 SERP 格式）</option>
        </select>
      </div>
      <p class="uc-hint">工具名与模型看到的结果格式都不变，只换谁去搜。tavily_search 工具始终走 Tavily。</p>
      <div class="uc-field">
        <span class="uc-field-label">SERP 额外延迟</span>
        <input type="number" min="0" max="300" step="0.1" data-setting="serpDelaySec" value="${escapeHtml(s.serpDelaySec)}"/>
        <span>s</span>
      </div>
      ${delayRow('get_client_location', '定位')}
      ${delayRow('amap_weather', '天气')}
      <p class="uc-hint">延迟只在工具成功返回前等待，模拟真实网络耗时；改完下一次工具调用生效。</p>`;
    el.settings.open = open;
  }

  function renderNotice() {
    const notes = [];
    if (state.defaults) {
      for (const { id, error } of unavailablePresets(state.defaults)) notes.push(`preset ${id} 不可用：${error}`);
    }
    const config = state.defaults?.uicard_tools_config;
    if (config) {
      if (!config.config_set) {
        notes.push('Demo 服务没有设置 UICARD_TOOLS_CONFIG：搜索、天气、地图工具会给模型回「未配置」的错误。');
      } else {
        const missing = Object.entries(config.configured || {}).filter(([, ok]) => !ok).map(([name]) => name);
        if (missing.length) notes.push(`未配置 key：${missing.join(' / ')}（这些工具会给模型回错误）`);
      }
    }
    const dataId = presetDataId(state.defaults, state.presetId);
    el.notice.innerHTML = [
      ...notes.map((note) => `<div class="uc-warn">${escapeHtml(note)}</div>`),
      dataId ? `<div class="uc-source">system / tools 取自训练样本 <code>${escapeHtml(dataId)}</code></div>` : '',
    ].join('');
  }

  function renderCards() {
    const current = state.current;
    if (!current) {
      el.cardSlot.innerHTML = `<div class="uc-empty">
          模型调用 <code>show_card</code> 之后，卡片会显示在这里。<br/>
          可以试着说：<em>「帮我看下北京今天的天气」</em>、<em>「找找附近的咖啡店」</em>、
          <em>「从中关村开车去北京西站怎么走」</em>、<em>「最近有什么科技新闻」</em>。
        </div>`;
      el.cardMeta.textContent = '';
    } else {
      el.cardSlot.innerHTML = renderCard(current.card);
      el.cardMeta.textContent = current.meta;
    }
    el.strip.innerHTML = state.shown.length
      ? `<span class="uc-strip-title">本会话的卡片</span>${state.shown.map((item, index) => `<button type="button"
            class="uc-chip ${item === current ? 'active' : ''}" data-action="show" data-index="${index}">${escapeHtml(item.label)}</button>`).join('')}`
      : '';
  }

  // ------------------------------------------------------------------ 消息区

  function clearMessages() {
    el.messages.innerHTML = '<div class="uc-placeholder">模型还没有输出</div>';
    state.toolMessages.clear();
  }

  function dropPlaceholder() {
    el.messages.querySelector('.uc-placeholder')?.remove();
  }

  function scrollMessages() {
    el.messages.scrollTop = el.messages.scrollHeight;
  }

  function appendMessageText(kind, text) {
    if (!text) return;
    ensureBuilt();
    dropPlaceholder();
    const last = el.messages.lastElementChild;
    if (last?.dataset.kind === kind && last.dataset.open === '1') {
      last.querySelector('.uc-msg-body').textContent += text;
    } else {
      const item = document.createElement('div');
      item.className = `uc-msg uc-msg-${kind}`;
      item.dataset.kind = kind;
      item.dataset.open = '1';
      item.innerHTML = `<div class="uc-msg-tag">${kind === 'speech' ? 'AI' : 'think'}</div><div class="uc-msg-body"></div>`;
      item.querySelector('.uc-msg-body').textContent = text;
      el.messages.appendChild(item);
    }
    scrollMessages();
  }

  function sealMessage(kind) {
    const last = el.messages?.lastElementChild;
    if (last?.dataset.kind === kind) last.dataset.open = '0';
  }

  function toolMessage(toolCallId) {
    ensureBuilt();
    if (toolCallId && state.toolMessages.has(toolCallId)) return state.toolMessages.get(toolCallId);
    dropPlaceholder();
    sealMessage('speech');
    sealMessage('think');
    const item = document.createElement('div');
    item.className = 'uc-msg uc-msg-tool streaming';
    item.dataset.kind = 'tool';
    item.dataset.open = '0';
    item.innerHTML = `<div class="uc-msg-tag">tool</div>
      <div class="uc-msg-body"><strong class="uc-tool-name">调用中…</strong><code class="uc-tool-args"></code></div>`;
    el.messages.appendChild(item);
    if (toolCallId) state.toolMessages.set(toolCallId, item);
    scrollMessages();
    return item;
  }

  function finalizeToolMessage(toolCallId, { name, args, fullText, error, via }) {
    const item = toolMessage(toolCallId);
    item.classList.remove('streaming');
    item.classList.toggle('uc-msg-error', Boolean(error));
    item.querySelector('.uc-tool-name').textContent = `${error ? `${name}（解析失败）` : name}${via ? ` · via ${via}` : ''}`;
    item.querySelector('.uc-tool-args').textContent = error ? String(fullText || error) : compactJson(args);
    if (error) {
      const note = document.createElement('span');
      note.className = 'uc-msg-error-text';
      note.textContent = error;
      item.querySelector('.uc-msg-body').appendChild(note);
    }
    scrollMessages();
  }

  function attachToolResult(toolCallId, result) {
    const item = state.toolMessages.get(toolCallId);
    if (!item) return;
    let details = item.querySelector('.uc-tool-result');
    if (!details) {
      details = document.createElement('details');
      details.className = 'uc-tool-result';
      details.innerHTML = '<summary></summary><code class="uc-tool-result-text"></code>';
      item.querySelector('.uc-msg-body').appendChild(details);
    }
    const summary = details.querySelector('summary');
    const body = details.querySelector('.uc-tool-result-text');
    if (!result) {
      summary.textContent = '工具返回 · 执行中…';
      body.textContent = '';
      return;
    }
    const elapsed = Number.isFinite(result.elapsed_ms) ? ` · ${result.elapsed_ms} ms` : '';
    const ids = (result.cards || []).map((entry) => entry.card_id).filter(Boolean);
    summary.textContent = `${result.error ? '工具返回 · 失败' : '工具返回'}${elapsed}${ids.length ? ` · 产出 ${ids.join(' ')}` : ''}`;
    details.classList.toggle('failed', Boolean(result.error));
    const text = String(result.error ? `${result.error}\n\n${result.content || ''}` : result.content || '');
    body.textContent = text.length > RESULT_PREVIEW_CHARS ? `${text.slice(0, RESULT_PREVIEW_CHARS)}…（共 ${text.length} 字）` : text;
    scrollMessages();
  }

  // ------------------------------------------------------------------ 卡片

  function showCard(card, { name, toolCallId, cardId }) {
    const time = new Date().toLocaleTimeString();
    const subject = card?.title || card?.city || card?.prompt || card?.caption || '';
    const label = `${cardId || card?.type || '?'}${subject ? ` · ${subject}` : ''}`;
    const item = {
      card,
      label: label.length > 24 ? `${label.slice(0, 24)}…` : label,
      meta: [name || 'show_card', card?.type || '?', cardId || '自写卡片', time, toolCallId].filter(Boolean).join(' · '),
    };
    state.shown.push(item);
    state.current = item;
    renderCards();
  }

  // ------------------------------------------------------------------ 工具执行

  function toolOptions(name) {
    const s = state.settings;
    const options = {
      search_provider: s.searchProvider,
      serp_response_delay_sec: Math.min(Math.max(Number(s.serpDelaySec) || 0, 0), 300),
    };
    for (const tool of ['get_client_location', 'amap_weather']) {
      options[`${tool}_delay_enabled`] = Boolean(s[tool].enabled);
      options[`${tool}_delay_min_ms`] = Number(s[tool].min) || 0;
      options[`${tool}_delay_max_ms`] = Number(s[tool].max) || 0;
    }
    const base = registry.indexBaseFor(name);
    if (base) options.card_index_base = base;
    return options;
  }

  function sendToolResult(toolCallId, content) {
    const client = getClient();
    if (!client || !toolCallId) return;
    client.sendToolResult({ toolCallId, content: String(content ?? '') });
  }

  function executeTool(name, args, toolCallId) {
    return registry.runExclusive(async () => {
      const response = await fetch(api.url('tools/execute'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, arguments: args, tool_call_id: toolCallId, options: toolOptions(name) }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
      registry.register(name, body.cards);
      return body;
    });
  }

  async function onToolCallDone(event) {
    const call = event.call || {};
    const name = call.name || 'unknown';
    const toolCallId = event.tool_call_id;
    const args = parseArguments(call.arguments);
    const via = name === 'serp_google_search' ? state.settings.searchProvider : null;
    finalizeToolMessage(toolCallId, { name, args, fullText: event.full_text, error: event.error, via });

    if (event.error) {
      // 解析失败的调用也带着 tool_call_id，Unit 会一直等结果；如实回一个错误。
      sendToolResult(toolCallId, JSON.stringify({ status: 'error', message: event.error }));
      return;
    }
    attachToolResult(toolCallId, null);
    let result;
    try {
      result = await executeTool(name, args, toolCallId);
    } catch (error) {
      const message = String(error?.message || error);
      result = { content: JSON.stringify({ status: 'error', message }), error: message, elapsed_ms: 0, cards: [] };
    }
    if (result.card_id) {
      const card = registry.get(result.card_id);
      if (card) {
        showCard(card, { name, toolCallId, cardId: result.card_id });
      } else {
        const message = unknownCardIdMessage(result.card_id);
        result = { ...result, error: message, content: JSON.stringify({ status: 'error', message }) };
      }
    } else if (result.card) {
      showCard(result.card, { name, toolCallId });
    }
    attachToolResult(toolCallId, result);
    sendToolResult(toolCallId, result.content);
  }

  // ------------------------------------------------------------------ 交互

  function onClick(event) {
    const target = event.target.closest('[data-action]');
    if (!target || !stageEl.contains(target)) return;
    const action = target.dataset.action;
    if (action === 'toggle-side') {
      state.sideCollapsed = !state.sideCollapsed;
      globalThis.localStorage?.setItem(SIDE_COLLAPSED_KEY, state.sideCollapsed ? '1' : '0');
      renderSide();
    } else if (action === 'clear') {
      clearMessages();
    } else if (action === 'show') {
      const item = state.shown[Number(target.dataset.index)];
      if (item) {
        state.current = item;
        renderCards();
      }
    }
  }

  function onSettingsChange(event) {
    const key = event.target.dataset.setting;
    if (!key) return;
    const value = event.target.type === 'checkbox' ? event.target.checked : event.target.value;
    const [head, field] = key.split('.');
    if (field) state.settings[head] = { ...state.settings[head], [field]: field === 'enabled' ? value : Number(value) };
    else state.settings[head] = head === 'serpDelaySec' ? Number(value) : value;
    try {
      globalThis.localStorage?.setItem(SETTINGS_KEY, JSON.stringify(state.settings));
    } catch {
      // 存储不可用时只是不记忆。
    }
  }

  return {
    pageTitle: 'MiniCPM-o · UI Card',
    panelTitle: 'UI Card',
    panelHint: 'amap_weather / map_search / serp / tavily → show_card',
    liveStatus: '对它说「帮我看下北京的天气」试试',
    presetOptions: PRESET_OPTIONS,
    fallbackQuickSetting: PRESET_1016,
    buildQuickSettings(defaults) {
      state.defaults = defaults;
      setMapDisplayConfig(defaults?.uicard_map_display || null);
      if (el.notice) renderNotice();
      return buildUiCardQuickSettings(defaults);
    },
    tools(defaults, presetId) {
      state.presetId = presetId || PRESET_1016;
      if (el.notice) renderNotice();
      return presetTools(defaults, state.presetId);
    },
    prepareSystem(segments) {
      return restampSegments(segments);
    },
    render,
    reset() {
      registry.reset();
      state.shown = [];
      state.current = null;
      ensureBuilt();
      clearMessages();
      render();
    },
    onSpokenDelta(event, text) {
      appendMessageText('speech', text);
    },
    onSpokenEnd(event) {
      if (event.reason === 'listen' || event.reason === 'turn_eos') sealMessage('speech');
    },
    onThinkDelta(event) {
      appendMessageText('think', textFromSteps(event.steps));
    },
    onThinkEnd() {
      sealMessage('think');
    },
    onToolCallBegin(event) {
      toolMessage(event.tool_call_id);
    },
    onToolCallDelta(event, text) {
      if (!text) return;
      toolMessage(event.tool_call_id).querySelector('.uc-tool-args').textContent += text;
      scrollMessages();
    },
    onToolCallDone,
  };
}
