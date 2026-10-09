// MVP 画板场景：display_object_on_board 工具 → 搜图 → 画板卡片 → 回 tool result。
// 逻辑从 fc-board-app.js 原样迁出，行为不变。

import { BoardState } from './board-state.js';
import {
  QUICK_SETTING_0730,
  buildBoardQuickSettings,
} from './quick-settings.js';
import { escapeHtml } from '../../../static/demo-gallery/core/ui-utils.js';

const DEFAULT_SYSTEM_TEXT = `你是一个可以一边听用户说话、一边思考并调用工具的语音助手。用户让你把故事或描述中出现的动物、植物或具体物体放到画板上时，使用 display_object_on_board 工具。不要等用户完全讲完才思考；在你确认具体对象后，可以调用工具把对象放到画板。`;

const DISPLAY_OBJECT_TOOL = {
  type: 'function',
  function: {
    name: 'display_object_on_board',
    description: 'Display a named concrete object on the visual board.',
    parameters: {
      type: 'object',
      properties: {
        name: { type: 'string', description: 'The object name to display on the board.' },
      },
      required: ['name'],
    },
  },
};

const PRESET_OPTIONS = [
  { id: '0730', label: '0730 · Current' },
  { id: '0807_sysalign', label: '0807 · SYS-ALIGN' },
  { id: '0808_family4_full', label: '0808 · Family4 Full (T1/T2/T4)' },
  { id: '0808_family4_short', label: '0808 · Family4 Short (T3)' },
  { id: 'latest_mvp_zh', label: '最新训练前缀 · MVP 中文 (F224, 45/25, tts_pad legacy)' },
  { id: 'latest_mvp_zh_revised', label: '最新训练前缀 · MVP 中文 (F224, 45/25, tts_pad revised)' },
];

export function createBoardScene({ stageEl, getClient, api }) {
  const state = new BoardState({ maxCards: 6 });

  function defaultSystem(defaults) {
    const system = defaults?.default_system;
    if (!system || !Array.isArray(system.segments) || !Array.isArray(system.tools)) {
      return {
        segments: [{ kind: 'text', text: DEFAULT_SYSTEM_TEXT }],
        tools: [DISPLAY_OBJECT_TOOL],
      };
    }
    return system;
  }

  function defaultTtsPromptAudioPath(defaults) {
    const audio = defaults?.default_tts_prompt_audio;
    if (audio?.source !== 'path' || typeof audio.file_path !== 'string') return '';
    return audio.file_path;
  }

  function render() {
    if (!state.cards.length) {
      stageEl.classList.add('is-empty');
      stageEl.innerHTML = `<div class="empty-state">
        <strong>先说一句指令</strong>（约 6-8 秒），比如：<br />
        <em>「我等下讲讲故事里出现的动物，提到的你就帮我放到画板上。」</em><br />
        <em>「等下我描述房间里的东西，提到的你就帮我展示出来。」</em><br />
        等 AI 应一声之后，再自由描述具体物体。<br />
        <span style="color:#999;font-size:11px">句式仿训练数据：\`等下我/我等下…提到 XX 你就 放到画板上\`</span>
      </div>`;
      return;
    }
    stageEl.classList.remove('is-empty');
    stageEl.innerHTML = state.cards.map((card) => {
      const image = card.image?.image_url
        ? `<img src="${escapeHtml(card.image.image_url)}" alt="${escapeHtml(card.query)}" />`
        : `<div class="placeholder">${card.status === 'searching' ? '搜图中…' : (card.error || '—')}</div>`;
      return `<article class="card ${escapeHtml(card.status)}" title="${escapeHtml(card.tool_call_id || '')}">
        ${image}
        <div class="card-body">
          <strong>${escapeHtml(card.query)}</strong>
          <span>${escapeHtml(card.status)}</span>
        </div>
      </article>`;
    }).join('');
  }

  async function executeDisplayObjectOnBoard({ query, toolCallId }) {
    let result;
    try {
      const response = await fetch(api.url('tools/display_object_on_board'), {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name: query, tool_call_id: toolCallId }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      result = await response.json();
    } catch (err) {
      result = {
        query,
        image: {
          query,
          asset_id: `client-tool:${toolCallId || query}`,
          image_url: placeholderImageDataUrl(query),
          source_url: null,
          title: query,
          elapsed_ms: 0,
          error: String(err?.message || err),
        },
        error: String(err?.message || err),
        tool_response_content: JSON.stringify({
          status: 'displayed',
          name: query,
          reason: '已在画板显示该对象。',
        }),
      };
    }
    state.upsert({
      card_id: cardIdFor(toolCallId),
      tool_call_id: toolCallId,
      query: result.query || query,
      status: result.error ? 'error' : 'ready',
      image: result.image,
      error: result.error,
    });
    render();
    sendToolResult({
      toolCallId,
      query: result.query || query,
      content: result.tool_response_content,
    });
  }

  function sendToolResult({ toolCallId, query, content }) {
    const client = getClient();
    if (!client || !toolCallId) return;
    let payload = content || {
      status: 'displayed',
      name: query,
      reason: '已在画板显示该对象。',
    };
    if (typeof payload === 'string') {
      try { payload = JSON.parse(payload); } catch (_) { payload = { text: payload }; }
    }
    client.sendToolResult({ toolCallId, content: payload });
  }

  return {
    panelTitle: 'Board · display_object_on_board',
    panelHint: '1 × 6 cards · FIFO',
    liveStatus: 'Listening via /v1/realtime · mention concrete objects for the board',
    presetOptions: PRESET_OPTIONS,
    fallbackQuickSetting: QUICK_SETTING_0730,

    buildQuickSettings(defaults) {
      return buildBoardQuickSettings(defaultSystem(defaults), defaultTtsPromptAudioPath(defaults), {
        nonSpokenScheduling: defaults?.non_spoken_scheduling,
        nonSpokenBudgetWhileListening: defaults?.non_spoken_budget_while_listening,
        nonSpokenBudgetWhileSpeaking: defaults?.non_spoken_budget_while_speaking,
      });
    },

    tools(defaults) {
      return JSON.parse(JSON.stringify(defaultSystem(defaults).tools));
    },

    render,

    reset() {
      state.cards = [];
      stageEl.innerHTML = '';
      render();
    },

    onThinkBegin() {},
    onThinkDelta() {},
    onThinkEnd() {},

    onToolCallDone(event) {
      const call = event.call || {};
      if (call.name !== 'display_object_on_board' || event.error) return;
      const args = parseArguments(call.arguments);
      const query = String(args.name || '').trim();
      if (!query) return;
      state.upsert({
        card_id: cardIdFor(event.tool_call_id),
        tool_call_id: event.tool_call_id,
        query,
        status: 'searching',
      });
      render();
      executeDisplayObjectOnBoard({ query, toolCallId: event.tool_call_id });
    },
  };
}

function cardIdFor(toolCallId) {
  return `card:${toolCallId || 'pending'}`;
}

function parseArguments(value) {
  if (!value) return {};
  if (typeof value === 'object') return value;
  try { return JSON.parse(value); } catch (_) { return {}; }
}

function placeholderImageDataUrl(label) {
  const safeLabel = escapeHtml(label || 'object');
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420">
      <rect width="640" height="420" fill="#0f172a"/>
      <rect x="42" y="42" width="556" height="336" rx="18" fill="#1e293b" stroke="#38bdf8" stroke-width="3"/>
      <text x="320" y="200" fill="#e2e8f0" font-family="sans-serif" font-size="36" font-weight="700" text-anchor="middle">${safeLabel}</text>
      <text x="320" y="250" fill="#94a3b8" font-family="sans-serif" font-size="20" text-anchor="middle">display_object_on_board</text>
    </svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}
