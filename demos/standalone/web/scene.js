// Standalone Event 场景：主面板上一个输入框，回车即发送一次 input.standalone_event。
//
// 只负责「注入」与「回执」：模型说了什么、想了什么由 shell 的 AI Speech / Think 面板如实展示，
// 这里不对模型输出做任何解读或补救。下方时间线列出每条注入及它被哪个 Unit 吃掉
// （response.unit.started.standalone_events 回执）。
// 本 Demo 不带任何工具；模型若仍发出 tool_call，时间线如实列出，不拦截、不回包。

import { escapeHtml } from '../../../static/demo-gallery/core/ui-utils.js';
import {
  EXAMPLES_BY_LANGUAGE,
  PRESET_OPTIONS,
  PRESET_ZH_THINK,
  buildStandaloneQuickSettings,
  presetLanguage,
} from './quick-settings.js';

export function createScene({ stageEl, getClient }) {
  const state = {
    entries: [],      // {kind:'event', id, text, status, unit} | {kind:'tool', name, args, unit}
    seq: 0,
    notice: '',
    presetId: PRESET_ZH_THINK,   // shell 经 tools(defaults, presetId) 告知当前 preset（只用来切示例语言）
  };
  let built = false;

  function build() {
    stageEl.innerHTML = `
      <div class="sa-compose">
        <input class="sa-input" type="text" autocomplete="off"
               placeholder="输入一段文本，回车 = 注入一次 standalone_event" />
        <button class="sa-send" type="button">注入</button>
      </div>
      <div class="sa-examples"></div>
      <div class="sa-notice" role="status"></div>
      <ol class="sa-timeline"></ol>`;
    const input = stageEl.querySelector('.sa-input');
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.isComposing) {
        event.preventDefault();
        send(input);
      }
    });
    stageEl.querySelector('.sa-send').addEventListener('click', () => send(input));
    stageEl.querySelector('.sa-examples').addEventListener('click', (event) => {
      const chip = event.target.closest('[data-example]');
      if (!chip) return;
      input.value = chip.dataset.example;
      input.focus();
    });
    built = true;
  }

  function send(input) {
    const text = input.value.trim();
    if (!text) return;
    const client = getClient();
    if (!client || !client.ready) {
      state.notice = '会话还没就绪：先点 Start mic live，等底部状态栏出现「Listening」再注入（system 预填要十几秒）。';
      render();
      return;
    }
    const id = `sa_${++state.seq}`;
    try {
      client.sendStandaloneEvent({ content: text, eventId: id });
    } catch (error) {
      state.notice = `发送失败：${error.message}`;
      render();
      return;
    }
    state.entries.push({ kind: 'event', id, text, status: 'pending', unit: null });
    state.notice = '';
    input.value = '';
    render();
  }

  function renderExamples() {
    const examples = EXAMPLES_BY_LANGUAGE[presetLanguage(state.presetId)] || [];
    return examples.length
      ? '<span class="sa-examples-label">示例（点一下填入）：</span>' + examples
        .map((text) => `<button type="button" class="sa-chip" data-example="${escapeHtml(text)}">${escapeHtml(text)}</button>`)
        .join('')
      : '';
  }

  function renderEntry(entry) {
    if (entry.kind === 'tool') {
      return `<li class="sa-row sa-tool">
        <span class="sa-tag">模型发出 tool_call</span>
        <span class="sa-unit">U${escapeHtml(String(entry.unit ?? '?'))}</span>
        <code>${escapeHtml(entry.name)}</code>
        <span class="sa-args">${escapeHtml(entry.args)}</span>
      </li>`;
    }
    const status = entry.status === 'consumed'
      ? `<span class="sa-status ok">已注入 Unit ${escapeHtml(String(entry.unit))}</span>`
      : '<span class="sa-status wait">已发送，等待下一个 Unit</span>';
    return `<li class="sa-row sa-event">
      <span class="sa-tag">你注入</span>
      <span class="sa-text">${escapeHtml(entry.text)}</span>
      ${status}
    </li>`;
  }

  function render() {
    if (!built) build();
    stageEl.querySelector('.sa-examples').innerHTML = renderExamples();
    stageEl.querySelector('.sa-notice').textContent = state.notice;
    const timeline = stageEl.querySelector('.sa-timeline');
    timeline.innerHTML = state.entries.length
      ? state.entries.slice().reverse().map(renderEntry).join('')
      : '<li class="sa-empty">还没有注入任何事件。开麦后在上面输入文本并回车；模型的反应看下方 AI Speech 与 Think 面板。</li>';
  }

  return {
    panelTitle: 'Standalone · input_event 注入',
    panelHint: '回车 = 一次 input.standalone_event',
    liveStatus: 'Listening via /v1/realtime · 输入框回车即注入 standalone_event',
    presetOptions: PRESET_OPTIONS,
    fallbackQuickSetting: PRESET_ZH_THINK,

    buildQuickSettings(defaults) {
      return buildStandaloneQuickSettings(defaults);
    },

    tools(defaults, presetId) {
      if (presetId && presetId !== state.presetId) {
        state.presetId = presetId;
        if (built) queueMicrotask(render);   // preset 切换时示例跟着换
      }
      return [];   // 不带任何工具
    },

    render,

    reset() {
      state.entries = [];
      state.notice = '';
      render();
    },

    onUnitStarted(event) {
      const consumed = (event.standalone_events || []).map((item) => item.event_id);
      if (!consumed.length) return;
      for (const entry of state.entries) {
        if (entry.kind === 'event' && entry.status === 'pending' && consumed.includes(entry.id)) {
          entry.status = 'consumed';
          entry.unit = event.unit_index;
        }
      }
      render();
    },

    onToolCallDone(event) {
      const call = event.call || {};
      let args = call.arguments;
      if (typeof args !== 'string') args = JSON.stringify(args ?? {});
      state.entries.push({ kind: 'tool', name: call.name || '(解析失败)', args, unit: event.unit_index });
      render();
    },
  };
}
