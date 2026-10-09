// Rejection（拒识）场景：无工具。每个 think 块流式解析出「模型认为现在是谁在说话」，
// 首次出现的人建卡（用户 / 说话人 N），已登记的人只高亮，表示模型当前识别到的说话人。

import {
  QUICK_SETTING_REJECTION_ZH,
  REJECTION_PRESET_OPTIONS,
  buildRejectionQuickSettings,
} from '../rejection-quick-settings.js';
import {
  SpeakerRegistry,
  detectThinkLanguage,
  parseSpeakerThink,
} from '../speaker-think-parser.js';
import { escapeHtml, textFromSteps } from '../ui-utils.js';

const LABELS = {
  zh: {
    user: '用户',
    speaker: (no) => `说话人 ${no}`,
    now: '当前识别',
    nobody: '还没有识别到说话人',
    registeredAt: (unit) => `登记于 Unit ${unit}`,
    hits: (n) => `识别 ${n} 次`,
    lastSeen: (unit) => `最近 Unit ${unit}`,
    claimed: '曾要求成为唯一对象 · 已拒绝',
    unregistered: '模型未先登记就直接引用',
    created: '新登记',
    unparsed: '未识别的 think',
    log: '识别记录',
    actions: {
      respond: '正常交流',
      interrupt_listen: '用户打断 → 停下来听',
      backchannel_continue: '用户附和 → 继续说',
      no_respond: '不回应',
      no_stop: '不停下',
    },
    reasons: {
      first_speaker: '第一个开口的人 → 登记为用户',
      requested: '应其要求 → 登记为用户',
    },
    claimEvent: '要求成为唯一对象 → 拒绝',
  },
  en: {
    user: 'User',
    speaker: (no) => `Speaker ${no}`,
    now: 'Now identified',
    nobody: 'No speaker identified yet',
    registeredAt: (unit) => `registered @ Unit ${unit}`,
    hits: (n) => `${n} hit${n === 1 ? '' : 's'}`,
    lastSeen: (unit) => `last @ Unit ${unit}`,
    claimed: 'asked to be the only one · refused',
    unregistered: 'referenced before registration',
    created: 'new',
    unparsed: 'Unparsed think',
    log: 'Identification log',
    actions: {
      respond: 'respond normally',
      interrupt_listen: 'interrupting → stop & listen',
      backchannel_continue: 'backchannel → keep talking',
      no_respond: "don't respond",
      no_stop: "don't stop",
    },
    reasons: {
      first_speaker: 'first to speak → registered as user',
      requested: 'asked for it → registered as user',
    },
    claimEvent: 'asks to be the only one → refused',
  },
};

const MAX_LOG_ROWS = 40;

export function createSpeakerScene({ stageEl }) {
  const registry = new SpeakerRegistry();
  const unparsed = [];
  let lang = 'zh';
  let blockSeq = 0;
  let currentBlock = null;

  function t() {
    return LABELS[lang] || LABELS.zh;
  }

  function speakerName(speaker) {
    return speaker.isUser ? t().user : t().speaker(speaker.speakerNo);
  }

  function avatarText(speaker) {
    if (speaker.isUser) return lang === 'zh' ? '用' : 'U';
    return `${lang === 'zh' ? '' : 'S'}${speaker.speakerNo}`;
  }

  function describe(parsed) {
    const L = t();
    if (parsed.kind === 'register_user') return L.reasons[parsed.reason] || L.reasons.first_speaker;
    if (parsed.kind === 'speaker_claims_user') {
      return `${L.claimEvent}${parsed.action ? ` · ${L.actions[parsed.action]}` : ''}`;
    }
    return parsed.action ? L.actions[parsed.action] : '…';
  }

  function ingest(text, { unitIndex, final }) {
    if (!currentBlock) return;
    const parsed = parseSpeakerThink(text);
    if (text.trim()) lang = detectThinkLanguage(text);
    if (!parsed) {
      if (final && text.trim()) {
        unparsed.push({ blockKey: currentBlock.key, unitIndex: currentBlock.unitIndex, text });
      }
      if (final) render();
      return;
    }
    const before = registry.events.length;
    registry.apply(parsed, { unitIndex: currentBlock.unitIndex ?? unitIndex, blockKey: currentBlock.key, text });
    // 只有身份首次确定或 think 结束时重绘，避免每个 token 都刷 DOM。
    if (registry.events.length !== before || final) render();
  }

  function render() {
    const L = t();
    const speakers = registry.orderedSpeakers();
    if (!speakers.length && !unparsed.length) {
      stageEl.classList.remove('speaker-mode');
      stageEl.classList.add('empty-board');
      stageEl.innerHTML = `<div class="empty-state">
        <strong>先由你开口说第一句</strong>（如「小钢炮，我来啦」/「Hi Mini, I'm here」）。<br />
        模型会在你开口约 0.9 秒后，在 think 里把你登记为<b>用户</b>。<br />
        之后让旁边的人插话、或外放一段别人的说话，观察它是否识别为<b>说话人 N</b> 且不回应；<br />
        你本人再说话时，对应的卡片会高亮。<br />
        <span style="color:#999;font-size:11px">卡片只由模型 think 里写下的身份判断驱动，不做任何推断。</span>
      </div>`;
      return;
    }
    stageEl.classList.remove('empty-board');
    // .board 默认是画板的 6 列卡片网格；说话人面板自己排版。
    stageEl.classList.add('speaker-mode');

    const lastEvent = registry.events[registry.events.length - 1] || null;
    const active = registry.activeKey ? registry.speakers.get(registry.activeKey) : null;
    const nowHtml = active && lastEvent
      ? `<div class="speaker-now ${active.isUser ? 'is-user' : 'is-other'}">
          <span class="speaker-now-label">${escapeHtml(L.now)}</span>
          <span class="speaker-avatar small">${escapeHtml(avatarText(active))}</span>
          <strong>${escapeHtml(speakerName(active))}</strong>
          <span class="speaker-now-action">${escapeHtml(describe(lastEvent.parsed))}</span>
          <span class="speaker-now-unit">Unit ${escapeHtml(lastEvent.unitIndex ?? '—')}</span>
        </div>`
      : `<div class="speaker-now"><span class="speaker-now-label">${escapeHtml(L.nobody)}</span></div>`;

    const cardsHtml = speakers.map((speaker) => {
      const isActive = speaker.key === registry.activeKey;
      const badges = [
        speaker.claimedUser ? `<span class="speaker-badge warn">${escapeHtml(L.claimed)}</span>` : '',
        speaker.registeredByModel ? '' : `<span class="speaker-badge warn">${escapeHtml(L.unregistered)}</span>`,
      ].join('');
      // key 里带 hits：同一个人被再次识别时 DOM 节点重建，高亮动画会重新播放一次。
      return `<article class="speaker-card ${speaker.isUser ? 'is-user' : 'is-other'} ${isActive ? 'active' : ''}"
                       data-speaker="${escapeHtml(speaker.key)}" data-hit="${speaker.hits}">
        <div class="speaker-avatar">${escapeHtml(avatarText(speaker))}</div>
        <div class="speaker-body">
          <strong>${escapeHtml(speakerName(speaker))}</strong>
          <span class="speaker-meta">${escapeHtml(L.registeredAt(speaker.registeredUnit ?? '—'))}
            · ${escapeHtml(L.hits(speaker.hits))} · ${escapeHtml(L.lastSeen(speaker.lastUnit ?? '—'))}</span>
          <span class="speaker-action">${escapeHtml(speaker.lastAction ? L.actions[speaker.lastAction] : '…')}</span>
          ${badges}
        </div>
      </article>`;
    }).join('');

    const rows = [
      ...registry.events.map((event) => ({ ...event, ok: true })),
      ...unparsed.map((item) => ({ ...item, ok: false })),
    ]
      .sort((a, b) => b.blockKey - a.blockKey)
      .slice(0, MAX_LOG_ROWS)
      .map((row) => {
        if (!row.ok) {
          return `<li class="speaker-log-row unparsed">
            <span class="unit">U${escapeHtml(row.unitIndex ?? '—')}</span>
            <span class="who">${escapeHtml(L.unparsed)}</span>
            <span class="text">${escapeHtml(row.text)}</span>
          </li>`;
        }
        const speaker = registry.speakers.get(row.parsed.speakerKey);
        return `<li class="speaker-log-row ${row.parsed.speakerKey === 'user' ? 'is-user' : 'is-other'}">
          <span class="unit">U${escapeHtml(row.unitIndex ?? '—')}</span>
          <span class="who">${escapeHtml(speaker ? speakerName(speaker) : row.parsed.speakerKey)}${row.created ? ` <em>${escapeHtml(L.created)}</em>` : ''}</span>
          <span class="what">${escapeHtml(describe(row.parsed))}</span>
          <span class="text" title="${escapeHtml(row.text)}">${escapeHtml(row.text)}</span>
        </li>`;
      }).join('');

    stageEl.innerHTML = `<div class="speaker-stage">
      ${nowHtml}
      <div class="speaker-grid">${cardsHtml}</div>
      <div class="speaker-log">
        <div class="speaker-log-title">${escapeHtml(L.log)}</div>
        <ol>${rows}</ol>
      </div>
    </div>`;
  }

  return {
    id: 'speakers',
    pageTitle: 'MiniCPM-o · Rejection',
    panelTitle: 'Speakers · think 中识别到的说话人',
    panelHint: '新出现的人建卡 · 已登记的人高亮',
    liveStatus: 'Listening via /v1/realtime · 先由你开口登记为用户，再让其他人插话',
    presetOptions: REJECTION_PRESET_OPTIONS,
    fallbackQuickSetting: QUICK_SETTING_REJECTION_ZH,

    buildQuickSettings() {
      return buildRejectionQuickSettings();
    },

    tools() {
      return [];
    },

    render,

    reset() {
      registry.reset();
      unparsed.length = 0;
      currentBlock = null;
      render();
    },

    onThinkBegin(event) {
      blockSeq += 1;
      currentBlock = { key: blockSeq, unitIndex: event.unit_index, text: '' };
    },

    onThinkDelta(event) {
      if (!currentBlock) this.onThinkBegin(event);
      currentBlock.text += textFromSteps(event.steps);
      ingest(currentBlock.text, { unitIndex: event.unit_index, final: false });
    },

    onThinkEnd(event) {
      if (!currentBlock) return;
      const text = typeof event.full_text === 'string' && event.full_text ? event.full_text : currentBlock.text;
      ingest(text, { unitIndex: event.unit_index, final: true });
      currentBlock = null;
    },

    onToolCallDone() {},
  };
}
