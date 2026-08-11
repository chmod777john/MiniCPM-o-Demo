// FC Board quick settings are UI projection templates, not protocol defaults.
// 0807 source: Family3 SYS-ALIGN-001.
// 0808 source: https://modelbest.feishu.cn/wiki/FbgHwNQGaiVtp6kQxqYcpulCnHf
// Family4 presets prepend their prefix and preserve the case's original segments.

export const QUICK_SETTING_CUSTOM = 'custom';
export const QUICK_SETTING_0730 = '0730';
export const QUICK_SETTING_0807_SYSALIGN = '0807_sysalign';
export const QUICK_SETTING_0808_FAMILY4_FULL = '0808_family4_full';
export const QUICK_SETTING_0808_FAMILY4_SHORT = '0808_family4_short';

export const SYSALIGN_0807_PREFIX_TEXT = [
  '扮演一个具有以上声音特征的助手。请认真、高质量地回复用户的问题。',
  '请用高自然度的方式和用户聊天。',
  '你处于双工 Agent 模式：可以一边听、一边说；',
  '可以在 non_spoken_slot 中并行地思考与工具调用，',
  '并在 input_event_slot 中接收工具返回结果。',
  '你是由面壁智能开发的人工智能助手：面壁小钢炮。\n',
].join('');

export const FAMILY4_0808_FULL_PREFIX_TEXT = [
  '你处于双工 Agent 模式，时间轴按约一秒一个的 unit 组织。',
  '你在 user_audio_slot 中接收当前 unit 的用户语音。',
  '若需要说话，则在 ai_spoken_slot 中作出 speak 决策并生成当前 unit 要说的文本；',
  '否则作出 listen 决策。',
  '你可以在 ai_non_spoken_slot 中生成 think 或 tool_call 内容；',
  '若无额外动作，则作出 no_action 决策。',
  '你在 input_event_slot 中接收工具返回结果或其他独立事件信息。',
  '使用以下音频中的声音说话。\n',
].join('');

export const FAMILY4_0808_SHORT_PREFIX_TEXT = '使用以下音频中的声音说话。\n';

const FAMILY3_RUNTIME = Object.freeze({
  nonSpokenScheduling: 'quality',
  nonSpokenBudgetWhileListening: 30,
  nonSpokenBudgetWhileSpeaking: 15,
});

const FAMILY4_RUNTIME = Object.freeze({
  nonSpokenScheduling: 'quality',
  nonSpokenBudgetWhileListening: 45,
  nonSpokenBudgetWhileSpeaking: 25,
});

export function buildBoardQuickSettings(
  defaultSystem,
  defaultTtsPromptAudioPath,
  defaultRuntime = {},
) {
  const sourceSegments = cloneSegments(defaultSystem?.segments || []);
  const audioSegments = sourceSegments.filter((segment) => segment.kind === 'audio');
  const textSegments = sourceSegments.filter((segment) => segment.kind === 'text');

  return new Map([
    [
      QUICK_SETTING_0730,
      {
        segments: sourceSegments,
        ttsPromptAudioPath: defaultTtsPromptAudioPath || '',
        runtime: cloneRuntime(defaultRuntime),
      },
    ],
    [
      QUICK_SETTING_0807_SYSALIGN,
      {
        segments: [
          { kind: 'text', text: SYSALIGN_0807_PREFIX_TEXT },
          ...cloneSegments(audioSegments),
          ...cloneSegments(textSegments),
        ],
        ttsPromptAudioPath: defaultTtsPromptAudioPath || firstAudioPath(audioSegments),
        runtime: cloneRuntime(FAMILY3_RUNTIME),
      },
    ],
    [
      QUICK_SETTING_0808_FAMILY4_FULL,
      buildPrependedSetting(
        FAMILY4_0808_FULL_PREFIX_TEXT,
        sourceSegments,
        defaultTtsPromptAudioPath,
        FAMILY4_RUNTIME,
      ),
    ],
    [
      QUICK_SETTING_0808_FAMILY4_SHORT,
      buildPrependedSetting(
        FAMILY4_0808_SHORT_PREFIX_TEXT,
        sourceSegments,
        defaultTtsPromptAudioPath,
        FAMILY4_RUNTIME,
      ),
    ],
  ]);
}

export function cloneQuickSetting(setting) {
  return {
    segments: cloneSegments(setting?.segments || []),
    ttsPromptAudioPath: setting?.ttsPromptAudioPath || '',
    runtime: cloneRuntime(setting?.runtime),
  };
}

function cloneSegments(segments) {
  return JSON.parse(JSON.stringify(segments));
}

function buildPrependedSetting(
  prefixText,
  sourceSegments,
  defaultTtsPromptAudioPath,
  runtime,
) {
  return {
    segments: [
      { kind: 'text', text: prefixText },
      ...cloneSegments(sourceSegments),
    ],
    ttsPromptAudioPath:
      defaultTtsPromptAudioPath || firstAudioPath(sourceSegments),
    runtime: cloneRuntime(runtime),
  };
}

function cloneRuntime(runtime = {}) {
  return {
    nonSpokenScheduling: runtime.nonSpokenScheduling,
    nonSpokenBudgetWhileListening: runtime.nonSpokenBudgetWhileListening,
    nonSpokenBudgetWhileSpeaking: runtime.nonSpokenBudgetWhileSpeaking,
  };
}

function firstAudioPath(segments) {
  const audio = segments.find((segment) => segment.kind === 'audio');
  return audio?.audio?.file_path || '';
}
