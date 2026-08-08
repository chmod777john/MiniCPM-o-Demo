// FC Board quick settings are UI projection templates, not protocol defaults.
// 0807 prefix source:
// swy-dev/omni_agent_research/minicpm_o5_training/issues/
// SYS-ALIGN-001-agent-duplex-three-source-system-prefix.md

export const QUICK_SETTING_CUSTOM = 'custom';
export const QUICK_SETTING_0730 = '0730';
export const QUICK_SETTING_0807_SYSALIGN = '0807_sysalign';

export const SYSALIGN_0807_PREFIX_TEXT = [
  '扮演一个具有以上声音特征的助手。请认真、高质量地回复用户的问题。',
  '请用高自然度的方式和用户聊天。',
  '你处于双工 Agent 模式：可以一边听、一边说；',
  '可以在 non_spoken_slot 中并行地思考与工具调用，',
  '并在 input_event_slot 中接收工具返回结果。',
  '你是由面壁智能开发的人工智能助手：面壁小钢炮。\n',
].join('');

export function buildBoardQuickSettings(defaultSystem, defaultTtsPromptAudioPath) {
  const sourceSegments = cloneSegments(defaultSystem?.segments || []);
  const audioSegments = sourceSegments.filter((segment) => segment.kind === 'audio');
  const textSegments = sourceSegments.filter((segment) => segment.kind === 'text');

  return new Map([
    [
      QUICK_SETTING_0730,
      {
        segments: sourceSegments,
        ttsPromptAudioPath: defaultTtsPromptAudioPath || '',
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
      },
    ],
  ]);
}

export function cloneQuickSetting(setting) {
  return {
    segments: cloneSegments(setting?.segments || []),
    ttsPromptAudioPath: setting?.ttsPromptAudioPath || '',
  };
}

function cloneSegments(segments) {
  return JSON.parse(JSON.stringify(segments));
}

function firstAudioPath(segments) {
  const audio = segments.find((segment) => segment.kind === 'audio');
  return audio?.audio?.file_path || '';
}
