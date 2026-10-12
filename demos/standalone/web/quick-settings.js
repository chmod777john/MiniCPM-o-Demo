// Standalone Demo 的 preset：[统一前缀, 参考音频] 两段，不带任何工具。
// 本 Demo 只演示「注入 standalone 事件」本身，与工具 / 子任务无关。

import {
  SYSTEM_PREFIX_EN_GENERAL,
  SYSTEM_PREFIX_EN_NO_THINK,
  SYSTEM_PREFIX_ZH_NO_THINK,
  SYSTEM_PREFIX_ZH_THINK,
} from '../../../static/demo-gallery/core/system-prefixes.js';

// 训练侧统一参考音频（swy-dev o5-ai-spoken-regulation skill 的 ref_audio_mapping.json 登记项）。
const REF_DIR = '/user/sunweiyue/lib/swy-dev/.claude/skills/o5-ai-spoken-regulation/media/ref_audio_in_training_data/MB-HT';
export const ZH_REF_AUDIO_PATH = `${REF_DIR}/ref_minicpm_signature_fix.wav`;          // 中文声优 F224
export const EN_REF_AUDIO_PATH = `${REF_DIR}/F409-379_clip_310-324_fix_16khz.wav`;    // 英文声优 F409

export const PRESET_ZH_THINK = 'zh_think';
export const PRESET_ZH_NOTHINK = 'zh_nothink';
export const PRESET_EN_GENERAL = 'en_general';
export const PRESET_EN_NOTHINK = 'en_nothink';

const PRESETS = [
  { id: PRESET_ZH_THINK, label: '中文 · think（前缀 #1 · F224）', prefix: SYSTEM_PREFIX_ZH_THINK, ref: ZH_REF_AUDIO_PATH },
  { id: PRESET_ZH_NOTHINK, label: '中文 · no-think（前缀 #2 · F224）', prefix: SYSTEM_PREFIX_ZH_NO_THINK, ref: ZH_REF_AUDIO_PATH },
  { id: PRESET_EN_GENERAL, label: 'English · general (prefix #3 · F409)', prefix: SYSTEM_PREFIX_EN_GENERAL, ref: EN_REF_AUDIO_PATH },
  { id: PRESET_EN_NOTHINK, label: 'English · no-think (prefix #6 · F409)', prefix: SYSTEM_PREFIX_EN_NO_THINK, ref: EN_REF_AUDIO_PATH },
];

export const PRESET_OPTIONS = PRESETS.map(({ id, label }) => ({ id, label }));

/** 输入框的一键示例（只填进输入框，不自动发送）。刻意与工具 / 子任务无关。 */
export const EXAMPLES_BY_LANGUAGE = {
  zh: ['[系统通知] 用户的快递已经送到门口了', '现在是下午三点半'],
  en: ['[NOTICE] The user\'s package has just been delivered.', 'It is now 3:30 PM.'],
};

export function presetLanguage(presetId) {
  return String(presetId || '').startsWith('en_') ? 'en' : 'zh';
}

/** 构造 quick setting（预算 / 调度 / tts_pad 由所选端点的模型决定，不在 preset 里）。 */
export function buildStandaloneQuickSettings() {
  return new Map(PRESETS.map(({ id, prefix, ref }) => [id, {
    segments: [
      { kind: 'text', text: prefix },
      { kind: 'audio', audio: { source: 'path', file_path: ref } },
    ],
    ttsPromptAudioPath: ref,
  }]));
}
