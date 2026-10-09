// Rejection（拒识）Demo 的系统提示词预设：与 rejection 0929 训练数据的 system 逐字一致。
// 数据来源（飞书登记表「启用」的 rejection 行）：
//   zh  /user/sunweiyue/dataset/o5_sdk_training_data/rejection/agent_duplex_rejection_0929_min2k_max8k_zh
//   en  /user/sunweiyue/dataset/o5_sdk_training_data/rejection/agent_duplex_rejection_0929_min1k_max8k_en
// system = [统一前缀, 参考音频, 规则文本]，无 tools，UnitPolicy 45/25；32k 档的参考音频与 8k 档逐字节相同。
// 盘上 bundle 只有后两段 —— 第一段「统一前缀」在训练侧注入，推理侧必须自己补（见 system-prefixes.js）。
// rejection 的 think 比例是 1.000（全 think），中文取 #1；英文取 #3 通用（用户 2026-10-09 定）。
// 数据里另有约 3% 样本只有身份句、不含规则句（「你是面壁小钢炮。」），这里取主形态（带规则）。

import {
  SYSTEM_PREFIX_EN_GENERAL,
  SYSTEM_PREFIX_ZH_THINK,
} from './system-prefixes.js';

export const QUICK_SETTING_REJECTION_ZH = 'rejection_zh';
export const QUICK_SETTING_REJECTION_EN = 'rejection_en';

export const REJECTION_ZH_SYSTEM_TEXT = "你是面壁小钢炮。在听到第一个和你语音对话的人开口 0.9s 的时候，你触发一段思考内容，将这个人在你的思考中登记为用户，此后只回复用户。";
export const REJECTION_EN_SYSTEM_TEXT = "You are Mini, a voice assistant developed by ModelBest. When the first person who talks to you by voice has spoken for 0.9 seconds, start a think content, register this person as the user in your thinking, and from then on respond only to the user.";

// 中文 F224 训练侧素材 B（sha256 b5b40ca8879bf167…）/ 英文 F409（sha256 8fa5df0d3dada485…）。
export const REJECTION_ZH_REF_AUDIO_PATH = "/user/sunweiyue/dataset/o5_sdk_training_data/rejection/agent_duplex_rejection_0929_min2k_max8k_zh/media/system_reference/ref_minicpm_signature_fix.wav";
export const REJECTION_EN_REF_AUDIO_PATH = "/user/sunweiyue/dataset/o5_sdk_training_data/rejection/agent_duplex_rejection_0929_min1k_max8k_en/media/system_reference/F409.wav";

const REJECTION_RUNTIME = Object.freeze({
  nonSpokenScheduling: 'quality',
  nonSpokenBudgetWhileListening: 45,
  nonSpokenBudgetWhileSpeaking: 25,
});

export const REJECTION_PRESET_OPTIONS = [
  { id: QUICK_SETTING_REJECTION_ZH, label: '拒识 · 中文（F224）' },
  { id: QUICK_SETTING_REJECTION_EN, label: 'Rejection · English (F409)' },
];

export function buildRejectionQuickSettings() {
  return new Map([
    [QUICK_SETTING_REJECTION_ZH,
      preset(SYSTEM_PREFIX_ZH_THINK, REJECTION_ZH_REF_AUDIO_PATH, REJECTION_ZH_SYSTEM_TEXT)],
    [QUICK_SETTING_REJECTION_EN,
      preset(SYSTEM_PREFIX_EN_GENERAL, REJECTION_EN_REF_AUDIO_PATH, REJECTION_EN_SYSTEM_TEXT)],
  ]);
}

function preset(prefixText, audioPath, text) {
  return {
    segments: [
      { kind: 'text', text: prefixText },
      { kind: 'audio', audio: { source: 'path', file_path: audioPath } },
      { kind: 'text', text },
    ],
    ttsPromptAudioPath: audioPath,
    runtime: { ...REJECTION_RUNTIME },
  };
}
