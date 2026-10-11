// subagent Demo 的 preset：system 取自 sub_agent 训练数据（后端 defaults 原样给出 [参考音频, per-case 正文]
// 与 tools），这里只按数据类型在最前拼上训练时注入的统一前缀。

import { buildPrependedSetting } from '../../../static/demo-gallery/core/quick-settings.js';
import {
  SYSTEM_PREFIX_EN_NO_THINK,
  SYSTEM_PREFIX_ZH_NO_THINK,
  SYSTEM_PREFIX_ZH_THINK,
} from '../../../static/demo-gallery/core/system-prefixes.js';

export const PRESET_ZH_NOTHINK = 'subagent_zh_nothink';
export const PRESET_ZH_THINK = 'subagent_zh_think';
export const PRESET_EN_NOTHINK = 'subagent_en_nothink';

export const PRESET_OPTIONS = [
  { id: PRESET_ZH_NOTHINK, label: 'sub_agent · 中文 no-think（前缀 #2）' },
  { id: PRESET_ZH_THINK, label: 'sub_agent · 中文 think（前缀 #1）' },
  { id: PRESET_EN_NOTHINK, label: 'sub_agent · English no-think (prefix #6)' },
];

// 与 F9-T3 data-mix 的源一一对应。
const PREFIX_BY_PRESET = {
  [PRESET_ZH_NOTHINK]: SYSTEM_PREFIX_ZH_NO_THINK,
  [PRESET_ZH_THINK]: SYSTEM_PREFIX_ZH_THINK,
  [PRESET_EN_NOTHINK]: SYSTEM_PREFIX_EN_NO_THINK,
};

function presetSource(defaults, presetId) {
  const preset = defaults?.subagent_presets?.[presetId];
  if (!preset || preset.error || !preset.system) return null;
  return preset;
}

export function presetLanguage(defaults, presetId) {
  return presetSource(defaults, presetId)?.language || (String(presetId).includes('_en_') ? 'en' : 'zh');
}

export function buildSubagentQuickSettings(defaults) {
  const runtime = {
    nonSpokenScheduling: defaults?.non_spoken_scheduling,
    nonSpokenBudgetWhileListening: defaults?.non_spoken_budget_while_listening,
    nonSpokenBudgetWhileSpeaking: defaults?.non_spoken_budget_while_speaking,
  };
  const settings = new Map();
  for (const { id } of PRESET_OPTIONS) {
    const preset = presetSource(defaults, id);
    if (!preset) continue;
    settings.set(id, buildPrependedSetting(
      PREFIX_BY_PRESET[id],
      preset.system.segments,
      preset.tts_prompt_audio?.file_path || '',
      runtime,
    ));
  }
  return settings;
}

/** 当前 preset 的工具（顺序是训练口径的一部分，不能混用别的 preset 的）。 */
export function presetTools(defaults, presetId) {
  const preset = presetSource(defaults, presetId) || presetSource(defaults, PRESET_ZH_NOTHINK);
  return JSON.parse(JSON.stringify(preset?.system?.tools || []));
}

export function unavailablePresets(defaults) {
  return PRESET_OPTIONS
    .map(({ id }) => [id, defaults?.subagent_presets?.[id]])
    .filter(([, preset]) => !preset || preset.error)
    .map(([id, preset]) => ({ id, error: preset?.error || 'defaults 里没有该 preset' }));
}
