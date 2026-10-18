// uicard 的 preset：每个对齐一份 ui_card 训练数据（以及用它训的端点）。
// system 取自该份数据的一条真实样本（后端 defaults 原样给出 [参考音频, per-case 正文] 与 tools），
// 这里在最前拼上训练时注入的统一前缀 #2（中文 no-think + 填充句；三个 data-mix 已逐字核对）。

import { buildPrependedSetting } from '../../../static/demo-gallery/core/quick-settings.js';
import { SYSTEM_PREFIX_ZH_NO_THINK } from '../../../static/demo-gallery/core/system-prefixes.js';

export const PRESET_1016 = 'uicard_1016_r002';
export const PRESET_1004 = 'uicard_1004_r011';

export const PRESET_OPTIONS = [
  { id: PRESET_1016, label: 'ui_card 1016·recipe_002（6 工具含 tavily）· 对齐 f5t16 / f5t17 端点' },
  { id: PRESET_1004, label: 'ui_card 1004·recipe_011（5 工具）· 对齐 merge1006 端点（0930 合版）' },
];

function presetSource(defaults, presetId) {
  const preset = defaults?.uicard_presets?.[presetId];
  if (!preset || preset.error || !preset.system) return null;
  return preset;
}

export function buildUiCardQuickSettings(defaults) {
  const settings = new Map();
  for (const { id } of PRESET_OPTIONS) {
    const preset = presetSource(defaults, id);
    if (!preset) continue;
    settings.set(id, buildPrependedSetting(
      SYSTEM_PREFIX_ZH_NO_THINK,
      preset.system.segments,
      preset.tts_prompt_audio?.file_path || '',
    ));
  }
  return settings;
}

/** 当前 preset 的工具：顺序与描述都是训练口径，原样取自样本。 */
export function presetTools(defaults, presetId) {
  const preset = presetSource(defaults, presetId) || presetSource(defaults, PRESET_1016);
  return JSON.parse(JSON.stringify(preset?.system?.tools || []));
}

export function presetDataId(defaults, presetId) {
  return presetSource(defaults, presetId)?.data_id || '';
}

export function unavailablePresets(defaults) {
  return PRESET_OPTIONS
    .map(({ id }) => [id, defaults?.uicard_presets?.[id]])
    .filter(([, preset]) => !preset || preset.error)
    .map(([id, preset]) => ({ id, error: preset?.error || 'defaults 里没有该 preset' }));
}
