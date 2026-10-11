// Standalone Demo 的 preset：system 取自训练过 standalone 的 sub_agent 数据（后端 defaults 原样给出
// [参考音频, per-case 正文] 与 tools），这里只按数据类型在最前拼上训练时注入的统一前缀。

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
  { id: PRESET_ZH_NOTHINK, label: 'sub_agent · 中文 no-think（训练里 standalone = 子任务进度）' },
  { id: PRESET_ZH_THINK, label: 'sub_agent · 中文 think（训练里 standalone = 文件上传通知）' },
  { id: PRESET_EN_NOTHINK, label: 'sub_agent · English no-think（progress updates）' },
];

// 与 F9-T3 data-mix 的源一一对应：zh no-think #2 / zh think #1 / en no-think #6。
const PREFIX_BY_PRESET = {
  [PRESET_ZH_NOTHINK]: SYSTEM_PREFIX_ZH_NO_THINK,
  [PRESET_ZH_THINK]: SYSTEM_PREFIX_ZH_THINK,
  [PRESET_EN_NOTHINK]: SYSTEM_PREFIX_EN_NO_THINK,
};

/** 每类数据里 standalone 的训练形态，作为输入框的一键示例（只填进输入框，不自动发送）。 */
export const EXAMPLES_BY_PRESET = {
  [PRESET_ZH_NOTHINK]: [
    '已找到两篇相关文献，正在核对其中的实验数据',
    '价格区间已经查到了，还在确认是否含运费，马上汇总',
  ],
  [PRESET_ZH_THINK]: ['[FILE_UPLOADED] report.pdf'],
  [PRESET_EN_NOTHINK]: ['Found the official documentation; now checking the version differences.'],
};

function presetSource(defaults, presetId) {
  const preset = defaults?.standalone_presets?.[presetId];
  if (!preset || preset.error || !preset.system) return null;
  return preset;
}

/** 从 /defaults 构造 quick setting；某个 preset 的数据不可读时跳过它（页面上会提示）。 */
export function buildStandaloneQuickSettings(defaults) {
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

/** 不可用的 preset 及原因，供场景提示。 */
export function unavailablePresets(defaults) {
  return PRESET_OPTIONS
    .map(({ id }) => [id, defaults?.standalone_presets?.[id]])
    .filter(([, preset]) => !preset || preset.error)
    .map(([id, preset]) => ({ id, error: preset?.error || 'defaults 里没有该 preset' }));
}
