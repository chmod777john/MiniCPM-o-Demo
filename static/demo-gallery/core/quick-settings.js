// Quick setting（快捷设置）的通用部分。
//
// 一个 quick setting 是**训练分布的原子投影**：选一次就必须同时切换
// ordered system segments + TTS prompt audio + scheduling + budget + tts_pad 口径。
// 禁止让使用者组合出「prompt 属于 A、budget 属于 B」的半切换状态。
// 具体有哪些 preset 由各 demo 自己定义，这里只放跨 demo 复用的工具。

/** 下拉框里「自定义」那一项的 id；任何 demo 的 preset 都不得占用它。 */
export const QUICK_SETTING_CUSTOM = 'custom';

/** 深拷贝一份 preset，避免页面编辑污染 demo 的模板常量。 */
export function cloneQuickSetting(setting) {
  return {
    segments: cloneSegments(setting?.segments || []),
    ttsPromptAudioPath: setting?.ttsPromptAudioPath || '',
    runtime: cloneRuntime(setting?.runtime),
  };
}

export function cloneSegments(segments) {
  return JSON.parse(JSON.stringify(segments));
}

export function cloneRuntime(runtime = {}) {
  return {
    nonSpokenScheduling: runtime.nonSpokenScheduling,
    nonSpokenBudgetWhileListening: runtime.nonSpokenBudgetWhileListening,
    nonSpokenBudgetWhileSpeaking: runtime.nonSpokenBudgetWhileSpeaking,
    ttsPadProtocol: runtime.ttsPadProtocol || 'legacy',
  };
}

export function firstAudioPath(segments) {
  const audio = segments.find((segment) => segment.kind === 'audio');
  return audio?.audio?.file_path || '';
}

/** 在 case 原始 segments 前面拼一段 prefix 文本，其余原样保留。 */
export function buildPrependedSetting(
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
    ttsPromptAudioPath: defaultTtsPromptAudioPath || firstAudioPath(sourceSegments),
    runtime: cloneRuntime(runtime),
  };
}
