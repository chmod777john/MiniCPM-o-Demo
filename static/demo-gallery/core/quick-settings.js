// Quick setting（快捷设置）的通用部分。
//
// 一个 quick setting = 一份训练分布的输入形态：有序 system segments + TTS prompt 音频，选一次同时切换。
// 预算、调度方式、tts_pad 口径不在这里：它们是 checkpoint 的训练事实，由所选 API 端点上部署的
// 模型决定（会话前从 <端点>/v1/info 查到后展示），Demo 不发送也不覆盖。
// 具体有哪些 preset 由各 demo 自己定义，这里只放跨 demo 复用的工具。
// 音频一律用 Demo 服务所在机器上的绝对路径描述；建会话前由 shell 读成 base64（协议 v4）。

/** 下拉框里「自定义」那一项的 id；任何 demo 的 preset 都不得占用它。 */
export const QUICK_SETTING_CUSTOM = 'custom';

/** 深拷贝一份 preset，避免页面编辑污染 demo 的模板常量。 */
export function cloneQuickSetting(setting) {
  return {
    segments: cloneSegments(setting?.segments || []),
    ttsPromptAudioPath: setting?.ttsPromptAudioPath || '',
  };
}

export function cloneSegments(segments) {
  return JSON.parse(JSON.stringify(segments));
}

export function firstAudioPath(segments) {
  const audio = segments.find((segment) => segment.kind === 'audio');
  return audio?.audio?.file_path || '';
}

/** 在 case 原始 segments 前面拼一段 prefix 文本，其余原样保留。 */
export function buildPrependedSetting(prefixText, sourceSegments, defaultTtsPromptAudioPath) {
  return {
    segments: [
      { kind: 'text', text: prefixText },
      ...cloneSegments(sourceSegments),
    ],
    ttsPromptAudioPath: defaultTtsPromptAudioPath || firstAudioPath(sourceSegments),
  };
}
