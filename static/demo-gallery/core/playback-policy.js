// 播放缓冲策略（纯函数，无 DOM）。
//
// 首包缓冲：SPEAK turn 开始播之前先攒多少音频，用来吸收单 Unit 处理时间的抖动。
// 600ms 来自实测回放：服务端说话 Unit p90 ≈1.4s > 1s 实时预算，200ms 吸收不住。
// 页面 Runtime 面板可现调；HTML 里 input 的默认值必须与此常量一致（有测试对拍）。

export const DEFAULT_PLAYBACK_DELAY_MS = 600;

/**
 * 把输入框的原始字符串解析成首包缓冲毫秒数。
 *
 * @param {unknown} raw 用户输入（可能是空串、非数字、负数）
 * @param {number} [fallback] 非法输入时的回退值
 * @returns {number} 非负整数毫秒；非法输入一律回退，绝不返回 NaN（NaN 会让 setTimeout 立即触发）
 */
export function resolvePlaybackDelayMs(raw, fallback = DEFAULT_PLAYBACK_DELAY_MS) {
  const parsed = Number.parseInt(String(raw ?? ''), 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}
