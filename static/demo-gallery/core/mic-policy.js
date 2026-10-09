// 麦克风输入策略（纯函数，无 DOM）。
//
// 训练数据里「开麦到用户第一次开口」最长 5.0s（rejection zh / mvp 各抽 400 条实测，中位 2.2s）。
// 超过这个时长还只有环境底噪，就已经在训练分布之外 —— 实测该 ckpt 会开始乱生成
// （复读同一句、或在 tools=[] 的 preset 里幻觉工具调用）。这里只产出提示文案，不做任何门控：
// 推理侧不补偿模型行为（见模块 Playbook「模型忠实推理」）。

export const MIC_OPENING_SILENCE_LIMIT_MS = 5000;
export const MIC_SPEECH_DB = -35;       // 高于它才算「有人在说话」，低于它是底噪

/**
 * 判断是否该提示「开麦很久了还没人说话」。
 *
 * @param {object} args
 * @param {string} args.micLiveState   麦克风状态机当前值；只有 'live' 才可能提示
 * @param {boolean} args.speechSeen    本次开麦以来是否出现过高于 MIC_SPEECH_DB 的电平
 * @param {number} args.waitedMs       开麦至今毫秒数
 * @param {number} args.peakDb         本次开麦以来的峰值（dBFS）
 * @returns {{visible: boolean, text: string}}
 */
export function openingSilenceWarning({ micLiveState, speechSeen, waitedMs, peakDb }) {
  const visible = micLiveState === 'live'
    && !speechSeen
    && waitedMs > MIC_OPENING_SILENCE_LIMIT_MS;
  if (!visible) return { visible: false, text: '' };
  return {
    visible: true,
    text: `⚠️ 已开麦 ${Math.round(waitedMs / 1000)}s 仍未检测到说话声`
      + `（峰值 ${peakDb.toFixed(0)} dB，需高于 ${MIC_SPEECH_DB} dB）。`
      + `训练数据里开场静音最长 5s，超出后模型输出不可预期（会复读或幻觉工具调用）。`
      + `请现在开口说话，或检查浏览器选的是哪个麦克风输入。`,
  };
}
