// 把 system 正文里的会话时间刷新成「现在」。
//
// ui_card 训练数据每条样本的正文都写着生成当时的会话时间（北京时间 / UTC / Unix 时间戳三处），
// 每条都不一样；这是会话的事实信息，不是固定 prompt。照搬某一条样本的旧时间，模型会把那天当「今天」，
// 所以建会话前（scene.prepareSystem）把这三处换成当前时间，格式逐字保持训练样本的写法。

const BEIJING_OFFSET_MS = 8 * 3600 * 1000;

function pad(value) {
  return String(value).padStart(2, '0');
}

function formatUtcParts(date) {
  return {
    date: `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}`,
    time: `${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}:${pad(date.getUTCSeconds())}`,
  };
}

/** 一个时刻在训练格式下的三处写法。 */
export function sessionTimeFields(nowMs) {
  const utc = formatUtcParts(new Date(nowMs));
  const beijing = formatUtcParts(new Date(nowMs + BEIJING_OFFSET_MS));
  return {
    local: `服务端本地时间：${beijing.date} ${beijing.time} CST`,
    utc: `UTC 时间：${utc.date}T${utc.time}Z`,
    unix: `Unix 时间戳：${Math.floor(nowMs / 1000)}`,
  };
}

const PATTERNS = {
  local: /服务端本地时间：\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} CST/g,
  utc: /UTC 时间：\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z/g,
  unix: /Unix 时间戳：\d+/g,
};

/** 刷新一段文本里的会话时间；没有会话时间块的文本原样返回。 */
export function restampSessionTime(text, nowMs = Date.now()) {
  if (typeof text !== 'string') return text;
  const fields = sessionTimeFields(nowMs);
  let out = text;
  for (const [key, pattern] of Object.entries(PATTERNS)) out = out.replace(pattern, fields[key]);
  return out;
}

/** 对 system segments 的每个 text 段刷新会话时间，返回新数组（不改入参）。 */
export function restampSegments(segments, nowMs = Date.now()) {
  return (segments || []).map((segment) => (
    segment?.kind === 'text' ? { ...segment, text: restampSessionTime(segment.text, nowMs) } : segment
  ));
}
