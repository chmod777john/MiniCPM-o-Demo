// API 端点：Demo 与 API 是两个服务，页面上选一个端点，之后只通过协议访问它。
//
// 端点地址的规范形态是 https://<host>:<port>/e/<端点名>；也接受直接粘贴的 realtime 地址
// （wss://…/e/<端点名>/v1/realtime?mode=audio）或 info 地址，统一规范化。

/**
 * @param {string} raw 用户填写 / 配置里的端点地址。
 * @returns {{base: string, info: string, realtime: string}}
 */
export function normalizeEndpoint(raw) {
  let text = String(raw || '').trim();
  if (!text) throw new Error('请填写 API 端点地址');
  if (!/^(https?|wss?):\/\//.test(text)) throw new Error('端点地址要以 https:// 开头（或 wss://）');
  text = text.replace(/[?#].*$/, '').replace(/\/+$/, '');
  text = text.replace(/\/v1\/(realtime|info)$/, '');
  const httpBase = text.replace(/^ws(s?):\/\//, 'http$1://');
  return {
    base: httpBase,
    info: `${httpBase}/v1/info`,
    realtime: `${httpBase.replace(/^http/, 'ws')}/v1/realtime?mode=audio`,
  };
}

function firstValue(value) {
  return Array.isArray(value) ? value[0] : value;
}

function budgetText(value) {
  const first = firstValue(value);
  if (first === undefined || first === null) return '';
  return typeof first === 'object' ? JSON.stringify(first) : String(first);
}

/** 从 <端点>/v1/info 提取页面要展示的内容（模型口径由部署决定，Demo 只展示不发送）。 */
export function summarizeEndpointInfo(info) {
  const model = info?.models?.[0]?.model || null;
  const policy = model?.unit_policy || {};
  const workers = Array.isArray(info?.workers) ? info.workers.length : 0;
  let statusText = '';
  if (info) {
    statusText = `${workers} 个 Worker · 可用 ${info.available_workers ?? 0} · 排队 ${info.queue_length ?? 0}`;
    if ((info.models || []).length > 1) statusText += ' · ⚠️ 该端点下有多个不同模型，以 session.created 为准';
    if (!workers) statusText += ' · 当前没有 Worker，开始会话会一直排队';
  }
  return {
    model,
    profileId: model?.profile_id || '',
    ttsPad: model?.tts_pad_protocol || '',
    scheduling: model?.non_spoken_scheduling || '',
    listening: budgetText(policy.non_spoken_budgets_while_listening),
    speaking: budgetText(policy.non_spoken_budgets_while_speaking),
    unitSec: policy.unit_sec ?? null,
    statusText,
  };
}
