// API 端点：Demo 与 API 是两个服务，页面上选一个端点，之后只通过协议访问它。
//
// 端点地址的规范形态是 https://<host>:<port>/e/<端点名>；也接受直接粘贴的 realtime 地址
// （wss://…/e/<端点名>/v1/realtime?mode=audio）或 info 地址，统一规范化。

/** 端点会话的两种输入模式(协议 v5):audio 只送音频,omni 每秒再带一帧画面。 */
export const INPUT_MODES = ['audio', 'omni'];

/** 某个输入模式的 realtime 地址。 */
export function realtimeUrl(endpoint, mode) {
  if (!INPUT_MODES.includes(mode)) throw new Error(`未知输入模式:${mode}`);
  return endpoint.realtime.replace(/\?mode=[a-z]+$/, `?mode=${mode}`);
}

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

/**
 * 规范化 API 服务地址（https://host:port）。也接受粘贴的端点 / realtime / info / overview 地址：
 * 取出服务地址，若带端点名一并返回规范化的端点地址。
 * @returns {{service: string, endpoint: string|null}}
 */
export function parseApiService(raw) {
  let text = String(raw || '').trim();
  if (!text) throw new Error('请填写 API 地址（https://<host>:<port>）');
  if (!/^(https?|wss?):\/\//.test(text)) throw new Error('API 地址要以 https:// 开头');
  text = text.replace(/[?#].*$/, '').replace(/^ws(s?):\/\//, 'http$1://').replace(/\/+$/, '');
  const match = text.match(/^(.*?)\/e\/([^/]+)/);
  if (match) return { service: match[1], endpoint: normalizeEndpoint(`${match[1]}/e/${match[2]}`).base };
  return { service: text.replace(/\/(overview|control|api-docs)(\/.*)?$/, '').replace(/\/v1(\/.*)?$/, ''), endpoint: null };
}

/**
 * 把 API 服务公开状态（<服务地址>/overview/api/state）转成端点选项。
 * @param {object} state overview 状态：{public_base, endpoints: [{name, model, available, queue_length, workers}]}
 * @param {string} serviceBase 发现用的服务地址；状态里没有 public_base 时用它拼端点地址。
 * @returns {{name: string, url: string, label: string, available: number}[]}
 */
export function endpointsFromOverview(state, serviceBase) {
  const base = String(state?.public_base || serviceBase || '').replace(/\/+$/, '');
  return (Array.isArray(state?.endpoints) ? state.endpoints : []).filter((e) => e?.name).map((e) => {
    const workers = Array.isArray(e.workers) ? e.workers.length : 0;
    const available = Number(e.available) || 0;
    const load = workers ? `可用 ${available}/${workers}${e.queue_length ? ` · 排队 ${e.queue_length}` : ''}` : '没有 Worker';
    return { name: e.name, url: `${base}/e/${e.name}`, label: `${e.name} · ${e.model || '未部署模型'} · ${load}`, available };
  });
}

/** 合并自动发现的端点与 Demo 配置里写死的端点（按规范化地址去重，发现到的在前、信息更全）。 */
export function mergeEndpointOptions(discovered, configured) {
  const seen = new Set();
  const out = [];
  for (const option of [...discovered, ...configured.map((c) => ({ ...c, label: c.label || c.name, available: null }))]) {
    let key;
    try { key = normalizeEndpoint(option.url).base; } catch { continue; }
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ ...option, url: key });
  }
  return out;
}

function firstValue(value) {
  return Array.isArray(value) ? value[0] : value;
}

function budgetText(value) {
  const first = firstValue(value);
  if (first === undefined || first === null) return '';
  return typeof first === 'object' ? JSON.stringify(first) : String(first);
}

/**
 * 从 <端点>/v1/info 提取页面要展示的内容。tts_pad / 预算 / 调度取 model.session_defaults：
 * Demo 不发这些参数，所以端点的默认值就是本次会话会用的值（建会话后以 session.created.session_config 为准）。
 */
export function summarizeEndpointInfo(info) {
  const model = info?.models?.[0]?.model || null;
  return { ...summarizeSessionConfig(model?.session_defaults), model, profileId: model?.profile_id || '', statusText: endpointStatus(info) };
}

/** tts_pad / Unit 策略 / 调度的展示值；输入是 model.session_defaults 或 session.created.session_config。 */
export function summarizeSessionConfig(config) {
  const policy = config?.unit_policy || {};
  return {
    ttsPad: config?.tts_pad_protocol || '',
    scheduling: config?.non_spoken_scheduling || '',
    listening: budgetText(policy.non_spoken_budgets_while_listening),
    speaking: budgetText(policy.non_spoken_budgets_while_speaking),
    unitSec: policy.unit_sec ?? null,
  };
}

function endpointStatus(info) {
  const workers = Array.isArray(info?.workers) ? info.workers.length : 0;
  let statusText = '';
  if (info) {
    statusText = `${workers} 个 Worker · 可用 ${info.available_workers ?? 0} · 排队 ${info.queue_length ?? 0}`;
    if ((info.models || []).length > 1) statusText += ' · ⚠️ 该端点下有多个不同模型，以 session.created 为准';
    if (!workers) statusText += ' · 当前没有 Worker，开始会话会一直排队';
  }
  return statusText;
}
