const randomId = () =>
  crypto.randomUUID?.() || `${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
const pageSessionId = `p_${randomId()}`;
function clientIdentity(url: URL) {
  let id = '';
  try {
    id = localStorage.getItem('minicpmo_client_id') || `c_${randomId()}`;
    localStorage.setItem('minicpmo_client_id', id);
  } catch {
    id = `c_tmp_${pageSessionId}`;
  }
  url.searchParams.set('client_id', id);
  url.searchParams.set('page_session_id', pageSessionId);
  url.searchParams.set('page_route', location.pathname + location.hash);
  url.searchParams.set('client_surface', 'desktop');
}
export function apiBase(value: string) {
  if (!value.trim()) throw new Error('请填写 API 地址。');
  const url = new URL(value.trim());
  if (!['http:', 'https:'].includes(url.protocol))
    throw new Error('API 地址需要使用 HTTP 或 HTTPS。');
  if (url.username || url.password || url.search || url.hash)
    throw new Error('请填写服务的 API Base，不包含凭证、查询参数或页面锚点。');
  if (/\/e(?:\/|$)/.test(url.pathname) || /\/v1\/.+/.test(url.pathname))
    throw new Error('请填写 API Base，并从模型列表选择模型。');
  url.pathname = url.pathname.replace(/\/+$/, '').replace(/\/v1$/, '');
  return url.toString().replace(/\/$/, '');
}
export function realtimeUrl(base: string, modelId: string, mode: string) {
  if (!modelId) throw new Error('请选择模型。');
  const url = new URL(`${apiBase(base)}/v1/realtime`);
  url.protocol = url.protocol.replace('http', 'ws');
  url.searchParams.set('model', modelId);
  url.searchParams.set('mode', mode);
  clientIdentity(url);
  return url.toString();
}
export function applicationUrl(base: string, path: string) {
  if (base && !/^https?:\/\//.test(base))
    throw new Error('工具服务地址需要以 http:// 或 https:// 开头。');
  return `${base.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
}
export async function readJson(url: string, signal?: AbortSignal) {
  const request = new AbortController();
  const cancel = () => request.abort(signal?.reason);
  if (signal?.aborted) cancel();
  else signal?.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(() => request.abort(new Error('服务响应超时，请稍后重试。')), 20000);
  try {
    const response = await fetch(url, { signal: request.signal, cache: 'no-store' });
    if (response.headers.get('content-type')?.includes('text/html'))
      throw new Error('服务返回了网页，暂时无法读取配置，请联系部署维护者检查服务。');
    const body = await response.json().catch(() => null);
    if (!response.ok || !body)
      throw new Error(body?.detail || `服务未就绪（HTTP ${response.status}），请稍后重试。`);
    return body;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}
