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
export function endpointBase(value: string) {
  if (!value.trim()) throw new Error('请先在连接设置中填写模型 API 地址。');
  const url = new URL(value.trim());
  if (!['http:', 'https:', 'ws:', 'wss:'].includes(url.protocol))
    throw new Error('API 地址需要使用 HTTP、HTTPS 或 WebSocket。');
  url.protocol = url.protocol.replace('ws', 'http');
  url.pathname = url.pathname.replace(/\/v1\/(realtime|info)\/?$/, '').replace(/\/$/, '');
  url.search = '';
  url.hash = '';
  return url.toString().replace(/\/$/, '');
}
export function realtimeUrl(base: string, mode: string) {
  const url = new URL(`${endpointBase(base)}/v1/realtime`);
  url.protocol = url.protocol.replace('http', 'ws');
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
  const timer = setTimeout(
    () => request.abort(new Error('读取配置超时，请检查应用服务地址。')),
    20000,
  );
  try {
    const response = await fetch(url, { signal: request.signal, cache: 'no-store' });
    if (response.headers.get('content-type')?.includes('text/html'))
      throw new Error(
        '配置请求返回了网页。请在连接设置填写应用／工具服务地址，或为预览配置对应的服务代理。',
      );
    const body = await response.json().catch(() => null);
    if (!response.ok || !body)
      throw new Error(body?.detail || `服务未就绪（HTTP ${response.status}），请检查地址。`);
    return body;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}
