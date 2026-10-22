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
  return url.toString();
}
export function applicationUrl(base: string, path: string) {
  if (base && !/^https?:\/\//.test(base))
    throw new Error('工具服务地址需要以 http:// 或 https:// 开头。');
  return `${base.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
}
export async function readJson(url: string, signal?: AbortSignal) {
  const response = await fetch(url, { signal });
  const body = await response.json().catch(() => null);
  if (!response.ok || !body)
    throw new Error(body?.detail || `服务未就绪（HTTP ${response.status}），请检查地址。`);
  return body;
}
