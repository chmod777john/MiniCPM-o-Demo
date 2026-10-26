import { createServer, request as httpRequest } from 'node:http';
import type { IncomingHttpHeaders, IncomingMessage, ServerResponse } from 'node:http';
import { request as httpsRequest } from 'node:https';
import { readFile, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { extname, resolve, sep } from 'node:path';
import { applicationHandler } from './application.ts';

const env = process.env;
const host = env.PLAYGROUND_APP_HOST || '0.0.0.0';
const port = Number(env.PORT || env.PLAYGROUND_APP_PORT || 8080);
const dist = fileURLToPath(new URL('../dist/', import.meta.url));
const localApplication = applicationHandler();
const maxUpload = 64 * 1024 * 1024;
const hopHeaders = new Set([
  'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
  'te', 'trailer', 'transfer-encoding', 'upgrade',
]);
const mime: Record<string, string> = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.webp': 'image/webp', '.ico': 'image/x-icon', '.woff2': 'font/woff2',
  '.wav': 'audio/wav', '.wasm': 'application/wasm',
};
function target(value: string | undefined): URL | null {
  if (!value) return null;
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password
      || url.search || url.hash || url.pathname !== '/')
    throw new Error('Application upstream must be an HTTP(S) origin');
  return url;
}
const gallery = target(env.PLAYGROUND_GALLERY_TARGET);
const uicard = target(env.PLAYGROUND_UICARD_TARGET) || gallery;
const sessions = target(env.PLAYGROUND_SESSION_TARGET) || gallery;
const inPath = (path: string, prefix: string) => path === prefix || path.startsWith(prefix + '/');
function upstream(path: string): URL | null | undefined {
  if (inPath(path, '/api/demos/uicard')) return uicard;
  if (inPath(path, '/api/demos') || inPath(path, '/demos') || inPath(path, '/static')) return gallery;
  if (inPath(path, '/api/sessions') || inPath(path, '/s')) return sessions;
  return undefined;
}
function json(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(body));
}
function headers(input: IncomingHttpHeaders) {
  const excluded = new Set([...hopHeaders,
    ...String(input.connection || '').split(',').map((s) => s.trim().toLowerCase())]);
  return Object.fromEntries(Object.entries(input).filter(([name]) => !excluded.has(name.toLowerCase())));
}
function proxy(req: IncomingMessage, res: ServerResponse, remote: URL) {
  if (Number(req.headers['content-length'] || 0) > maxUpload) {
    json(res, 413, { detail: 'Upload exceeds 64 MiB' });
    req.resume();
    return;
  }
  const forwarded = headers(req.headers);
  forwarded.host = remote.host;
  forwarded['x-forwarded-host'] = req.headers.host || '';
  forwarded['x-forwarded-proto'] = req.headers['x-forwarded-proto'] || 'http';
  const send = remote.protocol === 'https:' ? httpsRequest : httpRequest;
  const request = send(remote, {
    method: req.method, path: req.url, headers: forwarded,
    rejectUnauthorized: env.PLAYGROUND_PROXY_TLS_INSECURE !== '1',
  }, (response) => {
    const returned = headers(response.headers);
    const location = response.headers.location;
    if (location?.startsWith(remote.origin + '/')) returned.location = location.slice(remote.origin.length);
    res.writeHead(response.statusCode || 502, returned);
    response.on('error', () => res.destroy());
    response.pipe(res);
  });
  request.setTimeout(600_000, () => request.destroy(new Error('Application upstream timed out')));
  request.on('error', (error) => {
    if (res.destroyed || res.writableEnded) return;
    console.error('[application upstream]', remote.origin, error.message);
    if (res.headersSent) res.destroy();
    else json(res, 502, { detail: '应用工具或会话服务暂时不可用，请稍后重试。' });
  });
  let bytes = 0;
  req.on('data', (chunk: Buffer) => {
    bytes += chunk.length;
    if (bytes > maxUpload) {
      req.unpipe(request);
      if (!res.headersSent) json(res, 413, { detail: 'Upload exceeds 64 MiB' });
      request.destroy();
      req.resume();
    }
  });
  req.on('aborted', () => request.destroy());
  req.on('error', () => request.destroy());
  res.on('close', () => { if (!res.writableEnded) request.destroy(); });
  req.pipe(request);
}
async function handle(req: IncomingMessage, res: ServerResponse) {
  const raw = req.url || '/';
  if (!raw.startsWith('/') || raw.startsWith('//')) {
    json(res, 400, { detail: 'Invalid request path' });
    return;
  }
  const path = new URL(raw, 'http://localhost').pathname;
  if (path === '/health' && (req.method === 'GET' || req.method === 'HEAD')) {
    json(res, 200, { status: 'ok', service: 'o5-playground', inference: 'browser-direct',
      revision: env.SOURCE_COMMIT || 'development' });
    return;
  }
  if (await localApplication(req, res)) return;
  const remote = upstream(path);
  if (remote !== undefined) {
    if (remote) proxy(req, res, remote);
    else json(res, 503, { detail: 'Application upstream is not configured' });
    return;
  }
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    json(res, 405, { detail: 'Method not allowed' });
    return;
  }
  let decoded: string;
  try { decoded = decodeURIComponent(path); }
  catch { json(res, 400, { detail: 'Invalid request path' }); return; }
  const file = resolve(dist, '.' + (decoded === '/' ? '/index.html' : decoded));
  if (!file.startsWith(dist.replace(/\/$/, '') + sep) || decoded.includes('\0')) {
    json(res, 404, { detail: 'Not found' });
    return;
  }
  const info = await stat(file).catch(() => null);
  if (!info?.isFile()) { json(res, 404, { detail: 'Not found' }); return; }
  res.writeHead(200, {
    'Content-Type': mime[extname(file)] || 'application/octet-stream',
    'Content-Length': info.size,
    'Cache-Control': path.startsWith('/assets/') ? 'public, max-age=31536000, immutable' : 'no-cache',
    'X-Content-Type-Options': 'nosniff',
  });
  if (req.method === 'HEAD') res.end();
  else {
    const stream = createReadStream(file);
    stream.on('error', () => res.destroy());
    res.on('close', () => stream.destroy());
    stream.pipe(res);
  }
}
// Fail before listening if the frontend artifact or local preset catalog is missing.
await readFile(resolve(dist, 'index.html'));
await readFile(env.PLAYGROUND_CASES_FILE || fileURLToPath(new URL('./cases.json', import.meta.url)));
const server = createServer((req, res) => {
  void handle(req, res).catch((error) => {
    console.error('[playground]', error.message);
    if (res.headersSent) res.destroy();
    else json(res, 500, { detail: 'Application request failed' });
  });
});
server.on('upgrade', (_req, socket) => socket.end('HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n'));
server.listen(port, host, () => console.log(`Playground application listening on ${host}:${port}`));
for (const signal of ['SIGTERM', 'SIGINT'] as const) {
  process.on(signal, () => {
    server.close(() => process.exit(0));
    setTimeout(() => { server.closeAllConnections(); process.exit(0); }, 15_000).unref();
  });
}
