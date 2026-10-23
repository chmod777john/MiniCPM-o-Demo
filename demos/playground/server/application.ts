import { readFile } from 'node:fs/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

export const apiPrefix = '/api/demos/playground';
const defaultCatalog = fileURLToPath(new URL('./cases.json', import.meta.url));
const defaultAudioRoot = fileURLToPath(new URL('../../../assets/ref_audio/', import.meta.url));
const audioFiles: Record<string, string> = {
  'signature-zh': 'ref_minicpm_signature.wav',
  'voice-en': 'ref_en_dlc_1.wav',
};

type Environment = Record<string, string | undefined>;

/** Application configuration only; this service does not load models or proxy inference. */
export function applicationHandler(env: Environment = process.env) {
  const catalogPath = env.PLAYGROUND_CASES_FILE || defaultCatalog;
  const audioRoot = env.PLAYGROUND_AUDIO_ROOT || defaultAudioRoot;
  async function catalog() {
    const data = JSON.parse(await readFile(catalogPath, 'utf8'));
    if (data.schema_version !== 1 || !data.cases) throw new Error('Invalid application catalog');
    return data;
  }
  return async function handle(req: IncomingMessage, res: ServerResponse): Promise<boolean> {
    const path = new URL(req.url || '/', 'http://localhost').pathname;
    if (path !== apiPrefix && !path.startsWith(`${apiPrefix}/`)) return false;
    const json = (status: number, data: unknown) => {
      res.writeHead(status, {
        'Content-Type': 'application/json; charset=utf-8',
        'Cache-Control': 'no-store',
      });
      res.end(JSON.stringify(data));
    };
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.setHeader('Allow', 'GET, HEAD');
      json(405, { detail: '此接口只提供应用配置。' });
      return true;
    }
    try {
      const audio = path.match(/^\/api\/demos\/playground\/audio\/([a-z0-9-]+)$/);
      if (audio && audioFiles[audio[1]]) {
        const data = await readFile(resolve(audioRoot, audioFiles[audio[1]]));
        res.writeHead(200, {
          'Content-Type': 'audio/wav',
          'Content-Length': data.length,
          'Cache-Control': 'no-cache',
        });
        res.end(req.method === 'HEAD' ? undefined : data);
        return true;
      }
      if (path === `${apiPrefix}/health`) {
        await catalog();
        json(200, { status: 'ok', service: 'playground-application' });
        return true;
      }
      const match = path.match(/^\/api\/demos\/playground\/cases\/([a-z-]+)\/defaults$/);
      const data = await catalog();
      const config = match && Object.hasOwn(data.cases, match[1]) ? data.cases[match[1]] : null;
      if (!config) {
        json(404, { detail: '没有找到对应的体验配置或参考素材。' });
        return true;
      }
      json(200, {
        schema_version: data.schema_version,
        case_id: match![1],
        ...config,
        defaults: Object.fromEntries(
          Object.entries(config.settings_schema).map(([key, spec]: [string, any]) => [
            key,
            spec.default,
          ]),
        ),
        audio_assets: Object.fromEntries(
          Object.keys(audioFiles).map((id) => [
            id,
            { url: `${apiPrefix}/audio/${id}`, name: audioFiles[id] },
          ]),
        ),
      });
    } catch (error) {
      console.error('[playground application]', error instanceof Error ? error.message : error);
      json(503, { detail: 'Playground 应用配置或参考音频不可用，请检查应用服务配置。' });
    }
    return true;
  };
}
