import assert from 'node:assert/strict'
import test from 'node:test'
import vm from 'node:vm'
import { readFile } from 'node:fs/promises'
import { buildDuplexPreparePayload, DEFAULT_DUPLEX_OPTIONS } from '../src/shared/nofc-options.ts'

const source = (await readFile(new URL('../../../static/duplex/lib/realtime-session.js', import.meta.url), 'utf8'))
  .replace("import { AudioPlayer } from './audio-player.js';", '')
  .replace('export class RealtimeSession', 'globalThis.RealtimeSession = class RealtimeSession')

function fixture() {
  const sent = []
  const sockets = []
  class Socket {
    static OPEN = 1
    constructor(url) {
      this.url = String(url); this.readyState = 1; sockets.push(this)
      setTimeout(() => { this.onopen?.(); setTimeout(() => this.emit({ type: 'session.queue_done' }), 0) }, 0)
    }
    send(raw) {
      const msg = JSON.parse(raw); sent.push(msg)
      if (msg.type === 'session.init') setTimeout(() => this.emit({ type: 'session.created', session_id: 'mobile_contract', session_config: { unit_finalized_metrics: true } }), 0)
    }
    emit(msg) { this.onmessage?.({ data: JSON.stringify(msg) }) }
    close() { this.readyState = 3 }
  }
  class AudioPlayer { init() {} stop() {} }
  const context = vm.createContext({ AudioPlayer, WebSocket: Socket, URL, location: { href: 'https://demo.invalid/mobile/', protocol: 'https:', host: 'demo.invalid' }, window: {}, performance, setTimeout, clearTimeout, requestAnimationFrame: (fn) => fn(), console })
  vm.runInContext(source, context)
  return { Session: context.RealtimeSession, sent, sockets }
}

for (const mode of ['audio', 'video']) test(`mobile ${mode} duplex options and chunks traverse shared realtime client`, async () => {
  const { Session, sent, sockets } = fixture()
  const options = { ...DEFAULT_DUPLEX_OPTIONS, basicWindowHighTokens: 131072, basicWindowLowTokens: 8192 }
  const payload = buildDuplexPreparePayload(options, 'REFERENCE', mode === 'video')
  const session = new Session(mode, { getMaxKvTokens: () => options.maxKvTokens, getPlaybackDelayMs: () => 0, enableReconnect: true, getWsUrl: () => `wss://demo.invalid/v1/realtime?mode=${mode}` })
  const metrics = []; session.onMetrics = (data) => metrics.push(JSON.parse(JSON.stringify(data)))
  await session.start('SYSTEM_PROMPT', payload)
  assert.match(sockets[0].url, new RegExp(`mode=${mode}`))
  assert.match(sockets[0].url, /ordinary_reconnect=1/)
  assert.deepEqual(sent[0], { type: 'session.init', payload: { system_prompt: 'SYSTEM_PROMPT', ...payload } })
  session.sendChunk({ audio_base64: 'AUDIO', ...(mode === 'video' ? { frame_base64_list: ['FRAME'] } : {}) })
  assert.deepEqual(sent[1], { type: 'input.append', input: { audio: 'AUDIO', input_id: 'input_1', ...(mode === 'video' ? { video_frames: ['FRAME'] } : {}) } })
  sockets[0].emit({ type: 'response.metrics', input_id: 'input_1', unit_finalized: true, kv_cache_length: 8192, window_events: [{ count: 1, before: 131090, after: 8192, duration_ms: 235, mode: 'basic' }] })
  assert.equal(metrics.filter((m) => m.type === 'window')[0].windowEventCount, 1)
  assert.equal(metrics.filter((m) => m.type === 'result').at(-1).kvCacheLength, 8192)
  session.stop()
  assert.equal(sent.at(-1).type, 'session.close')
  assert.equal(sockets[0].readyState, 3)
})
