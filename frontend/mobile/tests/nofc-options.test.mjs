import assert from 'node:assert/strict'
import test from 'node:test'
import { readFile } from 'node:fs/promises'
import {
  buildChatPayload, buildDuplexPreparePayload, DEFAULT_DUPLEX_OPTIONS,
  applyFrontendDefaults, validateDuplexOptions,
} from '../src/shared/nofc-options.ts'

const chatOptions = { maxNewTokens: 256, ttsEnabled: true, refAudio: 'PCM_FLOAT32', hdImages: false, enableThinking: false }

test('simplex keeps complete ordered history for both output modes', () => {
  const messages = [{ role: 'system', content: 'system' }, { role: 'user', content: 'first' }, { role: 'assistant', content: 'answer' }, { role: 'user', content: 'followup' }]
  for (const streaming of [false, true]) {
    const body = buildChatPayload(messages, streaming, chatOptions)
    assert.equal(body.messages, messages)
    assert.equal(body.streaming, streaming)
    assert.deepEqual(body.generation, { max_new_tokens: 256, do_sample: true })
    assert.equal(body.tts.ref_audio_data, 'PCM_FLOAT32')
    assert.equal(body.tts.mode, 'audio_assistant')
    assert.equal(body.enable_thinking, false)
    assert.equal(body.omni_mode, false)
  }
})

test('simplex HD images use two slices and video uses omni TTS and one slice', () => {
  const image = [{ role: 'user', content: [{ type: 'image', data: 'IMAGE' }] }]
  assert.equal(buildChatPayload(image, true, { ...chatOptions, hdImages: true }).image.max_slice_nums, 2)
  const video = [...image, { role: 'user', content: [{ type: 'video', data: 'VIDEO' }, { type: 'audio', data: 'AUDIO' }] }]
  const body = buildChatPayload(video, true, { ...chatOptions, hdImages: true })
  assert.equal(body.omni_mode, true)
  assert.equal(body.image.max_slice_nums, 1)
  assert.equal(body.tts.mode, 'omni')
  assert.equal(body.messages, video)
})

test('disabled TTS does not forward a voice or template', () => {
  const body = buildChatPayload([], false, { ...chatOptions, ttsEnabled: false })
  assert.deepEqual(body.tts, { enabled: false })
  assert.equal(body.use_tts_template, false)
})

test('duplex defaults equal desktop policy and max KV is a client stop limit', () => {
  const body = buildDuplexPreparePayload(DEFAULT_DUPLEX_OPTIONS, 'REF', false)
  assert.deepEqual(body, { config: {
    sliding_window_mode: 'basic', basic_window_high_tokens: 8000, basic_window_low_tokens: 6000,
    context_max_units: 250, text_sink: 512, text_sliding_window: 10240, ls_decision_mode: 'binary', speak_threshold: 0.3,
  }, use_tts: true, ref_audio_base64: 'REF', tts_ref_audio_base64: 'REF' })
  assert.equal(DEFAULT_DUPLEX_OPTIONS.maxKvTokens, 262144)
  assert.equal('max_kv_tokens' in body.config, false)
})

test('service window and playback defaults override fallback values', () => {
  const options = applyFrontendDefaults(DEFAULT_DUPLEX_OPTIONS, { playback_delay_ms: 0, sliding_window_mode: 'off', basic_window_high_tokens: 131072, basic_window_low_tokens: 8192 })
  assert.equal(options.playbackDelayMs, 0)
  assert.equal(options.slidingWindowMode, 'off')
  assert.equal(options.basicWindowHighTokens, 131072)
  assert.equal(options.basicWindowLowTokens, 8192)
  assert.equal(options.maxKvTokens, 262144)
})

test('invalid basic threshold fails before starting devices/session', () => {
  assert.throws(() => validateDuplexOptions({ ...DEFAULT_DUPLEX_OPTIONS, basicWindowHighTokens: 6000 }), /threshold/)
  assert.doesNotThrow(() => validateDuplexOptions({ ...DEFAULT_DUPLEX_OPTIONS, slidingWindowMode: 'off', basicWindowHighTokens: 6000 }))
})

test('duplex HD fixed/strategy/off and TTS switches map to documented fields', () => {
  for (const visionHDMode of ['fixed', 'strategy', 'off']) {
    const body = buildDuplexPreparePayload({ ...DEFAULT_DUPLEX_OPTIONS, visionHDMode, useTts: false }, null, true)
    assert.equal(body.max_slice_nums, visionHDMode === 'fixed' ? 2 : 1)
    assert.equal(body.config.strategy_hd, visionHDMode === 'strategy')
    assert.equal(body.config.strategy_hd_max_slice_nums, 4)
    assert.equal(body.use_tts, false)
    assert.equal('ref_audio_base64' in body, false)
    assert.equal('deferred_finalize' in body, false)
  }
})

test('mobile HTML preserves every desktop element ID required by shared runtime', async () => {
  const desktop = await readFile(new URL('../../../static/omni/omni.html', import.meta.url), 'utf8')
  const mobile = await readFile(new URL('../../../static/mobile-omni/index.html', import.meta.url), 'utf8')
  const ids = (html) => [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]).sort()
  assert.deepEqual(ids(mobile), ids(desktop))
  assert.match(mobile, /body class="video-fullscreen mobile-omni"/)
  assert.match(mobile, /settings-widget.js/)
  assert.match(mobile, /mobile-bridge.js/)
})

test('all simplex output modes use the current chat WS endpoint', async () => {
  const app = await readFile(new URL('../src/App.tsx', import.meta.url), 'utf8')
  assert.doesNotMatch(app, /fetch\('\/api\/chat'/)
  assert.match(app, /\/v1\/realtime\?mode=chat/)
  assert.match(app, /buildChatRequestBody\(nextMessages, systemMessage, streaming\)/)
  assert.match(app, /type: 'input.append', input: requestPayload/)
})

test('independent TTS reference does not replace the LLM system voice', () => {
  const body = buildDuplexPreparePayload({ ...DEFAULT_DUPLEX_OPTIONS, ttsRefAudioBase64: 'TTS_VOICE' }, 'LLM_VOICE', false)
  assert.equal(body.ref_audio_base64, 'LLM_VOICE')
  assert.equal(body.tts_ref_audio_base64, 'TTS_VOICE')
})
