import assert from 'node:assert/strict'
import test from 'node:test'
import vm from 'node:vm'
import { readFile } from 'node:fs/promises'
import ts from 'typescript'

// Execute the actual App request/stop functions with WebSocket, player, state
// and timers replaced. No copied cancellation implementation or regex checks.
const appSource = await readFile(new URL('../src/App.tsx', import.meta.url), 'utf8')
const ast = ts.createSourceFile('App.tsx', appSource, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
const app = ast.statements.find((node) => ts.isFunctionDeclaration(node) && node.name?.text === 'App')
const selected = ['submitConversationStreaming', 'stopCurrentReply'].map((name) => {
  const fn = app.body.statements.find((node) => ts.isFunctionDeclaration(node) && node.name?.text === name)
  assert.ok(fn, `App function ${name} exists`)
  return fn.getText(ast)
}).join('\n')
const executable = ts.transpileModule(selected, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText

function fixture() {
  const sockets = []; const players = []; const timers = new Map(); let seq = 0
  const ui = { messages: [], pending: [], generating: [], playback: [], persisted: [] }
  class Socket {
    static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3
    constructor(url) { this.url = url; this.readyState = Socket.CONNECTING; this.sent = []; this.closeCount = 0; sockets.push(this) }
    send(raw) { if (this.readyState !== Socket.OPEN) throw new Error('not open'); this.sent.push(JSON.parse(raw)) }
    close() { if (this.readyState === Socket.CLOSED) return; this.closeCount++; this.readyState = Socket.CLOSED; this.onclose?.() }
    emit(msg) { this.onmessage?.({ data: JSON.stringify(msg) }) }
  }
  class Player {
    constructor() { this.disposed = 0; this.chunks = []; players.push(this) }
    pushBase64(chunk) { this.chunks.push(chunk) }
    getMergedFloat32() { return this.chunks.length ? new Float32Array([0.1]) : null }
    dispose() { this.disposed++; return Promise.resolve() }
    markFinished() { this.finished = true }
    disposeAfterDrain(callback) { this.onDrained = callback }
  }
  const context = vm.createContext({
    WebSocket: Socket, StreamingPcmPlayer: Player, window: { location: { protocol: 'https:', host: 'demo.invalid' } },
    settings: { turnTtsEnabled: true }, activeSessionIdRef: { current: 'original' }, streamingPlayerRef: { current: null }, streamingStopRef: { current: null }, streamingWsRef: { current: null },
    i18n: { generating: 'generating', stoppedReply: 'stopped', emptyReply: 'empty', wsClosed: 'wsClosed', wsError: 'wsError', connectFailed: (x) => x, sendFailed: (x) => x, requestFailedDetail: (x) => x },
    buildTurnSystemMessage: () => 'system', createId: (prefix) => `${prefix}-${++seq}`, appendClientIdentity: (url) => url,
    buildChatRequestBody: (messages, _system, streaming) => ({ messages, streaming }), getErrorMessage: (e) => String(e),
    setPendingReply: (value) => ui.pending.push(value), setIsGenerating: (value) => ui.generating.push(value), setMessages: (value) => ui.messages.push(value),
    setIsStreamAudioPlaying: (value) => ui.playback.push(value), setLastSessionId() {},
    persistEntryToSession: (id, messages) => ui.persisted.push({ id, messages }),
    float32ToWavBlobUrl: () => 'blob:partial', float32ToBase64: () => 'PARTIAL_PCM',
    setTimeout: (callback, delay) => { const id = ++seq; timers.set(id, { callback, delay }); return id }, clearTimeout: (id) => timers.delete(id), console,
  })
  vm.runInContext(executable, context)
  const begin = async () => { await context.submitConversationStreaming([{ role: 'user', content: 'hello' }], true); return sockets.at(-1) }
  const establish = (socket) => { socket.readyState = Socket.OPEN; socket.emit({ type: 'session.queue_done' }); socket.emit({ type: 'session.created' }) }
  const runTimer = () => { const [id, timer] = timers.entries().next().value; timers.delete(id); timer.callback(); return timer.delay }
  return { context, sockets, players, timers, ui, begin, establish, runTimer, Socket }
}

test('active stop sends protocol close once, immediately stops UI/audio and ignores late frames', async () => {
  const f = fixture(); const ws = await f.begin(); f.establish(ws)
  ws.emit({ type: 'response.output.delta', kind: 'text', text: 'partial' })
  ws.emit({ type: 'response.output.delta', kind: 'audio', audio: 'SAMPLES' })
  const cancel = f.context.streamingStopRef.current
  cancel(); cancel()
  assert.deepEqual(ws.sent.at(-1), { type: 'session.close', reason: 'client_cancel' })
  assert.equal(ws.sent.filter((msg) => msg.type === 'session.close').length, 1)
  assert.equal(ws.closeCount, 0)
  assert.equal(f.players[0].disposed, 1)
  assert.equal(f.ui.generating.at(-1), false)
  assert.equal(f.ui.pending.at(-1), null)
  assert.equal(f.ui.messages.at(-1).at(-1).text, 'partial')
  assert.equal(f.ui.messages.at(-1).at(-1).interrupted, true)
  assert.equal(f.ui.messages.at(-1).at(-1).audioBase64, 'PARTIAL_PCM')
  const count = f.ui.messages.length; const pendingCount = f.ui.pending.length
  ws.emit({ type: 'response.output.delta', kind: 'text', text: 'late' })
  ws.emit({ type: 'response.done', text: 'late final' })
  assert.equal(f.ui.messages.length, count); assert.equal(f.ui.pending.length, pendingCount)
  ws.emit({ type: 'session.closed', reason: 'client_cancel' })
  assert.equal(ws.closeCount, 1); assert.equal(f.timers.size, 0); assert.equal(f.sockets.length, 1)
})

test('active stop has a bounded fallback without server close acknowledgment', async () => {
  const f = fixture(); const ws = await f.begin(); f.establish(ws)
  f.context.stopCurrentReply()
  assert.equal(f.runTimer(), 1500)
  assert.equal(ws.closeCount, 1); assert.equal(f.timers.size, 0); assert.equal(f.sockets.length, 1)
})

for (const phase of ['connecting', 'queued']) test(`${phase} cancellation closes raw transport and never retries`, async () => {
  const f = fixture(); const ws = await f.begin()
  if (phase === 'queued') ws.readyState = f.Socket.OPEN
  f.context.stopCurrentReply()
  assert.equal(ws.sent.length, 0); assert.equal(ws.closeCount, 1)
  assert.equal(f.ui.generating.at(-1), false); assert.equal(f.ui.messages.at(-1).at(-1).text, 'stopped')
  ws.emit({ type: 'session.created' }); ws.onerror?.()
  assert.equal(ws.sent.length, 0); assert.equal(f.timers.size, 0); assert.equal(f.sockets.length, 1)
})

test('cancel during retry backoff clears retry and finishes locally', async () => {
  const f = fixture(); const ws = await f.begin(); ws.close()
  assert.equal(f.timers.size, 1)
  f.context.stopCurrentReply()
  assert.equal(f.timers.size, 0); assert.equal(f.ui.generating.at(-1), false); assert.equal(f.sockets.length, 1)
})

test('switching sessions saves partial result to original and cannot overwrite new request', async () => {
  const f = fixture(); const oldWs = await f.begin(); f.establish(oldWs)
  oldWs.emit({ type: 'response.output.delta', kind: 'text', text: 'original partial' })
  f.context.activeSessionIdRef.current = 'new'
  f.context.stopCurrentReply()
  assert.equal(f.ui.messages.length, 0); assert.equal(f.ui.persisted[0].id, 'original')
  assert.equal(f.ui.persisted[0].messages.at(-1).text, 'original partial')
  const nextWs = await f.begin(); f.establish(nextWs)
  const currentStop = f.context.streamingStopRef.current
  oldWs.emit({ type: 'response.done', text: 'late original' }); oldWs.emit({ type: 'session.closed' })
  assert.equal(f.context.streamingStopRef.current, currentStop)
  assert.equal(f.context.streamingWsRef.current, nextWs)
  assert.equal(f.ui.generating.at(-1), true)
})

test('unmount cancels gracefully without publishing UI or creating a partial audio blob', async () => {
  const f = fixture(); const ws = await f.begin(); f.establish(ws)
  ws.emit({ type: 'response.output.delta', kind: 'audio', audio: 'SAMPLES' })
  f.context.float32ToWavBlobUrl = () => { throw new Error('unmount must not create object URL') }
  const count = f.ui.generating.length
  f.context.streamingStopRef.current({ unmount: true })
  assert.equal(f.ui.generating.length, count); assert.equal(f.ui.messages.length, 0); assert.equal(f.ui.persisted.length, 0)
  assert.equal(f.players[0].disposed, 1); assert.equal(ws.closeCount, 0)
  assert.equal(ws.sent.at(-1).reason, 'client_cancel')
  ws.emit({ type: 'session.closed' }); assert.equal(ws.closeCount, 1)
})

test('completed audio remains stoppable during drain and old drain cannot clear new playback', async () => {
  const f = fixture(); const ws = await f.begin(); f.establish(ws)
  ws.emit({ type: 'response.output.delta', kind: 'audio', audio: 'SAMPLES' })
  ws.emit({ type: 'response.done', text: 'done' })
  const oldPlayer = f.players[0]
  assert.equal(f.context.streamingPlayerRef.current, oldPlayer)
  assert.equal(ws.sent.at(-1).reason, 'turn_done')
  f.context.stopCurrentReply(); assert.equal(oldPlayer.disposed, 1)
  await f.begin(); const nextPlayer = f.players[1]
  const count = f.ui.playback.length
  oldPlayer.onDrained()
  assert.equal(f.context.streamingPlayerRef.current, nextPlayer); assert.equal(f.ui.playback.length, count)
})

test('duplicate gateway readiness messages do not resubmit model input', async () => {
  const f = fixture(); const ws = await f.begin(); f.establish(ws)
  ws.emit({ type: 'session.queue_done' }); ws.emit({ type: 'session.created' })
  assert.equal(ws.sent.filter((msg) => msg.type === 'session.init').length, 1)
  assert.equal(ws.sent.filter((msg) => msg.type === 'input.append').length, 1)
})
