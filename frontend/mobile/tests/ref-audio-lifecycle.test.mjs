import assert from 'node:assert/strict'
import test from 'node:test'
import vm from 'node:vm'
import { readFile } from 'node:fs/promises'

const source = (await readFile(new URL('../../../static/duplex/ui/ref-audio-init.js', import.meta.url), 'utf8')).replace('export function initRefAudio', 'function initRefAudio')
function fixture() {
  const requests = []
  class Player {
    constructor(_container, options) { this.options = options; this.base64 = null }
    setAudio(b64) { this.base64 = b64 }
  }
  const context = vm.createContext({
    window: { RefAudioPlayer: Player }, document: { getElementById: () => ({}) }, console,
    fetch: () => new Promise((resolve) => requests.push((base64) => resolve({ json: async () => ({ base64, name: base64, duration: 1 }) }))),
  })
  vm.runInContext(source, context)
  return { ref: context.initRefAudio('test'), requests }
}
const flush = () => new Promise((resolve) => setImmediate(resolve))

test('default reference initializes when no manual selection intervenes', async () => {
  const { ref, requests } = fixture()
  requests[0]('DEFAULT')
  await flush()
  assert.equal(ref.getBase64(), 'DEFAULT')
  assert.equal(ref.isDefault(), true)
})
test('late initial default cannot overwrite a preset or mobile selection', async () => {
  const { ref, requests } = fixture()
  ref.setAudio('CUSTOM', 'custom', 1)
  requests[0]('DEFAULT')
  await flush()
  assert.equal(ref.getBase64(), 'CUSTOM')
  assert.equal(ref.rap.base64, 'CUSTOM')
})
test('late reset fetch cannot overwrite a newer upload', async () => {
  const { ref, requests } = fixture()
  requests[0]('DEFAULT')
  await flush()
  ref.rap.options.onRemove()
  ref.setAudio('NEW', 'new', 1)
  requests[1]('OLD_RESET')
  await flush()
  assert.equal(ref.getBase64(), 'NEW')
})
