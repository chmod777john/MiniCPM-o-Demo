import { useState } from 'react'
import { convertAudioBlobToFloat32Base64 } from './settings-utils'
import { useI18n } from '../i18n'
import type { DuplexOptions, SlidingWindowMode } from './nofc-options'

export function DuplexOptionsFields({ value, onChange, video = false, allowedModes = ['off', 'basic', 'kv_surgery'], disabled = false }: {
  value: DuplexOptions
  onChange: (value: DuplexOptions) => void
  video?: boolean
  allowedModes?: SlidingWindowMode[]
  disabled?: boolean
}) {
  const { lang } = useI18n()
  const [audioError, setAudioError] = useState('')
  const [loadingAudio, setLoadingAudio] = useState(false)
  const label = (zh: string, en: string) => lang === 'zh' ? zh : en
  const update = <K extends keyof DuplexOptions>(key: K, next: DuplexOptions[K]) => onChange({ ...value, [key]: next })
  const number = (key: keyof DuplexOptions, text: string, min: number, max: number, step = 1) => (
    <label className="settings-field" key={key}><span>{text}</span>
      <input className="settings-input" type="number" value={Number(value[key])} min={min} max={max} step={step} onChange={(event) => {
        const next = Number(event.target.value)
        if (event.target.value && Number.isFinite(next)) update(key, Math.max(min, Math.min(max, next)))
      }} />
    </label>
  )
  return <fieldset disabled={disabled} style={{ border: 0, padding: 0, margin: 0 }}>
    <p className="settings-chip">{label('会话参数在下次开始时生效', 'Session options apply on next start')}</p>
    <div className="settings-grid">
      {number('maxKvTokens', label('KV 上限（到达后停止）', 'Max KV (stop limit)'), 512, 262144, 512)}
      {number('playbackDelayMs', label('播放缓冲（ms）', 'Playback delay (ms)'), 0, 2000, 50)}
      <label className="settings-field"><span>{label('滑动窗口', 'Sliding window')}</span>
        <select className="settings-input" value={value.slidingWindowMode} onChange={(event) => update('slidingWindowMode', event.target.value as SlidingWindowMode)}>
          {allowedModes.map((mode) => <option key={mode} value={mode}>{mode === 'off' ? label('关闭', 'Off') : mode === 'basic' ? 'Basic' : 'KV surgery'}</option>)}
        </select>
      </label>
      {value.slidingWindowMode === 'basic' && <>
        {number('basicWindowHighTokens', label('触发阈值（token）', 'High threshold (tokens)'), 1, 262144)}
        {number('basicWindowLowTokens', label('保留数量（token）', 'Retained tokens'), 1, 262144)}
      </>}
      {value.slidingWindowMode === 'kv_surgery' && <>
        {number('contextMaxUnits', label('保留 unit 数', 'Keep units'), 1, 320)}
        {number('textSink', label('保留开头文本', 'Text sink'), 0, 32768)}
        {number('textSlidingWindow', label('保留近期文本', 'Recent text'), 1, 32768)}
      </>}
      {value.listenSpeakDecision && number('speakThreshold', label('开始说话阈值', 'Speak threshold'), 0.01, 0.99, 0.01)}
      {video && <>
        <label className="settings-field"><span>{label('高清策略', 'HD mode')}</span>
          <select className="settings-input" value={value.visionHDMode} onChange={(event) => update('visionHDMode', event.target.value as DuplexOptions['visionHDMode'])}>
            <option value="off">{label('关闭', 'Off')}</option><option value="fixed">{label('固定', 'Fixed')}</option><option value="strategy">{label('SPEAK 后高清', 'HD after SPEAK')}</option>
          </select>
        </label>
        {value.visionHDMode === 'fixed' && number('visionFixedSliceNums', label('每 unit 切片数', 'Slices per unit'), 1, 9)}
        {value.visionHDMode === 'strategy' && number('visionStrategySliceNums', label('SPEAK 后切片数', 'Slices after SPEAK'), 1, 9)}
      </>}
    </div>
    <label className="settings-toggle"><input type="checkbox" checked={value.listenSpeakDecision} onChange={(event) => update('listenSpeakDecision', event.target.checked)} /><span>{label('Listen/Speak 二选一判定', 'Binary Listen/Speak decision')}</span></label>
    <label className="settings-toggle"><input type="checkbox" checked={value.useTts} onChange={(event) => update('useTts', event.target.checked)} /><span>{label('语音输出', 'Voice reply')}</span></label>
    <label className="settings-field"><span>{label('TTS 独立音色（留空沿用系统参考音频）', 'Independent TTS voice (otherwise uses system reference)')}</span>
      <input type="file" accept="audio/*" disabled={loadingAudio} onChange={async (event) => {
        const file = event.target.files?.[0]
        event.target.value = ''
        if (!file) return
        setLoadingAudio(true)
        setAudioError('')
        try { const audio = await convertAudioBlobToFloat32Base64(file); update('ttsRefAudioBase64', audio) }
        catch (error) { setAudioError(error instanceof Error ? error.message : String(error)) }
        finally { setLoadingAudio(false) }
      }} />
      {value.ttsRefAudioBase64 && <button type="button" className="settings-mini-btn" onClick={() => update('ttsRefAudioBase64', null)}>{label('已设置 · 清除', 'Set · Clear')}</button>}
      {loadingAudio && <span>{label('处理中…', 'Processing…')}</span>}
      {audioError && <span role="alert">{audioError}</span>}
    </label>
    {value.slidingWindowMode === 'basic' && value.basicWindowHighTokens <= value.basicWindowLowTokens && <p role="alert">{label('触发阈值必须大于保留数量。', 'High threshold must exceed retained tokens.')}</p>}
  </fieldset>
}
