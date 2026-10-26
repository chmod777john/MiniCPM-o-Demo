import { numericSetting, settingOptions } from '../session/presets';
import { useEffect, useRef, useState } from 'react';
import { MixerPreview } from '../media/mixer-preview';
import { measureLUFS } from '../../../../static/duplex/lib/lufs.js';
import type { CaseDefinition, SessionSettings } from '../session/types';
import { Field, Section, Toggle } from './controls';
export function InputSettings({
  definition,
  settings: s,
  onSettings: change,
  disabled,
  onError,
  readLevels,
}: {
  definition: CaseDefinition;
  settings: SessionSettings;
  onSettings: (patch: Partial<SessionSettings>) => void;
  disabled: boolean;
  onError: (e: unknown) => void;
  readLevels: () => Record<string, number | undefined>;
}) {
  const [previewing, setPreviewing] = useState(false);
  const [levels, setLevels] = useState<Record<string, number | undefined>>({});
  const latestLevels = useRef(readLevels);
  latestLevels.current = readLevels;
  useEffect(() => {
    const timer = setInterval(
      () => setLevels(preview.current?.levels() || latestLevels.current()),
      150,
    );
    return () => clearInterval(timer);
  }, []);
  const [previewUrl, setPreviewUrl] = useState('');
  const preview = useRef<MixerPreview | null>(null);
  const previewLink = useRef('');
  useEffect(() => {
    preview.current?.update(s);
  }, [s]);
  useEffect(
    () => () => {
      preview.current?.dispose();
      if (previewLink.current) URL.revokeObjectURL(previewLink.current);
    },
    [],
  );
  useEffect(() => {
    if (disabled) {
      preview.current?.dispose();
      preview.current = null;
      setPreviewing(false);
      calibration.current();
      setCalibrating(false);
    }
  }, [disabled]);
  const [calibrating, setCalibrating] = useState(false);
  const calibration = useRef<() => void>(() => {});
  useEffect(() => () => calibration.current(), []);
  async function calibrate() {
    setCalibrating(true);
    let cancelled = false;
    let stream: MediaStream | undefined;
    let context: AudioContext | undefined;
    calibration.current = () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
      void context?.close().catch(() => {});
    };
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          ...(s.micDevice ? { deviceId: { exact: s.micDevice } } : {}),
        },
      });
      if (cancelled) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      context = new AudioContext({ sampleRate: 16000 });
      const node = context.createScriptProcessor(4096, 1, 1);
      const chunks: Float32Array[] = [];
      node.onaudioprocess = (e) => chunks.push(e.inputBuffer.getChannelData(0).slice());
      context.createMediaStreamSource(stream).connect(node);
      node.connect(context.destination);
      await context.resume();
      await new Promise((resolve) => setTimeout(resolve, 3000));
      if (cancelled) return;
      const samples = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
      let offset = 0;
      for (const c of chunks) {
        samples.set(c, offset);
        offset += c.length;
      }
      const value = measureLUFS(samples, 16000);
      change({ micMeasured: Number.isFinite(value) ? value : -23, micTrim: 0 });
    } catch (e) {
      if (!cancelled) onError(e);
    } finally {
      const wasCancelled = cancelled;
      calibration.current();
      if (!wasCancelled) setCalibrating(false);
    }
  }
  const [filePreview, setFilePreview] = useState('');
  useEffect(() => {
    if (!s.inputFile) {
      setFilePreview('');
      return;
    }
    const url = URL.createObjectURL(s.inputFile);
    setFilePreview(url);
    return () => URL.revokeObjectURL(url);
  }, [s.inputFile]);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  useEffect(() => {
    let active = true;
    const refresh = () =>
      void navigator.mediaDevices
        ?.enumerateDevices()
        .then((list) => {
          if (active) setDevices(list);
        })
        .catch(() => {});
    refresh();
    navigator.mediaDevices?.addEventListener('devicechange', refresh);
    return () => {
      active = false;
      navigator.mediaDevices?.removeEventListener('devicechange', refresh);
    };
  }, []);
  if (definition.protocol === 'chat') return null;
  const fc = definition.protocol === 'fc';
  const number = (
    key: keyof SessionSettings,
    label: string,
    min: number,
    max: number,
    step = 1,
  ) => {
    [min, max, step] = numericSetting(s, key, [min, max, step]);
    return (
      <Field label={label}>
        <input
          type="number"
          value={s[key] as number}
          min={min}
          max={max}
          step={step}
          disabled={
            disabled &&
            !(['micTarget', 'micTrim', 'fileTrim', 'monitor', 'fileVolume'] as string[]).includes(
              key,
            )
          }
          onChange={(e) => {
            if (e.target.value !== '')
              change({ [key]: Math.min(max, Math.max(min, Number(e.target.value))) });
          }}
        />
      </Field>
    );
  };
  return (
    <Section title="输入与设备">
      {fc && (
        <Toggle
          label="Omni · 摄像头画面"
          checked={s.fcVideo}
          disabled={disabled || s.inputMode === 'file'}
          onChange={(fcVideo) => change({ fcVideo })}
        />
      )}
      <Field label="输入来源">
        <select
          disabled={disabled || (fc && s.fcVideo)}
          value={s.inputMode}
          onChange={(e) => change({ inputMode: e.target.value as SessionSettings['inputMode'] })}
        >
          {settingOptions(s, 'inputMode', ['live', 'file']).map((mode) => (
            <option key={mode} value={mode}>
              {mode === 'live'
                ? `实时麦克风${definition.video ? '与摄像头' : ''}`
                : definition.video
                  ? '视频文件'
                  : '音频文件'}
            </option>
          ))}
        </select>
      </Field>
      {s.inputMode === 'file' && (
        <>
          <Field label={s.inputFile?.name || '选择文件'}>
            <input
              type="file"
              accept={definition.video ? 'video/*' : 'audio/*'}
              disabled={disabled}
              onChange={(e) => change({ inputFile: e.target.files?.[0] })}
            />
          </Field>
          {filePreview &&
            (definition.video ? (
              <video controls className="message-media" src={filePreview} />
            ) : (
              <audio controls className="message-audio" src={filePreview} />
            ))}
          {!fc && (
            <Field label="音频来源">
              <select
                value={s.fileAudioMode}
                disabled={disabled}
                onChange={(e) =>
                  change({ fileAudioMode: e.target.value as SessionSettings['fileAudioMode'] })
                }
              >
                {settingOptions(
                  s,
                  'fileAudioMode',
                  definition.video ? ['file', 'mic', 'mixed'] : ['file', 'mixed'],
                ).map((mode) => (
                  <option key={mode} value={mode}>
                    {({ file: '仅文件音频', mic: '仅麦克风', mixed: '文件 + 麦克风' } as any)[
                      mode
                    ] || mode}
                  </option>
                ))}
              </select>
            </Field>
          )}
          {number('padBefore', '前置静音 · s', 0, 30)}
          {number('padAfter', '尾部静音 · s', 0, 30)}
          {!fc && number('fileVolume', '本地文件播放音量', 0, 1, 0.05)}
          {definition.video && number('frameOffset', '每秒截帧位置', 0, 1, 0.05)}
          {!fc && s.fileAudioMode === 'mixed' && (
            <>
              <button
                className="secondary-button full"
                disabled={disabled || calibrating}
                onClick={() => void calibrate()}
              >
                {calibrating ? '请正常说话，采集 3 秒…' : '校准麦克风音量'}
              </button>
              <p className="field-hint">
                麦克风实测 {s.micMeasured.toFixed(1)} LUFS · 自动增益{' '}
                {(s.micTarget - s.micMeasured).toFixed(1)} dB
              </p>
              <p className="field-hint">
                文件实测 {Number.isFinite(levels.fileLUFS) ? levels.fileLUFS!.toFixed(1) : '—'} LUFS
                · 自动增益{' '}
                {Number.isFinite(levels.fileLUFS)
                  ? (s.fileTarget - levels.fileLUFS!).toFixed(1)
                  : '—'}{' '}
                dB
              </p>
              <div className="mixer-meters">
                {(['mic', 'file', 'mix'] as const).map((key) => (
                  <label key={key}>
                    <span>{{ mic: '麦克风', file: '文件', mix: '混音' }[key]}</span>
                    <meter min={-60} max={0} value={levels[key] ?? -60} />
                    <small>
                      {levels[key] === undefined
                        ? '—'
                        : `${Math.max(-60, levels[key]!).toFixed(1)} dB`}
                    </small>
                  </label>
                ))}
              </div>
              {number('micTarget', '麦克风目标 · LUFS', -60, 0)}
              {number('micTrim', '麦克风增益 · dB', -12, 12)}
              {number('fileTarget', '文件目标 · LUFS', -60, 0)}
              {number('fileTrim', '文件增益 · dB', -12, 12)}
              {number('monitor', '本地监听音量 · %', 0, 100)}
              <button
                className="secondary-button full"
                disabled={disabled || calibrating || !s.inputFile}
                onClick={() => {
                  if (previewing) {
                    void preview.current?.finish();
                    return;
                  }
                  setPreviewing(true);
                  const player = new MixerPreview((blob) => {
                    if (previewLink.current) URL.revokeObjectURL(previewLink.current);
                    const url = URL.createObjectURL(blob);
                    previewLink.current = url;
                    setPreviewUrl(url);
                    setPreviewing(false);
                    preview.current = null;
                  });
                  preview.current = player;
                  void player.start(s).catch((e) => {
                    setPreviewing(false);
                    preview.current = null;
                    onError(e);
                  });
                }}
              >
                {previewing ? '停止并保留试听' : '录制混音试听'}
              </button>
              {previewUrl && <audio className="message-audio" controls src={previewUrl} />}
            </>
          )}
        </>
      )}
      {!fc && (
        <>
          <button
            className="secondary-button full"
            disabled={disabled}
            onClick={() =>
              void (async () => {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                try {
                  setDevices(await navigator.mediaDevices.enumerateDevices());
                } finally {
                  stream.getTracks().forEach((track) => track.stop());
                }
              })().catch(onError)
            }
          >
            刷新音频设备
          </button>
          {(['micDevice', 'speakerDevice'] as const).map((key) => (
            <Field key={key} label={key === 'micDevice' ? '麦克风' : '扬声器'}>
              <select
                value={s[key]}
                disabled={disabled && key === 'micDevice'}
                onChange={(e) => change({ [key]: e.target.value })}
              >
                <option value="">系统默认</option>
                {devices
                  .filter((d) => d.kind === (key === 'micDevice' ? 'audioinput' : 'audiooutput'))
                  .map((d, i) => (
                    <option key={d.deviceId || i} value={d.deviceId}>
                      {d.label || `设备 ${i + 1}`}
                    </option>
                  ))}
              </select>
            </Field>
          ))}
        </>
      )}
    </Section>
  );
}
