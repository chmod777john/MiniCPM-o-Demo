import { numericSetting, settingOptions } from '../session/presets';
import type { CaseDefinition, SessionSettings } from '../session/types';
import { Field, Section, Toggle } from './controls';

export function ParameterSettings({
  definition,
  settings: s,
  onSettings: change,
  disabled,
}: {
  definition: CaseDefinition;
  settings: SessionSettings;
  onSettings: (patch: Partial<SessionSettings>) => void;
  disabled: boolean;
}) {
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
          min={min}
          max={max}
          step={step}
          value={s[key] as number}
          disabled={
            disabled &&
            ![
              'playbackDelay',
              'maxKvTokens',
              'screenSubtitleHeight',
              'screenSubtitleBottom',
              'screenSubtitleTop',
            ].includes(key)
          }
          onChange={(e) => {
            if (e.target.value !== '')
              change({
                [key]: Math.min(max, Math.max(min, Number(e.target.value))),
                ...(key === 'playbackDelay' ? { playbackPreference: true } : {}),
              });
          }}
        />
      </Field>
    );
  };
  return (
    <>
      {definition.protocol === 'ordinary' && (
        <>
          <Section title="上下文与滑窗">
            {number('maxKvTokens', 'Max KV tokens（客户端停止阈值）', 512, 262144, 512)}
            <Field label="滑窗模式">
              <select
                value={s.slidingWindowMode}
                disabled={disabled}
                onChange={(e) =>
                  change({
                    slidingWindowMode: e.target.value as SessionSettings['slidingWindowMode'],
                  })
                }
              >
                {s.allowedSlidingModes.map((mode) => (
                  <option value={mode} key={mode}>
                    {mode}
                  </option>
                ))}
              </select>
            </Field>
            {s.slidingWindowMode === 'basic' && (
              <>
                {number('basicHigh', '启动阈值 · tokens', 1, 262144)}
                {number('basicLow', '保留阈值 · tokens', 1, 262144)}
              </>
            )}
            {s.slidingWindowMode === 'kv_surgery' && (
              <>
                {number('contextMaxUnits', 'Context max units', 1, 320)}
                {number('textSink', 'Text sink', 0, 32768)}
                {number('textWindow', 'Text sliding window', 1, 32768)}
              </>
            )}
          </Section>
          <Section title="Listen / Speak 判定">
            <Toggle
              label="Binary 判定（关闭为 legacy）"
              checked={s.binaryDecision}
              disabled={disabled}
              onChange={(binaryDecision) => change({ binaryDecision })}
            />
            {number('speakThreshold', 'Speak threshold', 0.01, 0.99, 0.01)}
          </Section>
        </>
      )}
      {definition.video && (
        <Section title="屏幕字幕">
          {number('screenSubtitleHeight', '字幕区域高度 · %', 5, 80)}
          {number('screenSubtitleBottom', '底部不透明度 · %', 0, 100)}
          {number('screenSubtitleTop', '顶部不透明度 · %', 0, 100)}
        </Section>
      )}
      {definition.video && (
        <Section title="视频理解">
          <Field label="HD 模式">
            <select
              disabled={disabled}
              value={s.visionHD}
              onChange={(e) => change({ visionHD: e.target.value as SessionSettings['visionHD'] })}
            >
              {settingOptions(s, 'visionHD', ['off', 'fixed', 'strategy']).map((mode) => (
                <option key={mode} value={mode}>
                  {({ off: 'Off', fixed: 'Fixed HD', strategy: 'Strategy HD' } as any)[mode] ||
                    mode}
                </option>
              ))}
            </select>
          </Field>
          {s.visionHD === 'fixed' && number('fixedSlices', '每个 Unit 的 slices', 1, 9)}
          {s.visionHD === 'strategy' && number('strategySlices', 'SPEAK 后的 slices', 1, 9)}
          <p className="field-hint">
            保留采集画面的原始分辨率。Strategy 由服务端判断下一 Unit，开始后不能切换。
          </p>
          <Toggle
            label="前置摄像头"
            checked={s.frontCamera}
            disabled={disabled}
            onChange={(frontCamera) => change({ frontCamera })}
          />
          <Toggle
            label="镜像输入画面"
            checked={s.mirror}
            disabled={disabled}
            onChange={(mirror) => change({ mirror })}
          />
        </Section>
      )}
      {definition.protocol === 'chat' && (
        <Section title="生成设置">
          <Toggle
            label="流式输出"
            checked={s.streaming}
            disabled={disabled}
            onChange={(streaming) => change({ streaming })}
          />
          {number('maxTokens', 'Max new tokens', 1, 2048)}
          <Toggle
            label="图像 HD（2 slices）"
            checked={s.imageHD}
            disabled={disabled}
            onChange={(imageHD) => change({ imageHD })}
          />
          <p className="field-hint">
            上下文中包含视频时，自动使用 omni 和 1 slice，与原单工页一致。
          </p>
        </Section>
      )}
      {definition.protocol !== 'chat' && (
        <Section title="播放设置">
          {number(
            'playbackDelay',
            '播放预缓冲 · ms',
            0,
            definition.protocol === 'fc' ? 3000 : 2000,
            50,
          )}
        </Section>
      )}
    </>
  );
}
