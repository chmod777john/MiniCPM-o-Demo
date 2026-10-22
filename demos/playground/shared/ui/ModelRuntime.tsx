import {
  MODEL_IDENTITY_ROWS,
  MODEL_IDENTITY_HINTS,
} from '../../../../static/demo-gallery/core/model-identity.js';
import { summarizeSessionConfig } from '../../../../static/demo-gallery/core/api-endpoint.js';
import type { CaseDefinition, SessionSettings, SessionView } from '../session/types';
import { realtimeUrl } from '../session/urls';
import { Section } from './controls';

const shown = (value: unknown) =>
  value === undefined || value === null || value === '' ? '未返回' : String(value);
function Fact({ label, value, hint }: { label: string; value: unknown; hint?: string }) {
  return (
    <div className="identity-row" title={hint}>
      <span>{label}</span>
      <code>{shown(value)}</code>
    </div>
  );
}

/** Keep original Loaded model / Runtime facts explicit; JSON is supplementary. */
export function ModelRuntime({
  definition,
  settings,
  view,
}: {
  definition: CaseDefinition;
  settings: SessionSettings;
  view: SessionView;
}) {
  const fc = definition.protocol === 'fc';
  const identity = view.modelIdentity;
  const config = view.effectiveConfig;
  const runtime = summarizeSessionConfig(config);
  const mode = fc
    ? settings.fcVideo
      ? 'omni'
      : 'audio'
    : definition.video
      ? 'video'
      : definition.protocol === 'chat'
        ? 'chat'
        : 'audio';
  let socket = '';
  try {
    socket = realtimeUrl(settings.endpoint, mode);
  } catch {
    /* Address may still be edited. */
  }
  const source = (kind: SessionView['modelSource']) =>
    kind === 'session'
      ? `会话实际值 · ${view.sessionId || '未知会话'}`
      : kind === 'endpoint'
        ? '所选端点默认值 · /v1/info'
        : '尚未读取';
  let tools: string[] = [];
  try {
    tools = JSON.parse(settings.toolsJson || '[]').map(
      (t: any) => t.function?.name || t.name || 'tool',
    );
  } catch {
    /* The System editor shows parse errors. */
  }
  return (
    <>
      <Section title="Loaded model · 当前模型" open>
        <p className="field-hint">{source(view.modelSource)}</p>
        <Fact label="Checkpoint Profile" value={identity.profile_id} />
        {MODEL_IDENTITY_ROWS.map(([key, label]) => (
          <Fact
            key={key}
            label={label}
            value={identity[key]}
            hint={(MODEL_IDENTITY_HINTS as Record<string, string>)[key]}
          />
        ))}
        {identity.model_family && <Fact label="Model family" value={identity.model_family} />}
        {fc && <Fact label="Tools" value={tools.length ? tools.join(', ') : 'none'} />}
      </Section>
      <Section title="Runtime · 运行配置" open={fc}>
        <p className="field-hint">{source(view.configSource)}</p>
        {fc && (
          <>
            <Fact
              label="Mode · Non-spoken scheduling"
              value={runtime.scheduling}
              hint="quality / latency 调度方式，由所选端点决定。"
            />
            <Fact label="Checkpoint Profile" value={identity.profile_id} />
            <Fact label="Listening budget" value={runtime.listening} />
            <Fact label="Speaking budget" value={runtime.speaking} />
            <Fact label="TTS pad protocol" value={runtime.ttsPad} />
            <Fact label="Unit duration · s" value={runtime.unitSec} />
            <Fact
              label="Playback buffer · ms"
              value={settings.playbackDelay}
              hint="客户端播放设置，可在播放设置中修改，下一次 SPEAK 开始生效。"
            />
          </>
        )}
        <Fact label="Realtime endpoint · 输入模式" value={socket || undefined} />
        <p className="field-hint">
          {fc
            ? '模型配置由服务返回。完整路径可选择复制；会话恢复检查点在下方单独展示。'
            : '显示服务返回的实际配置。'}
        </p>
        <details className="runtime-raw">
          <summary>原始配置与模型信息</summary>
          <pre className="config-json">
            {JSON.stringify({ model: identity, session_config: config }, null, 2)}
          </pre>
        </details>
      </Section>
    </>
  );
}
