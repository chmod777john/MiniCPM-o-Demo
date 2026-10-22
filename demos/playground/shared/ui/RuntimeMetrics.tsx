import type { CaseDefinition, SessionView } from '../session/types';
import { Section } from './controls';
const ms = (n: unknown) =>
  typeof n === 'number' && Number.isFinite(n) ? `${Math.round(n)} ms` : '—';
const display = (v: unknown) => (v == null || v === '' ? '—' : String(v));
export function RuntimeMetrics({
  definition,
  view,
}: {
  definition: CaseDefinition;
  view: SessionView;
}) {
  const m = view.metrics;
  const rows: [string, unknown, string?][] = [
    ['State', m.sessionState || view.phase],
    ['SID', view.sessionId],
  ];
  if (definition.protocol === 'ordinary')
    rows.push(
      ['Model', m.modelState],
      ['Chunks sent', m.chunksSent ?? 0],
      ['KV', m.kvCacheLength == null ? undefined : `${m.kvCacheLength} / ${m.maxKvTokens ?? '—'}`],
      ['Windows', m.windowEventCount ?? 0],
      [
        'Last window',
        m.windowEvent ? `${m.windowEvent.before} → ${m.windowEvent.after}` : undefined,
        m.windowEvent
          ? `${m.windowEvent.mode}; input ${m.windowEvent.inputId}; ${m.windowEvent.duration_ms} ms`
          : '',
      ],
      [
        'Vision',
        m.visionSlices != null
          ? `${m.visionSlices} slices · ${m.visionTokens ?? m.visionSlices * 64} tokens`
          : m.effectiveMaxSliceNums != null
            ? `slice limit ${m.effectiveMaxSliceNums}`
            : undefined,
      ],
      [
        'P(speak)',
        m.speakProbability == null ? undefined : `${(m.speakProbability * 100).toFixed(1)}%`,
        m.lsDecision,
      ],
      ['Infer', ms(m.latencyMs), `cost_all: ${ms(m.costAllMs)}`],
      ['TTFS', ms(m.ttfsMs)],
      ['PDelay', ms(m.pdelay)],
      ['Ahead', ms(m.ahead)],
      ['Gaps', m.gapCount ?? 0],
      ['Shift', ms(m.totalShift)],
      ['Drift', ms(m.driftMs)],
    );
  if (definition.protocol !== 'chat')
    rows.push(['Mic dBFS', m.micDb?.toFixed(1)], ['Session peak dBFS', m.micPeak?.toFixed(1)]);
  if (definition.protocol === 'fc')
    rows.push(
      ['Chunks sent', m.chunksSent ?? 0],
      ['AI audio chunks', m.audioChunks ?? 0],
      ['Generation steps', m.generation_steps ?? 0],
      ['Budget updated', m.debugUpdated],
      ['Budget used', m.debug?.debug?.used],
      ['Budget max / 1s', m.debug?.debug?.estimated_max_budget_1s],
    );
  if (definition.protocol === 'chat')
    rows.push(['Input tokens', m.input_tokens], ['Generated tokens', m.generated_tokens]);
  return (
    <Section title="运行指标" open>
      {definition.protocol === 'fc' && <p className="field-hint">麦克风目标电平 −28 ～ −18 dBFS</p>}
      {definition.protocol === 'fc' && m.micWarning && (
        <p role="status" className="error-banner">
          {m.micWarning}
        </p>
      )}
      <dl className="runtime-metrics-grid">
        {rows.map(([label, value, hint]) => (
          <div key={label} title={hint}>
            <dt>{label}</dt>
            <dd>{display(value)}</dd>
          </div>
        ))}
      </dl>
      <details>
        <summary>原始指标</summary>
        <pre className="config-json">{JSON.stringify(m, null, 2)}</pre>
      </details>
    </Section>
  );
}
