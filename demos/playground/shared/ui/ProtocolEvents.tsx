import { useState } from 'react';
import { Section } from './controls';
import { MediaPreview } from './MediaPreview';
import type { SessionController } from '../session/controller';
import type { CaseDefinition, SessionView } from '../session/types';
export function ProtocolEvents({
  controller,
  definition,
  view,
  open = false,
}: {
  controller: SessionController;
  definition: CaseDefinition;
  view: SessionView;
  open?: boolean;
}) {
  const [filter, setFilter] = useState('all');
  const [expanded, setExpanded] = useState(false);
  const records = controller.protocolEvents;
  const shown = records.filter(
    ({ direction, event }) =>
      filter === 'all' ||
      filter === direction ||
      (filter === 'audio'
        ? event.audio || event.input?.audio_base64
        : String(event.type).includes(filter)),
  );
  return (
    <Section title="协议事件" badge={String(records.length)} open={open}>
      <p className="field-hint">
        保留最近 200 条完整事件及音频，最多约 24 MiB；导出范围与此处一致。
      </p>
      <div className="editor-add-row">
        <select
          aria-label="协议事件筛选"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          {[
            ['all', '全部'],
            ['client', 'TX 发送'],
            ['server', 'RX 接收'],
            ['audio', '音频'],
            ['think', 'Think'],
            ['tool', 'Tool'],
            ['unit', 'Unit'],
            ['session', 'Session'],
          ].map(([value, label]) => (
            <option value={value} key={value}>
              {label}
            </option>
          ))}
        </select>
        <button className="text-button" onClick={() => setExpanded(!expanded)}>
          {expanded ? '收起大窗' : '展开大窗'}
        </button>
        <button
          className="secondary-button"
          onClick={() => {
            const url = URL.createObjectURL(
              new Blob(
                [
                  JSON.stringify(
                    {
                      case: definition.id,
                      model: view.modelIdentity,
                      config: view.effectiveConfig,
                      messages: view.messages.map(({ audioPlayer, ...message }) => message),
                      retention: 'latest 200 events, 24 MiB JSON limit',
                      events: records,
                    },
                    null,
                    2,
                  ),
                ],
                { type: 'application/json' },
              ),
            );
            const a = document.createElement('a');
            a.href = url;
            a.download = `o5-${definition.id}-session.json`;
            a.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
          }}
        >
          导出诊断 JSON
        </button>
      </div>
      <div className={expanded ? 'protocol-overlay' : 'event-list'}>
        {expanded && (
          <button className="secondary-button" onClick={() => setExpanded(false)}>
            关闭大窗
          </button>
        )}
        {shown.map(({ direction, event, time, id }) => (
          <details className="protocol-row" key={id}>
            <summary>
              <strong>
                {direction === 'client' ? 'TX' : 'RX'} · {event.type || '(unknown)'}
              </strong>{' '}
              <time>{time.slice(11, 23)}</time>
            </summary>
            {(event.audio || event.input?.audio_base64) && (
              <MediaPreview
                sampleRate={Number(
                  event.sample_rate ||
                    event.input?.sample_rate ||
                    (direction === 'client' ? 16000 : 24000),
                )}
                part={{ type: 'audio', data: event.audio || event.input.audio_base64 }}
              />
            )}
            <pre className="config-json">{JSON.stringify(event, null, 2)}</pre>
          </details>
        ))}
      </div>
    </Section>
  );
}
