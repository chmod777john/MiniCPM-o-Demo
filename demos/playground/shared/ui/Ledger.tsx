import { useEffect, useRef } from 'react';
import { NsSegmentView } from '../../../../static/demo-gallery/core/ns-segment-view.js';
import { summarizeSessionConfig } from '../../../../static/demo-gallery/core/api-endpoint.js';

/** Same non-spoken state machine used by the original Gallery; React owns the boundary. */
export function Ledger({ events, config }: { events: any[]; config: any }) {
  const host = useRef<HTMLDivElement>(null);
  const state = useRef<{ view: any; count: number; first: any } | null>(null);
  useEffect(() => {
    if (!host.current) return;
    if (
      !state.current ||
      events.length < state.current.count ||
      (events[0] && state.current.first !== events[0])
    )
      state.current = { view: new NsSegmentView(host.current), count: 0, first: events[0] };
    const { view } = state.current;
    const summary = summarizeSessionConfig(config);
    view.setBudgets(summary.listening, summary.speaking);
    for (const e of events.slice(state.current.count)) {
      const kind = e.type.includes('tool_call') ? 'tool_call' : 'think';
      switch (e.type) {
        case 'response.think.begin':
        case 'response.tool_call.begin':
          view.beginSegment(kind, e.unit_index, { toolCallId: e.tool_call_id });
          break;
        case 'response.think.delta':
        case 'response.tool_call.delta':
          view.appendSteps(e.unit_index, e.steps);
          break;
        case 'response.think.end':
        case 'response.tool_call.done':
          view.endSegment(kind, e.unit_index, {
            fullText:
              e.type === 'response.tool_call.done' ? e.full_text || e.error || '' : e.full_text,
            toolCallId: e.tool_call_id,
          });
          break;
        case 'response.non_spoken.end':
          view.markNonSpokenEnd(e.unit_index, e.reason);
          break;
        case 'response.spoken.end':
          view.setUnitLane(e.unit_index, e.reason === 'listen' ? 'listen' : 'speak');
          break;
      }
    }
    state.current.count = events.length;
    state.current.first = events[0];
  }, [events, config]);
  return <div ref={host} className="protocol-ledger" />;
}
