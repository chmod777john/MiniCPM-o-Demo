import { useEffect, useId, useRef, useState } from 'react';
import { AudioLines, ChevronDown, GitBranch, Maximize2, Minimize2 } from 'lucide-react';
import { summarizeSessionConfig } from '../../../../static/demo-gallery/core/api-endpoint.js';
import type { SessionView } from '../session/types';
import { IconButton } from './controls';
import { Ledger } from './Ledger';
import { Messages } from './Messages';

const shown = (value: unknown) =>
  value === undefined || value === null || value === '' ? '—' : String(value);

/** Primary FC content, kept mounted when folded so the event ledger keeps its state. */
export function FcConversation({ view }: { view: SessionView }) {
  const [speechOpen, setSpeechOpen] = useState(true);
  const [traceOpen, setTraceOpen] = useState(true);
  const [fullscreen, setFullscreen] = useState(false);
  const [fullscreenError, setFullscreenError] = useState('');
  const trace = useRef<HTMLElement>(null);
  const id = useId();
  const summary = summarizeSessionConfig(view.effectiveConfig);
  const debug = view.metrics.debug?.debug || {};
  useEffect(() => {
    const update = () => setFullscreen(document.fullscreenElement === trace.current);
    document.addEventListener('fullscreenchange', update);
    return () => document.removeEventListener('fullscreenchange', update);
  }, []);
  async function expand() {
    setFullscreenError('');
    try {
      if (document.fullscreenElement === trace.current) await document.exitFullscreen();
      else {
        setTraceOpen(true);
        await trace.current?.requestFullscreen();
      }
    } catch {
      setFullscreenError('浏览器未允许全屏，可以在当前面板继续查看。');
    }
  }
  return (
    <div className={`fc-conversation ${!speechOpen && !traceOpen ? 'all-folded' : ''}`}>
      <section
        className="fc-observation-panel fc-speech-panel"
        aria-labelledby={`${id}-speech-title`}
      >
        <header className="fc-panel-heading">
          <AudioLines size={16} aria-hidden="true" />
          <h2 id={`${id}-speech-title`}>AI Speech</h2>
          <span className="fc-panel-subtitle">语音内容</span>
          <IconButton
            label={speechOpen ? '收起语音内容' : '展开语音内容'}
            aria-expanded={speechOpen}
            aria-controls={`${id}-speech`}
            onClick={() => setSpeechOpen(!speechOpen)}
          >
            <ChevronDown size={16} className={speechOpen ? '' : 'is-folded'} />
          </IconButton>
        </header>
        <div id={`${id}-speech`} className="fc-panel-body fc-speech-body" hidden={!speechOpen}>
          {view.messages.length ? (
            <Messages messages={view.messages} compact />
          ) : (
            <div className="fc-panel-empty">
              <AudioLines size={24} strokeWidth={1.5} aria-hidden="true" />
              <p>AI 还没开口</p>
              <span>开始对话后，语音内容会实时显示在这里。</span>
            </div>
          )}
        </div>
      </section>
      <section
        ref={trace}
        className="fc-observation-panel fc-trace-panel"
        aria-labelledby={`${id}-trace-title`}
      >
        <header className="fc-panel-heading">
          <GitBranch size={16} aria-hidden="true" />
          <h2 id={`${id}-trace-title`}>Think / Tool-Call</h2>
          <span className="fc-panel-subtitle">Unit · Token · Budget</span>
          <div className="fc-panel-actions">
            <IconButton
              label={fullscreen ? '退出轨迹全屏' : '全屏查看生成轨迹'}
              onClick={() => void expand()}
            >
              {fullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
            </IconButton>
            <IconButton
              label={traceOpen ? '收起生成轨迹' : '展开生成轨迹'}
              aria-expanded={traceOpen}
              aria-controls={`${id}-trace`}
              onClick={() => setTraceOpen(!traceOpen)}
            >
              <ChevronDown size={16} className={traceOpen ? '' : 'is-folded'} />
            </IconButton>
          </div>
        </header>
        <div className="fc-trace-content" id={`${id}-trace`} hidden={!traceOpen}>
          <dl
            className="fc-budget-strip"
            aria-label="生成预算"
            title={
              view.configSource === 'session'
                ? '预算来自本次会话实际配置'
                : '预算来自所选端点默认配置'
            }
          >
            <div>
              <dt>听预算</dt>
              <dd>{shown(summary.listening)}</dd>
            </div>
            <div>
              <dt>说预算</dt>
              <dd>{shown(summary.speaking)}</dd>
            </div>
            <div>
              <dt>最新 used</dt>
              <dd>{shown(debug.used)}</dd>
            </div>
            <div>
              <dt>max / 1s</dt>
              <dd>{shown(debug.estimated_max_budget_1s)}</dd>
            </div>
            {view.metrics.debugUpdated && (
              <div className="fc-budget-updated">
                <dt>更新</dt>
                <dd>{view.metrics.debugUpdated}</dd>
              </div>
            )}
          </dl>
          {fullscreenError && (
            <p className="fc-panel-notice" role="status">
              {fullscreenError}
            </p>
          )}
          <div className="fc-panel-body fc-ledger-body">
            <Ledger events={view.ledger} config={view.effectiveConfig} />
          </div>
        </div>
      </section>
    </div>
  );
}
