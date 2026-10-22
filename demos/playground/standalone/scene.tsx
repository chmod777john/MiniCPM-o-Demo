import { createRoot, type Root } from 'react-dom/client';
import { useSyncExternalStore } from 'react';
import { Check, RadioTower } from 'lucide-react';
import {
  PRESET_OPTIONS,
  PRESET_ZH_THINK,
  buildStandaloneQuickSettings,
} from '../../standalone/web/quick-settings.js';
import type { SceneFactory } from '../shared/session/types';

type EventItem = { id: string; text: string; unit: number | null };
class EventScene {
  entries: EventItem[] = [];
  listeners = new Set<() => void>();
  root: Root | null = null;
  constructor(private stage: HTMLElement) {}
  snapshot = () => this.entries;
  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  };
  update(next: EventItem[]) {
    this.entries = next;
    this.listeners.forEach((fn) => fn());
  }
  render = () => {
    if (!this.root) {
      this.root = createRoot(this.stage);
      this.root.render(<EventStage store={this} />);
    }
  };
  presetOptions = PRESET_OPTIONS;
  fallbackQuickSetting = PRESET_ZH_THINK;
  buildQuickSettings = () => buildStandaloneQuickSettings();
  tools = () => [];
  reset = () => this.update([]);
  dispose = () => {
    const root = this.root;
    this.root = null;
    queueMicrotask(() => root?.unmount());
    this.listeners.clear();
  };
  onExternalEvent = (id: string, text: string) =>
    this.update([...this.entries, { id, text, unit: null }]);
  onUnitStarted = (event: any) => {
    const ids = new Set((event.standalone_events || []).map((e: any) => e.event_id));
    this.update(
      this.entries.map((entry) =>
        ids.has(entry.id) ? { ...entry, unit: event.unit_index } : entry,
      ),
    );
  };
}
function EventStage({ store }: { store: EventScene }) {
  const entries = useSyncExternalStore(store.subscribe, store.snapshot);
  return (
    <div className="event-stage">
      <header>
        <RadioTower size={16} />
        外部消息
      </header>
      {entries.length ? (
        <ol className="injected-events">
          {entries.map((entry) => (
            <li key={entry.id}>
              <p>{entry.text}</p>
              <span>
                {entry.unit === null ? (
                  '已发送，等待模型接收'
                ) : (
                  <>
                    <Check size={12} />
                    已在 Unit {entry.unit} 接收
                  </>
                )}
              </span>
            </li>
          ))}
        </ol>
      ) : (
        <div className="task-empty">
          <RadioTower size={28} />
          <h3>为对话补充新的信息</h3>
          <p>开始会话后，在下方发送消息，接收状态会显示在这里。</p>
        </div>
      )}
    </div>
  );
}
export const factory: SceneFactory = {
  create: (context) => new EventScene(context.stageEl),
  css: '',
};
