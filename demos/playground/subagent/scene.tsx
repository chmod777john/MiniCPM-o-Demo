import { createRoot, type Root } from 'react-dom/client';
import { useSyncExternalStore } from 'react';
import { Check, Clock3, File, LoaderCircle, Upload, Workflow } from 'lucide-react';
import {
  buildSubagentQuickSettings,
  presetTools,
  presetLanguage,
  PRESET_OPTIONS,
  PRESET_ZH_NOTHINK,
} from '../../subagent/web/quick-settings.js';
import type { SceneContext, SceneFactory } from '../shared/session/types';
import { base64 } from '../shared/media/files';

type Entry = {
  id: string;
  name: string;
  args: string;
  status: 'running' | 'done' | 'error';
  result: string;
};
type State = {
  entries: Entry[];
  files: { name: string; size?: number }[];
  ready: boolean;
  notice: string;
};
class TaskScene {
  state: State = { entries: [], files: [], ready: false, notice: '' };
  listeners = new Set<() => void>();
  root: Root | null = null;
  abort = new AbortController();
  defaults: any = {};
  preset = PRESET_ZH_NOTHINK;
  workspace = '';
  generation = 0;
  handled = new Set<string>();
  constructor(private context: SceneContext) {}
  subscribe = (fn: () => void) => {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  };
  snapshot = () => this.state;
  update(patch: Partial<State>) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((fn) => fn());
  }
  async request(path: string, init: RequestInit = {}) {
    const response = await fetch(this.context.api.url(path), {
      ...init,
      signal: this.abort.signal,
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || `工具服务返回 ${response.status}`);
    return body;
  }
  render = () => {
    if (!this.root) {
      this.root = createRoot(this.context.stageEl);
      this.root.render(<TaskStage store={this} />);
    }
  };
  presetOptions = PRESET_OPTIONS;
  fallbackQuickSetting = PRESET_ZH_NOTHINK;
  buildQuickSettings = (defaults: any) => {
    this.defaults = defaults;
    return buildSubagentQuickSettings(defaults);
  };
  tools = (defaults: any, preset?: string) => {
    this.preset = preset || PRESET_ZH_NOTHINK;
    return presetTools(defaults, this.preset);
  };
  pause = () => {
    this.generation++;
    this.abort.abort();
    this.update({
      ready: false,
      entries: this.state.entries.map((entry) =>
        entry.status === 'running'
          ? { ...entry, status: 'error', result: '会话已结束；已提交的后台任务可能仍在执行。' }
          : entry,
      ),
    });
  };
  dispose = () => {
    this.pause();
    const root = this.root;
    this.root = null;
    queueMicrotask(() => root?.unmount());
    this.listeners.clear();
  };
  reset = async () => {
    this.pause();
    this.abort = new AbortController();
    this.workspace = `pg_${crypto.randomUUID().replaceAll('-', '')}`;
    this.handled.clear();
    this.update({ entries: [], files: [], ready: false, notice: '正在准备独立工作区…' });
    const generation = this.generation;
    await this.request(`workspaces/${this.workspace}`, { method: 'POST' })
      .then(() => {
        if (generation === this.generation) this.update({ ready: true, notice: '' });
      })
      .catch((error) => {
        if (generation === this.generation) {
          this.update({ notice: error.message });
          throw error;
        }
      });
  };
  async upload(file: globalThis.File) {
    if (!this.context.getClient()?.ready || !this.state.ready) {
      this.update({ notice: '会话和工作区准备好后，即可上传文件。' });
      return;
    }
    if (file.size > 32 * 1024 * 1024) {
      this.update({ notice: '文件请控制在 32 MB 以内。' });
      return;
    }
    const generation = this.generation;
    try {
      const data = base64(await file.arrayBuffer());
      if (generation !== this.generation) return;
      await this.request(`workspaces/${this.workspace}/files`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: file.name, content_base64: data }),
      });
      if (generation !== this.generation) return;
      const response = await this.request(`workspaces/${this.workspace}/files`);
      if (generation !== this.generation) return;
      this.update({ files: response.files || [], notice: '' });
      this.context.getClient()?.sendStandaloneEvent({
        content: `[FILE_UPLOADED] ${file.name}`,
        eventId: `pg_${crypto.randomUUID()}`,
      });
    } catch (error) {
      if (generation === this.generation)
        this.update({ notice: error instanceof Error ? error.message : '上传失败' });
    }
  }
  onToolCallDone = async (event: any) => {
    const client = this.context.getClient();
    const call = event.call || {};
    const id = event.tool_call_id;
    if (!client?.ready || !id || this.handled.has(id)) return;
    this.handled.add(id);
    const generation = this.generation;
    const update = (patch: Partial<Entry>) => {
      if (generation === this.generation)
        this.update({
          entries: this.state.entries.map((e) => (e.id === id ? { ...e, ...patch } : e)),
        });
    };
    const send = (text: string) => {
      if (generation === this.generation && this.context.getClient() === client && client.ready)
        client.sendToolResult({ toolCallId: id, content: text });
    };
    this.update({
      entries: [
        ...this.state.entries,
        {
          id,
          name: call.name || '工具调用',
          args:
            typeof call.arguments === 'string'
              ? call.arguments
              : JSON.stringify(call.arguments || {}),
          status: 'running',
          result: '',
        },
      ],
    });
    try {
      if (call.name === 'show_card') {
        update({ status: 'done', result: JSON.stringify(call.arguments, null, 2) });
        send('已展示');
        return;
      }
      const allowed = this.defaults.subagent_handled_tools || [
        'delegate_task',
        'web_search',
        'calculate',
      ];
      if (!allowed.includes(call.name)) throw new Error(`当前场景未配置工具：${call.name}`);
      if (!this.state.ready) throw new Error('工作区尚未就绪，请稍后重试。');
      const args =
        typeof call.arguments === 'string' ? JSON.parse(call.arguments) : call.arguments || {};
      const task = await this.request('tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          workspace_id: this.workspace,
          tool_call_id: id,
          name: call.name,
          arguments: args,
          language: presetLanguage(this.defaults, this.preset),
        }),
      });
      let after = 0;
      while (generation === this.generation && !this.abort.signal.aborted) {
        const result = await this.request(`tasks/${task.task_id}/events?after=${after}`);
        if (generation !== this.generation) return;
        for (const item of result.events || []) {
          after = item.seq + 1;
          if (item.kind === 'progress') {
            update({ result: item.text });
            this.context
              .getClient()
              ?.sendStandaloneEvent({ content: item.text, eventId: `pg_${crypto.randomUUID()}` });
          } else {
            update({ status: item.kind === 'result' ? 'done' : 'error', result: item.text });
            send(item.text);
          }
        }
        if (result.done) break;
        await new Promise((resolve) => setTimeout(resolve, 700));
      }
    } catch (error) {
      if (generation === this.generation) {
        const text = error instanceof Error ? error.message : '工具执行失败';
        update({ status: 'error', result: text });
        send(text);
      }
    }
  };
}
function TaskStage({ store }: { store: TaskScene }) {
  const state = useSyncExternalStore(store.subscribe, store.snapshot);
  return (
    <div className="task-stage">
      <header>
        <span>
          <Workflow size={16} />
          任务工作区
        </span>
        <label className={`task-upload ${!state.ready ? 'disabled' : ''}`}>
          <Upload size={14} />
          添加文件
          <input
            type="file"
            hidden
            disabled={!state.ready}
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = '';
              if (file) void store.upload(file);
            }}
          />
        </label>
      </header>
      {state.notice && <p className="task-notice">{state.notice}</p>}
      <div className="task-files">
        {state.files.map((file) => (
          <span key={file.name}>
            <File size={13} />
            {file.name}
          </span>
        ))}
      </div>
      {state.entries.length ? (
        <ol className="task-entries">
          {state.entries.map((entry) => (
            <li key={entry.id}>
              <div className="task-entry-head">
                {entry.status === 'running' ? (
                  <LoaderCircle size={15} className="spin" />
                ) : (
                  <Check size={15} />
                )}
                <strong>{entry.name}</strong>
                <span>
                  {entry.status === 'running'
                    ? '执行中'
                    : entry.status === 'done'
                      ? '已完成'
                      : '未完成'}
                </span>
              </div>
              <pre>{entry.args}</pre>
              {entry.result && <p>{entry.result}</p>}
            </li>
          ))}
        </ol>
      ) : (
        <div className="task-empty">
          <Clock3 size={27} />
          <h3>等待你的第一个任务</h3>
          <p>通过语音交代任务，执行过程会显示在这里。</p>
        </div>
      )}
    </div>
  );
}
export const factory: SceneFactory = { create: (context) => new TaskScene(context), css: '' };
