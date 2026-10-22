import { useRef, useState } from 'react';
import {
  ArrowUpRight,
  Check,
  ChevronRight,
  Download,
  LoaderCircle,
  Plug,
  Upload,
  X,
} from 'lucide-react';
import type { CaseDefinition, SessionSettings, SessionView } from '../session/types';
import type { SessionController } from '../session/controller';
import { wavReference } from '../media/files';
import { DesignIcon } from './DesignIcon';
import { Field, IconButton, Section, Toggle } from './controls';

export function DeveloperPanel({
  definition,
  settings,
  onSettings,
  controller,
  view,
  onClose,
  sceneReady,
}: {
  definition: CaseDefinition;
  settings: SessionSettings;
  onSettings: (patch: Partial<SessionSettings>) => void;
  controller: SessionController;
  view: SessionView;
  onClose: () => void;
  sceneReady: boolean;
}) {
  const [loading, setLoading] = useState('');
  const [notice, setNotice] = useState('');
  const [jsonDraft, setJsonDraft] = useState<'segments' | 'tools' | null>(null);
  const refInput = useRef<HTMLInputElement>(null);
  const fc = definition.protocol === 'fc';
  const busy = controller.busy;
  async function action(name: string, fn: () => Promise<void>) {
    setLoading(name);
    setNotice('');
    try {
      await fn();
      setNotice(name === 'endpoint' ? '连接信息读取成功' : '预设已加载');
    } catch (error) {
      controller.setError(error);
    } finally {
      setLoading('');
    }
  }
  function exportEvents() {
    const data = new Blob([JSON.stringify({ case: definition.id, events: view.events }, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(data);
    const a = document.createElement('a');
    a.href = url;
    a.download = `o5-${definition.id}-events.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <aside className="developer-panel" aria-label="开发者设置">
      <header className="developer-heading">
        <span>
          <DesignIcon name="settings" size={18} /> 开发者模式
        </span>
        <IconButton label="收起开发者面板" onClick={onClose}>
          <ChevronRight size={17} />
        </IconButton>
      </header>
      <div className="developer-scroll">
        <Section title="连接设置" badge={view.sessionId ? '已建立' : 'SESSION'} open>
          <Field label="模型 API 地址" hint="浏览器直接连接所选端点。">
            <input
              type="url"
              value={settings.endpoint}
              placeholder="https://api.example.com/e/model"
              disabled={busy}
              onChange={(e) => onSettings({ endpoint: e.target.value })}
            />
          </Field>
          <button
            className="secondary-button full"
            disabled={!!loading || !settings.endpoint || busy}
            onClick={() =>
              void action('endpoint', async () => {
                await controller.inspectEndpoint(settings);
              })
            }
          >
            {loading === 'endpoint' ? (
              <LoaderCircle className="spin" size={14} />
            ) : (
              <Plug size={14} />
            )}
            读取模型信息
            <ArrowUpRight size={13} />
          </button>
          {view.model && (
            <div className="meta-row">
              <span>模型</span>
              <strong title={view.model}>{view.model.split('/').pop()}</strong>
            </div>
          )}
          <div className="meta-row">
            <span>交互方式</span>
            <strong>{definition.protocol === 'chat' ? '单工对话' : '实时双工'}</strong>
          </div>
          {view.sessionId && (
            <div className="meta-row">
              <span>Session</span>
              <code title={view.sessionId}>{view.sessionId.slice(0, 12)}…</code>
            </div>
          )}
          {fc && (
            <Field label="工具服务地址" hint="留空使用同源 Gallery；与模型 API 分开配置。">
              <input
                type="url"
                placeholder="https://demo.example.com"
                value={settings.gallery}
                disabled={busy}
                onChange={(e) => onSettings({ gallery: e.target.value })}
              />
            </Field>
          )}
        </Section>
        {fc ? (
          <Section title="场景预设" open>
            {controller.presets.length > 0 && (
              <Field label="Preset">
                <select
                  value={settings.preset || controller.presets[0]?.id}
                  disabled={busy || !!loading || !sceneReady}
                  onChange={(e) => onSettings({ preset: e.target.value })}
                >
                  {controller.presets.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </Field>
            )}
            <button
              className="secondary-button full"
              disabled={busy || !!loading || !sceneReady}
              onClick={() =>
                void action('preset', async () => {
                  onSettings(await controller.loadPreset(settings, settings.preset));
                })
              }
            >
              {loading === 'preset' ? (
                <LoaderCircle className="spin" size={14} />
              ) : (
                <Download size={14} />
              )}
              加载场景配置
            </button>
            <p className="field-hint">读取这个场景的 system、工具定义和参考音色。</p>
          </Section>
        ) : (
          <Section title="System prompt">
            <textarea
              aria-label="系统提示词"
              rows={5}
              disabled={busy}
              value={settings.systemPrompt}
              onChange={(e) => onSettings({ systemPrompt: e.target.value })}
            />
          </Section>
        )}
        {fc && (
          <>
            <Section title="System segments" badge={settings.segmentsJson ? '已配置' : '待加载'}>
              <textarea
                className="code-input"
                aria-label="System segments JSON"
                rows={9}
                disabled={busy}
                value={settings.segmentsJson}
                onFocus={() => setJsonDraft('segments')}
                onChange={(e) => onSettings({ segmentsJson: e.target.value })}
                placeholder='[{"kind":"text","text":"…"}]'
              />
            </Section>
            <Section title="Tools" badge="JSON">
              <textarea
                className="code-input"
                aria-label="Tools JSON"
                rows={8}
                disabled={busy}
                value={settings.toolsJson}
                onFocus={() => setJsonDraft('tools')}
                onChange={(e) => onSettings({ toolsJson: e.target.value })}
                placeholder="[]"
              />
              {jsonDraft === 'tools' && (
                <small className="field-hint">保持服务端预设中的工具顺序与定义。</small>
              )}
            </Section>
          </>
        )}
        <Section title="声音与输入" open>
          <Toggle
            label="语音回复"
            checked={settings.generateAudio}
            onChange={(generateAudio) => onSettings({ generateAudio })}
            disabled={busy}
          />
          {fc && (
            <Toggle
              label="加入摄像头画面"
              checked={settings.fcVideo}
              onChange={(fcVideo) => onSettings({ fcVideo })}
              disabled={busy}
            />
          )}
          <Field label="参考音色" hint="可上传音频，自动转换为 16 kHz 单声道 WAV。">
            <input
              className="visually-hidden"
              ref={refInput}
              type="file"
              accept="audio/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                e.target.value = '';
                if (file)
                  void action('audio', async () => {
                    onSettings({ referenceAudio: await wavReference(file) });
                  });
              }}
            />
            <button
              className="upload-button"
              disabled={busy || !!loading}
              onClick={() => refInput.current?.click()}
            >
              <Upload size={15} />
              <span>{settings.referenceAudio?.name || '上传参考音频'}</span>
              {settings.referenceAudio && <Check size={14} />}
            </button>
          </Field>
          {settings.referenceAudio && (
            <button
              className="text-button"
              disabled={busy}
              onClick={() => onSettings({ referenceAudio: undefined })}
            >
              <X size={12} />
              移除上传音色
            </button>
          )}
          {fc && (
            <Field label="服务端参考音频路径">
              <input
                value={settings.referencePath}
                disabled={busy}
                onChange={(e) => onSettings({ referencePath: e.target.value })}
                placeholder="由场景预设提供"
              />
            </Field>
          )}
        </Section>
        <Section title="运行参数">
          {definition.protocol === 'chat' && (
            <>
              <Toggle
                label="流式输出"
                checked={settings.streaming}
                onChange={(streaming) => onSettings({ streaming })}
                disabled={busy}
              />
              <Field label="最大生成 tokens">
                <input
                  type="number"
                  min={16}
                  max={8192}
                  step={64}
                  value={settings.maxTokens}
                  disabled={busy}
                  onChange={(e) =>
                    onSettings({
                      maxTokens: Math.max(16, Math.min(8192, Number(e.target.value) || 512)),
                    })
                  }
                />
              </Field>
            </>
          )}
          <Field label="播放缓冲（ms）">
            <input
              type="number"
              min={0}
              max={3000}
              step={50}
              value={settings.playbackDelay}
              disabled={busy}
              onChange={(e) =>
                onSettings({
                  playbackDelay: Math.max(0, Math.min(3000, Number(e.target.value) || 0)),
                })
              }
            />
          </Field>
          {fc && <p className="field-hint">推理预算与调度参数沿用模型端点默认值。</p>}
        </Section>
        <Section title="事件记录" badge={String(view.events.length)}>
          <div className="event-list">
            {view.events.length ? (
              view.events
                .slice(-25)
                .reverse()
                .map((event) => (
                  <div className="event" key={event.id}>
                    <time>{event.time}</time>
                    <code>{event.type}</code>
                    {event.text && <p>{event.text}</p>}
                  </div>
                ))
            ) : (
              <p className="field-hint">开始会话后，连接与工具事件会显示在这里。</p>
            )}
          </div>
          <button className="text-button" disabled={!view.events.length} onClick={exportEvents}>
            <Download size={12} />
            导出记录
          </button>
        </Section>
        {notice && (
          <p className="settings-notice" role="status">
            <Check size={13} />
            {notice}
          </p>
        )}
      </div>
      <footer className="developer-footer">
        <span className="status-dot" /> 设置仅作用于当前体验
      </footer>
    </aside>
  );
}
