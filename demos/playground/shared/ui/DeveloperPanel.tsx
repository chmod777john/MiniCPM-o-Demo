import { useEffect, useRef, useState } from 'react';
import { ChevronRight, Download, LoaderCircle, Plug, Upload, X } from 'lucide-react';
import type {
  CaseDefinition,
  SessionSettings,
  SessionView,
  AudioReference,
} from '../session/types';
import type { SessionController } from '../session/controller';
import { loadDefaultReference } from '../session/presets';
import { MediaPreview } from './MediaPreview';
import { wavReference } from '../media/files';
import { defaultsFor, ttsReference } from '../session/options';
import { RuntimeMetrics } from './RuntimeMetrics';
import { ModelRuntime } from './ModelRuntime';
import { DesignIcon } from './DesignIcon';
import { Field, IconButton, Section, Toggle } from './controls';
import { ContentEditor, SegmentEditor } from './ContentEditor';
import { ParameterSettings } from './ParameterSettings';
import { SessionArtifacts } from './SessionArtifacts';
import { ProtocolEvents } from './ProtocolEvents';
import { InputSettings } from './InputSettings';

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
  const fc = definition.protocol === 'fc';
  const chat = definition.protocol === 'chat';
  const busy = controller.busy;
  const disabled = busy || !!loading || view.configLoading || view.recordingPending;
  const custom = (patch: Partial<SessionSettings>) => onSettings(controller.customSettings(patch));
  async function action(name: string, fn: () => Promise<void>) {
    setLoading(name);
    setNotice('');
    try {
      await fn();
      setNotice('已更新配置');
    } catch (error) {
      controller.setError(error);
    } finally {
      setLoading('');
    }
  }
  const loadPreset = (preset?: string) =>
    action('preset', async () => onSettings(await controller.loadPreset(settings, preset)));
  return (
    <aside className="developer-panel" aria-label="开发者设置">
      <header className="developer-heading">
        <span>
          <DesignIcon name="settings" size={18} />
          开发者模式
        </span>
        <IconButton label="收起开发者面板" onClick={onClose}>
          <ChevronRight size={17} />
        </IconButton>
      </header>
      <div className="developer-scroll">
        <Section title="连接设置" badge={view.sessionId ? '已建立' : 'SESSION'} open>
          <Field label="模型 API 端点" hint="浏览器直接连接此端点。">
            <input
              value={settings.endpoint}
              disabled={disabled}
              placeholder="https://api.example.com/e/model"
              onChange={(e) => onSettings({ endpoint: e.target.value })}
              onBlur={() => {
                if (settings.endpoint)
                  void action('endpoint', async () => {
                    await controller.inspectEndpoint(settings);
                  });
              }}
            />
          </Field>
          <>
            <Field label="API 服务地址" hint="用于发现同一服务下的多个模型端点。">
              <input
                value={settings.apiService}
                disabled={disabled}
                placeholder="https://api.example.com"
                onChange={(e) => onSettings({ apiService: e.target.value })}
              />
            </Field>
            <button
              className="secondary-button full"
              disabled={disabled || (!settings.apiService && !settings.endpoint)}
              onClick={() =>
                void action('discover', async () => {
                  const patch = await controller.discoverEndpoints(settings);
                  onSettings(patch);
                  await controller.inspectEndpoint({ ...settings, ...patch });
                })
              }
            >
              发现模型端点
            </button>
            {!!controller.endpoints.length && (
              <Field label="可选端点">
                <select
                  value={settings.endpoint}
                  disabled={disabled}
                  onChange={(e) => {
                    const endpoint = e.target.value;
                    onSettings({ endpoint });
                    void action('endpoint', async () => {
                      await controller.inspectEndpoint({ ...settings, endpoint });
                    });
                  }}
                >
                  <option value={settings.endpoint}>当前：{settings.endpoint || '未选择'}</option>
                  {controller.endpoints
                    .filter((e) => e.url !== settings.endpoint)
                    .map((e) => (
                      <option key={e.url} value={e.url}>
                        {e.label}
                      </option>
                    ))}
                </select>
              </Field>
            )}
          </>
          <button
            className="secondary-button full"
            disabled={disabled || !settings.endpoint}
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
          </button>
          {!fc && (
            <Field
              label="会话分享服务地址"
              hint="可选。提供会话存储和分享页面，独立于预设服务；留空使用同源。"
            >
              <input
                value={settings.shareSource}
                disabled={disabled}
                placeholder="https://sessions.example.com"
                onChange={(e) => onSettings({ shareSource: e.target.value })}
              />
            </Field>
          )}
          <Field
            label={fc ? '工具服务地址' : '应用／预设服务地址'}
            hint={
              fc
                ? '提供场景预设、音频文件和工具接口；留空使用同源服务。'
                : '提供当前体验的预设、默认设置和参考音频；留空使用本应用服务。'
            }
          >
            <input
              value={fc ? settings.gallery : settings.presetSource}
              disabled={disabled}
              placeholder="https://demo.example.com"
              onChange={(e) =>
                onSettings(
                  fc
                    ? { gallery: e.target.value }
                    : { presetSource: e.target.value, configReady: false },
                )
              }
            />
          </Field>
          {view.sessionId && (
            <div className="meta-row">
              <span>Session</span>
              <code>{view.sessionId}</code>
            </div>
          )}
        </Section>
        <ModelRuntime definition={definition} settings={settings} view={view} />
        <Section title="场景预设" open>
          <Field label="当前预设">
            <select
              value={settings.preset || ''}
              disabled={disabled || (fc && !sceneReady)}
              onChange={(e) => void loadPreset(e.target.value)}
            >
              <option value="" disabled>
                尚未加载
              </option>
              <option value="custom" disabled>
                自定义
              </option>
              {controller.presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </Field>
          {!fc && (
            <p className="field-hint">
              {controller.presets.find((p) => p.id === settings.preset)?.description ||
                '预设提供提示词和参考音色；切换时保留设备与播放选择。'}
            </p>
          )}
          <button
            className="secondary-button full"
            disabled={disabled || (fc && !sceneReady)}
            onClick={() =>
              void loadPreset(
                settings.preset && settings.preset !== 'custom' ? settings.preset : undefined,
              )
            }
          >
            {loading === 'preset' || view.configLoading ? (
              <LoaderCircle className="spin" size={14} />
            ) : (
              <Download size={14} />
            )}
            {fc ? '加载预设与默认配置' : '重新加载当前预设'}
          </button>
          {notice && (
            <p className="field-hint" role="status">
              {notice}
            </p>
          )}
        </Section>
        <Section title={fc ? 'System segments' : chat ? 'System 内容' : 'System prompt'}>
          {fc ? (
            <>
              <SegmentEditor
                value={settings.segmentsJson}
                onChange={(segmentsJson) => custom({ segmentsJson })}
                disabled={disabled}
                onError={(e) => controller.setError(e)}
                onReference={(referencePath) =>
                  custom({ referencePath, referenceAudio: undefined })
                }
              />
              <button
                className="text-button"
                disabled={disabled}
                onClick={() => {
                  try {
                    onSettings(controller.resetFcPart(settings, 'system'));
                  } catch (e) {
                    controller.setError(e);
                  }
                }}
              >
                恢复当前预设的 System 与音色
              </button>
              <details>
                <summary>原始 JSON</summary>
                <textarea
                  className="code-input"
                  aria-label="System segments JSON"
                  rows={8}
                  value={settings.segmentsJson}
                  disabled={disabled}
                  onChange={(e) => custom({ segmentsJson: e.target.value })}
                />
              </details>
            </>
          ) : chat ? (
            <ContentEditor
              maxAudioSeconds={20}
              value={settings.systemContent}
              onChange={(systemContent) => custom({ systemContent })}
              disabled={disabled}
              onError={(e) => controller.setError(e)}
            />
          ) : (
            <>
              <textarea
                rows={5}
                aria-label="System prompt"
                disabled={disabled}
                value={settings.systemPrompt}
                onChange={(e) => custom({ systemPrompt: e.target.value })}
              />
              <AudioField
                label="LLM 参考音频"
                value={settings.referenceAudio}
                disabled={disabled}
                onChange={(referenceAudio) => {
                  if (referenceAudio) custom({ referenceAudio });
                  else
                    void action('reference', async () => {
                      custom({
                        referenceAudio: await loadDefaultReference(settings, definition.id),
                      });
                    });
                }}
                onError={(e) => controller.setError(e)}
              />
              <button
                className="text-button"
                disabled={disabled}
                onClick={() =>
                  void action('reference', async () => {
                    custom({ referenceAudio: await loadDefaultReference(settings, definition.id) });
                  })
                }
              >
                恢复服务默认参考音频
              </button>
            </>
          )}
        </Section>
        {fc && (
          <Section title="Tools" badge="来自预设">
            <pre className="config-json">{settings.toolsJson || '[]'}</pre>
            <p className="field-hint">工具定义随场景预设加载，保留原顺序。</p>
          </Section>
        )}
        <Section title="语音回复" open>
          {!fc && (
            <Toggle
              label="生成语音"
              checked={settings.generateAudio}
              disabled={disabled}
              onChange={(generateAudio) => onSettings({ generateAudio })}
            />
          )}
          {fc ? (
            <>
              <Field label="TTS 参考音频路径" hint="从工具服务读取，转换为 base64 WAV 后发送。">
                <input
                  value={settings.referencePath}
                  disabled={disabled}
                  onChange={(e) =>
                    custom({ referencePath: e.target.value, referenceAudio: undefined })
                  }
                />
              </Field>
              <AudioField
                label="或上传 TTS 参考音频"
                value={settings.referenceAudio}
                disabled={disabled}
                onChange={(referenceAudio) => custom({ referenceAudio })}
                onError={(e) => controller.setError(e)}
              />
            </>
          ) : (
            <>
              <Field label="TTS 参考音频来源">
                <select
                  value={settings.ttsReferenceMode}
                  disabled={disabled}
                  onChange={(e) =>
                    onSettings({
                      ttsReferenceMode: e.target.value as SessionSettings['ttsReferenceMode'],
                    })
                  }
                >
                  <option value="extract">{chat ? '从 System 提取' : '复用 LLM 参考音频'}</option>
                  <option value="independent">独立上传</option>
                </select>
              </Field>
              {settings.ttsReferenceMode === 'independent' ? (
                <AudioField
                  label="独立 TTS 参考音频"
                  value={settings.ttsReferenceAudio}
                  disabled={disabled}
                  onChange={(ttsReferenceAudio) => onSettings({ ttsReferenceAudio })}
                  onError={(e) => controller.setError(e)}
                />
              ) : (
                <p className="field-hint">
                  {ttsReference(settings, chat)
                    ? '已有可用参考音频。'
                    : chat
                      ? 'System 中需要恰好一段不超过 20 秒的音频；不满足时不发送 TTS 参考音频。'
                      : '尚未选择 LLM 参考音频。'}
                </p>
              )}
            </>
          )}
        </Section>
        <InputSettings
          readLevels={() => controller.mixerLevels()}
          definition={definition}
          settings={settings}
          onSettings={onSettings}
          disabled={disabled}
          onError={(e) => controller.setError(e)}
        />
        {fc && (
          <div className="editor-add-row">
            <button
              className="text-button"
              disabled={disabled}
              onClick={() => {
                try {
                  onSettings(controller.resetFcPart(settings, 'tts'));
                } catch (e) {
                  controller.setError(e);
                }
              }}
            >
              仅恢复预设 TTS 音色
            </button>
            <button
              className="text-button"
              disabled={disabled}
              onClick={() => {
                try {
                  const first = JSON.parse(settings.segmentsJson).find(
                    (s: any) => s.kind === 'audio',
                  );
                  if (!first) throw new Error('System 没有音频片段。');
                  custom({ referencePath: first.audio.file_path, referenceAudio: undefined });
                } catch (e) {
                  controller.setError(e);
                }
              }}
            >
              使用 System 第一个音频
            </button>
          </div>
        )}
        <ParameterSettings
          definition={definition}
          settings={settings}
          onSettings={onSettings}
          disabled={disabled}
        />
        {!fc && (
          <SessionArtifacts
            settings={settings}
            view={view}
            definition={definition}
            controller={controller}
            onSettings={onSettings}
          />
        )}
        <button
          className="text-button"
          disabled={disabled}
          onClick={() =>
            void action('reset', async () => {
              const fresh = {
                ...defaultsFor(definition.id),
                endpoint: settings.endpoint,
                gallery: settings.gallery,
                presetSource: settings.presetSource,
                apiService: settings.apiService,
                shareSource: settings.shareSource,
                referenceAudio: undefined,
                ttsReferenceAudio: undefined,
                inputFile: undefined,
              };
              const loaded = await controller.loadPreset(fresh, undefined, 'reset');
              onSettings({ ...fresh, ...loaded });
            })
          }
        >
          恢复此体验的默认配置
        </button>
        {fc && (
          <Section title="会话检查点">
            <p className="field-hint">
              {view.metrics.checkpoint?.status === 'available'
                ? `Unit ${view.metrics.unit_index} 可恢复`
                : `当前端点未提供可恢复检查点${view.metrics.checkpoint?.reason ? `：${view.metrics.checkpoint.reason}` : ''}`}
            </p>
            <button
              className="secondary-button full"
              disabled={
                !['listening', 'speaking'].includes(view.phase) ||
                view.metrics.checkpoint?.status !== 'available' ||
                settings.inputMode === 'file'
              }
              onClick={() =>
                void controller.resumeCheckpoint(settings).catch((e) => controller.setError(e))
              }
            >
              从检查点重新连接
            </button>
          </Section>
        )}
        {!fc && (
          <>
            <RuntimeMetrics definition={definition} view={view} />
            <ProtocolEvents controller={controller} definition={definition} view={view} />
          </>
        )}
      </div>
    </aside>
  );
}
function AudioField({
  label,
  value,
  onChange,
  disabled,
  onError,
}: {
  label: string;
  value?: AudioReference;
  onChange: (value?: AudioReference) => void;
  disabled: boolean;
  onError: (e: unknown) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);
  const lifecycle = useRef({ mounted: true, value, disabled });
  lifecycle.current.value = value;
  lifecycle.current.disabled = disabled;
  useEffect(() => {
    lifecycle.current.mounted = true;
    return () => {
      lifecycle.current.mounted = false;
    };
  }, []);
  return (
    <Field label={label}>
      <input
        ref={input}
        type="file"
        className="visually-hidden"
        accept="audio/*"
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = '';
          if (file) {
            setLoading(true);
            const initial = value;
            void wavReference(file)
              .then((audio) => {
                const current = lifecycle.current;
                if (current.mounted && !current.disabled && current.value === initial)
                  onChange(audio);
              })
              .catch((error) => {
                if (lifecycle.current.mounted) onError(error);
              })
              .finally(() => {
                if (lifecycle.current.mounted) setLoading(false);
              });
          }
        }}
      />
      <button
        className="upload-button"
        disabled={disabled || loading}
        onClick={() => input.current?.click()}
      >
        <Upload size={14} />
        <span>{loading ? '转换中…' : value?.name || '上传音频'}</span>
      </button>
      {value?.pcm && <MediaPreview part={{ type: 'audio', data: value.pcm, name: value.name }} />}
      {value && (
        <button
          className="text-button"
          disabled={disabled || loading}
          onClick={() => onChange(undefined)}
        >
          <X size={12} />
          移除{value.duration ? ` · ${value.duration.toFixed(1)}s` : ''}
        </button>
      )}
    </Field>
  );
}
