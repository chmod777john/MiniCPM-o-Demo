import { useState } from 'react';
import { ChevronDown, RefreshCw } from 'lucide-react';
import type { CaseDefinition, SessionSettings, SessionView } from '../session/types';
import type { SessionController } from '../session/controller';
import { IconButton } from './controls';

type ConnectionProps = {
  settings: SessionSettings;
  controller: SessionController;
  openSettings: () => void;
};

export function EndpointPicker({
  definition,
  settings,
  onSettings,
  controller,
  view,
  openSettings,
}: ConnectionProps & {
  definition: CaseDefinition;
  onSettings: (patch: Partial<SessionSettings>) => void;
  view: SessionView;
}) {
  const [loading, setLoading] = useState(false);
  const disabled = controller.busy || view.configLoading || view.recordingPending || loading;
  const fc = definition.protocol === 'fc';
  const options = controller.endpoints;
  const known = options.some((option) => option.url === settings.endpoint);
  async function select(endpoint: string) {
    onSettings({ endpoint });
    controller.dismissError();
    setLoading(true);
    try {
      await controller.inspectEndpoint({ ...settings, endpoint });
    } catch (error) {
      controller.setError(error);
    } finally {
      setLoading(false);
    }
  }
  const label = view.model
    ? view.model.split('/').pop()
    : settings.endpoint
      ? '模型连接设置'
      : '选择模型 API';
  if (!fc) {
    return (
      <button className="endpoint-chip" onClick={openSettings} title="模型与连接设置">
        <span className={`status-dot ${controller.busy ? 'connected' : ''}`} />
        <span>{label}</span>
        <ChevronDown size={12} />
      </button>
    );
  }
  return (
    <div className="endpoint-picker">
      <label className="endpoint-chip endpoint-select">
        <span className={`status-dot ${controller.busy ? 'connected' : ''}`} />
        <select
          aria-label="选择模型 API 端点"
          title={
            options.find((option) => option.url === settings.endpoint)?.label || settings.endpoint
          }
          value={settings.endpoint}
          disabled={disabled || (!options.length && !settings.endpoint)}
          onChange={(event) => void select(event.target.value)}
        >
          {!known && (
            <option value={settings.endpoint}>
              {view.configLoading ? '正在加载模型…' : settings.endpoint || '尚未配置模型 API'}
            </option>
          )}
          {options.map((option) => (
            <option key={option.url} value={option.url}>
              {option.label}
            </option>
          ))}
        </select>
        <ChevronDown size={12} aria-hidden="true" />
      </label>
      <IconButton
        label="刷新模型端点列表"
        disabled={disabled || (!settings.apiService && !settings.endpoint)}
        onClick={() => {
          setLoading(true);
          controller.dismissError();
          void (async () => {
            try {
              const patch = await controller.discoverEndpoints(settings);
              onSettings(patch);
              if (patch.endpoint) await controller.inspectEndpoint({ ...settings, ...patch });
            } catch (error) {
              controller.setError(error);
            } finally {
              setLoading(false);
            }
          })();
        }}
      >
        <RefreshCw size={14} className={loading ? 'spin' : ''} />
      </IconButton>
      {!options.length && (
        <button className="endpoint-setup" onClick={openSettings}>
          配置连接
        </button>
      )}
    </div>
  );
}

export function EndpointNotice({ controller, openSettings }: ConnectionProps) {
  if (!controller.endpointListError) return null;
  return (
    <div className="endpoint-notice" role="status">
      <span>
        {controller.endpointListError}。若浏览器提示证书问题，请先打开 API
        服务确认访问，再刷新端点列表。
      </span>
      {controller.endpointListService && (
        <a href={`${controller.endpointListService}/overview`} target="_blank" rel="noreferrer">
          打开 API 服务
        </a>
      )}
      <button onClick={openSettings}>连接设置</button>
    </div>
  );
}
