import { ChevronDown } from 'lucide-react';
import type { CaseDefinition, SessionSettings, SessionView } from '../session/types';
import type { SessionController } from '../session/controller';

interface ConnectionProps {
  definition: CaseDefinition;
  settings: SessionSettings;
  view: SessionView;
  controller: SessionController;
  onSettings: (patch: Partial<SessionSettings>) => void;
  openSettings: () => void;
}
export function EndpointPicker({
  settings,
  view,
  controller,
  onSettings,
  openSettings,
}: ConnectionProps) {
  const options = controller.models;
  const known = options.some((model) => model.id === settings.modelId);
  return (
    <div className="endpoint-picker">
      <label className="endpoint-chip endpoint-select">
        <span className={`status-dot ${controller.busy ? 'connected' : ''}`} />
        <select
          aria-label="选择模型"
          title={settings.modelId}
          value={settings.modelId}
          disabled={
            controller.busy || view.recordingPending || controller.modelLoading || !options.length
          }
          onChange={(event) => {
            controller.selectModel(event.target.value);
            onSettings({ modelId: event.target.value });
          }}
        >
          {!known && (
            <option value={settings.modelId}>
              {controller.modelLoading ? '正在加载模型…' : settings.modelId || '请选择模型'}
            </option>
          )}
          {options.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name || model.id}
            </option>
          ))}
        </select>
        <ChevronDown size={12} aria-hidden="true" />
      </label>
      {!options.length && (
        <button className="endpoint-setup" onClick={openSettings}>
          配置连接
        </button>
      )}
    </div>
  );
}
export function EndpointNotice({
  controller,
  settings,
  openSettings,
}: Pick<ConnectionProps, 'controller' | 'settings' | 'openSettings'>) {
  if (!controller.modelError) return null;
  let catalogUrl = '';
  try {
    const url = new URL(settings.apiBase);
    if (['http:', 'https:'].includes(url.protocol))
      catalogUrl = `${url.toString().replace(/\/$/, '').replace(/\/v1$/, '')}/v1/models`;
  } catch {
    /* Incomplete address while typing. */
  }
  return (
    <div className="endpoint-notice" role="status">
      <span>{controller.modelError}</span>
      {catalogUrl && (
        <a href={catalogUrl} target="_blank" rel="noreferrer">
          打开 API 服务
        </a>
      )}
      <button onClick={openSettings}>连接设置</button>
    </div>
  );
}
