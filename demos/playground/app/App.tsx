import { useEffect, useRef, useState } from 'react';
import { AudioLines, Camera, Code2, MessagesSquare, Sparkles } from 'lucide-react';
import { cases } from './registry';
import { Sidebar } from './Sidebar';
import { Experience } from './Experience';
import { SessionController } from '../shared/session/controller';
import { type CaseId, type SessionSettings } from '../shared/session/types';
import { defaultsFor, persistedSettings } from '../shared/session/options';
import { Dialog } from '../shared/ui/controls';
import { apiBase } from '../shared/session/urls';

const fromHash = () =>
  cases.find((c) => c.id === location.hash.replace('#', ''))?.id || 'audio-duplex';
function loadSettings(): Record<string, SessionSettings> {
  try {
    const stored = JSON.parse(localStorage.getItem('o5-playground-settings-v2') || '{}');
    const deployment = import.meta.env.VITE_PLAYGROUND_DEFAULT_API || '';
    const defaultModel = import.meta.env.VITE_PLAYGROUND_DEFAULT_MODEL || '';
    const deploymentKey = `${deployment}|${defaultModel}`;
    const changed =
      deployment && localStorage.getItem('o5-playground-deployment') !== deploymentKey;
    if (deployment) {
      const base = apiBase(deployment);
      for (const item of cases) {
        if (changed || !stored[item.id]?.apiBase)
          stored[item.id] = {
            ...stored[item.id],
            apiBase: base,
            modelId: defaultModel,
          };
      }
    }
    return Object.fromEntries(
      cases.map((c) => [
        c.id,
        {
          ...defaultsFor(c.id),
          ...Object.fromEntries(
            Object.entries(persistedSettings(c.id, stored[c.id] || {})).filter(
              ([, v]) => v !== undefined,
            ),
          ),
          userOverrides:
            stored[c.id]?.userOverrides?.filter(
              (key: string) =>
                !['gallery', 'presetSource', 'shareSource', 'endpoint', 'apiService'].includes(key),
            ) ||
            Object.keys(persistedSettings(c.id, stored[c.id] || {})).filter(
              (key) => stored[c.id]?.[key] !== undefined,
            ),
          segmentsJson: '',
          toolsJson: '',
          referencePath: '',
          preset:
            c.id === 'video-duplex' ||
            (c.id !== 'audio-duplex' && stored[c.id]?.preset === 'custom')
              ? ''
              : stored[c.id]?.preset || '',
          referenceAudio: undefined,
        },
      ]),
    );
  } catch {
    return Object.fromEntries(cases.map((c) => [c.id, defaultsFor(c.id)]));
  }
}
export function App() {
  const [id, setId] = useState<CaseId>(fromHash);
  const [revision, setRevision] = useState(0);
  const [collapsed, setCollapsed] = useState(false);
  const [developer, setDeveloper] = useState(false);
  const [dialog, setDialog] = useState<'about' | 'help' | null>(null);
  const [pending, setPending] = useState<{ id: CaseId; reset?: boolean } | null>(null);
  const [settings, setSettings] = useState(loadSettings);
  const [controller, setController] = useState(
    () => new SessionController(cases.find((c) => c.id === id)!),
  );
  const controllerRef = useRef(controller);
  controllerRef.current = controller;
  const definition = cases.find((c) => c.id === id)!;
  const currentSettings = settings[id];
  useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (controllerRef.current.busy) {
        event.preventDefault();
        event.returnValue = '';
      }
    };
    const pageHide = () => controllerRef.current.stop();
    window.addEventListener('beforeunload', beforeUnload);
    window.addEventListener('pagehide', pageHide);
    return () => {
      window.removeEventListener('beforeunload', beforeUnload);
      window.removeEventListener('pagehide', pageHide);
      controllerRef.current.dispose();
    };
  }, []);
  useEffect(() => {
    const safe = Object.fromEntries(
      Object.entries(settings).map(([key, value]) => [
        key,
        persistedSettings(key as CaseId, value),
      ]),
    );
    try {
      localStorage.setItem('o5-playground-settings-v2', JSON.stringify(safe));
      if (import.meta.env.VITE_PLAYGROUND_DEFAULT_API)
        localStorage.setItem(
          'o5-playground-deployment',
          `${import.meta.env.VITE_PLAYGROUND_DEFAULT_API}|${import.meta.env.VITE_PLAYGROUND_DEFAULT_MODEL || ''}`,
        );
    } catch {
      /* storage may be disabled */
    }
  }, [settings]);
  useEffect(() => {
    function onHash() {
      const next = fromHash();
      if (next !== id) {
        if (controller.busy) {
          history.replaceState(null, '', `#${id}`);
          setPending({ id: next });
        } else activate(next);
      }
    }
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, [id, controller]);
  function activate(next: CaseId) {
    controller.dispose();
    const target = cases.find((c) => c.id === next)!;
    setSettings((all) => ({
      ...all,
      [next]: {
        ...all[next],
        apiBase: all[next].apiBase || all[id].apiBase,
        modelId:
          all[next].apiBase && all[next].apiBase !== all[id].apiBase
            ? all[next].modelId
            : all[next].modelId || all[id].modelId,
      },
    }));
    setController(new SessionController(target));
    setId(next);
    setRevision((v) => v + 1);
    setPending(null);
    history.replaceState(null, '', `#${next}`);
  }
  function select(next: CaseId) {
    if (next === id) return;
    if (controller.busy || controller.snapshot().messages.length) setPending({ id: next });
    else activate(next);
  }
  function newSession() {
    if (controller.busy || controller.snapshot().dirty) setPending({ id, reset: true });
    else activate(id);
  }
  return (
    <div
      className="playground-app"
      style={{ '--case-accent': definition.accent } as React.CSSProperties}
    >
      <Sidebar
        selected={id}
        onSelect={select}
        collapsed={collapsed}
        onCollapse={() => setCollapsed((v) => !v)}
        onNew={newSession}
        onAbout={() => setDialog('about')}
      />
      <Experience
        key={`${id}-${revision}`}
        definition={definition}
        settings={currentSettings}
        onSettings={(patch) =>
          setSettings((all) => ({
            ...all,
            [id]: {
              ...all[id],
              ...patch,
              userOverrides:
                patch.userOverrides ??
                (patch.configReady === undefined
                  ? [
                      ...new Set([
                        ...(all[id].userOverrides || []),
                        ...(Object.keys(patch) as (keyof SessionSettings)[]),
                      ]),
                    ]
                  : all[id].userOverrides),
            },
          }))
        }
        controller={controller}
        developer={developer}
        setDeveloper={setDeveloper}
        openHelp={() => setDialog('help')}
      />
      {pending && (
        <Dialog
          title={pending.reset ? '开始一段新对话？' : '切换体验？'}
          onClose={() => setPending(null)}
        >
          <p>当前会话将结束，麦克风和摄像头会关闭。这一页的对话与工具结果将清空。</p>
          <div className="dialog-actions">
            <button className="secondary-button" autoFocus onClick={() => setPending(null)}>
              继续当前对话
            </button>
            <button className="primary-button" onClick={() => activate(pending.id)}>
              {pending.reset ? '新建对话' : '结束并切换'}
            </button>
          </div>
        </Dialog>
      )}
      {dialog && (
        <Dialog
          title={dialog === 'about' ? '一个窗口，探索更多可能' : '开始你的第一次体验'}
          onClose={() => setDialog(null)}
        >
          {dialog === 'about' ? (
            <>
              <div className="about-brand">
                <Sparkles size={28} />
                <strong>
                  MiniCPM-o <span>5</span>
                </strong>
              </div>
              <p>看见眼前的世界，听懂自然的对话，在交流中调用工具、完成任务。</p>
              <div className="about-features">
                <span>
                  <MessagesSquare size={17} />
                  多模态对话
                </span>
                <span>
                  <AudioLines size={17} />
                  实时双工
                </span>
                <span>
                  <Camera size={17} />
                  视觉理解
                </span>
                <span>
                  <Code2 size={17} />
                  Agent 工具
                </span>
              </div>
            </>
          ) : (
            <>
              <ol className="help-steps">
                <li>
                  <strong>选择一种体验</strong>
                  <p>左侧切换单工、语音、视频，或选择一个 Agent 场景。</p>
                </li>
                <li>
                  <strong>选择模型</strong>
                  <p>
                    Agent
                    场景和对话体验会自动加载应用预设。右上角选择已配置的兼容模型，或在开发者面板填写对应的
                    API 地址。
                  </p>
                </li>
                <li>
                  <strong>开始交流</strong>
                  <p>单工支持文字与附件；双工点击开始并允许麦克风权限。视频体验还需要摄像头。</p>
                </li>
              </ol>
              <p className="field-hint">
                使用 HTTPS 或 localhost 打开以启用音视频。切换体验前会结束当前会话。
              </p>
            </>
          )}
          <div className="dialog-actions">
            <button
              className="primary-button"
              onClick={() => {
                setDialog(null);
                if (dialog === 'help') setDeveloper(true);
              }}
            >
              {dialog === 'help' ? '打开连接设置' : '开始探索'}
            </button>
          </div>
        </Dialog>
      )}
    </div>
  );
}
