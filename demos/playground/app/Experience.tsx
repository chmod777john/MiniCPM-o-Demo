import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import {
  AudioLines,
  Camera,
  ChevronDown,
  CircleHelp,
  Activity,
  Maximize2,
  Mic,
  MicOff,
  Pause,
  Play,
  Ear,
  FlipHorizontal,
  SwitchCamera,
  Captions,
  Plug,
  CirclePlus,
  Square,
  Volume2,
  X,
} from 'lucide-react';
import type {
  CaseDefinition,
  ContentPart,
  SessionPhase,
  SessionSettings,
} from '../shared/session/types';
import { defaultsFor } from '../shared/session/options';
import { SessionController } from '../shared/session/controller';
import { DeveloperPanel } from '../shared/ui/DeveloperPanel';
import { EndpointPicker, EndpointNotice } from '../shared/ui/EndpointPicker';
import { SceneHost } from '../shared/ui/SceneHost';
import { FcConversation } from '../shared/ui/FcConversation';
import { ProtocolEvents } from '../shared/ui/ProtocolEvents';
import { RuntimeMetrics } from '../shared/ui/RuntimeMetrics';
import microphone from '../shared/assets/figma-microphone.svg';
import { DesignIcon } from '../shared/ui/DesignIcon';
import { Orb } from '../shared/ui/Orb';
import { Messages } from '../shared/ui/Messages';
import { Composer } from '../shared/ui/Composer';
import { Dialog, IconButton } from '../shared/ui/controls';

const phaseLabels: Record<SessionPhase, string> = {
  idle: '准备就绪',
  connecting: '正在连接',
  queued: '等待接入',
  listening: '正在聆听',
  speaking: '正在回答',
  generating: '正在思考',
  error: '连接未完成',
};

export function Experience({
  definition,
  settings,
  onSettings,
  controller,
  developer,
  setDeveloper,
  openHelp,
}: {
  definition: CaseDefinition;
  settings: SessionSettings;
  onSettings: (patch: Partial<SessionSettings>) => void;
  controller: SessionController;
  developer: boolean;
  setDeveloper: (value: boolean) => void;
  openHelp: () => void;
}) {
  const view = useSyncExternalStore(controller.subscribe, controller.snapshot);
  const [draft, setDraft] = useState('');
  const [sceneReady, setSceneReady] = useState(false);
  const [now, setNow] = useState(Date.now());
  const [showTranscript, setShowTranscript] = useState(true);
  const [diagnostics, setDiagnostics] = useState(false);
  const [subtitles, setSubtitles] = useState(true);
  const [topic, setTopic] = useState('');
  const [previewReady, setPreviewReady] = useState(false);
  const [pageActive, setPageActive] = useState(true);
  const pageActiveRef = useRef(true);
  const [previewFolded, setPreviewFolded] = useState(false);
  const previewStream = useRef<MediaStream | null>(null);
  const [previewSize, setPreviewSize] = useState('');
  const video = useRef<HTMLVideoElement>(null);
  const videoBox = useRef<HTMLDivElement>(null);
  const live = controller.busy;
  const fc = definition.protocol === 'fc';
  const chat = definition.protocol === 'chat';
  const wantsVideo = definition.video || (fc && settings.fcVideo);
  const active = ['listening', 'speaking'].includes(view.phase);
  useEffect(() => {
    if (!view.startedAt || !live) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [view.startedAt, live]);
  useEffect(() => controller.updateSettings(settings), [controller, settings]);
  useEffect(() => {
    const hide = () => {
      pageActiveRef.current = false;
      setPageActive(false);
      previewStream.current?.getTracks().forEach((track) => track.stop());
      previewStream.current = null;
      setPreviewReady(false);
    };
    const show = () => {
      pageActiveRef.current = true;
      setPageActive(true);
    };
    window.addEventListener('pagehide', hide);
    window.addEventListener('pageshow', show);
    return () => {
      window.removeEventListener('pagehide', hide);
      window.removeEventListener('pageshow', show);
    };
  }, []);
  useEffect(() => {
    if (
      !pageActive ||
      !pageActiveRef.current ||
      !wantsVideo ||
      live ||
      settings.inputMode !== 'live' ||
      !video.current
    )
      return;
    let cancelled = false;
    const element = video.current;
    let acquired: MediaStream | null = null;
    setPreviewReady(false);
    void navigator.mediaDevices
      ?.getUserMedia({
        audio: false,
        video: {
          facingMode: settings.frontCamera ? 'user' : 'environment',
          ...(fc ? { width: { ideal: 1280 }, height: { ideal: 720 } } : {}),
        },
      })
      .then(async (stream) => {
        if (cancelled || !pageActiveRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        acquired = stream;
        previewStream.current = stream;
        element.removeAttribute('src');
        element.srcObject = stream;
        element.style.transform = (fc ? settings.mirror : settings.frontCamera !== settings.mirror)
          ? 'scaleX(-1)'
          : 'none';
        await element.play();
        if (!cancelled) {
          setPreviewReady(true);
          setPreviewSize(`${element.videoWidth} × ${element.videoHeight}`);
        }
      })
      .catch((error) => {
        acquired?.getTracks().forEach((track) => track.stop());
        if (element.srcObject === acquired) element.srcObject = null;
        if (previewStream.current === acquired) previewStream.current = null;
        if (!cancelled && pageActiveRef.current) {
          controller.setError(error);
          if (fc) onSettingsRef.current({ fcVideo: false });
        }
      });
    return () => {
      cancelled = true;
      acquired?.getTracks().forEach((track) => track.stop());
      if (element.srcObject === acquired) element.srcObject = null;
      if (previewStream.current === acquired) previewStream.current = null;
    };
  }, [
    controller,
    wantsVideo,
    live,
    settings.inputMode,
    settings.frontCamera,
    settings.mirror,
    fc,
    pageActive,
  ]);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const onSettingsRef = useRef(onSettings);
  onSettingsRef.current = onSettings;
  useEffect(() => {
    controller.onDraft = setDraft;
    controller.onSettings = (patch) => onSettingsRef.current(patch);
    return () => {
      controller.onSettings = undefined;
      controller.onDraft = undefined;
    };
  }, [controller]);
  useEffect(() => {
    if (fc && !sceneReady) return;
    const current = settingsRef.current;
    let active = true;
    void controller
      .loadPreset(
        current,
        current.preset && current.preset !== 'custom' ? current.preset : undefined,
        'initialize',
      )
      .then((patch) => {
        if (active)
          onSettingsRef.current(
            !fc
              ? patch
              : current.configReady || current.preset === 'custom'
                ? {
                    ...patch,
                    ...current,
                    apiBase: settingsRef.current.apiBase,
                    modelId: settingsRef.current.modelId,
                  }
                : patch,
          );
      })
      .catch((error) => {
        if (active) controller.setError(error);
      });
    return () => {
      active = false;
    };
  }, [controller, sceneReady, fc]);
  useEffect(() => {
    let active = true;
    const load = async () => {
      if (!settingsRef.current.apiBase) return;
      const patch = await controller.loadModels(settingsRef.current);
      if (active && Object.keys(patch).length) onSettingsRef.current(patch);
    };
    const delay = setTimeout(() => void load(), 450);
    const timer = setInterval(() => void load(), 30000);
    return () => {
      active = false;
      clearTimeout(delay);
      clearInterval(timer);
    };
  }, [controller, settings.apiBase]);
  const elapsed = view.startedAt ? Math.max(0, Math.floor((now - view.startedAt) / 1000)) : 0;
  const duration = `${String(Math.floor(elapsed / 60)).padStart(2, '0')}:${String(elapsed % 60).padStart(2, '0')}`;
  const currentText = [...view.messages].reverse().find((m) => m.role === 'assistant')?.text;
  async function start() {
    if (view.configLoading || view.recordingPending) return;
    if (!fc && !settings.configReady) {
      setDeveloper(true);
      controller.setError('请先加载此体验的应用配置。');
      return;
    }
    if (!settings.apiBase || !settings.modelId || (fc && !settings.segmentsJson)) {
      setDeveloper(true);
      controller.setError(
        !settings.apiBase || !settings.modelId
          ? '请填写 API 地址并选择模型。'
          : '先加载当前场景配置，即可开始体验。',
      );
      return;
    }
    try {
      previewStream.current?.getTracks().forEach((track) => track.stop());
      previewStream.current = null;
      setPreviewReady(false);
      await controller.start(settings, video.current);
    } catch (error) {
      controller.setError(error);
      setDeveloper(true);
    }
  }
  async function send(text: string, files: File[], ordered?: ContentPart[]) {
    if (!settings.configReady) {
      setDeveloper(true);
      controller.setError('请先加载此体验的应用配置。');
      return false;
    }
    if (!settings.apiBase || !settings.modelId) {
      setDeveloper(true);
      controller.setError('请填写 API 地址并选择模型。');
      return false;
    }
    try {
      return await controller.send(settings, text, files, ordered);
    } catch (error) {
      controller.setError(error);
      return false;
    }
  }
  const working = chat ? view.messages.length > 0 : live || view.dirty;
  return (
    <div className={`experience ${developer ? 'with-developer' : ''}`}>
      <main
        className={`workspace ${fc ? 'fc-workspace' : ''}`}
        id="experience-panel"
        role="tabpanel"
        aria-labelledby={`tab-${definition.id}`}
      >
        <header className="workspace-header">
          <div className="breadcrumb">
            <strong>{definition.label}</strong>
          </div>
          <div className="workspace-actions">
            <EndpointPicker
              definition={definition}
              settings={settings}
              onSettings={onSettings}
              controller={controller}
              view={view}
              openSettings={() => setDeveloper(true)}
            />
            <span className="header-divider" />
            <IconButton label="使用说明" onClick={openHelp}>
              <CircleHelp size={17} />
            </IconButton>
            <button
              className={`developer-toggle ${developer ? 'pressed' : ''}`}
              onClick={() => setDeveloper(!developer)}
              aria-expanded={developer}
            >
              <DesignIcon name="settings" size={18} />
              <span>开发者模式</span>
            </button>
          </div>
        </header>
        <div className="workspace-content">
          <div className="view-heading">
            <span className="mode-pill">
              <definition.icon size={13} />
              {chat ? '多模态对话' : fc ? '实时 Agent' : wantsVideo ? '实时视觉' : '实时语音'}
            </span>
            <div className={`connection-status ${active ? 'is-live' : ''}`}>
              <span className="status-dot" />
              {phaseLabels[view.phase]}
              {live && view.startedAt && <time>{duration}</time>}
            </div>
          </div>
          <EndpointNotice
            settings={settings}
            controller={controller}
            openSettings={() => setDeveloper(true)}
          />
          {view.error && (
            <div className="error-banner" role="alert">
              <span>{view.error}</span>
              <IconButton label="关闭提示" onClick={() => controller.dismissError()}>
                <X size={15} />
              </IconButton>
            </div>
          )}
          {fc && (
            <div className="fc-scene-heading">
              <div>
                <h2>{definition.title}</h2>
                <p>{definition.description}</p>
              </div>
              <button className="secondary-button" onClick={() => setDiagnostics(true)}>
                <Activity size={15} />
                协议与指标
              </button>
            </div>
          )}
          {fc && !working && (
            <div className="fc-prompt-row" aria-label="试试这些话题">
              {definition.prompts.map((prompt) => (
                <button
                  key={prompt}
                  className={topic === prompt ? 'selected' : ''}
                  title="选择后，开始对话时可以试着说"
                  onClick={() =>
                    definition.id === 'standalone' ? setDraft(prompt) : setTopic(prompt)
                  }
                >
                  {prompt}
                </button>
              ))}
              {topic && <span>开始后试着说：{topic}</span>}
            </div>
          )}
          <div
            className={`content-stage ${working || fc ? 'working' : 'welcoming'} ${fc ? 'fc-stage' : ''} ${wantsVideo ? 'with-video' : ''}`}
          >
            {!working && !fc && (
              <section className="welcome">
                <Orb />
                <div className="welcome-copy">
                  <h1>{definition.title}</h1>
                  <p>{definition.description}</p>
                </div>
                {(chat || fc) && (
                  <div className="prompt-grid">
                    {definition.prompts.map((prompt, i) => {
                      const PromptIcon = definition.promptIcons?.[i] || definition.icon;
                      return (
                        <button
                          key={prompt}
                          className="prompt-card"
                          onClick={() =>
                            chat
                              ? setDraft(prompt)
                              : definition.id === 'standalone'
                                ? setDraft(prompt)
                                : setTopic(prompt)
                          }
                        >
                          <span className={`prompt-number n${i}`}>
                            {chat ? (
                              <DesignIcon
                                name={(['document', 'palette', 'code'] as const)[i]}
                                size={21}
                              />
                            ) : (
                              <PromptIcon size={21} strokeWidth={1.5} />
                            )}
                          </span>
                          <span>{prompt}</span>
                          <CirclePlus size={20} strokeWidth={1.2} />
                        </button>
                      );
                    })}
                  </div>
                )}
                {fc && (
                  <div className="welcome-note">
                    {topic
                      ? `开始对话后，可以试着说：${topic}`
                      : '选择一个话题，开始后用语音交代任务'}
                  </div>
                )}
              </section>
            )}
            {chat && working && (
              <Messages
                messages={view.messages}
                canRegenerate={!live && !view.configLoading}
                onRegenerate={(id) =>
                  void controller.regenerate(settings, id).catch((e) => controller.setError(e))
                }
              />
            )}
            {wantsVideo && (
              <div
                className={`video-panel ${working || previewReady ? 'shown' : 'dormant'} ${previewFolded ? 'folded' : ''}`}
                ref={videoBox}
              >
                <video ref={video} playsInline muted autoPlay />
                <div className="video-label">
                  <Camera size={13} />
                  {view.camera
                    ? '实时画面'
                    : previewReady
                      ? `预览 · ${previewSize}`
                      : '摄像头未开启'}
                </div>
                {!view.camera && !previewReady && (
                  <div className="video-empty">
                    <Camera size={28} />
                    <p>开始后，共享眼前的画面</p>
                  </div>
                )}
                {live && subtitles && (
                  <div
                    className="fullscreen-subtitles"
                    style={{ height: `${settings.screenSubtitleHeight}%` }}
                  >
                    {view.messages
                      .filter((m) => m.role === 'assistant')
                      .slice(-12)
                      .map((m, i, all) => (
                        <p
                          key={m.id}
                          style={{
                            opacity:
                              all.length === 1
                                ? settings.screenSubtitleBottom / 100
                                : (settings.screenSubtitleTop +
                                    ((settings.screenSubtitleBottom - settings.screenSubtitleTop) *
                                      i) /
                                      (all.length - 1)) /
                                  100,
                          }}
                        >
                          {m.text}
                        </p>
                      ))}
                  </div>
                )}
                {live && (
                  <IconButton
                    className="video-subtitles"
                    label={subtitles ? '隐藏字幕' : '显示字幕'}
                    onClick={() => setSubtitles(!subtitles)}
                  >
                    <Captions size={16} />
                  </IconButton>
                )}
                <button
                  className="preview-fold text-button"
                  onClick={() => setPreviewFolded(!previewFolded)}
                >
                  {previewFolded ? '展开预览' : '折叠预览'}
                </button>
                {!live && previewReady && (
                  <div className="preview-controls">
                    <IconButton
                      label="翻转摄像头"
                      onClick={() =>
                        onSettings({
                          frontCamera: !settings.frontCamera,
                          ...(fc ? { mirror: !settings.frontCamera } : {}),
                        })
                      }
                    >
                      <SwitchCamera size={16} />
                    </IconButton>
                    <IconButton
                      label="镜像画面"
                      onClick={() => onSettings({ mirror: !settings.mirror })}
                    >
                      <FlipHorizontal size={16} />
                    </IconButton>
                  </div>
                )}
                <div className="fullscreen-controls">
                  <button
                    className="secondary-button"
                    onClick={() => (live ? controller.stop() : void start())}
                  >
                    {live ? '结束' : '开始'}
                  </button>
                  {active && !fc && (
                    <>
                      <button className="secondary-button" onClick={() => controller.pause()}>
                        {view.paused ? '继续' : '暂停'}
                      </button>
                      <button className="secondary-button" onClick={() => controller.forceListen()}>
                        {view.forceListen ? '取消强制聆听' : '强制聆听'}
                      </button>
                      <button
                        className="secondary-button"
                        disabled={settings.visionHD === 'strategy'}
                        onClick={() => controller.toggleHD()}
                      >
                        HD {view.hdEnabled ? '开' : '关'}
                      </button>
                    </>
                  )}
                  <button
                    className="secondary-button"
                    onClick={() => void document.exitFullscreen()}
                  >
                    退出全屏
                  </button>
                </div>
                <IconButton
                  className="video-expand"
                  label="全屏预览"
                  onClick={() => {
                    videoBox.current
                      ?.requestFullscreen?.()
                      .catch((error) => controller.setError(error));
                  }}
                >
                  <Maximize2 size={16} />
                </IconButton>
              </div>
            )}
            {fc && (
              <SceneHost
                definition={definition}
                controller={controller}
                settings={settings}
                visible={fc || working}
                onReady={() => setSceneReady(true)}
              />
            )}
            {!chat && !fc && working && !wantsVideo && (
              <section className="voice-stage">
                <Orb active={active} level={view.micLevel} />
                <span className="voice-stage-label">
                  {view.paused
                    ? '已暂停'
                    : view.forceListen
                      ? '强制聆听'
                      : view.muted
                        ? '麦克风已静音'
                        : phaseLabels[view.phase]}
                </span>
                <p>
                  {topic && view.phase === 'listening'
                    ? `试着说：${topic}`
                    : view.phase === 'speaking'
                      ? '你可以随时开口打断'
                      : view.phase === 'listening'
                        ? '我在，继续说吧'
                        : view.detail || '我们准备好时，就开始聊聊'}
                </p>
                {currentText && <div className="live-caption">{currentText}</div>}
              </section>
            )}
          </div>
          {fc && <FcConversation view={view} />}
          {!chat && !fc && working && view.messages.length > 0 && (
            <div className="transcript-section">
              <button
                className="transcript-heading"
                onClick={() => setShowTranscript(!showTranscript)}
              >
                <AudioLines size={13} />
                对话记录
                <ChevronDown
                  size={13}
                  style={{ transform: showTranscript ? undefined : 'rotate(-90deg)' }}
                />
              </button>
              {showTranscript && <Messages messages={view.messages} compact />}
            </div>
          )}
          {!chat && !fc && wantsVideo && currentText && (
            <div className="video-caption">
              <Volume2 size={14} />
              <span>{currentText}</span>
            </div>
          )}
        </div>
        {fc && diagnostics && (
          <Dialog title="协议事件与运行指标" onClose={() => setDiagnostics(false)}>
            <div className="fc-diagnostics">
              <ProtocolEvents controller={controller} definition={definition} view={view} open />
              <RuntimeMetrics definition={definition} view={view} />
            </div>
          </Dialog>
        )}
        <footer className={`interaction-dock ${chat ? 'chat-dock' : ''}`}>
          {chat ? (
            <Composer
              disabled={view.configLoading}
              value={draft}
              onChange={setDraft}
              onSend={send}
              onStop={() => controller.stop()}
              busy={live}
              placeholder="发送消息，或添加图片、音频、视频…"
            />
          ) : (
            <>
              {definition.id === 'standalone' && (
                <Composer
                  value={draft}
                  onChange={setDraft}
                  onSend={send}
                  onStop={() => controller.stop()}
                  busy={false}
                  disabled={!active}
                  attachments={false}
                  placeholder="输入要注入的外部消息…"
                />
              )}
              <div className="live-controls">
                <div className="dock-side">
                  <span className="dock-live-dot" />
                  {live ? (
                    <span>
                      {view.paused
                        ? '已暂停'
                        : view.forceListen
                          ? '强制聆听'
                          : view.muted
                            ? '麦克风已静音'
                            : phaseLabels[view.phase]}
                    </span>
                  ) : (
                    <span>准备好，就开始吧</span>
                  )}
                </div>
                <div className="main-live-controls">
                  {active && !fc && (
                    <>
                      <IconButton
                        label={view.paused ? '继续输入' : '暂停输入'}
                        onClick={() => controller.pause()}
                      >
                        {view.paused ? <Play size={18} /> : <Pause size={18} />}
                      </IconButton>
                      <IconButton
                        label={view.forceListen ? '取消强制聆听' : '强制聆听'}
                        className={view.forceListen ? 'muted' : ''}
                        onClick={() => controller.forceListen()}
                      >
                        <Ear size={18} />
                      </IconButton>
                    </>
                  )}
                  {active && wantsVideo && settings.inputMode === 'live' && (
                    <>
                      <IconButton
                        label="翻转摄像头"
                        onClick={() =>
                          void controller.flipCamera().catch((e) => controller.setError(e))
                        }
                      >
                        <SwitchCamera size={18} />
                      </IconButton>
                      <IconButton label="镜像画面" onClick={() => controller.mirrorCamera()}>
                        <FlipHorizontal size={18} />
                      </IconButton>
                    </>
                  )}
                  {active && definition.video && (
                    <button
                      className="secondary-button"
                      disabled={settings.visionHD === 'strategy'}
                      onClick={() => controller.toggleHD()}
                    >
                      {settings.visionHD === 'strategy'
                        ? 'Strategy HD'
                        : view.hdEnabled
                          ? 'HD 开'
                          : 'HD 关'}
                    </button>
                  )}

                  {active &&
                    (settings.inputMode === 'live' ||
                      (!fc && settings.fileAudioMode !== 'file')) && (
                      <IconButton
                        label={view.muted ? '取消静音' : '静音麦克风'}
                        className={view.muted ? 'muted' : ''}
                        onClick={() => controller.mute()}
                      >
                        {view.muted ? <MicOff size={19} /> : <Mic size={19} />}
                      </IconButton>
                    )}
                  <button
                    className={`start-button ${live ? 'stop' : ''}`}
                    onClick={() => (live ? controller.stop() : void start())}
                    aria-label={live ? '结束对话' : '开始对话'}
                    title={live ? '结束对话' : '开始对话'}
                    disabled={view.configLoading || view.recordingPending || (fc && !sceneReady)}
                  >
                    {live ? (
                      <>
                        <Square size={15} fill="currentColor" />
                        <span>结束对话</span>
                      </>
                    ) : (
                      <>
                        <img src={microphone} alt="" className="start-microphone" />
                      </>
                    )}
                  </button>
                  {live && (
                    <span
                      className={`audio-bars ${view.muted ? 'silent' : ''}`}
                      aria-label="麦克风音量"
                    >
                      {[0.45, 0.8, 1, 0.65, 0.4].map((factor, i) => (
                        <i
                          key={i}
                          style={{ height: `${5 + Math.min(view.micLevel * 400, 24) * factor}px` }}
                        />
                      ))}
                    </span>
                  )}
                </div>
                <div className="dock-side right">
                  <button onClick={() => setDeveloper(true)}>
                    <Plug size={14} />
                    {settings.apiBase ? '连接设置' : '配置连接'}
                  </button>
                </div>
              </div>
              <div className="live-footnote">
                {live
                  ? '结束对话后，麦克风与摄像头将自动释放'
                  : '点击麦克风开始对话' + (wantsVideo ? ' · 将使用摄像头' : '')}
              </div>
            </>
          )}
        </footer>
      </main>
      {developer && (
        <DeveloperPanel
          definition={definition}
          settings={settings}
          onSettings={onSettings}
          controller={controller}
          view={view}
          onClose={() => setDeveloper(false)}
          sceneReady={sceneReady}
        />
      )}
    </div>
  );
}
