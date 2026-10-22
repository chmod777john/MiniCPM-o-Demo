import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import {
  AudioLines,
  Camera,
  ChevronDown,
  CircleHelp,
  Maximize2,
  Mic,
  MicOff,
  Plug,
  CirclePlus,
  Square,
  Volume2,
  X,
} from 'lucide-react';
import type { CaseDefinition, SessionPhase, SessionSettings } from '../shared/session/types';
import { SessionController } from '../shared/session/controller';
import { DeveloperPanel } from '../shared/ui/DeveloperPanel';
import { SceneHost } from '../shared/ui/SceneHost';
import microphone from '../shared/assets/figma-microphone.svg';
import { DesignIcon } from '../shared/ui/DesignIcon';
import { Orb } from '../shared/ui/Orb';
import { Messages } from '../shared/ui/Messages';
import { Composer } from '../shared/ui/Composer';
import { IconButton } from '../shared/ui/controls';

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
  const [topic, setTopic] = useState('');
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
  const elapsed = view.startedAt ? Math.max(0, Math.floor((now - view.startedAt) / 1000)) : 0;
  const duration = `${String(Math.floor(elapsed / 60)).padStart(2, '0')}:${String(elapsed % 60).padStart(2, '0')}`;
  const currentText = [...view.messages].reverse().find((m) => m.role === 'assistant')?.text;
  async function start() {
    if (!settings.endpoint || (fc && !settings.segmentsJson)) {
      setDeveloper(true);
      controller.setError(
        !settings.endpoint
          ? '先填写模型 API 地址，即可开始体验。'
          : '先加载当前场景配置，即可开始体验。',
      );
      return;
    }
    try {
      await controller.start(settings, video.current);
    } catch (error) {
      controller.setError(error);
      setDeveloper(true);
    }
  }
  async function send(text: string, files: File[]) {
    if (!settings.endpoint) {
      setDeveloper(true);
      controller.setError('请先填写模型 API 地址。');
      return false;
    }
    try {
      return await controller.send(settings, text, files);
    } catch (error) {
      controller.setError(error);
      return false;
    }
  }
  const working = chat ? view.messages.length > 0 : live || view.dirty;
  return (
    <div className={`experience ${developer ? 'with-developer' : ''}`}>
      <main
        className="workspace"
        id="experience-panel"
        role="tabpanel"
        aria-labelledby={`tab-${definition.id}`}
      >
        <header className="workspace-header">
          <div className="breadcrumb">
            <strong>{definition.label}</strong>
          </div>
          <div className="workspace-actions">
            <button
              className="endpoint-chip"
              onClick={() => setDeveloper(!developer)}
              title="模型与连接设置"
            >
              <span className={`status-dot ${active ? 'connected' : ''}`} />
              <span>{view.model ? view.model.split('/').pop() : 'MiniCPM-o 5'}</span>
              <ChevronDown size={12} />
            </button>
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
          {view.error && (
            <div className="error-banner" role="alert">
              <span>{view.error}</span>
              <IconButton label="关闭提示" onClick={() => controller.dismissError()}>
                <X size={15} />
              </IconButton>
            </div>
          )}
          <div
            className={`content-stage ${working ? 'working' : 'welcoming'} ${fc ? 'fc-stage' : ''} ${wantsVideo ? 'with-video' : ''}`}
          >
            {!working && (
              <section className="welcome">
                <Orb />
                <div className="welcome-copy">
                  <h1>{definition.title}</h1>
                  <p>{definition.description}</p>
                </div>
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
                {!chat && (
                  <div className="welcome-note">
                    {topic
                      ? `开始对话后，可以试着说：${topic}`
                      : fc
                        ? '选择一个话题，开始后用语音交代任务'
                        : '选择一个话题，开始后直接开口交流'}
                  </div>
                )}
              </section>
            )}
            {chat && working && <Messages messages={view.messages} />}
            {wantsVideo && (
              <div className={`video-panel ${working ? 'shown' : 'dormant'}`} ref={videoBox}>
                <video ref={video} playsInline muted autoPlay />
                <div className="video-label">
                  <Camera size={13} />
                  {view.camera ? '实时画面' : '摄像头未开启'}
                </div>
                {!view.camera && (
                  <div className="video-empty">
                    <Camera size={28} />
                    <p>开始后，共享眼前的画面</p>
                  </div>
                )}
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
                visible={working}
                onReady={() => setSceneReady(true)}
              />
            )}
            {!chat && !fc && working && !wantsVideo && (
              <section className="voice-stage">
                <Orb active={active} level={view.micLevel} />
                <span className="voice-stage-label">
                  {view.muted ? '麦克风已静音' : phaseLabels[view.phase]}
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
          {!chat && fc && working && view.messages.length > 0 && (
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
        <footer className={`interaction-dock ${chat ? 'chat-dock' : ''}`}>
          {chat ? (
            <Composer
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
                    <span>{view.muted ? '麦克风已静音' : phaseLabels[view.phase]}</span>
                  ) : (
                    <span>准备好，就开始吧</span>
                  )}
                </div>
                <div className="main-live-controls">
                  {live && (
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
                    disabled={fc && !sceneReady}
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
                    {settings.endpoint ? '连接设置' : '配置连接'}
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
