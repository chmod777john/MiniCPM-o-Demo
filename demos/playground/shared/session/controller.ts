import { RealtimeSession } from '../../../../static/duplex/lib/realtime-session.js';
import { FcRealtimeClient } from '../../../../static/demo-gallery/core/realtime-client.js';
import { LiveMicProvider } from '../../../../static/demo-gallery/core/live-mic-provider.js';
import { CameraCapture } from '../../../../static/demo-gallery/core/camera-capture.js';
import { AudioPlayer } from '../../../../static/demo-gallery/core/audio-player.js';
import { textFromSteps } from '../../../../static/demo-gallery/core/ui-utils.js';
import { ChatTransport } from '../../turnbased/transport';
import { base64, contentFiles } from '../media/files';
import { applicationUrl, endpointBase, readJson, realtimeUrl } from './urls';
import {
  emptyView,
  type CaseDefinition,
  type Message,
  type Scene,
  type SceneFactory,
  type SessionSettings,
  type SessionView,
} from './types';

const uid = () => crypto.randomUUID();
const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** DOM-independent session store. Desktop and future mobile layouts subscribe to the same state. */
export class SessionController {
  private view = emptyView();
  private listeners = new Set<() => void>();
  private epoch = 0;
  private alive = true;
  private eventNumber = 0;
  private currentSpeech = '';
  private mic: any = null;
  private camera: any = null;
  private ordinary: any = null;
  private fc: any = null;
  private chat: ChatTransport | null = null;
  private player: any = null;
  private scene: Scene | null = null;
  private sceneFactory: SceneFactory | null = null;
  private stage: HTMLElement | null = null;
  private fcTimer?: ReturnType<typeof setTimeout>;
  private fetchAbort?: AbortController;
  defaults: any = null;
  presets: { id: string; label: string }[] = [];
  constructor(readonly definition: CaseDefinition) {}
  snapshot = () => this.view;
  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };
  private update(patch: Partial<SessionView>) {
    if (!this.alive) return;
    this.view = { ...this.view, ...patch };
    this.listeners.forEach((fn) => fn());
  }
  private log(type: string, text: string) {
    this.update({
      events: [
        ...this.view.events.slice(-149),
        {
          id: ++this.eventNumber,
          type,
          text: text.slice(0, 1600),
          time: new Date().toLocaleTimeString('zh-CN', { hour12: false }),
        },
      ],
    });
  }
  setError(error: unknown) {
    this.update({ error: messageOf(error) });
  }
  dismissError() {
    this.update({ error: '' });
  }
  get busy() {
    return !['idle', 'error'].includes(this.view.phase);
  }
  private speech(text: string, final = false) {
    if (!text && !final) return;
    if (!this.currentSpeech && text) {
      this.currentSpeech = uid();
      this.update({
        messages: [
          ...this.view.messages,
          { id: this.currentSpeech, role: 'assistant', text: '', pending: true },
        ],
      });
    }
    const id = this.currentSpeech;
    if (id)
      this.update({
        messages: this.view.messages.map((m) =>
          m.id === id ? { ...m, text: m.text + text, pending: !final } : m,
        ),
      });
    if (final) this.currentSpeech = '';
  }
  private async sceneSetup(settings: SessionSettings, activeEpoch = this.epoch) {
    if (!this.sceneFactory || !this.stage) return;
    const element = document.createElement('div');
    element.className = 'stage';
    this.scene?.dispose?.();
    this.stage.replaceChildren(element);
    const root = applicationUrl(settings.gallery, `/api/demos/${this.definition.id}`);
    const assets = applicationUrl(settings.gallery, `/demos/${this.definition.id}/web`);
    this.scene = this.sceneFactory.create({
      stageEl: element,
      demoId: this.definition.id,
      getClient: () => (this.alive && this.epoch === activeEpoch ? this.fc : null),
      api: { base: root, url: (path) => `${root}/${path.replace(/^\//, '')}` },
      assets: { base: assets, url: (path) => `${assets}/${path.replace(/^\//, '')}` },
      shell: {
        setStatus: (text) => {
          if (this.epoch === activeEpoch) this.update({ detail: text });
        },
      },
    });
    this.scene.render();
    if (this.defaults) this.scene.buildQuickSettings?.(this.defaults);
  }
  async mountScene(stage: HTMLElement, factory: SceneFactory, settings: SessionSettings) {
    this.stage = stage;
    this.sceneFactory = factory;
    await this.sceneSetup(settings);
    this.presets = this.scene?.presetOptions || [];
  }
  unmountScene() {
    this.scene?.dispose?.();
    this.scene = null;
    this.stage = null;
  }
  async loadPreset(settings: SessionSettings, preset?: string): Promise<Partial<SessionSettings>> {
    this.fetchAbort?.abort();
    const abort = new AbortController();
    this.fetchAbort = abort;
    const defaults = await readJson(
      applicationUrl(settings.gallery, `/api/demos/${this.definition.id}/defaults`),
      abort.signal,
    );
    if (abort.signal.aborted || !this.alive) return {};
    this.defaults = defaults;
    await this.sceneSetup(settings);
    const presets = this.scene?.buildQuickSettings?.(defaults);
    const selected = preset || this.scene?.fallbackQuickSetting || this.presets[0]?.id || '';
    const value = presets?.get(selected);
    if (!value && this.presets.length)
      throw new Error('工具服务尚未配置这个预设，请选择可用预设或检查服务配置。');
    const tools = this.scene?.tools?.(defaults, selected) || defaults.system?.tools || [];
    const segments = value?.segments || defaults.system?.segments || [];
    this.log('配置', '已从工具服务读取场景预设');
    return {
      preset: selected,
      segmentsJson: JSON.stringify(segments, null, 2),
      toolsJson: JSON.stringify(tools, null, 2),
      referencePath: value?.ttsPromptAudioPath || defaults.tts_prompt_audio?.file_path || '',
      ...(!settings.endpoint && defaults.default_api_endpoint
        ? { endpoint: defaults.default_api_endpoint }
        : {}),
    };
  }
  async inspectEndpoint(settings: SessionSettings) {
    const data = await readJson(`${endpointBase(settings.endpoint)}/v1/info`);
    if (!this.alive) return;
    const model =
      data.model?.name || data.model?.profile_id || data.model?.checkpoint || data.model_id || '';
    this.update({ model: String(model), detail: 'API 信息已读取' });
    this.log('API', '已读取模型信息');
    return data;
  }
  private async resolveAudio(settings: SessionSettings, path: string) {
    const audio = await readJson(
      applicationUrl(settings.gallery, `/api/demos/_audio?path=${encodeURIComponent(path)}`),
    );
    if (!audio.data) throw new Error('工具服务未返回参考音频。');
    return { source: 'base64', format: 'wav', data: audio.data };
  }
  private async fcPayload(settings: SessionSettings) {
    if (!settings.segmentsJson.trim())
      throw new Error('请先加载场景预设，或在开发者面板填写 system segments。');
    let segments = JSON.parse(settings.segmentsJson);
    const tools = JSON.parse(settings.toolsJson || '[]');
    if (!Array.isArray(segments) || !Array.isArray(tools))
      throw new Error('System segments 和 tools 必须是 JSON 数组。');
    segments = (await this.scene?.prepareSystem?.(segments)) || segments;
    segments = await Promise.all(
      segments.map(async (segment: any) => {
        if (segment.kind !== 'audio' || segment.audio?.source !== 'path') return segment;
        return { ...segment, audio: await this.resolveAudio(settings, segment.audio.file_path) };
      }),
    );
    let reference = settings.referenceAudio
      ? { source: 'base64', format: 'wav', data: settings.referenceAudio.data }
      : null;
    if (!reference && settings.referencePath)
      reference = await this.resolveAudio(settings, settings.referencePath);
    if (settings.generateAudio && !reference)
      throw new Error('请上传参考音色，或加载包含参考音频的场景预设。');
    return {
      protocol_version: '5',
      mode: 'full_duplex',
      fc_duplex: true,
      system: { segments, tools },
      ...(reference ? { tts_prompt_audio: reference } : {}),
      generate_audio: settings.generateAudio,
      config: {
        runtime: 'fc_duplex',
        auto_execute_tools: false,
        sample_rate: 16000,
        max_spoken_tokens: 24,
        decode_mode: 'greedy',
        decoding_policy: 'constrained_decoding',
      },
    };
  }
  async start(settings: SessionSettings, video: HTMLVideoElement | null) {
    if (this.busy) return;
    endpointBase(settings.endpoint);
    this.stop();
    const epoch = this.epoch;
    const current = () => this.alive && this.epoch === epoch;
    this.update({ phase: 'connecting', error: '', detail: '正在准备连接', dirty: true });
    const wantsVideo = Boolean(
      this.definition.video || (this.definition.protocol === 'fc' && settings.fcVideo),
    );
    const fail = (error: unknown) => {
      if (current()) {
        this.stop();
        this.update({ phase: 'error', error: messageOf(error) });
      }
    };
    try {
      if (this.definition.protocol === 'fc') {
        this.player = new (AudioPlayer as any)({
          getPlaybackDelayMs: () => settings.playbackDelay,
        });
        this.player.init();
        await this.player.ctx?.resume();
      }
      if (!current()) return;
      if (wantsVideo) {
        if (!video) throw new Error('视频预览尚未准备好。');
        const camera = new (CameraCapture as any)(video);
        this.camera = camera;
        await camera.open();
        if (!current()) {
          camera.close();
          return;
        }
        camera.lockCaptureSize();
        this.update({ camera: true });
      }
      const mic = new (LiveMicProvider as any)({
        onChunk: (samples: Float32Array) => {
          if (!current()) return;
          try {
            const audio = base64(
              (this.view.muted ? new Float32Array(samples.length) : samples).buffer,
            );
            const frame = wantsVideo ? this.camera?.captureFrame() : null;
            if (wantsVideo && !frame) throw new Error('摄像头画面中断，请重新连接。');
            if (this.fc?.ready) this.fc.appendAudio({ audioBase64: audio, videoFrame: frame });
            if (this.ordinary?.running)
              this.ordinary.sendChunk({
                audio_base64: audio,
                ...(frame ? { frame_base64_list: [frame], max_slice_nums: 1 } : {}),
              });
          } catch (error) {
            fail(error);
          }
        },
        onLevel: (value: number) => {
          if (current()) this.update({ micLevel: this.view.muted ? 0 : value });
        },
        onPermissionHint: (text: string) => {
          if (current()) this.update({ detail: text });
        },
        onState: () => {},
        onStatus: () => {},
      });
      this.mic = mic;
      await mic.start();
      if (!current()) {
        mic.stop();
        return;
      }
      if (this.definition.protocol === 'ordinary') {
        const session = new (RealtimeSession as any)('playground', {
          getWsUrl: () => realtimeUrl(settings.endpoint, wantsVideo ? 'video' : 'audio'),
          getPlaybackDelayMs: () => settings.playbackDelay,
          enableReconnect: true,
        });
        this.ordinary = session;
        session.audioPlayer.init();
        await session.audioPlayer.ctx?.resume();
        if (!current()) {
          session.cancelQueue();
          void session.audioPlayer.ctx?.close().catch(() => {});
          return;
        }
        // Cleanup may be called by the transport itself, including a terminal server close.
        // Defer to avoid re-entering cleanup while it is still releasing its resources.
        session.onCleanup = () =>
          queueMicrotask(() => {
            if (current()) fail(new Error('会话已结束，请重新开始。'));
          });
        session.onQueueUpdate = (queue: any) => {
          if (current() && queue) this.update({ phase: 'queued', detail: '正在等待服务接入' });
        };
        session.onSpeakStart = (text: string) => {
          if (current()) {
            this.update({ phase: 'speaking' });
            this.speech(text);
          }
          return {};
        };
        session.onSpeakUpdate = (_: any, text: string) => {
          if (!current()) return;
          if (!this.currentSpeech) this.speech('');
          if (!this.currentSpeech) {
            this.currentSpeech = uid();
            this.update({
              messages: [
                ...this.view.messages,
                { id: this.currentSpeech, role: 'assistant', text, pending: true },
              ],
            });
          } else
            this.update({
              messages: this.view.messages.map((m) =>
                m.id === this.currentSpeech ? { ...m, text } : m,
              ),
            });
        };
        session.onSpeakEnd = () => {
          if (current()) {
            this.speech('', true);
            this.update({ phase: 'listening' });
          }
        };
        session.onListenResult = () => {
          if (current()) this.update({ phase: 'listening' });
        };
        session.onProtocolEvent = (event: any) => {
          if (current() && event.type !== 'input.append') this.log(event.type, event.summary || '');
        };
        session.onMetrics = (data: any) => {
          if (!current()) return;
          if (data.connectionState === 'failed') fail(new Error('连接中断，自动重连未成功。'));
          else if (data.connectionState === 'reconnecting')
            this.update({ phase: 'connecting', detail: '正在恢复连接' });
          else if (data.connectionState === 'connected')
            this.update({ phase: 'listening', detail: '' });
          if (data.chunksSent !== undefined) this.update({ unit: data.chunksSent });
        };
        await session.start(
          settings.systemPrompt,
          {
            config: { ls_decision_mode: 'binary', speak_threshold: 0.3 },
            use_tts: settings.generateAudio,
            ...(settings.referenceAudio ? { ref_audio_base64: settings.referenceAudio.pcm } : {}),
          },
          async () => {
            if (!current()) throw new Error('启动已取消。');
          },
        );
        if (!current()) {
          session.cancelQueue();
          return;
        }
        this.update({
          phase: 'listening',
          startedAt: Date.now(),
          sessionId: session.sessionId,
          detail: '',
        });
      } else {
        await this.sceneSetup(settings, epoch);
        this.scene?.buildQuickSettings?.(this.defaults || {});
        this.scene?.tools?.(this.defaults || {}, settings.preset);
        const payload = await this.fcPayload(settings);
        if (!current()) return;
        const client = new (FcRealtimeClient as any)({
          onEvent: (event: any) => {
            if (!current()) return;
            try {
              this.handleFc(event, fail);
            } catch (error) {
              fail(error);
            }
          },
          onStatus: (text: string) => {
            if (current() && text === 'WebSocket closed')
              fail(new Error('会话连接已断开，请重新开始。'));
          },
        });
        this.fc = client;
        this.fcTimer = setTimeout(
          () => fail(new Error('会话准备超时，请重试或检查模型服务。')),
          180000,
        );
        await client.connect(realtimeUrl(settings.endpoint, wantsVideo ? 'omni' : 'audio'));
        if (!current()) {
          client.disconnect();
          return;
        }
        await this.scene?.reset?.();
        if (!current()) {
          client.disconnect();
          return;
        }
        client.initSession(payload);
      }
    } catch (error) {
      fail(error);
    }
  }
  private handleFc(event: any, fail: (error: unknown) => void) {
    const type = event.type;
    if (type === 'session.queued' || type === 'session.queue_update')
      this.update({ phase: 'queued', detail: '正在等待服务接入' });
    if (type === 'session.created') {
      clearTimeout(this.fcTimer);
      this.update({
        phase: 'listening',
        detail: '',
        startedAt: Date.now(),
        sessionId: event.session_id || '',
        model: event.model?.profile_id || this.view.model,
      });
    }
    if (type === 'error' || type === 'session.failed' || type === 'session.closed') {
      fail(new Error(event.error?.message || event.message || event.reason || '会话已结束'));
      return;
    }
    const mapping: Record<string, keyof Scene> = {
      'response.unit.started': 'onUnitStarted',
      'response.think.begin': 'onThinkBegin',
      'response.think.delta': 'onThinkDelta',
      'response.think.end': 'onThinkEnd',
      'response.tool_call.begin': 'onToolCallBegin',
      'response.tool_call.done': 'onToolCallDone',
      'response.spoken.end': 'onSpokenEnd',
    };
    const name = mapping[type];
    if (name && typeof this.scene?.[name] === 'function') {
      const result = (this.scene[name] as (event: any) => any)(event);
      if (result?.catch) {
        const epoch = this.epoch;
        result.catch((error: unknown) => {
          if (this.epoch === epoch) this.setError(error);
        });
      }
    }
    if (type === 'response.unit.committed') this.update({ unit: Number(event.unit_index) + 1 });
    if (type === 'response.spoken.delta') {
      const text = textFromSteps(event.steps);
      this.speech(text);
      this.scene?.onSpokenDelta?.(event, text);
      if (text || event.audio) this.update({ phase: 'speaking' });
      if (event.audio) {
        if (!this.player.turnActive) this.player.beginTurn();
        this.player.playChunk(event.audio);
      }
    }
    if (type === 'response.spoken.end' && ['listen', 'turn_eos'].includes(event.reason)) {
      this.speech('', true);
      this.player?.endTurn();
      this.update({ phase: 'listening' });
    }
    if (type === 'response.tool_call.delta')
      this.scene?.onToolCallDelta?.(event, textFromSteps(event.steps));
    if (
      ![
        'response.spoken.delta',
        'response.think.delta',
        'response.tool_call.delta',
        'response.unit.started',
        'response.unit.committed',
      ].includes(type)
    )
      this.log(type, event.full_text || event.call?.name || event.reason || '');
  }
  async send(settings: SessionSettings, text: string, files: File[]): Promise<boolean> {
    if (this.definition.protocol !== 'chat') {
      if (this.definition.id !== 'standalone' || !this.fc?.ready)
        throw new Error('请先开始会话，再通过语音与助手交流。');
      const eventId = `pg_${uid()}`;
      this.fc.sendStandaloneEvent({ content: text, eventId });
      this.scene?.onExternalEvent?.(eventId, text);
      this.update({ messages: [...this.view.messages, { id: uid(), role: 'user', text }] });
      return true;
    }
    if (this.busy) return false;
    endpointBase(settings.endpoint);
    const history = this.view.messages.filter(
      (m) => !m.failed && !m.pending && (m.text !== '' || Array.isArray(m.content)),
    );
    this.stop();
    const epoch = this.epoch;
    this.update({ phase: 'generating', error: '', dirty: true });
    try {
      const attachments = await contentFiles(files);
      if (epoch !== this.epoch || !this.alive) return false;
      const content = attachments.length
        ? [...(text ? [{ type: 'text' as const, text }] : []), ...attachments]
        : text;
      const user: Message = {
        id: uid(),
        role: 'user',
        text,
        content,
        files: files.map((f) => f.name),
      };
      const assistantId = uid();
      this.update({
        messages: [
          ...this.view.messages,
          user,
          { id: assistantId, role: 'assistant', text: '', pending: true },
        ],
        startedAt: Date.now(),
      });
      this.player = new (AudioPlayer as any)({ getPlaybackDelayMs: () => settings.playbackDelay });
      this.player.init();
      const updateText = (text: string, append: boolean, pending: boolean) =>
        this.update({
          messages: this.view.messages.map((m) =>
            m.id === assistantId ? { ...m, text: append ? m.text + text : text, pending } : m,
          ),
        });
      const hasVideo =
        attachments.some((a) => a.type === 'video') ||
        history.some((m) => Array.isArray(m.content) && m.content.some((c) => c.type === 'video'));
      let streamed = false;
      this.chat = new ChatTransport({
        onEvent: (event) => {
          if (epoch !== this.epoch) return;
          if (event.type === 'session.created') this.update({ sessionId: event.session_id || '' });
          if (event.type === 'session.queued') this.update({ phase: 'queued' });
          if (event.type === 'response.output.delta') {
            this.update({ phase: 'generating' });
            if (event.kind === 'text') {
              streamed = true;
              updateText(event.text || '', true, true);
            }
            if (event.kind === 'audio' && event.audio) {
              if (!this.player.turnActive) this.player.beginTurn();
              this.player.playChunk(event.audio);
            }
          } else if (event.type === 'response.done') {
            if (!streamed) updateText(event.text || '', false, false);
            else updateText('', true, false);
            // Nonstreaming response.audio_data is a WAV container, unlike streaming float32 audio.
            if (!settings.streaming && (event.audio_data || event.audio))
              this.playWav(event.audio_data || event.audio);
            this.player?.endTurn();
            this.update({ phase: 'idle', detail: '' });
          } else this.log(event.type, event.message || event.reason || '');
        },
        onError: (error) => {
          if (epoch === this.epoch) {
            this.update({
              phase: 'error',
              error,
              messages: this.view.messages.map((m) =>
                m.id === assistantId ? { ...m, pending: false, failed: true } : m,
              ),
            });
          }
        },
        onClosed: () => {
          if (epoch === this.epoch && this.view.phase !== 'error') this.update({ phase: 'idle' });
        },
      });
      this.chat.start(settings.endpoint, {
        messages: [
          ...(settings.systemPrompt ? [{ role: 'system', content: settings.systemPrompt }] : []),
          ...history.map((m) => ({ role: m.role, content: m.content || m.text })),
          { role: 'user', content },
        ],
        streaming: settings.streaming,
        generation: { max_new_tokens: settings.maxTokens, do_sample: true },
        image: { max_slice_nums: 1 },
        omni_mode: hasVideo,
        ...(settings.generateAudio
          ? {
              use_tts_template: true,
              tts: {
                enabled: true,
                mode: hasVideo ? 'omni' : 'audio_assistant',
                ...(settings.referenceAudio ? { ref_audio_data: settings.referenceAudio.pcm } : {}),
              },
            }
          : {}),
      });
      return true;
    } catch (error) {
      if (epoch === this.epoch) {
        this.stop();
        this.update({ phase: 'error', error: messageOf(error) });
      }
      return false;
    }
  }
  private wavAudio: HTMLAudioElement | null = null;
  private playWav(data: string) {
    this.wavAudio?.pause();
    this.wavAudio = new Audio(`data:audio/wav;base64,${data}`);
    this.wavAudio.play().catch(() => this.setError('浏览器阻止了音频播放。'));
  }
  mute() {
    this.update({ muted: !this.view.muted });
  }
  stop() {
    this.epoch++;
    clearTimeout(this.fcTimer);
    this.scene?.pause?.();
    this.mic?.stop();
    this.mic = null;
    this.camera?.close();
    this.camera = null;
    if (this.ordinary) {
      const ordinary = this.ordinary;
      if (ordinary.running) ordinary.stop();
      else ordinary.cancelQueue();
      void ordinary.audioPlayer.ctx?.close().catch(() => {});
      this.ordinary = null;
    }
    if (this.fc) {
      this.fc.onEvent = null;
      this.fc.onStatus = null;
      this.fc.close('client_closed');
      const ws = this.fc.ws;
      setTimeout(() => {
        if (ws && ws.readyState < 2) ws.close();
      }, 1500);
      this.fc = null;
    }
    this.chat?.stop();
    this.chat = null;
    this.player?.stop();
    void this.player?.ctx?.close().catch(() => {});
    this.player = null;
    this.wavAudio?.pause();
    this.wavAudio = null;
    this.speech('', true);
    this.update({
      phase: 'idle',
      micLevel: 0,
      camera: false,
      muted: false,
      detail: '',
      messages: this.view.messages.map((m) =>
        m.pending ? { ...m, pending: false, failed: true } : m,
      ),
    });
  }
  clear() {
    this.stop();
    this.update(emptyView());
  }
  dispose() {
    this.stop();
    this.fetchAbort?.abort();
    this.alive = false;
    this.listeners.clear();
    this.scene?.dispose?.();
    this.scene = null;
    this.stage = null;
  }
}
