import {
  startDingDongLoop,
  playAlarmBell,
  playSessionChime,
} from '../../../../static/duplex/lib/queue-chimes.js';
import {
  openingSilenceWarning,
  MIC_SPEECH_DB,
} from '../../../../static/demo-gallery/core/mic-policy.js';
import { encodeWav } from '../../../../static/duplex/lib/mixer-controller.js';
import { SessionRecorder } from '../../../../static/duplex/lib/session-recorder.js';
import { SessionVideoRecorder } from '../media/video-recorder.js';
import { RealtimeSession } from '../../../../static/duplex/lib/realtime-session.js';
import { FcRealtimeClient } from '../../../../static/demo-gallery/core/realtime-client.js';
import { LiveMicProvider } from '../../../../static/demo-gallery/core/live-mic-provider.js';
import { CameraCapture } from '../../../../static/demo-gallery/core/camera-capture.js';
import { AudioPlayer } from '../../../../static/demo-gallery/core/audio-player.js';
import { textFromSteps } from '../../../../static/demo-gallery/core/ui-utils.js';
import { OrdinaryLive, ordinaryFile, type InputSource, type MediaChunk } from '../media/ordinary';
import { decodeFileToChunks } from '../../../../static/demo-gallery/core/file-audio-provider.js';
import { ordinaryPayload, ttsReference, wireContent } from './options';
import { loadConversationPresets, type PresetLoadMode } from './presets';
import {
  parseApiService,
  endpointsFromOverview,
  mergeEndpointOptions,
} from '../../../../static/demo-gallery/core/api-endpoint.js';
import { ChatAudioPlayer, outputWav } from '../../turnbased/audio';
import { ChatTransport } from '../../turnbased/transport';
import { assistantText } from './text';
import { base64, contentFiles } from '../media/files';
import {
  applicationUrl,
  endpointBase,
  modelInfoEndpoint,
  ordinaryEndpoint,
  readJson,
  realtimeUrl,
} from './urls';
import {
  emptyView,
  type CaseDefinition,
  type Message,
  type ContentPart,
  type Scene,
  type SceneFactory,
  type SessionSettings,
  type SessionView,
} from './types';

const uid = () =>
  crypto.randomUUID?.() || `${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** DOM-independent session store. Desktop and future mobile layouts subscribe to the same state. */
export class SessionController {
  private view = emptyView();
  private listeners = new Set<() => void>();
  private epoch = 0;
  private alive = true;
  private eventNumber = 0;
  protocolEvents: { id: number; direction: string; event: any; time: string; bytes: number }[] = [];
  private protocolSequence = 0;
  private captureEvent(direction: string, event: any) {
    const bytes = JSON.stringify(event).length * 2;
    const limit = 24 * 1024 * 1024;
    const recorded =
      bytes > limit
        ? { type: event.type, omitted: 'Event exceeds 24 MiB capture limit', originalBytes: bytes }
        : event;
    this.protocolEvents.push({
      id: ++this.protocolSequence,
      direction: /^(client|tx|sent)$/.test(direction) ? 'client' : 'server',
      event: recorded,
      bytes: bytes > limit ? 200 : bytes,
      time: new Date().toISOString(),
    });
    let total = this.protocolEvents.reduce((n, e) => n + e.bytes, 0);
    while (this.protocolEvents.length > 200 || total > limit)
      total -= this.protocolEvents.shift()!.bytes;
    this.update({});
  }
  private latestSettings?: SessionSettings;
  private queueTimer?: ReturnType<typeof setInterval>;
  private stopQueueChime?: () => void;
  private queueUpdate(queue: any) {
    clearInterval(this.queueTimer);
    if (!queue) {
      this.stopQueueChime?.();
      this.stopQueueChime = undefined;
      return;
    }
    const until = Date.now() + Number(queue.estimated_wait_s || 0) * 1000;
    const render = () => {
      const wait = Math.ceil((until - Date.now()) / 1000);
      this.update({
        phase: 'queued',
        detail: `${queue.position != null ? `排队第 ${queue.position} 位` : '等待接入'}${queue.estimated_wait_s != null ? (wait >= 0 ? ` · 预计 ${wait} 秒` : ` · 超出预计 ${-wait} 秒`) : ''}`,
      });
    };
    render();
    this.queueTimer = setInterval(render, 1000);
    if (this.definition.protocol === 'ordinary' && queue.position === 1 && !this.stopQueueChime)
      this.stopQueueChime = startDingDongLoop();
  }
  private micLevel(value: number) {
    const level = this.view.muted ? 0 : value;
    const db = Math.max(-50, level > 0 ? 20 * Math.log10(level) : -50);
    const peak = Math.max(this.view.metrics.micPeak ?? -50, db);
    const speechSeen = this.view.metrics.micSpeechSeen || db >= MIC_SPEECH_DB;
    const started = this.view.metrics.micStarted || Date.now();
    const warning = openingSilenceWarning({
      micLiveState: 'live',
      speechSeen,
      waitedMs: Date.now() - started,
      peakDb: peak,
    });
    this.update({
      micLevel: level,
      metrics: {
        ...this.view.metrics,
        micDb: db,
        micPeak: peak,
        micSpeechSeen: speechSeen,
        micStarted: started,
        micWarning: warning.visible ? warning.text : '',
      },
    });
  }
  private currentSpeech = '';
  private mic: any = null;
  private recorder: any = null;
  private recordingBlob: Blob | null = null;
  recordingExtension = 'wav';
  private input: InputSource | null = null;
  private fileTimer?: ReturnType<typeof setTimeout>;
  private restartInput?: () => Promise<void>;
  private readyInput?: () => Promise<void>;
  private camera: any = null;
  private ordinary: any = null;
  private fc: any = null;
  private fileFinishing = false;
  private pendingSceneSends: { epoch: number; method: string; args: any[] }[] = [];
  private sceneClient(epoch: number) {
    if (!this.alive || this.epoch !== epoch || !this.fc) return null;
    const current = this.fc;
    return new Proxy(current, {
      get: (_target, key: string) => {
        if (key === 'sendToolResult' || key === 'sendStandaloneEvent')
          return (...args: any[]) => {
            if (!this.alive || epoch !== this.epoch) return;
            if (this.fc?.ready) this.fc[key](...args);
            else this.pendingSceneSends.push({ epoch, method: key, args });
          };
        const value = this.fc?.[key];
        return typeof value === 'function' ? value.bind(this.fc) : value;
      },
    });
  }
  resetFcPart(settings: SessionSettings, part: 'system' | 'tts') {
    const presets = this.scene?.buildQuickSettings?.(this.defaults || {});
    const selected = presets?.has(settings.preset)
      ? settings.preset
      : this.scene?.fallbackQuickSetting;
    const value = presets?.get(selected || '');
    if (!value) throw new Error('当前场景没有可恢复的预设，请先加载配置。');
    return part === 'tts'
      ? { referencePath: value.ttsPromptAudioPath || '', referenceAudio: undefined }
      : {
          segmentsJson: JSON.stringify(value.segments || [], null, 2),
          toolsJson: JSON.stringify(
            this.scene?.tools?.(this.defaults || {}, selected) || [],
            null,
            2,
          ),
          referencePath: value.ttsPromptAudioPath || '',
          referenceAudio: undefined,
          preset: selected,
        };
  }
  private chat: ChatTransport | null = null;
  private chatTurn?: { id: string; start: number; first?: number; requestId: string; mode: string };
  private player: any = null;
  private scene: Scene | null = null;
  private sceneFactory: SceneFactory | null = null;
  private stage: HTMLElement | null = null;
  private fcTimer?: ReturnType<typeof setTimeout>;
  private fetchAbort?: AbortController;
  private infoAbort?: AbortController;
  private infoEndpoint = '';
  private endpointIdentity: Record<string, any> = {};
  defaults: any = null;
  private connectionDefaults: any = null;
  endpoints: { url: string; label: string; available?: number }[] = [];
  endpointListError = '';
  endpointListService = '';
  private endpointRequest = 0;
  private serviceEndpoints(state: any, service: string) {
    const configured = (this.connectionDefaults?.api_endpoints || []).filter((e: any) => {
      try {
        return parseApiService(e.url).service === parseApiService(service).service;
      } catch {
        return false;
      }
    });
    return mergeEndpointOptions(endpointsFromOverview(state, service), configured);
  }
  onDraft?: (text: string) => void;
  onSettings?: (patch: Partial<SessionSettings>) => void;
  presets: { id: string; label: string; description?: string }[] = [];
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
  updateSettings(settings: SessionSettings) {
    const previous = this.latestSettings;
    this.latestSettings = settings;
    if (previous && previous.apiService !== settings.apiService) this.endpointRequest++;
    if (
      previous &&
      previous.endpoint !== settings.endpoint &&
      this.infoEndpoint !== settings.endpoint
    ) {
      this.infoAbort?.abort();
      this.endpointIdentity = {};
      this.update({
        model: '',
        modelIdentity: {},
        effectiveConfig: {},
        modelSource: 'unavailable',
        configSource: 'unavailable',
      });
    }
    if (this.busy) this.input?.updateMixer?.(settings);
    if (this.busy && previous && previous.speakerDevice !== settings.speakerDevice) {
      const ctx = this.ordinary?.audioPlayer.ctx;
      const epoch = this.epoch;
      void (async () => {
        if (ctx?.setSinkId) await ctx.setSinkId(settings.speakerDevice || '');
        else if (settings.speakerDevice)
          throw new Error('此浏览器不支持选择扬声器，请使用系统默认。');
        await this.input?.setSpeaker?.(settings.speakerDevice);
      })().catch((error) => {
        if (epoch === this.epoch) this.setError(error);
      });
    }
  }
  customSettings(patch: Partial<SessionSettings>) {
    const tools =
      this.definition.protocol === 'fc'
        ? this.scene?.tools?.(this.defaults || {}, this.scene?.fallbackQuickSetting)
        : undefined;
    return {
      ...patch,
      preset: 'custom',
      ...(tools ? { toolsJson: JSON.stringify(tools, null, 2) } : {}),
    };
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
      getClient: () => this.sceneClient(activeEpoch),
      api: { base: root, url: (path) => `${root}/${path.replace(/^\//, '')}` },
      assets: { base: assets, url: (path) => `${assets}/${path.replace(/^\//, '')}` },
      shell: {
        setDraft: (text) => this.onDraft?.(text),
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
  async loadPreset(
    settings: SessionSettings,
    preset?: string,
    mode: PresetLoadMode = 'preset',
  ): Promise<Partial<SessionSettings>> {
    this.update({ configLoading: true });
    const result = this.readPreset(settings, preset, mode);
    const request = this.fetchAbort;
    try {
      return await result;
    } finally {
      if (this.fetchAbort === request) this.update({ configLoading: false });
    }
  }
  private async readPreset(
    settings: SessionSettings,
    preset?: string,
    mode: PresetLoadMode = 'preset',
  ): Promise<Partial<SessionSettings>> {
    this.fetchAbort?.abort();
    const abort = new AbortController();
    this.fetchAbort = abort;
    if (this.definition.protocol !== 'fc') {
      const result = await loadConversationPresets(
        settings,
        this.definition.id,
        preset,
        abort.signal,
        mode,
      );
      if (abort.signal.aborted || !this.alive) return {};
      this.presets = result.options;
      this.defaults = result.config;
      this.dismissError();
      const connection = await this.loadModelSelection(settings, abort.signal);
      if (abort.signal.aborted || !this.alive) return {};
      this.log('配置', '已加载应用提供的体验配置与预设');
      return { ...result.patch, ...connection, configReady: true };
    }
    const defaults = await readJson(
      applicationUrl(settings.gallery, `/api/demos/${this.definition.id}/defaults`),
      abort.signal,
    );
    if (abort.signal.aborted || !this.alive) return {};
    this.defaults = defaults;
    await this.sceneSetup(settings);
    if (abort.signal.aborted || !this.alive) return {};
    const presets = this.scene?.buildQuickSettings?.(defaults);
    this.presets = this.scene?.presetOptions || [];
    const selected =
      preset && preset !== 'custom'
        ? preset
        : presets?.has(defaults.default_quick_setting)
          ? defaults.default_quick_setting
          : this.scene?.fallbackQuickSetting || this.presets[0]?.id || '';
    const value = presets?.get(selected);
    if (!value && this.presets.length)
      throw new Error('工具服务尚未配置这个预设，请选择可用预设或检查服务配置。');
    const tools = this.scene?.tools?.(defaults, selected) || defaults.system?.tools || [];
    const segments = value?.segments || defaults.system?.segments || [];
    const connection = await this.loadModelSelection(settings, abort.signal);
    if (abort.signal.aborted || !this.alive) return {};
    this.log('配置', '已从工具服务读取场景预设');
    return {
      configReady: true,
      preset: selected,
      referenceAudio: undefined,
      ...connection,
      segmentsJson: JSON.stringify(segments, null, 2),
      toolsJson: JSON.stringify(tools, null, 2),
      referencePath: value?.ttsPromptAudioPath || defaults.tts_prompt_audio?.file_path || '',
    };
  }
  private async loadModelSelection(settings: SessionSettings, signal: AbortSignal) {
    const defaults = await readJson(applicationUrl(settings.gallery, '/api/demos/runtime'), signal);
    if (signal.aborted || !this.alive) return {};
    this.connectionDefaults = defaults;
    let endpoint = settings.endpoint || defaults.default_api_endpoint || '';
    let apiService = settings.apiService || endpoint || defaults.default_api_service || '';
    if (apiService) {
      const parsed = parseApiService(apiService);
      apiService = parsed.service;
      endpoint = parsed.endpoint || endpoint;
      const listed = await this.readEndpointOptions(apiService, signal);
      if (listed === null) return {};
      let preferred = '';
      try {
        preferred = endpoint ? endpointBase(endpoint) : '';
      } catch {
        /* Invalid saved address: use the current service's discovered options. */
      }
      const sameService = preferred && parseApiService(preferred).service === apiService;
      endpoint =
        this.endpoints.find((option) => sameService && option.url === preferred)?.url ||
        (!listed && sameService ? preferred : '') ||
        this.endpoints.find((option) => (option.available || 0) > 0)?.url ||
        this.endpoints[0]?.url ||
        '';
    } else {
      this.endpoints = mergeEndpointOptions([], defaults.api_endpoints || []);
    }
    if (!this.alive || signal.aborted) return {};
    if (endpoint)
      await this.inspectEndpoint({ ...settings, endpoint }).catch((error) => this.setError(error));
    return { endpoint, apiService };
  }
  async inspectEndpoint(settings: SessionSettings) {
    if (this.busy) return;
    const epoch = this.epoch;
    this.infoAbort?.abort();
    const abort = new AbortController();
    this.infoAbort = abort;
    this.infoEndpoint = settings.endpoint;
    this.update({
      model: '',
      modelIdentity: {},
      effectiveConfig: {},
      modelSource: 'unavailable',
      configSource: 'unavailable',
    });
    this.endpointIdentity = {};
    let data;
    try {
      data = await readJson(`${modelInfoEndpoint(settings.endpoint)}/v1/info`, abort.signal);
    } catch (error) {
      if (abort.signal.aborted) return;
      throw error;
    }
    if (
      !this.alive ||
      abort.signal.aborted ||
      this.infoAbort !== abort ||
      epoch !== this.epoch ||
      this.busy
    )
      return;
    const identity = data.models?.[0]?.model || data.model || {};
    this.endpointIdentity = identity;
    const model =
      identity.profile_id || identity.name || identity.checkpoint || data.model_id || '';
    this.update({
      model: String(model),
      modelIdentity: identity,
      modelSource: Object.keys(identity).length ? 'endpoint' : 'unavailable',
      effectiveConfig: identity.session_defaults || {},
      configSource: identity.session_defaults ? 'endpoint' : 'unavailable',
      detail: `${Array.isArray(data.workers) && !data.workers.length ? '此端点目前无 Worker，开始后将排队。' : 'API 信息已读取。'}${data.models?.length > 1 ? '此端点存在多个模型，以会话实际返回为准。' : ''}`,
    });
    this.log('API', '已读取模型信息');
    return data;
  }
  private async readEndpointOptions(service: string, signal?: AbortSignal) {
    const request = ++this.endpointRequest;
    this.endpointListService = service;
    try {
      const state = await readJson(`${service}/overview/api/state`, signal);
      if (!this.alive || signal?.aborted || request !== this.endpointRequest) return null;
      this.endpoints = this.serviceEndpoints(state, service);
      this.endpointListError = '';
      this.update({});
      return true;
    } catch (error) {
      if (!this.alive || signal?.aborted || request !== this.endpointRequest) return null;
      const configured = this.serviceEndpoints({}, service);
      // Standalone gateways may omit the control-plane overview. The application
      // still lists endpoints; query each gateway info route for its live state.
      const available = await Promise.all(
        configured.map(async (option) => {
          try {
            const info = await readJson(`${modelInfoEndpoint(option.url)}/v1/info`, signal);
            const idle = Array.isArray(info.workers)
              ? info.workers.filter((worker: any) => worker.status === 'idle' && !worker.draining)
                  .length
              : undefined;
            return { ...option, available: idle };
          } catch {
            return null;
          }
        }),
      );
      if (!this.alive || signal?.aborted || request !== this.endpointRequest) return null;
      this.endpoints = available.filter((option) => option !== null);
      this.endpointListError = this.endpoints.length
        ? ''
        : `无法读取 API 端点列表：${messageOf(error)}`;
      if (!this.endpoints.length) this.endpoints = configured;
      this.update({});
      return this.endpoints.length > 0 && !this.endpointListError;
    }
  }
  async discoverEndpoints(settings: SessionSettings) {
    if (this.busy) return {};
    const parsed = parseApiService(settings.apiService || settings.endpoint);
    const listed = await this.readEndpointOptions(parsed.service);
    if (listed === null) return {};
    let preferred = parsed.endpoint || '';
    try {
      if (!preferred && settings.endpoint) preferred = endpointBase(settings.endpoint);
    } catch {
      /* Fall back to discovered endpoints. */
    }
    const sameService = preferred && parseApiService(preferred).service === parsed.service;
    const remembered = this.endpoints.find((option) => sameService && option.url === preferred);
    this.log('API', `发现 ${this.endpoints.length} 个端点`);
    return {
      apiService: parsed.service,
      endpoint:
        remembered?.url ||
        (!listed && sameService ? preferred : '') ||
        this.endpoints.find((e) => (e.available || 0) > 0)?.url ||
        this.endpoints[0]?.url ||
        '',
    };
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
    const selected = this.scene?.buildQuickSettings?.(this.defaults || {})?.has(settings.preset)
      ? settings.preset
      : this.scene?.fallbackQuickSetting;
    const tools = this.scene?.tools?.(this.defaults || {}, selected) || [];
    this.onSettings?.({ toolsJson: JSON.stringify(tools, null, 2) });
    if (!Array.isArray(segments) || !Array.isArray(tools))
      throw new Error('System segments 和 tools 必须是 JSON 数组。');
    segments = (await this.scene?.prepareSystem?.(structuredClone(segments))) || segments;
    this.onSettings?.({ segmentsJson: JSON.stringify(segments, null, 2) });
    for (const segment of segments) {
      if (segment?.kind === 'text' && typeof segment.text === 'string') continue;
      if (
        segment?.kind === 'audio' &&
        ((segment.audio?.source === 'path' && segment.audio.file_path?.trim()) ||
          (segment.audio?.source === 'base64' && segment.audio.data))
      )
        continue;
      throw new Error('System segment 必须是 text 或具有音频来源的 audio。');
    }
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
    if (!reference) throw new Error('请上传参考音色，或加载包含参考音频的场景预设。');
    return {
      protocol_version: '5',
      mode: 'full_duplex',
      fc_duplex: true,
      system: { segments, tools },
      ...(reference ? { tts_prompt_audio: reference } : {}),
      generate_audio: true,
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
    if (this.busy || this.view.configLoading || this.view.recordingPending) return;
    endpointBase(settings.endpoint);
    if (this.definition.protocol !== 'fc') ordinaryEndpoint(settings.endpoint);
    this.infoAbort?.abort();
    this.stop();
    const epoch = this.epoch;
    const current = () => this.alive && this.epoch === epoch;
    this.protocolEvents = [];
    this.update({
      phase: 'connecting',
      error: '',
      detail: '正在准备连接',
      dirty: true,
      ledger: [],
      ...(this.definition.video || this.definition.protocol === 'fc' ? { messages: [] } : {}),
      metrics: {},
      recordingSessionId: '',
      sessionId: '',
      modelIdentity: this.endpointIdentity,
      effectiveConfig: this.endpointIdentity.session_defaults || {},
      modelSource: Object.keys(this.endpointIdentity).length ? 'endpoint' : 'unavailable',
      configSource: this.endpointIdentity.session_defaults ? 'endpoint' : 'unavailable',
      model: this.endpointIdentity.profile_id || this.endpointIdentity.name || '',
      hdEnabled: settings.visionHD !== 'off',
    });
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
          getPlaybackDelayMs: () => this.latestSettings?.playbackDelay ?? settings.playbackDelay,
        });
        this.player.init();
        await this.player.ctx?.resume();
      }
      if (!current()) return;
      if (this.definition.protocol === 'ordinary') {
        if (this.view.recordingUrl) URL.revokeObjectURL(this.view.recordingUrl);
        this.recordingBlob = null;
        this.update({ recordingUrl: '', recordingPending: false });
      }
      if (this.definition.protocol === 'ordinary' && settings.recordSession) {
        this.recorder =
          wantsVideo && video
            ? new (SessionVideoRecorder as any)(video)
            : new (SessionRecorder as any)();
      }
      const pendingInput: MediaChunk[] = [];
      const sendChunk = (chunk: MediaChunk) => {
        if (!current()) return;
        try {
          const samples =
            this.definition.protocol === 'fc' && this.view.muted
              ? new Float32Array(chunk.audio.length)
              : chunk.audio;
          const audio = base64(samples.buffer);
          const frame = chunk.frameBase64;
          if (wantsVideo && !frame) throw new Error('视频输入没有可用画面。');
          if (this.fc?.ready) {
            this.fc.appendAudio({ audioBase64: audio, videoFrame: frame || null });
            this.update({
              metrics: {
                ...this.view.metrics,
                chunksSent: (this.view.metrics.chunksSent || 0) + 1,
              },
            });
          }
          if (this.ordinary && !this.ordinary.running) {
            pendingInput.push(chunk);
            return;
          }
          this.recorder?.pushLeft(samples);
          if (this.ordinary?.running)
            this.ordinary.sendChunk({
              audio_base64: audio,
              ...(frame
                ? {
                    frame_base64_list: [frame],
                    ...(settings.visionHD === 'strategy'
                      ? {}
                      : { max_slice_nums: this.view.hdEnabled ? settings.fixedSlices : 1 }),
                  }
                : {}),
            });
        } catch (error) {
          fail(error);
        }
      };
      const startInput = async () => {
        if (!current()) return;
        if (this.definition.protocol === 'ordinary') {
          const onEnd = () => {
            if (current()) {
              this.stop();
              this.update({ detail: '文件回放结束' });
            }
          };
          const input: InputSource =
            settings.inputMode === 'file'
              ? ordinaryFile(settings, wantsVideo ? video : null, sendChunk, onEnd, (text) =>
                  this.log('输入', text),
                )
              : new OrdinaryLive(
                  settings,
                  wantsVideo ? video : null,
                  sendChunk,
                  (value) => this.micLevel(value),
                  fail,
                );
          this.input = input;
          if (this.recorder && !wantsVideo) this.recorder.start();
          await input.start();
          input.setMuted?.(this.view.muted);
          if (!current()) {
            input.stop();
            return;
          }
          if (this.recorder && wantsVideo) {
            const recorder = this.recorder;
            await recorder.start(
              {
                mimeType: settings.recordingMime,
                videoBitsPerSecond: settings.recordingBitrate,
                subtitle: settings.recordingSubtitles,
                subtitleHeight: settings.subtitleHeight,
                subtitleOpacityBottom: settings.subtitleBottom / 100,
                subtitleOpacityTop: settings.subtitleTop / 100,
              },
              {
                leftAudioStream: input.getAudioOutputStream?.(),
                leftAudioBuffer: input.getFullAudioBuffer?.(),
              },
            );
            if (!current()) {
              await recorder.stop();
              return;
            }
          }
          this.update({ camera: wantsVideo });
        } else if (settings.inputMode === 'file') {
          if (wantsVideo) throw new Error('FC 文件回放只支持 Audio 模式，与原页面一致。');
          if (!settings.inputFile) throw new Error('请先选择音频文件。');
          const chunks = await decodeFileToChunks(settings.inputFile, {
            padBeforeSec: settings.padBefore,
            padAfterSec: settings.padAfter,
          });
          if (!current()) return;
          const audio = new Audio(URL.createObjectURL(settings.inputFile));
          this.wavAudio = audio;
          void audio.play().catch(() => {});
          let index = 0;
          const feed = () => {
            if (!current()) return;
            if (index === chunks.length) {
              URL.revokeObjectURL(audio.src);
              this.fileFinishing = true;
              this.fc?.close('file_replay_finished');
              this.fcTimer = setTimeout(() => this.finishFileReplay(), 10000);
              this.update({ detail: '文件已发送完，正在接收最后的输出' });
              return;
            }
            sendChunk({ audio: chunks[index++] });
            this.update({ detail: `发送音频 ${index}/${chunks.length} chunks` });
            this.fileTimer = setTimeout(feed, 1000);
          };
          feed();
        } else {
          if (wantsVideo && !this.camera?.active) {
            if (!video) throw new Error('视频预览尚未准备好。');
            this.camera?.close();
            const camera = new (CameraCapture as any)(video);
            this.camera = camera;
            await camera.open(settings.frontCamera ? 'user' : 'environment');
            if (camera.mirror !== settings.mirror) camera.toggleMirror();
            if (!current()) {
              camera.close();
              return;
            }
            camera.lockCaptureSize();
            this.update({ camera: true });
          }
          const mic = new (LiveMicProvider as any)({
            onChunk: (audio: Float32Array) =>
              sendChunk({
                audio,
                ...(wantsVideo ? { frameBase64: this.camera?.captureFrame() } : {}),
              }),
            onLevel: (value: number) => {
              if (current()) this.micLevel(value);
            },
            onPermissionHint: (text: string) => {
              if (current()) this.update({ detail: text });
            },
            onState: () => {},
            onStatus: () => {},
          });
          this.mic = mic;
          await mic.start();
          if (!current()) mic.stop();
        }
      };
      if (this.definition.protocol === 'ordinary') {
        const session = new (RealtimeSession as any)('playground', {
          getWsUrl: () =>
            realtimeUrl(ordinaryEndpoint(settings.endpoint), wantsVideo ? 'video' : 'audio'),
          getPlaybackDelayMs: () => this.latestSettings?.playbackDelay ?? settings.playbackDelay,
          getMaxKvTokens: () => this.latestSettings?.maxKvTokens ?? settings.maxKvTokens,
          enableReconnect: true,
        });
        this.ordinary = session;
        session.audioPlayer.onRawAudio = (samples: Float32Array, sr: number, timestamp: number) =>
          this.recorder?.pushRight(samples, sr, timestamp);
        session.audioPlayer.init();
        await session.audioPlayer.ctx?.resume();
        if (settings.speakerDevice) {
          try {
            if (!session.audioPlayer.ctx?.setSinkId)
              throw new Error('此浏览器不支持选择扬声器，将使用系统默认。');
            await session.audioPlayer.ctx.setSinkId(settings.speakerDevice);
          } catch (error) {
            this.log('输出设备', messageOf(error));
          }
        }
        if (!current()) {
          session.cancelQueue();
          void session.audioPlayer.ctx?.close().catch(() => {});
          return;
        }
        // Cleanup may be called by the transport itself, including a terminal server close.
        // Defer to avoid re-entering cleanup while it is still releasing its resources.
        let closingReason = '会话已结束';
        session.onSystemLog = (text: string) => {
          if (!current()) return;
          this.log('system', text);
          if (/Session closed|Session stopped|Timeout:|reached limit|Connection ended/.test(text))
            closingReason = text;
          this.update({ detail: text });
        };
        session.onCleanup = () =>
          queueMicrotask(() => {
            if (current()) {
              this.stop();
              this.update({ detail: closingReason });
            }
          });
        session.onQueueUpdate = (queue: any) => {
          if (current()) this.queueUpdate(queue);
        };
        session.onQueueDone = () => {
          if (current()) {
            this.queueUpdate(null);
            playAlarmBell();
            this.update({ phase: 'connecting', detail: '已接入，正在准备会话' });
          }
        };
        session.onPrepared = () => playSessionChime();
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
        session.onSpeakDelta = (text: string) => {
          if (current()) this.recorder?.addSubtitleUnit?.(text);
        };
        session.onSpeakEnd = () => {
          this.recorder?.finalizeSubtitle?.();
          if (current()) {
            this.speech('', true);
            this.update({ phase: 'listening' });
          }
        };
        session.onListenResult = (result: any) => {
          if (current())
            this.update({
              phase: 'listening',
              ...(result.text
                ? {
                    messages: [
                      ...this.view.messages,
                      { id: uid(), role: 'user', text: result.text },
                    ],
                  }
                : {}),
            });
        };
        session.onProtocolEvent = (event: any) => {
          if (!current()) return;
          if (event.type === 'session.config')
            this.update({ effectiveConfig: event.full || {}, configSource: 'session' });
          this.captureEvent(
            event.dir || 'server',
            event.full || { type: event.type, summary: event.summary },
          );
          if (event.type !== 'input.append') this.log(event.type, event.summary || '');
        };
        session.onPauseStateChange = (state: string) => {
          if (current()) {
            this.update({ paused: state === 'paused' });
            if (state === 'paused') {
              this.input?.pause?.();
              this.recorder?.pause();
            } else {
              this.input?.resume?.();
              this.recorder?.resume();
            }
          }
        };
        session.onForceListenChange = (forceListen: boolean) => {
          if (current()) this.update({ forceListen });
        };
        session.onMetrics = (data: any) => {
          if (!current()) return;
          const metrics = { ...this.view.metrics };
          for (const [key, value] of Object.entries(data)) {
            if (value !== undefined && value !== null) metrics[key] = value;
          }
          this.update({ metrics });
          if (data.connectionState === 'failed') fail(new Error('连接中断，自动重连未成功。'));
          else if (data.connectionState === 'reconnecting')
            this.update({ phase: 'connecting', detail: '正在恢复连接' });
          else if (data.connectionState === 'connected')
            this.update({ phase: 'listening', detail: '' });
          if (data.chunksSent !== undefined) this.update({ unit: data.chunksSent });
        };
        await session.start(
          settings.systemPrompt,
          ordinaryPayload(settings, wantsVideo),
          async () => {
            if (!current()) throw new Error('启动已取消。');
            await startInput();
          },
        );
        if (!current()) {
          session.cancelQueue();
          return;
        }
        pendingInput.splice(0).forEach(sendChunk);
        this.update({
          phase: 'listening',
          startedAt: Date.now(),
          sessionId: session.sessionId,
          recordingSessionId: session.recordingSessionId || session.sessionId,
          detail: '',
        });
      } else {
        await this.sceneSetup(settings, epoch);
        this.scene?.buildQuickSettings?.(this.defaults || {});
        this.scene?.tools?.(this.defaults || {}, settings.preset);
        const payload = await this.fcPayload(settings);
        if (!current()) return;
        const client = new (FcRealtimeClient as any)({
          onSend: (event: any) => this.captureEvent('client', event),
          onEvent: (event: any) => {
            if (!current()) return;
            try {
              this.handleFc(event, fail);
            } catch (error) {
              fail(error);
            }
          },
          onStatus: (text: string) => {
            if (current() && text === 'WebSocket closed') {
              if (this.fileFinishing) this.finishFileReplay();
              else fail(new Error('会话连接已断开，请重新开始。'));
            }
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
        this.restartInput = startInput;
        this.readyInput = startInput;
        client.initSession(payload);
      }
    } catch (error) {
      fail(error);
    }
  }
  private finishFileReplay() {
    const playback = this.player;
    this.player = null;
    this.stop();
    this.player = playback;
    playback?.endTurn();
    this.update({ detail: '文件回放结束，已生成的音频继续播放' });
  }
  private handleFc(event: any, fail: (error: unknown) => void) {
    const type = event.type;
    this.captureEvent('server', event);
    if (Array.isArray(event.steps))
      this.update({
        metrics: {
          ...this.view.metrics,
          generation_steps: (this.view.metrics.generation_steps || 0) + event.steps.length,
        },
      });
    if (/^response\.(think|tool_call|non_spoken)\./.test(type) || type === 'response.spoken.end') {
      this.update({ ledger: [...this.view.ledger, event] });
    }
    if (type === 'session.queued' || type === 'session.queue_update') this.queueUpdate(event);
    if (type === 'session.created' || type === 'session.resumed') {
      this.queueUpdate(null);
      for (const item of this.pendingSceneSends.splice(0))
        if (item.epoch === this.epoch) this.fc?.[item.method](...item.args);
      this.scene?.onReady?.();
      clearTimeout(this.fcTimer);
      const startInput = this.readyInput;
      this.readyInput = undefined;
      const inputEpoch = this.epoch;
      if (startInput)
        void startInput()
          .then(() => {
            if (inputEpoch === this.epoch) this.update({ phase: 'listening', detail: '' });
          })
          .catch(fail);
      this.update({
        phase: startInput ? 'connecting' : 'listening',
        detail: startInput ? '正在准备媒体输入' : '',
        startedAt: type === 'session.resumed' ? this.view.startedAt : Date.now(),
        sessionId: event.session_id || this.view.sessionId,
        model: event.model?.profile_id || this.view.model,
        modelIdentity: event.model || (type === 'session.resumed' ? this.view.modelIdentity : {}),
        modelSource: event.model
          ? 'session'
          : type === 'session.resumed'
            ? this.view.modelSource
            : 'unavailable',
        effectiveConfig:
          event.session_config || (type === 'session.resumed' ? this.view.effectiveConfig : {}),
        configSource: event.session_config
          ? 'session'
          : type === 'session.resumed'
            ? this.view.configSource
            : 'unavailable',
      });
    }
    if (type === 'session.closed' && !event.error && !event.diagnostic) {
      if (this.fileFinishing) {
        this.finishFileReplay();
        return;
      }
      this.stop();
      this.update({ detail: `会话结束：${event.reason || 'closed'}` });
      return;
    }
    if (type === 'error' || type === 'session.failed' || type === 'session.closed') {
      fail(
        new Error(
          event.error?.message ||
            event.diagnostic?.message ||
            event.message ||
            event.reason ||
            '会话已结束',
        ),
      );
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
    if (type === 'response.unit.committed')
      this.update({
        unit: Number(event.unit_index) + 1,
        metrics: {
          ...this.view.metrics,
          checkpoint: event.resume || {},
          unit_index: event.unit_index,
        },
      });
    if (type === 'response.warning')
      this.update({ detail: event.message || event.code || '模型返回警告' });
    if (type === 'response.debug')
      this.update({
        metrics: {
          ...this.view.metrics,
          debug: event,
          debugUpdated: new Date().toLocaleTimeString(),
        },
      });
    if (type === 'response.spoken.delta') {
      const text = textFromSteps(event.steps);
      this.speech(text);
      this.scene?.onSpokenDelta?.(event, text);
      if (text || event.audio) this.update({ phase: 'speaking' });
      if (event.audio) {
        this.update({
          metrics: { ...this.view.metrics, audioChunks: (this.view.metrics.audioChunks || 0) + 1 },
        });
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
      this.log(type, event.full_text || event.message || event.call?.name || event.reason || '');
  }
  async send(
    settings: SessionSettings,
    text: string,
    files: File[],
    ordered?: ContentPart[],
    historyOverride?: Message[],
  ): Promise<boolean> {
    if (this.definition.protocol !== 'chat') {
      if (this.definition.id !== 'standalone' || !this.fc?.ready)
        throw new Error('请先开始会话，再通过语音与助手交流。');
      const eventId = `pg_${uid()}`;
      this.fc.sendStandaloneEvent({ content: text, eventId });
      this.scene?.onExternalEvent?.(eventId, text);
      this.update({ messages: [...this.view.messages, { id: uid(), role: 'user', text }] });
      return true;
    }
    if (this.busy || this.view.configLoading) return false;
    endpointBase(settings.endpoint);
    ordinaryEndpoint(settings.endpoint);
    const history =
      historyOverride ||
      this.view.messages.filter(
        (m) => !m.failed && !m.pending && (m.text !== '' || Array.isArray(m.content)),
      );
    this.stop();
    const epoch = this.epoch;
    this.protocolEvents = [];
    this.update({
      phase: 'generating',
      error: '',
      dirty: true,
      metrics: {},
    });
    try {
      const attachments = ordered
        ? ordered.filter((p) => p.type !== 'text')
        : await contentFiles(files);
      if (epoch !== this.epoch || !this.alive) return false;
      const content = ordered || [
        ...(text ? [{ type: 'text' as const, text }] : []),
        ...attachments,
      ];
      const user: Message = {
        id: uid(),
        role: 'user',
        text,
        content,
        files: attachments.map((part) => part.name || part.type),
      };
      if (historyOverride) {
        const retained = new Set(historyOverride.map((m) => m.id));
        this.view.messages
          .filter((m) => !retained.has(m.id))
          .forEach((m) => m.audioPlayer?.dispose());
      }
      const assistantId = uid();
      const requestStart = performance.now();
      const requestId = `req_${Date.now()}`;
      let firstText: number | undefined;
      this.chatTurn = {
        id: assistantId,
        start: requestStart,
        requestId,
        mode: settings.streaming ? 'streaming' : 'chat',
      };
      this.update({
        messages: [
          ...(historyOverride || this.view.messages),
          user,
          { id: assistantId, role: 'assistant', text: '', pending: true },
        ],
        startedAt: Date.now(),
      });
      this.player = new ChatAudioPlayer();
      this.player.init();
      let rawText = '';
      const updateText = (text: string, append: boolean, pending: boolean) => {
        rawText = append ? rawText + text : text;
        this.update({
          messages: this.view.messages.map((m) =>
            m.id === assistantId ? { ...m, text: assistantText(rawText, pending), pending } : m,
          ),
        });
      };
      const hasVideo =
        settings.systemContent.some((part) => part.type === 'video' && part.data) ||
        attachments.some((a) => a.type === 'video') ||
        history.some((m) => Array.isArray(m.content) && m.content.some((c) => c.type === 'video'));
      let streamed = false;
      const audioChunks: Float32Array[] = [];
      this.chat = new ChatTransport({
        onSend: (event) => this.captureEvent('client', event),
        onEvent: (event) => {
          if (epoch !== this.epoch) return;
          this.captureEvent('server', event);
          if (event.type === 'session.created') this.update({ sessionId: event.session_id || '' });
          if (event.type === 'session.queued') this.update({ phase: 'queued' });
          if (event.type === 'response.output.delta') {
            firstText ??= performance.now() - requestStart;
            if (this.chatTurn) this.chatTurn.first = firstText;
            this.update({ phase: 'generating' });
            if (event.kind === 'text') {
              firstText ??= performance.now() - requestStart;
              streamed = true;
              updateText(event.text || '', true, true);
            }
            if (event.kind === 'audio' && event.audio) {
              const bytes = Uint8Array.from(atob(event.audio), (c) => c.charCodeAt(0));
              audioChunks.push(new Float32Array(bytes.buffer));
              if (!this.player.turnActive) this.player.beginTurn();
              this.player.playChunk(event.audio);
              this.update({
                messages: this.view.messages.map((m) =>
                  m.id === assistantId ? { ...m, audioPlayer: this.player } : m,
                ),
              });
            }
          } else if (event.type === 'response.done') {
            const totalMs = performance.now() - requestStart;
            this.update({
              messages: this.view.messages.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      chatMeta: {
                        mode: settings.streaming ? 'streaming' : 'chat',
                        requestId,
                        totalMs,
                        ttfsMs: firstText,
                        processMs: settings.streaming
                          ? firstText == null
                            ? undefined
                            : totalMs - firstText
                          : (event.duration_ms ?? totalMs),
                        inputTokens: event.input_tokens,
                        generatedTokens: event.generated_tokens,
                      },
                    }
                  : m,
              ),
            });
            this.update({
              recordingSessionId: event.recording_session_id || '',
              metrics: {
                ...this.view.metrics,
                input_tokens: event.input_tokens,
                generated_tokens: event.generated_tokens,
              },
            });
            if (!streamed) updateText(event.text || '', false, false);
            else updateText('', true, false);
            // Nonstreaming audio is a WAV container. As on the original page,
            // expose playback controls without automatically starting it.
            if (event.audio_data || event.audio || audioChunks.length) {
              const attach = (url: string) => {
                if (
                  this.alive &&
                  this.view.messages.some((m) => m.id === assistantId && !m.pending && !m.failed)
                )
                  this.update({
                    messages: this.view.messages.map((m) =>
                      m.id === assistantId ? { ...m, audioUrl: url } : m,
                    ),
                  });
              };
              if (event.audio_data || event.audio)
                void outputWav(event.audio_data || event.audio)
                  .then((blob) => blob.arrayBuffer())
                  .then((buffer) => attach(`data:audio/wav;base64,${base64(buffer)}`))
                  .catch((error) => {
                    if (epoch === this.epoch) this.setError(error);
                  });
              else {
                const pcm = new Float32Array(audioChunks.reduce((n, c) => n + c.length, 0));
                let offset = 0;
                for (const c of audioChunks) {
                  pcm.set(c, offset);
                  offset += c.length;
                }
                void encodeWav(pcm, 24000)
                  .arrayBuffer()
                  .then((buffer: ArrayBuffer) => attach(`data:audio/wav;base64,${base64(buffer)}`));
              }
            }
            this.chatTurn = undefined;
            this.player?.endTurn();
            this.update({ phase: 'idle', detail: '' });
          } else this.log(event.type, event.message || event.reason || '');
        },
        onError: (error) => {
          if (epoch === this.epoch) {
            this.player?.endTurn();
            this.player?.pause();
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
      this.chat.start(ordinaryEndpoint(settings.endpoint), {
        messages: [
          ...(wireContent(settings.systemContent).length
            ? [{ role: 'system', content: wireContent(settings.systemContent) }]
            : []),
          ...history.map((m) => ({
            role: m.role,
            content: Array.isArray(m.content) ? wireContent(m.content, false) : m.content || m.text,
          })),
          { role: 'user', content: Array.isArray(content) ? wireContent(content, false) : content },
        ],
        streaming: settings.streaming,
        generation: { max_new_tokens: settings.maxTokens, do_sample: true },
        image: { max_slice_nums: hasVideo ? 1 : settings.imageHD ? 2 : 1 },
        omni_mode: hasVideo,
        ...(settings.generateAudio
          ? {
              use_tts_template: true,
              tts: {
                enabled: true,
                mode: hasVideo ? 'omni' : 'audio_assistant',
                ...(ttsReference(settings, true)
                  ? { ref_audio_data: ttsReference(settings, true) }
                  : {}),
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
  async regenerate(settings: SessionSettings, id: string) {
    if (this.busy) return;
    const index = this.view.messages.findIndex((m) => m.id === id);
    if (this.view.messages[index]?.role !== 'assistant') return;
    const userIndex = index - 1;
    const user = this.view.messages[userIndex];
    if (!user || user.role !== 'user') return;
    const content = Array.isArray(user.content)
      ? user.content
      : [{ type: 'text' as const, text: user.text }];
    await this.send(settings, user.text, [], content, this.view.messages.slice(0, userIndex));
  }
  private wavAudio: HTMLAudioElement | null = null;
  async refreshEndpoints(settings: SessionSettings) {
    if (!settings.apiService || this.view.configLoading) return;
    const service = parseApiService(settings.apiService).service;
    await this.readEndpointOptions(service);
  }
  async resumeCheckpoint(settings: SessionSettings) {
    const old = this.fc;
    const index = this.view.metrics.unit_index;
    if (
      !old ||
      !['listening', 'speaking'].includes(this.view.phase) ||
      this.view.metrics.checkpoint?.status !== 'available' ||
      settings.inputMode === 'file'
    )
      return;
    const payload = old.buildResumePayload(index);
    this.mic?.stop();
    this.mic = null;
    old.onEvent = null;
    old.onStatus = null;
    old.disconnect('demo_resume_test');
    this.update({ phase: 'connecting', detail: `正在从 Unit ${index} 恢复` });
    const epoch = this.epoch;
    const fail = (error: unknown) => {
      if (epoch === this.epoch) {
        this.stop();
        this.update({ phase: 'error', error: messageOf(error) });
      }
    };
    const client = new (FcRealtimeClient as any)({
      onEvent: (event: any) => {
        if (epoch !== this.epoch) return;
        if (event.type === 'session.resume.failed') fail(new Error(event.message || event.code));
        else this.handleFc(event, fail);
      },
      onSend: (event: any) => this.captureEvent('client', event),
      onStatus: (text: string) => {
        if (text === 'WebSocket closed') fail(new Error('恢复连接中断'));
      },
    });
    this.fc = client;
    try {
      await new Promise((resolve) => setTimeout(resolve, 600));
      if (epoch !== this.epoch) return;
      await client.connect(realtimeUrl(settings.endpoint, settings.fcVideo ? 'omni' : 'audio'));
      if (epoch !== this.epoch) {
        client.disconnect();
        return;
      }
      this.readyInput = this.restartInput;
      this.fcTimer = setTimeout(() => fail(new Error('恢复会话超时')), 120000);
      client.resumeSession(payload);
    } catch (error) {
      fail(error);
    }
  }
  mixerLevels() {
    const result: Record<string, number | undefined> = { fileLUFS: this.input?.fileLUFS?.() };
    for (const [key, analyser] of Object.entries(this.input?.meterNodes?.() || {})) {
      if (!analyser) continue;
      const samples = new Float32Array(analyser.fftSize);
      analyser.getFloatTimeDomainData(samples);
      const rms = Math.sqrt(samples.reduce((n, v) => n + v * v, 0) / samples.length);
      result[key] = rms > 0 ? 20 * Math.log10(rms) : -60;
    }
    return result;
  }
  private cameraFlipPending = false;
  async flipCamera() {
    if (this.cameraFlipPending) return;
    this.cameraFlipPending = true;
    try {
      if (this.camera) {
        await this.camera.flip();
        this.onSettings?.({
          frontCamera: this.camera.facing === 'user',
          mirror: this.camera.mirror,
        });
      } else if (this.input?.flip) {
        await this.input.flip();
        this.onSettings?.({ frontCamera: !this.latestSettings?.frontCamera });
      }
    } finally {
      this.cameraFlipPending = false;
    }
  }
  mirrorCamera() {
    if (this.camera) this.camera.toggleMirror();
    else this.input?.toggleMirror?.();
    this.onSettings?.({ mirror: this.camera ? this.camera.mirror : !this.latestSettings?.mirror });
  }
  pause() {
    this.ordinary?.pauseToggle();
  }
  forceListen() {
    this.ordinary?.toggleForceListen();
  }
  toggleHD() {
    this.update({ hdEnabled: !this.view.hdEnabled });
  }
  mute() {
    const muted = !this.view.muted;
    this.input?.setMuted?.(muted);
    this.update({ muted });
  }
  stop() {
    if (this.recorder) {
      const recorder = this.recorder;
      this.recorder = null;
      recorder.cancelPending?.();
      if (recorder.recording) {
        this.update({ recordingPending: true });
        void Promise.resolve(recorder.stop())
          .then((result: any) => {
            if (!this.alive) return;
            this.recordingBlob = result.blob;
            this.recordingExtension = result.blob.type.includes('mp4')
              ? 'mp4'
              : result.blob.type.includes('webm')
                ? 'webm'
                : 'wav';
            if (this.view.recordingUrl) URL.revokeObjectURL(this.view.recordingUrl);
            this.update({
              recordingUrl: URL.createObjectURL(result.blob),
              recordingPending: false,
            });
          })
          .catch((e) => {
            this.setError(e);
            this.update({ recordingPending: false });
          });
      }
    }
    if (this.definition.protocol === 'chat') {
      const turn = this.chatTurn;
      if (turn)
        this.update({
          messages: this.view.messages.map((m) =>
            m.id === turn.id && m.pending
              ? {
                  ...m,
                  chatMeta: {
                    mode: turn.mode,
                    requestId: turn.requestId,
                    totalMs: performance.now() - turn.start,
                    ttfsMs: turn.first,
                    processMs:
                      turn.first == null ? undefined : performance.now() - turn.start - turn.first,
                  },
                }
              : m,
          ),
        });
      this.chatTurn = undefined;
      this.update({
        messages: this.view.messages.flatMap((m) =>
          !m.pending
            ? [m]
            : m.text
              ? [
                  {
                    ...m,
                    pending: false,
                    cancelled: true,
                    audioPlayer: this.player?.duration ? this.player : undefined,
                  },
                ]
              : [],
        ),
      });
      this.player?.endTurn?.();
    }
    this.epoch++;
    this.fileFinishing = false;
    this.pendingSceneSends = [];
    this.queueUpdate(null);
    clearTimeout(this.fcTimer);
    clearTimeout(this.fileTimer);
    this.readyInput = undefined;
    this.restartInput = undefined;
    this.input?.stop();
    this.input = null;
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
    this.view.messages.forEach((m) => m.audioPlayer?.stop());
    this.player?.stop();
    void this.player?.ctx?.close().catch(() => {});
    this.player = null;
    this.wavAudio?.pause();
    if (this.wavAudio?.src.startsWith('blob:')) URL.revokeObjectURL(this.wavAudio.src);
    this.wavAudio = null;
    this.speech('', true);
    this.update({
      phase: 'idle',
      micLevel: 0,
      camera: false,
      muted: false,
      paused: false,
      forceListen: false,
      detail: '',
      messages: this.view.messages.map((m) =>
        m.pending ? { ...m, pending: false, failed: true } : m,
      ),
    });
  }
  async share(settings: SessionSettings, comment: string) {
    const id = this.view.recordingSessionId;
    if (!id) throw new Error('服务未返回可分享的会话编号。');
    const base = settings.shareSource;
    const path = `/api/sessions/${encodeURIComponent(id)}`;
    const recordingBlob = this.recordingBlob;
    const recordingExtension = this.recordingExtension;
    if (comment.trim()) {
      try {
        const response = await fetch(applicationUrl(base, `${path}/comment`), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ comment: comment.trim() }),
        });
        if (!response.ok) throw new Error(`保存备注失败（${response.status}）`);
      } catch (error) {
        this.setError(error);
      }
    }
    if (recordingBlob?.size) {
      try {
        const form = new FormData();
        form.append('file', recordingBlob, `recording.${recordingExtension}`);
        const response = await fetch(applicationUrl(base, `${path}/upload-recording`), {
          method: 'POST',
          body: form,
        });
        if (!response.ok) throw new Error(`上传录制失败（${response.status}）`);
      } catch (error) {
        if (this.definition.video) throw error;
        this.setError(error);
      }
    }
    try {
      const key = 'minicpmo45_recent_sessions';
      const stored = JSON.parse(localStorage.getItem(key) || '[]');
      const list = Array.isArray(stored)
        ? stored.filter((item) => item && typeof item.id === 'string')
        : [];
      const appType =
        this.definition.id === 'audio-duplex'
          ? 'audio_duplex'
          : this.definition.video
            ? 'omni_duplex'
            : 'streaming';
      localStorage.setItem(
        key,
        JSON.stringify(
          [
            { id, appType, savedAt: new Date().toISOString() },
            ...list.filter((item: any) => item.id !== id),
          ].slice(0, 20),
        ),
      );
    } catch {
      /* Local history is optional. */
    }
    const url = new URL(applicationUrl(base, `/s/${encodeURIComponent(id)}`), location.href).href;
    void navigator.clipboard?.writeText(url).catch(() => {});
    return url;
  }
  clear() {
    this.stop();
    if (this.view.recordingUrl) URL.revokeObjectURL(this.view.recordingUrl);
    this.recordingBlob = null;
    this.protocolEvents = [];
    this.view.messages.forEach((m) => m.audioPlayer?.dispose());
    this.update(emptyView());
  }
  dispose() {
    this.stop();
    this.fetchAbort?.abort();
    this.infoAbort?.abort();
    this.view.messages.forEach((m) => m.audioPlayer?.dispose());
    this.alive = false;
    if (this.view.recordingUrl) URL.revokeObjectURL(this.view.recordingUrl);
    this.listeners.clear();
    this.scene?.dispose?.();
    this.scene = null;
    this.stage = null;
  }
}
