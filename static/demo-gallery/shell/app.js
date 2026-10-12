import { decodeFileToChunks, float32ToBase64 } from '../core/file-audio-provider.js';
import { FcRealtimeClient } from '../core/realtime-client.js';
import { LiveMicProvider } from '../core/live-mic-provider.js';
import { AudioPlayer } from '../core/audio-player.js';
import { NsSegmentView } from '../core/ns-segment-view.js';
import { QUICK_SETTING_CUSTOM, cloneQuickSetting } from '../core/quick-settings.js';
import { getRegisteredDemo } from '../core/registry.js';
import { normalizeScene } from '../core/scene-contract.js';
import { DEFAULT_PLAYBACK_DELAY_MS, resolvePlaybackDelayMs } from '../core/playback-policy.js';
import { MIC_SPEECH_DB, openingSilenceWarning } from '../core/mic-policy.js';
import { MODEL_IDENTITY_HINTS, MODEL_IDENTITY_ROWS } from '../core/model-identity.js';
import { escapeHtml, textFromSteps } from '../core/ui-utils.js';
import { normalizeEndpoint, summarizeEndpointInfo, summarizeSessionConfig } from '../core/api-endpoint.js';

// 后端注入的当前 demo 元信息（入口模块 URL / API 前缀 / 标题）。
const bootstrap = JSON.parse(document.getElementById('gallery-bootstrap').textContent);

const streamEvents = new Map();
const audioPlayer = new AudioPlayer({
  outputSampleRate: 24000,
  getPlaybackDelayMs: () => resolvePlaybackDelayMs(
    el.playbackDelayMs?.value, DEFAULT_PLAYBACK_DELAY_MS,
  ),
});

let liveClient = null;
let micProvider = null;
let audioPlayerReady = false;
let sentChunkCount = 0;
let aiAudioCount = 0;
let generationStepCount = 0;
let latestCheckpoint = null;
let resumeInProgress = false;
// Mic level display uses a LUFS-like unit (dBFS) instead of raw RMS.
//
// 换算依据（`scripts/diagnostics/lufs_survey.py` 采样训练数据，见 o45-fc 分支）：
//   training set: RMS mean = 0.069  ↔  LUFS mean = -23.0
//   公式 20·log10(RMS) 在这个 RMS 上算得 -23.2 —— 跟真 LUFS 只差 0.2 dB
// 结论：对语音信号，`20·log10(rms)` 就够当 LUFS 近似值显示了（K-weighting
// 对语音频谱几乎无影响，真 LUFS 需要 400ms 窗 + 门控，UI 场景不必如此严格）。
//
// bar 映射范围：-50 dB → -10 dB（覆盖训练分布 -23 附近 ±15 dB）
// target zone：-28 dB → -18 dB（训练集 5th → 95th percentile RMS 换算）
const MIC_METER_MIN_DB = -50;    // 静音底
const MIC_METER_MAX_DB = -10;    // 满格
const MIC_TARGET_LO_DB = -28;    // 训练集 p5 附近
const MIC_TARGET_HI_DB = -18;    // 训练集 p95 附近

function rmsToDb(rms) {
  if (!(rms > 0)) return MIC_METER_MIN_DB;
  const db = 20 * Math.log10(rms);
  return Math.max(MIC_METER_MIN_DB, db);   // 静音底
}

let micPeakDbSinceStart = MIC_METER_MIN_DB;
let micLiveStartedAt = 0;
let micSpeechSeen = false;
let fcBoardDefaults = null;
let systemSegments = [];
let quickSettings = new Map();
// 所选 API 端点（normalizeEndpoint 的结果）与它的 /v1/info。模型口径（预算、调度、tts_pad）
// 由端点上部署的模型决定：这里只用于展示，不进 session.init。
let apiEndpoint = null;
let endpointInfo = null;
const API_ENDPOINT_KEY = 'demo_gallery_api_endpoint';
// 音频路径 → v4 base64 音频对象（由 Demo 服务读本机文件）；同一页面内复用。
const audioCache = new Map();
let micLiveState = 'idle';
let streamEventSeq = 0;
const speechQueue = [];
let speechDrainer = null;

const el = {
  status: document.getElementById('status'),
  micLevelBar: document.getElementById('micLevelBar'),
  micLevelText: document.getElementById('micLevelText'),
  statusLamp: document.getElementById('statusLamp'),
  modeBadge: document.getElementById('modeBadge'),
  wsState: document.getElementById('wsState'),
  sentChunks: document.getElementById('sentChunks'),
  aiAudioCount: document.getElementById('aiAudioCount'),
  generationSteps: document.getElementById('generationSteps'),
  resumeCheckpoint: document.getElementById('resumeCheckpoint'),
  resumeSummary: document.getElementById('resumeSummary'),
  liveHint: document.getElementById('liveHint'),
  audioFileInput: document.getElementById('audioFileInput'),
  runFileReplay: document.getElementById('runFileReplay'),
  startMicLive: document.getElementById('startMicLive'),
  stopMicLive: document.getElementById('stopMicLive'),
  userAudioPlayer: document.getElementById('userAudioPlayer'),
  aiAudioList: document.getElementById('aiAudioList'),
  padBeforeSec: document.getElementById('padBeforeSec'),
  padAfterSec: document.getElementById('padAfterSec'),
  timeline: document.getElementById('timeline'),
  streamFeed: document.getElementById('streamFeed'),
  streamFilter: document.getElementById('streamFilter'),
  openStreamDialog: document.getElementById('openStreamDialog'),
  streamDialog: document.getElementById('streamDialog'),
  streamModalCount: document.getElementById('streamModalCount'),
  streamModalList: document.getElementById('streamModalList'),
  stage: document.getElementById('demoStage'),
  pageTitle: document.getElementById('pageTitle'),
  scenePanelTitle: document.getElementById('scenePanelTitle'),
  scenePanelHint: document.getElementById('scenePanelHint'),
  ttsPadProtocolView: document.getElementById('ttsPadProtocolView'),
  playbackDelayMs: document.getElementById('playbackDelayMs'),
  micSilenceWarn: document.getElementById('micSilenceWarn'),
  modelIdentity: document.getElementById('modelIdentity'),
  kvProfileId: document.getElementById('kvProfileId'),
  aiSpeech: document.getElementById('aiSpeech'),
  nsStream: document.getElementById('nonSpokenStream'),
  kvMode: document.getElementById('kvMode'),
  apiEndpoint: document.getElementById('apiEndpoint'),
  apiEndpointOptions: document.getElementById('apiEndpointOptions'),
  applyApiEndpoint: document.getElementById('applyApiEndpoint'),
  apiEndpointStatus: document.getElementById('apiEndpointStatus'),
  kvCkpt: document.getElementById('kvCkpt'),
  kvTools: document.getElementById('kvTools'),
  quickSetting: document.getElementById('quickSetting'),
  systemSegments: document.getElementById('systemSegments'),
  ttsPromptAudioPath: document.getElementById('ttsPromptAudioPath'),
  nonSpokenScheduling: document.getElementById('nonSpokenScheduling'),
  checkpointProfileId: document.getElementById('checkpointProfileId'),
  nonSpokenBudgetWhileListening: document.getElementById('nonSpokenBudgetWhileListening'),
  nonSpokenBudgetWhileSpeaking: document.getElementById('nonSpokenBudgetWhileSpeaking'),
  debugBudgetUsed: document.getElementById('debugBudgetUsed'),
  debugBudgetMax: document.getElementById('debugBudgetMax'),
  debugBudgetUpdated: document.getElementById('debugBudgetUpdated'),
  addTextSegment: document.getElementById('addTextSegment'),
  addAudioSegment: document.getElementById('addAudioSegment'),
  resetSystemSegments: document.getElementById('resetSystemSegments'),
  resetTtsPromptAudio: document.getElementById('resetTtsPromptAudio'),
  useFirstSystemAudio: document.getElementById('useFirstSystemAudio'),
  debugToggle: document.getElementById('debugToggle'),
  debugDrawer: document.getElementById('debugDrawer'),
};

// non-spoken segment 的 3 轨审计视图（think / tool_call 渲染）
const nsView = new NsSegmentView(el.nsStream);

// 动态 import 当前 demo 的入口模块 —— import 的副作用就是自注册（见 core/registry.js）。
await import(bootstrap.entry);

const demoUrl = (base) => (path) => `${base}/${String(path).replace(/^\//, '')}`;
const scene = normalizeScene(
  getRegisteredDemo(bootstrap.demoId).createScene({
    demoId: bootstrap.demoId,
    stageEl: el.stage,
    getClient: () => liveClient,
    api: { base: bootstrap.api.base, url: demoUrl(bootstrap.api.base) },
    assets: { base: bootstrap.assetsBase, url: demoUrl(bootstrap.assetsBase) },
    shell: { setStatus },
  }),
  { demoId: bootstrap.demoId, title: bootstrap.title },
);

initPage();

function initPage() {
  el.modeBadge.textContent = 'Realtime API';
  el.modeBadge.className = 'mode-tag mode-real';
  el.kvMode.textContent = '—';
  el.kvCkpt.textContent = '—';
  document.title = scene.pageTitle;
  if (el.pageTitle) el.pageTitle.textContent = scene.pageTitle;
  if (el.scenePanelTitle) el.scenePanelTitle.textContent = scene.panelTitle;
  if (el.scenePanelHint) el.scenePanelHint.textContent = scene.panelHint;
  renderQuickSettingOptions();
  applyDefaults({});
  setWsState('idle');
  setMicLiveState('idle');
  setStatus('Loading defaults…');
  scene.render();
  loadFcBoardDefaults();
}

function renderQuickSettingOptions() {
  if (!el.quickSetting) return;
  el.quickSetting.innerHTML = [
    ...scene.presetOptions,
    { id: QUICK_SETTING_CUSTOM, label: 'Custom' },
  ].map((option) => (
    `<option value="${escapeHtml(option.id)}">${escapeHtml(option.label)}</option>`
  )).join('');
}

el.addTextSegment?.addEventListener('click', () => {
  systemSegments.push({ kind: 'text', text: '' });
  markQuickSettingCustom();
  renderSystemSegments();
});

el.addAudioSegment?.addEventListener('click', () => {
  systemSegments.push({ kind: 'audio', audio: { source: 'path', file_path: '' } });
  markQuickSettingCustom();
  renderSystemSegments();
});

el.quickSetting?.addEventListener('change', () => {
  if (el.quickSetting.value === QUICK_SETTING_CUSTOM) return;
  applyQuickSetting(el.quickSetting.value);
});

el.resetSystemSegments?.addEventListener('click', () => {
  if (hasQuickSettings()) applyQuickSetting(selectedQuickSetting());
});

el.resetTtsPromptAudio?.addEventListener('click', () => {
  const setting = quickSettings.get(selectedQuickSetting());
  if (!setting || !el.ttsPromptAudioPath) return;
  el.ttsPromptAudioPath.value = setting.ttsPromptAudioPath || '';
});

el.useFirstSystemAudio?.addEventListener('click', () => {
  const firstAudio = systemSegments.find((segment) => segment.kind === 'audio');
  if (!firstAudio) {
    setStatus('System has no audio segment');
    return;
  }
  el.ttsPromptAudioPath.value = firstAudio.audio.file_path;
  markQuickSettingCustom();
});

el.systemSegments?.addEventListener('input', (event) => {
  const input = event.target.closest('[data-segment-value]');
  if (!input) return;
  const index = Number(input.dataset.segmentIndex);
  const segment = systemSegments[index];
  if (!segment) return;
  if (segment.kind === 'text') segment.text = input.value;
  else segment.audio.file_path = input.value;
  markQuickSettingCustom();
});

el.ttsPromptAudioPath?.addEventListener('input', () => {
  markQuickSettingCustom();
});

el.systemSegments?.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-segment-action]');
  if (!button) return;
  const index = Number(button.dataset.segmentIndex);
  const action = button.dataset.segmentAction;
  if (!Number.isInteger(index) || index < 0 || index >= systemSegments.length) return;
  if (action === 'remove') systemSegments.splice(index, 1);
  if (action === 'up' && index > 0) {
    [systemSegments[index - 1], systemSegments[index]] = [systemSegments[index], systemSegments[index - 1]];
  }
  if (action === 'down' && index < systemSegments.length - 1) {
    [systemSegments[index + 1], systemSegments[index]] = [systemSegments[index], systemSegments[index + 1]];
  }
  markQuickSettingCustom();
  renderSystemSegments();
});

el.debugToggle?.addEventListener('click', () => {
  el.debugDrawer?.classList.toggle('open');
});

el.streamFilter?.addEventListener('change', applyStreamFilter);

el.openStreamDialog?.addEventListener('click', () => {
  openStreamDialog();
});

el.streamDialog?.addEventListener('click', (event) => {
  if (event.target === el.streamDialog) el.streamDialog.close();
});

el.startMicLive.addEventListener('click', async () => {
  if (micLiveState !== 'idle' && micLiveState !== 'stopped' && micLiveState !== 'closed' && micLiveState !== 'error') return;
  let sessionInitPayload;
  try {
    setStatus('Preparing session (reading reference audio)…');
    sessionInitPayload = await buildSessionInitPayload();
  } catch (err) {
    setStatus(`Cannot start: ${err.message}`);
    return;
  }
  setMicLiveState('starting');
  clearViews();
  resetMicPeak();
  try {
    audioPlayer.init();
    audioPlayerReady = true;
  } catch (err) {
    console.warn('[AudioPlayer] init failed:', err);
  }
  try {
    liveClient = await createRealtimeSession(sessionInitPayload);
    micProvider = createMicProvider(liveClient);
    await micProvider.start();
    setMicLiveState('live');
    setStatus(scene.liveStatus);
  } catch (err) {
    setStatus(`Mic live error: ${err.message}`);
    stopMicLive('error');
  }
});

el.stopMicLive.addEventListener('click', () => {
  if (micLiveState === 'idle' || micLiveState === 'stopped' || micLiveState === 'closed') return;
  stopMicLive();
  setStatus('Stopped');
});

el.resumeCheckpoint?.addEventListener('click', async () => {
  if (
    resumeInProgress
    || !liveClient
    || latestCheckpoint?.resume?.status !== 'available'
  ) return;

  resumeInProgress = true;
  updateControlState();
  const throughUnitIndex = Number(latestCheckpoint.unit_index);
  try {
    const resumePayload = liveClient.buildResumePayload(throughUnitIndex);
    if (micProvider) {
      micProvider.stop();
      micProvider = null;
    }
    const disconnectedClient = liveClient;
    liveClient = null;
    disconnectedClient.disconnect('demo_resume_test');
    setMicLiveState('starting');
    setWsState('reconnecting');
    setStatus(`Reconnecting from Unit ${throughUnitIndex} checkpoint…`);
    await sleep(600);

    liveClient = await resumeRealtimeSession(resumePayload);
    micProvider = createMicProvider(liveClient);
    await micProvider.start();
    setMicLiveState('live');
    setWsState('connected');
    setStatus(`Resume succeeded · continuing after Unit ${throughUnitIndex}`);
  } catch (err) {
    console.error('[FC resume]', err);
    setMicLiveState('error');
    setWsState('closed');
    setStatus(`Resume failed: ${err.message}`);
  } finally {
    resumeInProgress = false;
    updateControlState();
  }
});

el.audioFileInput.addEventListener('change', () => {
  const file = el.audioFileInput.files?.[0];
  if (!file) return;
  el.userAudioPlayer.src = URL.createObjectURL(file);
});

el.runFileReplay.addEventListener('click', async () => {
  const file = el.audioFileInput.files?.[0];
  if (!file) {
    setStatus('Please choose an audio file');
    return;
  }
  let sessionInitPayload;
  try {
    sessionInitPayload = await buildSessionInitPayload();
  } catch (err) {
    setStatus(`Cannot start: ${err.message}`);
    return;
  }
  clearViews();
  setStatus('Decoding file...');
  const chunks = await decodeFileToChunks(file, {
    padBeforeSec: Number(el.padBeforeSec.value || 0),
    padAfterSec: Number(el.padAfterSec.value || 2),
  });
  const client = await createRealtimeSession(sessionInitPayload);
  setStatus(`Streaming ${chunks.length} chunks through /v1/realtime...`);
  el.userAudioPlayer.currentTime = 0;
  el.userAudioPlayer.play().catch(() => {});
  for (let i = 0; i < chunks.length; i++) {
    client.appendAudio({ audioBase64: float32ToBase64(chunks[i]), sampleRate: 16000 });
    sentChunkCount += 1;
    updateStats();
    setStatus(`Sent chunk ${i + 1}/${chunks.length}`);
    await sleep(1000);
  }
  client.close('file_replay_finished');
  setWsState('closed');
  setStatus('File replay finished');
});

async function createRealtimeSession(payload) {
  let resolveCreated;
  let rejectCreated;
  const created = new Promise((resolve, reject) => {
    resolveCreated = resolve;
    rejectCreated = reject;
  });
  const client = await createConnectedClient((event) => {
    if (event.type === 'session.created') resolveCreated(event);
    if (event.type === 'session.closed' || event.type === 'error') {
      rejectCreated(new Error(
        `${event.reason || event.error || 'session initialization failed'}`,
      ));
    }
  });
  client.initSession(payload);
  const timeout = new Promise((_, reject) => {
    setTimeout(() => reject(new Error('session.init timed out')), 120000);
  });
  await Promise.race([created, timeout]);
  return client;
}

async function createConnectedClient(onControlEvent = null) {
  const client = new FcRealtimeClient({
    onEvent: (event) => {
      applyApiEvent(event);
      onControlEvent?.(event);
    },
    onSend: (message) => appendStreamEvent('tx', message),
    onStatus: setStatus,
  });
  if (!apiEndpoint) throw new Error('No API endpoint selected');
  setStatus(`Connecting to ${apiEndpoint.realtime} …`);
  await client.connect(apiEndpoint.realtime);
  setWsState('connected');
  return client;
}

async function resumeRealtimeSession(payload) {
  let resolveResume;
  let rejectResume;
  const resumed = new Promise((resolve, reject) => {
    resolveResume = resolve;
    rejectResume = reject;
  });
  const client = await createConnectedClient((event) => {
    if (event.type === 'session.resumed') resolveResume(event);
    if (event.type === 'session.resume.failed') {
      rejectResume(new Error(`${event.code || 'resume_failed'}: ${event.message || ''}`));
    }
  });
  client.resumeSession(payload);
  const timeout = new Promise((_, reject) => {
    setTimeout(() => reject(new Error('session.resume timed out')), 120000);
  });
  await Promise.race([resumed, timeout]);
  return client;
}

function createMicProvider(client) {
  return new LiveMicProvider({
    onChunk: (chunk) => {
      client.appendAudio({ audioBase64: float32ToBase64(chunk), sampleRate: 16000 });
      sentChunkCount += 1;
      updateStats();
    },
    onLevel: updateMicLevel,
    onPermissionHint: (text) => {
      if (el.liveHint) {
        el.liveHint.textContent = text;
        el.liveHint.classList.add('warning');
      }
    },
    onState: setMicState,
    onStatus: setStatus,
  });
}

/** 把 Demo 侧的路径音频读成协议 v4 的 base64 WAV（Demo 服务读本机白名单目录里的文件）。 */
async function resolveAudio(filePath) {
  const path = String(filePath || '').trim();
  if (!path) throw new Error('Audio path is empty');
  if (!audioCache.has(path)) {
    const response = await fetch(`/api/demos/_audio?path=${encodeURIComponent(path)}`, { cache: 'no-store' });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `HTTP ${response.status} reading ${path}`);
    audioCache.set(path, { source: 'base64', format: 'wav', data: body.data });
  }
  return audioCache.get(path);
}

async function buildSessionInitPayload() {
  if (!apiEndpoint) throw new Error('No API endpoint selected');
  const validatedSegments = systemSegments.map((segment, index) => (
    validateSystemSegment(segment, index)
  ));
  const ttsPromptAudioPath = (el.ttsPromptAudioPath?.value || '').trim();
  if (!ttsPromptAudioPath) {
    throw new Error('TTS prompt audio path is required when generate_audio is true');
  }
  const segments = [];
  for (const segment of validatedSegments) {
    segments.push(segment.kind === 'audio'
      ? { kind: 'audio', audio: await resolveAudio(segment.audio.file_path) }
      : segment);
  }
  // 协议 v4：音频以 base64 上传；预算、调度、tts_pad、checkpoint profile 由端点上的模型决定，不发送。
  const payload = {
    protocol_version: '4',
    mode: 'full_duplex',
    fc_duplex: true,
    system: {
      segments,
      tools: validatedTools(),
    },
    tts_prompt_audio: await resolveAudio(ttsPromptAudioPath),
    generate_audio: true,
    config: {
      runtime: 'fc_duplex',
      auto_execute_tools: false,
      sample_rate: 16000,
      max_spoken_tokens: 24,
      decode_mode: 'greedy',
      // Demo 用协议状态机约束解码（FDB 这类测基本能力的评测才用 raise_protocol_error）。
      decoding_policy: 'constrained_decoding',
    },
  };
  const summary = summarizeEndpointInfo(endpointInfo);
  nsView.setBudgets(Number(summary.listening) || 0, Number(summary.speaking) || 0);
  return payload;
}

async function loadFcBoardDefaults() {
  try {
    const response = await fetch(bootstrap.api.defaults, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    fcBoardDefaults = await response.json();
    applyDefaults(fcBoardDefaults);
    setStatus('Click Start to begin.');
  } catch (err) {
    console.warn('[gallery] failed to load defaults:', err);
    fcBoardDefaults = null;
    applyDefaults({});
    setStatus('Click Start to begin. Defaults endpoint unavailable; using fallback prompt.');
  }
  setupApiEndpoints(fcBoardDefaults || {});
}

// ---------------------------------------------------------------- API 端点

function setupApiEndpoints(defaults) {
  const options = Array.isArray(defaults.api_endpoints) ? defaults.api_endpoints : [];
  if (el.apiEndpointOptions) {
    el.apiEndpointOptions.innerHTML = options.map((option) => (
      `<option value="${escapeHtml(option.url)}">${escapeHtml(option.name)}</option>`
    )).join('');
  }
  const saved = globalThis.localStorage?.getItem(API_ENDPOINT_KEY);
  const initial = saved || defaults.default_api_endpoint || options[0]?.url || '';
  if (el.apiEndpoint) el.apiEndpoint.value = initial;
  if (initial) selectApiEndpoint(initial);
  else renderEndpoint('请填写 API 端点地址（https://<host>:<port>/e/<端点名>）');
}

async function selectApiEndpoint(raw) {
  try {
    apiEndpoint = normalizeEndpoint(raw);
  } catch (err) {
    apiEndpoint = null;
    endpointInfo = null;
    renderEndpoint(err.message);
    return;
  }
  globalThis.localStorage?.setItem(API_ENDPOINT_KEY, apiEndpoint.base);
  if (el.apiEndpoint) el.apiEndpoint.value = apiEndpoint.base;
  el.kvMode.textContent = apiEndpoint.realtime;
  renderEndpoint('查询端点信息…');
  try {
    const response = await fetch(apiEndpoint.info, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    endpointInfo = await response.json();
    renderEndpoint();
  } catch (err) {
    endpointInfo = null;
    renderEndpoint(`查不到端点信息（${err.message}）。若是自签证书，先在新标签页打开 ${apiEndpoint.info} 并选择继续访问。`);
  }
}

function renderEndpoint(message = '') {
  const summary = summarizeEndpointInfo(endpointInfo);
  if (el.apiEndpointStatus) el.apiEndpointStatus.textContent = message || summary.statusText;
  if (el.kvCkpt) el.kvCkpt.textContent = summary.profileId || '—';
  if (el.checkpointProfileId) el.checkpointProfileId.value = summary.profileId;
  renderSessionConfig(summary);
  renderModelIdentity(summary.model);
}

/** 展示 tts_pad / 预算 / 调度：会话前是端点默认值，建会话后是 session.created.session_config 的实际生效值。 */
function renderSessionConfig(summary) {
  if (el.nonSpokenBudgetWhileListening) el.nonSpokenBudgetWhileListening.value = summary.listening;
  if (el.nonSpokenBudgetWhileSpeaking) el.nonSpokenBudgetWhileSpeaking.value = summary.speaking;
  if (el.nonSpokenScheduling && summary.scheduling) el.nonSpokenScheduling.value = summary.scheduling;
  if (el.ttsPadProtocolView) el.ttsPadProtocolView.value = summary.ttsPad;
  if (summary.listening || summary.speaking) nsView.setBudgets(Number(summary.listening) || 0, Number(summary.speaking) || 0);
}

el.applyApiEndpoint?.addEventListener('click', () => selectApiEndpoint(el.apiEndpoint?.value));
el.apiEndpoint?.addEventListener('change', () => selectApiEndpoint(el.apiEndpoint?.value));

function applyDefaults(defaults) {
  quickSettings = scene.buildQuickSettings(defaults);
  // 部署方按 checkpoint 指定启动 preset；不属于本场景的 preset id 回退到场景自己的默认值。
  // 没有任何 preset 的 demo（例如脚手架）是合法的：此时跳过 quick setting，不要去查空表。
  const initialSetting = quickSettings.has(defaults.default_quick_setting)
    ? defaults.default_quick_setting
    : scene.fallbackQuickSetting;
  if (initialSetting !== null && quickSettings.has(initialSetting)) {
    applyQuickSetting(initialSetting);
  } else {
    systemSegments = [];
    renderSystemSegments();
  }
  renderToolsSummary();
}

/** Session 面板的 Tools 一栏；工具可以随 preset 变，所以切 preset 时也要刷新。 */
function renderToolsSummary() {
  if (!el.kvTools) return;
  const names = validatedTools().map(tool => tool?.function?.name || tool?.name || 'tool');
  el.kvTools.textContent = names.length ? names.join(', ') : 'none';
}

/** 展示端点上部署的模型（会话前来自 /v1/info，会话建立后以 session.created.model 为准）。 */
function renderModelIdentity(identity) {
  if (el.kvProfileId) el.kvProfileId.textContent = identity?.profile_id || '—';
  if (!el.modelIdentity) return;
  if (!identity) {
    el.modelIdentity.innerHTML = '<dd class="placeholder">端点信息不可用</dd>';
    return;
  }
  el.modelIdentity.innerHTML = MODEL_IDENTITY_ROWS.map(([key, label]) => {
    const value = identity[key];
    const shown = (value === null || value === undefined || value === '') ? '—' : String(value);
    const hint = MODEL_IDENTITY_HINTS[key];
    const dtTitle = hint ? ` title="${escapeHtml(hint)}"` : '';
    return `<dt${dtTitle}>${escapeHtml(label)}${hint ? ' <span class="hint-mark">?</span>' : ''}</dt>`
      + `<dd title="${escapeHtml(hint ? `${shown}\n\n${hint}` : shown)}">${escapeHtml(shown)}</dd>`;
  }).join('');
}

function selectedQuickSetting() {
  const selected = el.quickSetting?.value;
  return quickSettings.has(selected) ? selected : scene.fallbackQuickSetting;
}

/** 本 demo 有没有可选的 preset。没有 preset 是合法状态（例如纯展示型 demo）。 */
function hasQuickSettings() {
  return quickSettings.size > 0;
}

function applyQuickSetting(settingId) {
  const setting = quickSettings.get(settingId);
  if (!setting) throw new Error(`Unknown quick setting: ${settingId}`);
  const cloned = cloneQuickSetting(setting);
  systemSegments = cloneSystemSegments(cloned.segments);
  if (el.ttsPromptAudioPath) {
    el.ttsPromptAudioPath.value = cloned.ttsPromptAudioPath;
  }
  if (el.quickSetting) el.quickSetting.value = settingId;
  renderSystemSegments();
  renderToolsSummary();
}

function markQuickSettingCustom() {
  if (el.quickSetting) el.quickSetting.value = QUICK_SETTING_CUSTOM;
}

function validatedTools() {
  const tools = scene.tools(fcBoardDefaults || {}, selectedQuickSetting());
  if (!Array.isArray(tools)) throw new Error('System tools must be an array');
  return JSON.parse(JSON.stringify(tools));
}

function cloneSystemSegments(segments) {
  if (!Array.isArray(segments)) return [];
  return segments.map((segment, index) => validateSystemSegment(segment, index));
}

function validateSystemSegment(segment, index) {
  if (!segment || typeof segment !== 'object' || Array.isArray(segment)) {
    throw new Error(`System segment ${index + 1} must be an object`);
  }
  if (segment.kind === 'text') {
    if (typeof segment.text !== 'string') {
      throw new Error(`Text segment ${index + 1} must contain a string`);
    }
    return { kind: 'text', text: segment.text };
  }
  if (segment.kind === 'audio') {
    if (
      !segment.audio
      || typeof segment.audio !== 'object'
      || segment.audio.source !== 'path'
      || typeof segment.audio.file_path !== 'string'
      || !segment.audio.file_path.trim()
    ) {
      throw new Error(`Audio segment ${index + 1} requires a nonempty path source`);
    }
    return {
      kind: 'audio',
      audio: {
        source: 'path',
        file_path: segment.audio.file_path.trim(),
      },
    };
  }
  throw new Error(`System segment ${index + 1} has unsupported kind`);
}

function renderSystemSegments() {
  if (!el.systemSegments) return;
  el.systemSegments.innerHTML = systemSegments.map((segment, index) => {
    const controls = `
      <div class="segment-controls">
        <button type="button" data-segment-action="up" data-segment-index="${index}" ${index === 0 ? 'disabled' : ''}>↑</button>
        <button type="button" data-segment-action="down" data-segment-index="${index}" ${index === systemSegments.length - 1 ? 'disabled' : ''}>↓</button>
        <button type="button" data-segment-action="remove" data-segment-index="${index}">Remove</button>
      </div>`;
    if (segment.kind === 'text') {
      return `<article class="segment-item">
        <div class="segment-head"><strong>${index + 1}. Text</strong>${controls}</div>
        <textarea data-segment-value data-segment-index="${index}" rows="4" spellcheck="false">${escapeHtml(segment.text)}</textarea>
      </article>`;
    }
    const filePath = segment?.audio?.file_path || '';
    return `<article class="segment-item">
      <div class="segment-head"><strong>${index + 1}. Audio path</strong>${controls}</div>
      <input data-segment-value data-segment-index="${index}" type="text" spellcheck="false" value="${escapeHtml(filePath)}" placeholder="absolute path to audio" />
    </article>`;
  }).join('') || '<div class="segment-empty">No system segments. Add text or audio.</div>';
}

function applyApiEvent(event) {
  appendStreamEvent('rx', event);
  appendTimeline(`${event.type} ${event.kind || ''} ${event.tool_call_id || ''}`.trim());
  switch (event.type) {
    case 'session.created':
      if (event.model) {
        renderModelIdentity(event.model);
        if (el.kvCkpt) el.kvCkpt.textContent = event.model.profile_id || '—';
      }
      if (event.session_config) renderSessionConfig(summarizeSessionConfig(event.session_config));
      setStatus('Session created');
      return;
    case 'session.resumed':
      setStatus(`Session resumed after Unit ${event.through_unit_index}`);
      return;
    case 'session.resume.failed':
      setStatus(`Resume failed: ${event.code || 'unknown'}`);
      return;
    case 'response.warning':
      setStatus(`Warning: ${event.message || event.code || 'unknown'}`);
      return;
    case 'session.closed':
      setWsState('closed');
      releaseMicLive('closed');
      setStatus(`Session closed: ${event.reason || 'closed'}`);
      return;
    case 'response.unit.started':
      scene.onUnitStarted(event);
      return;
    case 'response.non_spoken.end':
      nsView.markNonSpokenEnd(event.unit_index, event.reason);
      return;
    case 'response.unit.committed':
      handleUnitCheckpoint(event);
      return;
    case 'response.spoken.delta':
      handleSpokenDelta(event);
      return;
    case 'response.spoken.end':
      handleSpokenEnd(event);
      return;
    case 'response.think.begin':
      nsView.beginSegment('think', event.unit_index);
      scene.onThinkBegin(event);
      return;
    case 'response.think.delta':
      generationStepCount += (Array.isArray(event.steps) ? event.steps.length : 0);
      nsView.appendSteps(event.unit_index, event.steps);
      scene.onThinkDelta(event);
      updateStats();
      return;
    case 'response.think.end':
      nsView.endSegment('think', event.unit_index, { fullText: event.full_text });
      scene.onThinkEnd(event);
      return;
    case 'response.tool_call.begin':
      nsView.beginSegment('tool_call', event.unit_index, { toolCallId: event.tool_call_id });
      return;
    case 'response.tool_call.delta':
      generationStepCount += (Array.isArray(event.steps) ? event.steps.length : 0);
      nsView.appendSteps(event.unit_index, event.steps);
      updateStats();
      return;
    case 'response.tool_call.done':
      handleToolCallDone(event);
      return;
    case 'response.debug':
      handleDebugEvent(event);
      return;
    default:
      return;
  }
}

function handleSpokenDelta(event) {
  const steps = Array.isArray(event.steps) ? event.steps : [];
  generationStepCount += steps.length;
  const text = textFromSteps(steps);
  if (text) enqueueSpeech(text, 0);
  if (event.audio) {
    handleSpokenOutput({
      isListen: false,
      isSpeaking: true,
      audioBase64: event.audio,
      sampleRate: event.sample_rate || 24000,
    });
  }
  updateStats();
}

function handleSpokenEnd(event) {
  // 该 Unit 的 spoken 决策决定 non-spoken 用哪档 budget：listen → listening，其余 → speaking
  const lane = event.reason === 'listen' ? 'listen' : 'speak';
  nsView.setUnitLane(event.unit_index, lane);
  if (event.reason === 'listen' || event.reason === 'turn_eos') {
    if (audioPlayerReady && audioPlayer.turnActive) audioPlayer.endTurn();
  }
}

function handleUnitCheckpoint(event) {
  latestCheckpoint = event;
  const unitIndex = Number(event.unit_index);
  const resume = event.resume || {};
  const available = resume.status === 'available';
  const detail = available
    ? `Unit ${unitIndex} · available`
    : `Unit ${unitIndex} · unavailable (${resume.reason || 'unknown'})`;
  if (el.kvCkpt) el.kvCkpt.textContent = detail;
  if (el.resumeSummary) {
    el.resumeSummary.textContent = available
      ? `Unit ${unitIndex} can reconnect without server state`
      : detail;
    el.resumeSummary.classList.toggle('available', available);
    el.resumeSummary.classList.toggle('failed', !available);
  }
  updateControlState();
}

function handleDebugEvent(event) {
  const debug = event.debug || {};
  if ('used' in debug && el.debugBudgetUsed) {
    el.debugBudgetUsed.textContent = formatMaybeNumber(debug.used);
  }
  if ('estimated_max_budget_1s' in debug && el.debugBudgetMax) {
    el.debugBudgetMax.textContent = formatMaybeNumber(debug.estimated_max_budget_1s);
  }
  if (el.debugBudgetUpdated) {
    el.debugBudgetUpdated.textContent = new Date().toLocaleTimeString();
  }
}

function handleToolCallDone(event) {
  nsView.endSegment('tool_call', event.unit_index, { fullText: event.full_text || event.error || '' });
  scene.onToolCallDone(event);
}

function handleSpokenOutput({ isListen, isSpeaking, audioBase64, sampleRate }) {
  if (!audioPlayerReady) return;
  if (isListen) {
    if (audioPlayer.turnActive) audioPlayer.endTurn();
    return;
  }
  if (isSpeaking && audioBase64) {
    if (!audioPlayer.turnActive) audioPlayer.beginTurn();
    audioPlayer.playChunk(audioBase64, performance.now());
    aiAudioCount += 1;
    updateStats();
    appendAiAudio(aiAudioCount, audioBase64, sampleRate || 24000);
  }
}

function enqueueSpeech(text, audioMs) {
  if (!text) return;
  const chars = [...text];
  const pendingBudget = Math.max(0, audioMs || chars.length * 80);
  const backlog = speechQueue.length;
  const stepMs = Math.max(15, Math.round(pendingBudget / Math.max(1, chars.length + backlog * 0.5)));
  for (const ch of chars) speechQueue.push({ ch, stepMs });
  pumpSpeechQueue();
}

function pumpSpeechQueue() {
  if (speechDrainer) return;
  const step = () => {
    const item = speechQueue.shift();
    if (!item) {
      speechDrainer = null;
      return;
    }
    appendSpeechChar(item.ch);
    speechDrainer = setTimeout(step, item.stepMs);
  };
  speechDrainer = setTimeout(step, 0);
}

function appendSpeechChar(ch) {
  const placeholder = el.aiSpeech.querySelector('.placeholder');
  if (placeholder) placeholder.remove();
  const last = el.aiSpeech.lastElementChild;
  const now = Date.now();
  if (last && last.classList.contains('speech-row') && (now - Number(last.dataset.ts || 0)) < 1500) {
    last.textContent = (last.textContent || '') + ch;
    last.dataset.ts = String(now);
  } else {
    const item = document.createElement('div');
    item.className = 'speech-row';
    item.dataset.ts = String(now);
    item.textContent = ch;
    el.aiSpeech.appendChild(item);
  }
  const speechScroller = el.aiSpeech.parentElement;
  if (speechScroller) requestAnimationFrame(() => { speechScroller.scrollTop = speechScroller.scrollHeight; });
}

function appendAiAudio(index, audioBase64, sampleRate) {
  if (!el.debugDrawer?.classList.contains('open')) return;
  const wavBase64 = float32Base64ToWavBase64(audioBase64, sampleRate || 24000);
  const wrap = document.createElement('div');
  wrap.className = 'audio-item';
  const label = document.createElement('span');
  label.textContent = `AI chunk ${index}`;
  const audio = document.createElement('audio');
  audio.controls = true;
  audio.src = `data:audio/wav;base64,${wavBase64}`;
  wrap.appendChild(label);
  wrap.appendChild(audio);
  el.aiAudioList.appendChild(wrap);
}

function stopMicLive(finalState = 'stopped') {
  setMicLiveState('stopping');
  if (micProvider) {
    micProvider.stop();
    micProvider = null;
  }
  if (liveClient) {
    try { liveClient.close('mic_live_stopped'); } catch (_) {}
    liveClient = null;
  }
  setMicLiveState(finalState);
  setWsState('closed');
}

function releaseMicLive(finalState = 'closed') {
  if (micProvider) {
    micProvider.stop();
    micProvider = null;
  }
  liveClient = null;
  setMicLiveState(finalState);
}

function clearViews() {
  scene.reset();
  nsView.reset();
  el.timeline.innerHTML = '';
  el.streamFeed.innerHTML = '<div class="placeholder">还没有数据流</div>';
  speechQueue.length = 0;
  if (speechDrainer) { clearTimeout(speechDrainer); speechDrainer = null; }
  el.aiSpeech.innerHTML = '<div class="placeholder">AI 还没开口</div>';
  el.aiAudioList.innerHTML = '';
  sentChunkCount = 0;
  aiAudioCount = 0;
  generationStepCount = 0;
  latestCheckpoint = null;
  if (el.kvCkpt) el.kvCkpt.textContent = 'No checkpoint';
  if (el.resumeSummary) {
    el.resumeSummary.textContent = 'No resumable checkpoint';
    el.resumeSummary.classList.remove('available', 'failed');
  }
  updateStats();
  updateMicLevel(0);
}

function setMicState(state) {
  if (!el.statusLamp) return;
  const label = el.statusLamp.querySelector('.label');
  if (state === 'live') {
    el.statusLamp.classList.add('live');
    if (label) label.textContent = 'LIVE';
  } else {
    el.statusLamp.classList.remove('live');
    if (label) label.textContent = state ? state.toUpperCase() : 'IDLE';
  }
}

function setMicLiveState(state) {
  micLiveState = state;
  setMicState(state);
  updateControlState();
}

function updateControlState() {
  const active = micLiveState === 'starting' || micLiveState === 'live' || micLiveState === 'stopping';
  if (el.startMicLive) el.startMicLive.disabled = active;
  if (el.stopMicLive) el.stopMicLive.disabled = !active || micLiveState === 'stopping';
  if (el.resumeCheckpoint) {
    el.resumeCheckpoint.disabled = (
      resumeInProgress
      || micLiveState !== 'live'
      || !liveClient
      || latestCheckpoint?.resume?.status !== 'available'
    );
  }
}

function setWsState(state) {
  if (!el.wsState) return;
  el.wsState.textContent = state;
  el.wsState.classList.remove('connected', 'closed');
  if (state === 'connected') el.wsState.classList.add('connected');
  else if (state === 'closed' || state === 'idle') el.wsState.classList.add('closed');
}

function updateMicLevel(level) {
  // `level` 是 live-mic-provider 传来的原始 RMS ∈ [0, 1]
  const db = rmsToDb(level);
  const clamped = Math.max(0, Math.min(1, (db - MIC_METER_MIN_DB) / (MIC_METER_MAX_DB - MIC_METER_MIN_DB)));
  if (el.micLevelBar) el.micLevelBar.style.width = `${Math.round(clamped * 100)}%`;
  // 用户反馈过"模型不响应"往往是麦克风采到了静音（Chrome noiseSuppression
  // 太狠 / 权限没给 / 硬件哑了）。session peak 让用户一眼就能判断音频真的
  // 有没有进来：peak < -50 dB 基本可以断定是静音。
  if (db > micPeakDbSinceStart) micPeakDbSinceStart = db;
  if (db >= MIC_SPEECH_DB) micSpeechSeen = true;
  if (el.micLevelText) {
    const inTarget = (db >= MIC_TARGET_LO_DB && db <= MIC_TARGET_HI_DB) ? '✓' : ' ';
    el.micLevelText.textContent = `${db.toFixed(1)} dB ${inTarget}  (peak ${micPeakDbSinceStart.toFixed(1)}, target ${MIC_TARGET_LO_DB}~${MIC_TARGET_HI_DB} ≈ -23 LUFS)`;
  }
  updateOpeningSilenceWarning();
}

function updateOpeningSilenceWarning() {
  if (!el.micSilenceWarn) return;
  const warning = openingSilenceWarning({
    micLiveState,
    speechSeen: micSpeechSeen,
    waitedMs: micLiveStartedAt ? Date.now() - micLiveStartedAt : 0,
    peakDb: micPeakDbSinceStart,
  });
  el.micSilenceWarn.hidden = !warning.visible;
  if (warning.visible) el.micSilenceWarn.textContent = warning.text;
}

function resetMicPeak() {
  micPeakDbSinceStart = MIC_METER_MIN_DB;
  micLiveStartedAt = Date.now();
  micSpeechSeen = false;
  if (el.micSilenceWarn) el.micSilenceWarn.hidden = true;
}

function updateStats() {
  el.sentChunks.textContent = String(sentChunkCount);
  el.aiAudioCount.textContent = String(aiAudioCount);
  if (el.generationSteps) {
    el.generationSteps.textContent = String(generationStepCount);
  }
}

function setStatus(text) {
  el.status.textContent = text;
}

function appendTimeline(text) {
  const item = document.createElement('div');
  item.className = 'timeline-row';
  item.textContent = text;
  el.timeline.appendChild(item);
  el.timeline.scrollTop = el.timeline.scrollHeight;
}

function appendStreamEvent(direction, event) {
  if (!el.streamFeed) return;
  const placeholder = el.streamFeed.querySelector('.placeholder');
  if (placeholder) placeholder.remove();
  const eventId = `stream_evt_${++streamEventSeq}`;
  streamEvents.set(eventId, { direction, event });
  const row = document.createElement('details');
  row.className = `stream-row ${direction}`;
  row.dataset.eventId = eventId;
  const type = event.type || '(unknown)';
  row.dataset.category = streamEventCategory(direction, event);
  row.innerHTML = renderStreamRowContent({ direction, event, type, timeText: new Date().toLocaleTimeString() });
  el.streamFeed.appendChild(row);
  applyStreamFilter();
  el.streamFeed.scrollTop = el.streamFeed.scrollHeight;
}

function openStreamDialog() {
  if (!el.streamDialog || !el.streamModalList) return;
  const records = [...streamEvents.values()];
  if (el.streamModalCount) {
    el.streamModalCount.textContent = `${records.length} event${records.length === 1 ? '' : 's'}`;
  }
  el.streamModalList.innerHTML = records.map((record, index) => {
    const type = record.event.type || '(unknown)';
    return `<details class="stream-row ${record.direction} modal-stream-row">
      ${renderStreamRowContent({
        direction: record.direction,
        event: record.event,
        type,
        timeText: `#${index + 1}`,
      })}
    </details>`;
  }).join('') || '<div class="placeholder">No stream events yet</div>';
  el.streamDialog.showModal();
}

function renderStreamRowContent({ direction, event, type, timeText }) {
  const summary = summarizeStreamEvent(event);
  const json = prettyJson(event);
  const media = renderStreamEventMedia(event);
  return `
    <summary class="stream-head">
      <span class="stream-dir">${direction.toUpperCase()}</span>
      <span class="stream-type">${escapeHtml(type)}</span>
      <span class="stream-time">${escapeHtml(timeText)}</span>
    </summary>
    <div class="stream-body">${escapeHtml(summary)}</div>
    ${media}
    <pre class="stream-json">${escapeHtml(json)}</pre>
  `;
}

function renderStreamEventMedia(event) {
  let audioBase64 = '';
  let sampleRate = 24000;
  if (event.type === 'response.spoken.delta' && event.audio) {
    audioBase64 = event.audio;
    sampleRate = Number(event.sample_rate || 24000);
  } else if (event.type === 'input.append' && event.input?.audio_base64) {
    audioBase64 = event.input.audio_base64;
    sampleRate = Number(event.input.sample_rate || 16000);
  }
  if (!audioBase64) return '';
  try {
    const wavBase64 = float32Base64ToWavBase64(audioBase64, sampleRate);
    return `<div class="stream-media"><audio controls src="data:audio/wav;base64,${wavBase64}"></audio></div>`;
  } catch (err) {
    return `<div class="stream-media error">audio decode failed: ${escapeHtml(err?.message || err)}</div>`;
  }
}

function applyStreamFilter() {
  if (!el.streamFeed) return;
  const selected = el.streamFilter?.value || 'all';
  for (const row of el.streamFeed.querySelectorAll('.stream-row')) {
    const category = row.dataset.category || '';
    const direction = row.classList.contains('tx') ? 'tx' : 'rx';
    const visible = selected === 'all' || selected === category || selected === direction;
    row.classList.toggle('hidden', !visible);
  }
}

function streamEventCategory(direction, event) {
  const type = event.type || '';
  if (type === 'error' || type.endsWith('.error')) return 'error';
  if (type === 'response.warning') return 'warning';
  if (type.startsWith('session.')) return 'session';
  if (type.startsWith('input.')) return 'input';
  if (type.startsWith('response.unit.')) return 'unit';
  if (type.startsWith('response.spoken.')) return 'spoken';
  if (type === 'response.debug') return 'debug';
  if (type.startsWith('response.think')) return 'think';
  if (type.startsWith('response.tool_call')) return 'tool_call';
  return direction;
}

function summarizeStreamEvent(event) {
  const type = event.type || '';
  if (type === 'input.append') {
    const audio = event.input?.audio_base64 || '';
    return `audio_base64=${audio.length} chars · sample_rate=${event.input?.sample_rate || '-'}`;
  }
  if (type === 'session.init') {
    const payload = event.payload || {};
    return [
      `protocol=${payload.protocol_version || '-'}`,
      `mode=${payload.mode || '-'}`,
      `fc_duplex=${Boolean(payload.fc_duplex)}`,
      `segments=${Array.isArray(payload.system?.segments) ? payload.system.segments.length : 0}`,
      `tools=${(payload.system?.tools || []).map((tool) => tool?.function?.name || '?').join(',') || '-'}`,
      `generate_audio=${Boolean(payload.generate_audio)}`,
    ].join(' · ');
  }
  if (type.endsWith('.delta')) {
    const steps = Array.isArray(event.steps) ? event.steps : [];
    return `unit=${event.unit_index ?? '-'} · text=${textFromSteps(steps)} · steps=${steps.length} · audio=${String(event.audio || '').length}`;
  }
  if (type === 'response.unit.started') {
    return `unit=${event.unit_index ?? '-'} · input_id=${event.input_id || '-'} · tool_events=${(event.tool_events || []).length}`;
  }
  if (type === 'response.non_spoken.end') {
    return `unit=${event.unit_index ?? '-'} · reason=${event.reason || '-'}`;
  }
  if (type === 'response.unit.committed') {
    return `unit=${event.unit_index ?? '-'} · resume=${event.resume?.status || '-'}${event.resume?.reason ? ` (${event.resume.reason})` : ''}`;
  }
  if (type === 'response.warning') {
    return `unit=${event.unit_index ?? '-'} · code=${event.code || '-'} · ${event.message || ''}`;
  }
  if (type.startsWith('response.tool_call')) {
    return `tool_call_id=${event.tool_call_id || '-'} · unit=${event.unit_index ?? '-'} · ${event.full_text || event.error || ''}`;
  }
  if (type.startsWith('response.think')) {
    return `unit=${event.unit_index ?? '-'} · ${event.full_text || textFromSteps(event.steps)}`;
  }
  if (type === 'response.debug') {
    const debug = event.debug || {};
    return `used=${formatMaybeNumber(debug.used)} · max/1s=${formatMaybeNumber(debug.estimated_max_budget_1s)}`;
  }
  if (type === 'session.created') {
    return `session_id=${event.session_id || '-'} · mode=${event.mode || '-'}`;
  }
  if (type === 'session.resumed') {
    return `session_id=${event.session_id || '-'} · through_unit=${event.through_unit_index ?? '-'} · next_unit=${event.next_unit_index ?? '-'}`;
  }
  if (type === 'session.resume.failed') {
    return `code=${event.code || '-'} · unit=${event.unit_index ?? '-'} · ${event.message || ''}`;
  }
  if (type === 'session.queued' || type === 'session.queue_update') {
    return `position=${event.position ?? '-'} · eta=${event.estimated_wait_s ?? '-'}s`;
  }
  if (type === 'session.close' || type === 'session.closed') {
    return `reason=${event.reason || '-'}`;
  }
  return compactJson(event);
}

function compactJson(value) {
  try { return JSON.stringify(value); } catch (_) { return String(value); }
}

function requirePositiveInt(value, label) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must come from a valid Checkpoint Profile`);
  }
  return parsed;
}

function prettyJson(value) {
  try { return JSON.stringify(value, null, 2); } catch (_) { return String(value); }
}

function float32Base64ToWavBase64(audioBase64, sampleRate) {
  const binary = atob(audioBase64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const samples = new Float32Array(bytes.buffer);
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  const wav = new ArrayBuffer(44 + pcm.byteLength);
  const view = new DataView(wav);
  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + pcm.byteLength, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, pcm.byteLength, true);
  new Uint8Array(wav, 44).set(new Uint8Array(pcm.buffer));
  let out = '';
  const wavBytes = new Uint8Array(wav);
  for (let i = 0; i < wavBytes.length; i += 0x8000) {
    out += String.fromCharCode(...wavBytes.subarray(i, i + 0x8000));
  }
  return btoa(out);
}

function writeAscii(view, offset, value) {
  for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatMaybeNumber(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number' && Number.isFinite(value)) return String(Math.round(value));
  return String(value);
}
