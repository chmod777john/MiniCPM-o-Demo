import type { ChatAudioPlayer } from '../../turnbased/audio';
import type { LucideIcon } from 'lucide-react';

export type CaseId =
  | 'turnbased'
  | 'audio-duplex'
  | 'video-duplex'
  | 'board'
  | 'uicard'
  | 'subagent'
  | 'rejection'
  | 'standalone';
export type SessionPhase =
  'idle' | 'connecting' | 'queued' | 'listening' | 'speaking' | 'generating' | 'error';
export type ContentPart = {
  type: 'text' | 'image' | 'audio' | 'video';
  text?: string;
  data?: string;
  name?: string;
  duration?: number;
};
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  content?: string | ContentPart[];
  pending?: boolean;
  failed?: boolean;
  files?: string[];
  audioUrl?: string;
  audioPlayer?: ChatAudioPlayer;
  cancelled?: boolean;
  chatMeta?: {
    mode: string;
    requestId: string;
    totalMs: number;
    ttfsMs?: number;
    processMs?: number;
    inputTokens?: number;
    generatedTokens?: number;
  };
}
export interface CaseDefinition {
  id: CaseId;
  title: string;
  label: string;
  description: string;
  icon: LucideIcon;
  group: 'conversation' | 'agent';
  protocol: 'chat' | 'ordinary' | 'fc';
  accent: string;
  prompts: string[];
  promptIcons?: LucideIcon[];
  video?: boolean;
  loadScene?: () => Promise<SceneFactory>;
}
export type AudioReference = { name: string; data: string; pcm: string; duration?: number };
export interface SessionSettings {
  configReady: boolean;
  endpoint: string;
  gallery: string;
  presetSource: string;
  apiService: string;
  systemPrompt: string;
  systemContent: ContentPart[];
  generateAudio: boolean;
  streaming: boolean;
  maxTokens: number;
  playbackDelay: number;
  playbackPreference: boolean;
  fcVideo: boolean;
  segmentsJson: string;
  toolsJson: string;
  referenceAudio?: AudioReference;
  ttsReferenceAudio?: AudioReference;
  ttsReferenceMode: 'extract' | 'independent';
  referencePath: string;
  preset: string;
  maxKvTokens: number;
  slidingWindowMode: 'off' | 'basic' | 'kv_surgery';
  allowedSlidingModes: string[];
  basicHigh: number;
  basicLow: number;
  contextMaxUnits: number;
  textSink: number;
  textWindow: number;
  binaryDecision: boolean;
  speakThreshold: number;
  imageHD: boolean;
  visionHD: 'off' | 'fixed' | 'strategy';
  fixedSlices: number;
  strategySlices: number;
  recordSession: boolean;
  recordingMime: string;
  recordingBitrate: number;
  recordingSubtitles: boolean;
  subtitleHeight: number;
  subtitleBottom: number;
  subtitleTop: number;
  screenSubtitleHeight: number;
  screenSubtitleBottom: number;
  screenSubtitleTop: number;
  inputMode: 'live' | 'file';
  inputFile?: File;
  fileAudioMode: 'file' | 'mic' | 'mixed';
  padBefore: number;
  padAfter: number;
  frameOffset: number;
  fileVolume: number;
  micTarget: number;
  micTrim: number;
  micMeasured: number;
  fileTarget: number;
  fileTrim: number;
  monitor: number;
  micDevice: string;
  speakerDevice: string;
  mirror: boolean;
  frontCamera: boolean;
}
export interface SessionView {
  phase: SessionPhase;
  configLoading: boolean;
  detail: string;
  error: string;
  messages: Message[];
  events: { id: number; type: string; text: string; time: string }[];
  startedAt: number | null;
  recordingSessionId: string;
  recordingUrl: string;
  recordingPending: boolean;
  ledger: any[];
  sessionId: string;
  unit: number;
  micLevel: number;
  muted: boolean;
  camera: boolean;
  model: string;
  modelIdentity: Record<string, any>;
  modelSource: 'unavailable' | 'endpoint' | 'session';
  configSource: 'unavailable' | 'endpoint' | 'session';
  effectiveConfig: Record<string, any>;
  metrics: Record<string, any>;
  paused: boolean;
  forceListen: boolean;
  hdEnabled: boolean;
  dirty: boolean;
}
// Existing scene modules remain isolated at this boundary. New views can be React-native.
export interface Scene {
  render(): void;
  reset?(): void | Promise<void>;
  pause?(): void;
  dispose?(): void;
  onExternalEvent?(id: string, text: string): void;
  prepareSystem?(segments: any[]): any[] | Promise<any[]>;
  presetOptions?: { id: string; label: string }[];
  fallbackQuickSetting?: string;
  buildQuickSettings?(defaults: any): Map<string, any>;
  tools?(defaults: any, preset?: string): any[];
  onReady?(): void;
  onUnitStarted?(event: any): void;
  onThinkBegin?(event: any): void;
  onThinkDelta?(event: any): void;
  onThinkEnd?(event: any): void;
  onToolCallBegin?(event: any): void;
  onToolCallDelta?(event: any, text: string): void;
  onToolCallDone?(event: any): void | Promise<void>;
  onSpokenDelta?(event: any, text: string): void;
  onSpokenEnd?(event: any): void;
}
export interface SceneContext {
  stageEl: HTMLElement;
  getClient: () => any;
  api: { base: string; url: (path: string) => string };
  assets: { base: string; url: (path: string) => string };
  demoId: string;
  shell: { setStatus: (text: string) => void; setDraft?: (text: string) => void };
}
export interface SceneFactory {
  create: (context: SceneContext) => Scene;
  css: string;
}
export const emptyView = (): SessionView => ({
  phase: 'idle',
  configLoading: false,
  detail: '',
  error: '',
  messages: [],
  events: [],
  startedAt: null,
  recordingSessionId: '',
  recordingUrl: '',
  recordingPending: false,
  ledger: [],
  sessionId: '',
  unit: 0,
  micLevel: 0,
  muted: false,
  camera: false,
  model: '',
  modelIdentity: {},
  modelSource: 'unavailable',
  configSource: 'unavailable',
  effectiveConfig: {},
  metrics: {},
  paused: false,
  forceListen: false,
  hdEnabled: true,
  dirty: false,
});
