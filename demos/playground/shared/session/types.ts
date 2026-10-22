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
};
export interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  content?: string | ContentPart[];
  pending?: boolean;
  failed?: boolean;
  files?: string[];
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
export interface SessionSettings {
  endpoint: string;
  gallery: string;
  systemPrompt: string;
  generateAudio: boolean;
  streaming: boolean;
  maxTokens: number;
  playbackDelay: number;
  fcVideo: boolean;
  segmentsJson: string;
  toolsJson: string;
  referenceAudio?: { name: string; data: string; pcm: string };
  referencePath: string;
  preset: string;
}
export interface SessionView {
  phase: SessionPhase;
  detail: string;
  error: string;
  messages: Message[];
  events: { id: number; type: string; text: string; time: string }[];
  startedAt: number | null;
  sessionId: string;
  unit: number;
  micLevel: number;
  muted: boolean;
  camera: boolean;
  model: string;
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
  shell: { setStatus: (text: string) => void };
}
export interface SceneFactory {
  create: (context: SceneContext) => Scene;
  css: string;
}
export const defaultSettings: SessionSettings = {
  endpoint: '',
  gallery: '',
  systemPrompt: '你是 MiniCPM-o，一个友好、自然的多模态助手。',
  generateAudio: true,
  streaming: true,
  maxTokens: 512,
  playbackDelay: 300,
  fcVideo: false,
  segmentsJson: '',
  toolsJson: '',
  referencePath: '',
  preset: '',
};
export const emptyView = (): SessionView => ({
  phase: 'idle',
  detail: '',
  error: '',
  messages: [],
  events: [],
  startedAt: null,
  sessionId: '',
  unit: 0,
  micLevel: 0,
  muted: false,
  camera: false,
  model: '',
  dirty: false,
});
