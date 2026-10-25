import type { CaseId, ContentPart, SessionSettings } from './types';

// Bootstrap values only. Conversation defaults/schema are loaded from the application before use.
export function defaultsFor(id: CaseId): SessionSettings {
  return {
    configReady: false,
    userOverrides: [],
    shareSource: '',
    apiBase: '',
    modelId: '',
    gallery: '',
    presetSource: '',
    systemPrompt: '',
    systemContent: [],
    generateAudio: id !== 'turnbased',
    streaming: true,
    maxTokens: 1024,
    playbackDelay: 200,
    playbackPreference: false,
    fcVideo: false,
    segmentsJson: '',
    toolsJson: '',
    referencePath: '',
    preset: '',
    ttsReferenceMode: 'extract',
    maxKvTokens: 262144,
    slidingWindowMode: 'basic',
    allowedSlidingModes: ['off', 'basic', 'kv_surgery'],
    basicHigh: 8000,
    basicLow: 6000,
    contextMaxUnits: 250,
    textSink: 512,
    textWindow: 10240,
    binaryDecision: true,
    speakThreshold: 0.3,
    imageHD: false,
    visionHD: 'fixed',
    fixedSlices: 2,
    strategySlices: 4,
    recordSession: true,
    recordingMime: 'auto',
    recordingBitrate: 2000000,
    recordingSubtitles: true,
    subtitleHeight: 25,
    subtitleBottom: 90,
    subtitleTop: 15,
    screenSubtitleHeight: 20,
    screenSubtitleBottom: 80,
    screenSubtitleTop: 30,
    inputMode: 'live',
    fileAudioMode: 'file',
    padBefore: 0,
    padAfter: 2,
    frameOffset: 0.5,
    micTarget: -23,
    micTrim: 0,
    micMeasured: -23,
    fileTarget: -33,
    fileTrim: 0,
    monitor: 50,
    fileVolume: id === 'audio-duplex' ? 1 : 0.3,
    micDevice: '',
    speakerDevice: '',
    mirror: !['turnbased', 'audio-duplex', 'video-duplex'].includes(id),
    frontCamera: !['turnbased', 'audio-duplex', 'video-duplex'].includes(id),
  };
}
export function wireContent(parts: ContentPart[], system = true) {
  return parts.flatMap((part) => {
    if (part.type === 'text') return part.text ? [{ type: 'text', text: part.text }] : [];
    if (!part.data) return [];
    return [
      {
        type: part.type,
        data: part.data,
        ...(!system ? { name: part.name } : {}),
        ...(part.type === 'video' || (!system && part.type === 'audio')
          ? { duration: part.duration }
          : {}),
      },
    ] as any[];
  });
}
export function ttsReference(settings: SessionSettings, chat: boolean) {
  if (settings.ttsReferenceMode === 'independent') return settings.ttsReferenceAudio?.pcm;
  if (!chat) return settings.referenceAudio?.pcm;
  const audios = settings.systemContent.filter((part) => part.type === 'audio' && part.data);
  return audios.length === 1 && (!audios[0].duration || audios[0].duration <= 20)
    ? audios[0].data
    : undefined;
}
export function ordinaryPayload(s: SessionSettings, video: boolean) {
  if (s.slidingWindowMode === 'basic' && s.basicHigh <= s.basicLow)
    throw new Error('Basic 滑窗启动阈值必须大于保留阈值。');
  const tts = ttsReference(s, false);
  return {
    config: {
      sliding_window_mode: s.slidingWindowMode,
      basic_window_high_tokens: s.basicHigh,
      basic_window_low_tokens: s.basicLow,
      context_max_units: s.contextMaxUnits,
      text_sink: s.textSink,
      text_sliding_window: s.textWindow,
      ls_decision_mode: s.binaryDecision ? 'binary' : 'legacy',
      speak_threshold: Math.max(0.01, Math.min(0.99, s.speakThreshold)),
      ...(video
        ? { strategy_hd: s.visionHD === 'strategy', strategy_hd_max_slice_nums: s.strategySlices }
        : {}),
    },
    ...(video
      ? {
          max_slice_nums:
            s.visionHD === 'strategy' ? null : s.visionHD === 'fixed' ? s.fixedSlices : 1,
        }
      : {}),
    use_tts: s.generateAudio,
    ...(s.referenceAudio?.pcm ? { ref_audio_base64: s.referenceAudio.pcm } : {}),
    ...(tts && tts !== s.referenceAudio?.pcm ? { tts_ref_audio_base64: tts } : {}),
  };
}

// Keep the old pages' configuration persistence separate from transient session/media state.
export function persistedSettings(id: CaseId, s: SessionSettings): Partial<SessionSettings> {
  const keys: (keyof SessionSettings)[] = ['apiBase', 'modelId'];
  if (id === 'audio-duplex')
    keys.push(
      'preset',
      'inputMode',
      'fileAudioMode',
      'padBefore',
      'padAfter',
      'playbackDelay',
      'playbackPreference',
      'basicHigh',
      'basicLow',
      'contextMaxUnits',
      'textSink',
      'textWindow',
      'binaryDecision',
      'speakThreshold',
      'systemPrompt',
      'ttsReferenceMode',
      'recordSession',
      'fileTarget',
      'fileTrim',
      'micTarget',
      'micTrim',
      'monitor',
      'micDevice',
      'speakerDevice',
    );
  else if (id === 'turnbased') keys.push('preset');

  const result = Object.fromEntries(keys.map((key) => [key, s[key]]));
  // Custom uploaded audio/segments cannot be reconstructed from a preset ID after reload.
  if (result.preset === 'custom' && id !== 'audio-duplex') delete result.preset;
  result.userOverrides = (s.userOverrides || []).filter((key) => keys.includes(key));
  return result;
}
