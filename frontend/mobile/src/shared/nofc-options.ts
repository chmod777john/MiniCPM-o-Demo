/** Wire options shared by the React pages and the desktop Omni bridge.
 * Inference stays in the HF capability; these functions only build API input.
 */
export type SlidingWindowMode = 'off' | 'basic' | 'kv_surgery'
export type DuplexOptions = {
  maxKvTokens: number
  playbackDelayMs: number
  slidingWindowMode: SlidingWindowMode
  basicWindowHighTokens: number
  basicWindowLowTokens: number
  contextMaxUnits: number
  textSink: number
  textSlidingWindow: number
  listenSpeakDecision: boolean
  speakThreshold: number
  useTts: boolean
  ttsRefAudioBase64: string | null
  visionHDMode: 'off' | 'fixed' | 'strategy'
  visionFixedSliceNums: number
  visionStrategySliceNums: number
}

export const DEFAULT_DUPLEX_OPTIONS: DuplexOptions = {
  maxKvTokens: 262144,
  playbackDelayMs: 200,
  slidingWindowMode: 'basic',
  basicWindowHighTokens: 8000,
  basicWindowLowTokens: 6000,
  contextMaxUnits: 250,
  textSink: 512,
  textSlidingWindow: 10240,
  listenSpeakDecision: true,
  speakThreshold: 0.3,
  useTts: true,
  ttsRefAudioBase64: null,
  visionHDMode: 'fixed',
  visionFixedSliceNums: 2,
  visionStrategySliceNums: 4,
}

export type FrontendDefaults = {
  max_kv_tokens?: number
  playback_delay_ms?: number
  sliding_window_mode?: SlidingWindowMode
  sliding_window_modes?: SlidingWindowMode[]
  basic_window_high_tokens?: number
  basic_window_low_tokens?: number
}

export function applyFrontendDefaults(options: DuplexOptions, defaults: FrontendDefaults): DuplexOptions {
  return {
    ...options,
    maxKvTokens: defaults.max_kv_tokens ?? options.maxKvTokens,
    playbackDelayMs: defaults.playback_delay_ms ?? options.playbackDelayMs,
    slidingWindowMode: defaults.sliding_window_mode ?? options.slidingWindowMode,
    basicWindowHighTokens: defaults.basic_window_high_tokens ?? options.basicWindowHighTokens,
    basicWindowLowTokens: defaults.basic_window_low_tokens ?? options.basicWindowLowTokens,
  }
}

export function validateDuplexOptions(options: DuplexOptions): void {
  if (options.slidingWindowMode === 'basic' && options.basicWindowHighTokens <= options.basicWindowLowTokens) {
    throw new Error('Basic sliding window: high threshold must exceed retained tokens / 滑窗触发阈值必须大于保留 token 数。')
  }
}

export function buildDuplexPreparePayload(
  options: DuplexOptions, refAudio: string | null, withVideo: boolean,
): Record<string, unknown> {
  validateDuplexOptions(options)
  return {
    config: {
      sliding_window_mode: options.slidingWindowMode,
      basic_window_high_tokens: options.basicWindowHighTokens,
      basic_window_low_tokens: options.basicWindowLowTokens,
      context_max_units: options.contextMaxUnits,
      text_sink: options.textSink,
      text_sliding_window: options.textSlidingWindow,
      ls_decision_mode: options.listenSpeakDecision ? 'binary' : 'legacy',
      speak_threshold: options.speakThreshold,
      ...(withVideo ? {
        strategy_hd: options.visionHDMode === 'strategy',
        strategy_hd_max_slice_nums: options.visionStrategySliceNums,
      } : {}),
    },
    use_tts: options.useTts,
    ...(withVideo ? {
      max_slice_nums: options.visionHDMode === 'fixed' ? options.visionFixedSliceNums : 1,
    } : {}),
    ...(refAudio ? { ref_audio_base64: refAudio } : {}),
    ...((options.ttsRefAudioBase64 || refAudio) ? { tts_ref_audio_base64: options.ttsRefAudioBase64 || refAudio } : {}),
  }
}

type ChatMessage = { role: string; content: string | { type: string }[] }
export type ChatOptions = {
  maxNewTokens: number
  lengthPenalty: number
  ttsEnabled: boolean
  refAudio: string | null
  hdImages: boolean
  enableThinking: boolean
}

export function buildChatPayload<M extends ChatMessage>(messages: M[], streaming: boolean, options: ChatOptions) {
  const hasVideo = messages.some((message) => Array.isArray(message.content) && message.content.some((item) => item.type === 'video'))
  return {
    messages,
    streaming,
    generation: {
      max_new_tokens: options.maxNewTokens,
      do_sample: true,
      length_penalty: options.lengthPenalty,
    },
    image: { max_slice_nums: hasVideo ? 1 : options.hdImages ? 2 : 1 },
    omni_mode: hasVideo,
    enable_thinking: options.enableThinking,
    use_tts_template: options.ttsEnabled,
    tts: {
      enabled: options.ttsEnabled,
      ...(options.ttsEnabled ? {
        mode: hasVideo ? 'omni' : 'audio_assistant',
        ...(options.refAudio ? { ref_audio_data: options.refAudio } : {}),
      } : {}),
    },
  }
}
