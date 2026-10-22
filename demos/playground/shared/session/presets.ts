import { applicationUrl, readJson } from './urls';
import type { CaseId, SessionSettings } from './types';

export const presetPage = (id: CaseId) =>
  id === 'audio-duplex' ? 'audio_duplex' : id === 'video-duplex' ? 'omni' : 'turnbased';
export async function loadOrdinaryPresets(
  s: SessionSettings,
  id: CaseId,
  selected?: string,
  signal?: AbortSignal,
  loadPolicy = false,
) {
  const base = s.presetSource;
  const [all, frontend, fallback] = await Promise.all([
    readJson(applicationUrl(base, '/api/presets'), signal).catch((e) => {
      if (signal?.aborted) throw e;
      return {};
    }),
    readJson(applicationUrl(base, '/api/frontend_defaults'), signal).catch((error) => {
      if (signal?.aborted) throw error;
      return null;
    }),
    readJson(applicationUrl(base, '/api/default_ref_audio'), signal).catch((e) => {
      if (signal?.aborted) throw e;
      return null;
    }),
  ]);
  const page = presetPage(id);
  const options = all[page] || [];
  const value =
    options.find((p: any) => p.id === selected) ||
    options.find((p: any) => p.id.includes('chinese')) ||
    options[0];
  if (!value && !frontend && !fallback) throw new Error('该地址未提供原页面的预设或默认配置。');
  const needsAudio =
    value &&
    ((value.ref_audio?.path && !value.ref_audio?.data) ||
      value.system_content?.some((part: any) => part.type === 'audio' && part.path && !part.data));
  const audio: any = needsAudio
    ? await readJson(
        applicationUrl(base, `/api/presets/${page}/${encodeURIComponent(value.id)}/audio`),
        signal,
      )
    : {};
  if (!audio.ref_audio && value?.ref_audio?.data) audio.ref_audio = value.ref_audio;
  const patch: Partial<SessionSettings> = { preset: value?.id || 'default' };
  if (id === 'turnbased') {
    let index = 0;
    patch.systemContent = (value?.system_content || s.systemContent).map((part: any) => {
      if (part.type !== 'audio') return { ...part };
      const resolved =
        (!part.data && part.path ? audio.system_content_audio?.[index++] : null) ||
        (part.data
          ? part
          : !value && fallback
            ? { data: fallback.base64, name: fallback.name, duration: fallback.duration }
            : part);
      if (value && part.path && !resolved?.data)
        throw new Error('预设中的参考音频未加载成功，请检查应用服务。');
      return {
        type: 'audio',
        data: resolved?.data,
        name: resolved?.name || part.name,
        duration: resolved?.duration || part.duration,
      };
    });
  } else {
    patch.systemPrompt = value?.system_prompt ?? s.systemPrompt;
    if (value?.ref_audio && !audio.ref_audio?.data)
      throw new Error('预设中的参考音频未加载成功，请检查应用服务。');
    if (!audio.ref_audio?.data && !s.referenceAudio && fallback?.base64)
      audio.ref_audio = { data: fallback.base64, name: fallback.name, duration: fallback.duration };
    patch.referenceAudio = audio.ref_audio?.data
      ? {
          name: audio.ref_audio.name,
          pcm: audio.ref_audio.data,
          data: '',
          duration: audio.ref_audio.duration,
        }
      : s.referenceAudio;
  }
  if ((loadPolicy || !selected) && frontend) {
    const mapping = {
      ...(id === 'audio-duplex' && s.playbackPreference
        ? {}
        : { playback_delay_ms: 'playbackDelay' }),
      sliding_window_mode: 'slidingWindowMode',
      basic_window_high_tokens: 'basicHigh',
      basic_window_low_tokens: 'basicLow',
      sliding_window_modes: 'allowedSlidingModes',
    } as const;
    for (const [wire, key] of Object.entries(mapping))
      if (frontend[wire] !== undefined) (patch as any)[key] = frontend[wire];
  }
  return {
    patch,
    options: (options.length ? options : [{ id: 'default', name: '服务默认' }]).map((p: any) => ({
      id: p.id,
      label: p.name || p.id,
    })),
  };
}
