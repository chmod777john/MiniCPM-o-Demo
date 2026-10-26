import { applicationUrl, readJson } from './urls';
import { wavReference } from '../media/files';
import type { CaseId, SessionSettings, SettingsSchema, AudioReference } from './types';

export interface ConversationDefaults {
  schema_version: 1;
  case_id: CaseId;
  protocol: 'chat' | 'ordinary';
  default_preset: string;
  default_audio: string;
  defaults: Partial<SessionSettings>;
  settings_schema: SettingsSchema;
  presets: { id: string; label: string; description?: string; settings: Record<string, any> }[];
  audio_assets: Record<string, { url: string; name: string }>;
}
export type PresetLoadMode = 'initialize' | 'preset' | 'reset';

export function numericSetting(s: SessionSettings, key: keyof SessionSettings, fallback: number[]) {
  const spec = s.settingsSchema?.[key];
  return [spec?.min ?? fallback[0], spec?.max ?? fallback[1], spec?.step ?? fallback[2] ?? 1];
}
export function settingOptions(s: SessionSettings, key: keyof SessionSettings, fallback: any[]) {
  return s.settingsSchema?.[key]?.options || fallback;
}

async function readDefaults(s: SessionSettings, id: CaseId, signal?: AbortSignal) {
  const path = `/api/demos/playground/cases/${id}/defaults`;
  let config: ConversationDefaults;
  try {
    config = await readJson(applicationUrl(s.presetSource, path), signal);
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error(
      `无法加载此体验的应用配置（${path}）：${error instanceof Error ? error.message : error}`,
    );
  }
  if (
    config.schema_version !== 1 ||
    config.case_id !== id ||
    !config.settings_schema ||
    !config.defaults ||
    !Array.isArray(config.presets) ||
    !config.presets.length ||
    config.protocol !== (id === 'turnbased' ? 'chat' : 'ordinary')
  )
    throw new Error('应用服务返回了不匹配的体验配置，请检查服务版本。');
  return config;
}

async function resolveAudio(
  base: string,
  config: ConversationDefaults,
  id: string,
  signal?: AbortSignal,
): Promise<AudioReference> {
  const asset = config.audio_assets[id];
  if (!asset || !asset.url.startsWith('/api/demos/playground/audio/'))
    throw new Error(`预设引用了未提供的音频：${id}`);
  const timeout = AbortSignal.timeout(20000);
  const response = await fetch(applicationUrl(base, asset.url), {
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
  });
  if (!response.ok || !response.headers.get('content-type')?.startsWith('audio/'))
    throw new Error(`参考音频 ${asset.name} 加载失败（HTTP ${response.status}）。`);
  const result = await wavReference(
    new File([await response.blob()], asset.name, { type: 'audio/wav' }),
  );
  signal?.throwIfAborted();
  return result;
}

export async function loadDefaultReference(s: SessionSettings, id: CaseId, signal?: AbortSignal) {
  const config = await readDefaults(s, id, signal);
  return resolveAudio(s.presetSource, config, config.default_audio, signal);
}

/** Initial defaults, explicit preset switching and full reset have separate merge rules. */
export async function loadConversationPresets(
  s: SessionSettings,
  id: CaseId,
  selected?: string,
  signal?: AbortSignal,
  mode: PresetLoadMode = 'preset',
) {
  const config = await readDefaults(s, id, signal);
  const preserveCustom = mode === 'initialize' && s.preset === 'custom';
  const preset =
    config.presets.find((p) => p.id === selected) ||
    config.presets.find((p) => p.id === config.default_preset) ||
    config.presets[0];
  if (mode === 'preset' && selected && !config.presets.some((p) => p.id === selected))
    throw new Error('此预设已不在应用配置中，请重新选择。');
  const patch: Partial<SessionSettings> = {};
  if (!s.configReady || mode === 'reset') {
    for (const key of Object.keys(config.settings_schema) as (keyof SessionSettings)[]) {
      if (Object.hasOwn(config.defaults, key) && Object.hasOwn(s, key))
        (patch as any)[key] = config.defaults[key];
    }
    // Persisted/user-edited values take priority on initialization, never on full reset.
    if (mode !== 'reset')
      for (const key of s.userOverrides || []) {
        if (Object.hasOwn(config.settings_schema, key)) (patch as any)[key] = s[key];
      }
  }
  if (mode !== 'initialize' || !s.configReady) {
    const audioCache = new Map<string, Promise<AudioReference>>();
    const audio = (id: string) => {
      if (!audioCache.has(id)) audioCache.set(id, resolveAudio(s.presetSource, config, id, signal));
      return audioCache.get(id)!;
    };
    if (!preserveCustom) {
      if (id === 'turnbased') {
        patch.systemContent = await Promise.all(
          preset.settings.systemContent.map(async (part: any) => {
            if (part.type !== 'audio') return { ...part };
            const value = await audio(part.asset);
            return {
              type: 'audio' as const,
              data: value.pcm,
              name: value.name,
              duration: value.duration,
            };
          }),
        );
      } else {
        patch.systemPrompt = preset.settings.systemPrompt;
        patch.referenceAudio = preset.settings.referenceAudio
          ? await audio(preset.settings.referenceAudio.asset)
          : undefined;
      }
      // Future presets may include request settings explicitly; devices/media choices remain local.
      for (const [key, value] of Object.entries(preset.settings)) {
        if (config.settings_schema[key]?.owner === 'request') (patch as any)[key] = value;
      }
      patch.preset = preset.id;
    } else if (id !== 'turnbased' && !s.referenceAudio) {
      patch.referenceAudio = await audio(config.default_audio);
    }
  }
  const merged = { ...s, ...patch };
  // Validate choices against current application support while keeping valid user values.
  for (const [key, spec] of Object.entries(config.settings_schema)) {
    const value = (merged as any)[key];
    if (spec.options && !spec.options.includes(value)) (patch as any)[key] = spec.default;
    if (spec.type === 'boolean' && typeof value !== 'boolean') (patch as any)[key] = spec.default;
    if (spec.type === 'number') {
      (patch as any)[key] = Number.isFinite(value)
        ? Math.max(spec.min ?? -Infinity, Math.min(spec.max ?? Infinity, value))
        : spec.default;
    }
  }
  patch.settingsSchema = config.settings_schema;
  patch.allowedSlidingModes = (config.settings_schema.slidingWindowMode?.options as string[]) || [];
  patch.configReady = true;
  if (mode === 'reset') patch.userOverrides = [];
  else if (mode === 'preset')
    patch.userOverrides = (s.userOverrides || []).filter(
      (key) => !Object.hasOwn(preset.settings, key),
    );
  return {
    patch,
    config,
    options: config.presets.map(({ id, label, description }) => ({ id, label, description })),
  };
}
