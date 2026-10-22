import type { ContentPart } from '../session/types';

export function base64(buffer: ArrayBufferLike) {
  const bytes = new Uint8Array(buffer);
  let text = '';
  for (let i = 0; i < bytes.length; i += 8192)
    text += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(text);
}
export async function audioPcm(file: Blob): Promise<string> {
  const ctx = new AudioContext();
  try {
    const audio = await ctx.decodeAudioData(await file.arrayBuffer());
    const output = new Float32Array(Math.ceil(audio.duration * 16000));
    const channels = Array.from({ length: audio.numberOfChannels }, (_, i) =>
      audio.getChannelData(i),
    );
    for (let i = 0; i < output.length; i++) {
      const position = (i * audio.sampleRate) / 16000;
      const lo = Math.floor(position);
      const blend = position - lo;
      output[i] =
        channels.reduce(
          (sum, data) =>
            sum +
            (data[Math.min(lo, data.length - 1)] * (1 - blend) +
              data[Math.min(lo + 1, data.length - 1)] * blend),
          0,
        ) / channels.length;
    }
    return base64(output.buffer);
  } finally {
    await ctx.close();
  }
}
export async function contentFiles(files: File[]): Promise<ContentPart[]> {
  return Promise.all(
    files.map(async (file) => {
      if (file.size > 32 * 1024 * 1024)
        throw new Error(`${file.name} 超过 32 MB，请使用更小的文件。`);
      const type = file.type.startsWith('image/')
        ? 'image'
        : file.type.startsWith('audio/')
          ? 'audio'
          : file.type.startsWith('video/')
            ? 'video'
            : null;
      if (!type) throw new Error(`暂不支持 ${file.name} 的文件类型。`);
      return {
        type,
        data: type === 'audio' ? await audioPcm(file) : base64(await file.arrayBuffer()),
        name: file.name,
      };
    }),
  );
}
export async function wavReference(file: File) {
  if (file.size > 16 * 1024 * 1024) throw new Error('参考音频请使用 16 MB 以内的 WAV。');
  const pcm = Uint8Array.from(atob(await audioPcm(file)), (c) => c.charCodeAt(0));
  const floats = new Float32Array(pcm.buffer);
  const buffer = new ArrayBuffer(44 + floats.length * 2);
  const view = new DataView(buffer);
  const word = (offset: number, text: string) =>
    [...text].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  word(0, 'RIFF');
  view.setUint32(4, buffer.byteLength - 8, true);
  word(8, 'WAVE');
  word(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true);
  view.setUint32(28, 32000, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  word(36, 'data');
  view.setUint32(40, floats.length * 2, true);
  floats.forEach((sample, i) =>
    view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, sample)) * 32767), true),
  );
  return { name: file.name, data: base64(buffer), pcm: base64(pcm.buffer) };
}
