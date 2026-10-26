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
    const offline = new OfflineAudioContext(1, Math.ceil(audio.duration * 16000), 16000);
    const source = offline.createBufferSource();
    source.buffer = audio;
    source.connect(offline.destination);
    source.start();
    const output = await offline.startRendering();
    return base64(output.getChannelData(0).buffer);
  } finally {
    await ctx.close();
  }
}
export async function contentFiles(files: File[]): Promise<ContentPart[]> {
  return Promise.all(
    files.map(async (file) => {
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
        ...(type === 'video' || type === 'audio'
          ? { duration: await mediaDuration(file, type) }
          : {}),
      };
    }),
  );
}
export async function wavReference(file: File, maxSeconds = 20) {
  const pcm = Uint8Array.from(atob(await audioPcm(file)), (c) => c.charCodeAt(0));
  const floats = new Float32Array(pcm.buffer);
  if (floats.length > maxSeconds * 16000) throw new Error('参考音频不能超过 20 秒。');
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
  return {
    name: file.name,
    data: base64(buffer),
    pcm: base64(pcm.buffer),
    duration: floats.length / 16000,
  };
}

async function mediaDuration(file: File, type: 'video' | 'audio'): Promise<number> {
  const element = document.createElement(type);
  const url = URL.createObjectURL(file);
  try {
    return await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('读取媒体时长超时。')), 15000);
      element.onloadedmetadata = () => {
        clearTimeout(timer);
        resolve(Number.isFinite(element.duration) ? element.duration : 0);
      };
      element.onerror = () => {
        clearTimeout(timer);
        reject(new Error('无法读取媒体文件。'));
      };
      element.preload = 'metadata';
      element.src = url;
    });
  } finally {
    element.removeAttribute('src');
    URL.revokeObjectURL(url);
  }
}
