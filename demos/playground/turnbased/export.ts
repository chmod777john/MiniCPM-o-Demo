import { encodeWav } from '../../../static/duplex/lib/mixer-controller.js';
import type { ContentPart, Message } from '../shared/session/types';
const bytes = (value: string) => Uint8Array.from(atob(value), (c) => c.charCodeAt(0));
async function part(item: ContentPart): Promise<any> {
  if (item.type === 'text') return item.text ? { type: 'text', text: item.text } : null;
  if (!item.data) return null;
  const data = bytes(item.data);
  if (item.type === 'audio')
    return {
      type: 'audio',
      audio: new Uint8Array(await encodeWav(new Float32Array(data.buffer), 16000).arrayBuffer()),
      format: 'wav',
    };
  return { type: item.type, [item.type]: data, format: item.type === 'image' ? 'png' : 'mp4' };
}
export async function exportConversation(system: ContentPart[], messages: Message[]) {
  const rows: any[] = [];
  const systemContent = (
    await Promise.all(system.filter((p) => ['text', 'audio'].includes(p.type)).map(part))
  ).filter(Boolean);
  if (systemContent.length) rows.push({ role: 'system', content: systemContent });
  for (const message of messages) {
    if (message.failed || message.pending) continue;
    let content: any[];
    if (message.role === 'assistant' && (message.audioPlayer || message.audioUrl)) {
      const wav = message.audioPlayer
        ? await message.audioPlayer.wav().arrayBuffer()
        : await (await fetch(message.audioUrl!)).arrayBuffer();
      content = [
        { type: 'text_and_audio', text: message.text, audio: new Uint8Array(wav), format: 'wav' },
      ];
    } else
      content = (
        await Promise.all(
          (Array.isArray(message.content)
            ? message.content
            : [{ type: 'text' as const, text: message.text }]
          ).map(part),
        )
      ).filter(Boolean);
    if (content.length) rows.push({ role: message.role, content });
  }
  if (!messages.length) throw new Error('还没有可导出的对话。');
  const { encode } = await import('../../../static/lib/msgpack.mjs');
  const data = encode({ messages: rows });
  const url = URL.createObjectURL(new Blob([data as BlobPart], { type: 'application/x-msgpack' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `conversation_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.msgpack`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
