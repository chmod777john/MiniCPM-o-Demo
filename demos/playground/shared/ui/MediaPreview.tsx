import { useEffect, useState } from 'react';
import { encodeWav } from '../../../../static/duplex/lib/mixer-controller.js';
import type { ContentPart } from '../session/types';
export function MediaPreview({
  part,
  sampleRate = 16000,
}: {
  part: ContentPart;
  sampleRate?: number;
}) {
  const [url, setUrl] = useState('');
  useEffect(() => {
    if (!part.data) {
      setUrl('');
      return;
    }
    let local = '';
    try {
      if (part.type === 'audio') {
        const pcm = new Float32Array(
          Uint8Array.from(atob(part.data), (c) => c.charCodeAt(0)).buffer,
        );
        local = URL.createObjectURL(encodeWav(pcm, sampleRate));
        setUrl(local);
      } else
        setUrl(
          `data:${part.type === 'image' ? (part.data.startsWith('iVBOR') ? 'image/png' : 'image/jpeg') : 'video/mp4'};base64,${part.data}`,
        );
    } catch {
      setUrl('');
    }
    return () => {
      if (local) URL.revokeObjectURL(local);
    };
  }, [part.data, part.type, sampleRate]);
  if (!url) return null;
  if (part.type === 'audio') return <audio controls className="message-audio" src={url} />;
  if (part.type === 'video') return <video className="message-media" controls src={url} />;
  return (
    <a href={url} target="_blank" rel="noreferrer">
      <img className="message-media" src={url} alt={part.name || '输入图像'} />
    </a>
  );
}
