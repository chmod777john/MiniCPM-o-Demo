// TTS framing belongs to the protocol, not the assistant's visible answer.
const ttsMarkers = ['<|tts_bos|>', '<|tts_eos|>'];

export function assistantText(text: string, pending = false): string {
  let visible = text.replace(/<\|tts_(?:bos|eos)\|>/g, '');
  // Stream chunks can split a marker anywhere. Hold only a known marker prefix;
  // the caller retains the raw accumulated text until the next chunk arrives.
  if (pending) {
    const start = visible.lastIndexOf('<|');
    if (start >= 0 && ttsMarkers.some((marker) => marker.startsWith(visible.slice(start))))
      visible = visible.slice(0, start);
    else if (visible.endsWith('<')) visible = visible.slice(0, -1);
  }
  return visible;
}
