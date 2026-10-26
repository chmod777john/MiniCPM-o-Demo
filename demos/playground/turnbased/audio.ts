import { encodeWav } from '../../../static/duplex/lib/mixer-controller.js';
/** Native simplex PCM timeline. Playback controls are independent of generation. */
export class ChatAudioPlayer {
  ctx: AudioContext | null = null;
  turnActive = false;
  playing = false;
  finished = false;
  private chunks: Float32Array[] = [];
  private sources = new Set<AudioBufferSourceNode>();
  private samples = 0;
  private cursor = 0;
  private origin = 0;
  private started = 0;
  private next = 0;
  private revision = 0;
  private listeners = new Set<() => void>();
  private timer?: ReturnType<typeof setInterval>;
  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };
  snapshot = () => this.revision;
  private changed() {
    this.revision++;
    this.listeners.forEach((fn) => fn());
  }
  get duration() {
    return this.samples / 24000;
  }
  get position() {
    return this.playing && this.ctx
      ? Math.min(
          this.duration,
          this.origin + Math.max(0, Math.min(this.ctx.currentTime, this.next) - this.started),
        )
      : this.cursor;
  }
  init() {
    if (!this.ctx || this.ctx.state === 'closed') this.ctx = new AudioContext();
  }
  beginTurn() {
    this.turnActive = true;
  }
  endTurn() {
    this.turnActive = false;
    this.finished = true;
    this.changed();
  }
  playChunk(base64: string) {
    const pcm = new Float32Array(Uint8Array.from(atob(base64), (c) => c.charCodeAt(0)).buffer);
    if (!pcm.length) return;
    const first = !this.samples;
    this.chunks.push(pcm);
    this.samples += pcm.length;
    if (first) this.play();
    else if (this.playing) this.schedule(pcm);
    this.changed();
  }
  private schedule(pcm: Float32Array) {
    if (!this.ctx) return;
    const buffer = this.ctx.createBuffer(1, pcm.length, 24000);
    buffer.getChannelData(0).set(pcm);
    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.ctx.destination);
    const start = Math.max(this.next, this.ctx.currentTime);
    // Exclude time spent waiting for the next generated chunk from the audio cursor.
    this.started += Math.max(0, start - this.next);
    source.start(start);
    this.next = start + buffer.duration;
    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      source.disconnect();
      this.checkEnd();
    };
  }
  private checkEnd() {
    if (
      this.playing &&
      this.finished &&
      !this.sources.size &&
      this.position >= this.duration - 0.01
    ) {
      this.cursor = this.duration;
      this.playing = false;
      clearInterval(this.timer);
      this.changed();
    }
  }
  play() {
    this.init();
    if (!this.ctx || !this.samples) return;
    this.stopSources();
    if (this.cursor >= this.duration && this.finished) this.cursor = 0;
    void this.ctx.resume();
    this.origin = this.cursor;
    this.started = this.next = this.ctx.currentTime;
    this.playing = true;
    let consumed = 0;
    for (const chunk of this.chunks) {
      const end = consumed + chunk.length / 24000;
      if (end > this.cursor)
        this.schedule(chunk.subarray(Math.max(0, Math.floor((this.cursor - consumed) * 24000))));
      consumed = end;
    }
    clearInterval(this.timer);
    this.timer = setInterval(() => {
      this.checkEnd();
      this.changed();
    }, 100);
    this.changed();
  }
  pause() {
    this.cursor = this.position;
    this.playing = false;
    this.stopSources();
    clearInterval(this.timer);
    this.changed();
  }
  seek(seconds: number) {
    const playing = this.playing;
    this.pause();
    this.cursor = Math.max(0, Math.min(seconds, this.duration));
    if (playing) this.play();
    this.changed();
  }
  private stopSources() {
    for (const source of this.sources) {
      source.onended = null;
      try {
        source.stop();
      } catch {}
      source.disconnect();
    }
    this.sources.clear();
  }
  wav(): Blob {
    const pcm = new Float32Array(this.samples);
    let offset = 0;
    for (const c of this.chunks) {
      pcm.set(c, offset);
      offset += c.length;
    }
    return encodeWav(pcm, 24000);
  }
  stop() {
    this.pause();
    this.turnActive = false;
  }
  dispose() {
    this.stop();
    void this.ctx?.close().catch(() => {});
    this.listeners.clear();
  }
}

/** Legacy nonstream responses can contain a standard container or raw mono 24 kHz PCM. */
export async function outputWav(encoded: string): Promise<Blob> {
  const data = Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0));
  const context = new AudioContext();
  try {
    try {
      const audio = await context.decodeAudioData(data.buffer.slice(0));
      return encodeWav(audio.getChannelData(0), audio.sampleRate);
    } catch {
      return encodeWav(new Float32Array(data.buffer), 24000);
    }
  } finally {
    await context.close();
  }
}
