import { measureLUFS } from '../../../../static/duplex/lib/lufs.js';
import { encodeWav } from '../../../../static/duplex/lib/mixer-controller.js';
import workletUrl from '../../../../static/duplex/lib/capture-processor.js?url';
import { audioPcm } from './files';
import type { SessionSettings } from '../session/types';

/** Offline mic + file preview, with the original mixer's LUFS and monitoring policy. */
export class MixerPreview {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: AudioBufferSourceNode | null = null;
  private node: AudioWorkletNode | null = null;
  private chunks: Float32Array[] = [];
  private cancelled = false;
  private done = false;
  private nodes: Record<string, AnalyserNode> = {};
  private gains: { mic?: GainNode; file?: GainNode; monitor?: GainNode } = {};
  private autoDb = 0;
  fileLUFS?: number;
  levels() {
    const result: Record<string, number | undefined> = { fileLUFS: this.fileLUFS };
    for (const [key, node] of Object.entries(this.nodes)) {
      const samples = new Float32Array(node.fftSize);
      node.getFloatTimeDomainData(samples);
      const rms = Math.sqrt(samples.reduce((n, v) => n + v * v, 0) / samples.length);
      result[key] = rms > 0 ? 20 * Math.log10(rms) : -60;
    }
    return result;
  }
  update(settings: SessionSettings) {
    if (this.gains.mic)
      this.gains.mic.gain.value = Math.pow(
        10,
        (settings.micTarget - settings.micMeasured + settings.micTrim) / 20,
      );
    if (this.gains.file) this.gains.file.gain.value = Math.pow(10, settings.fileTrim / 20);
    if (this.gains.monitor)
      this.gains.monitor.gain.value =
        (settings.monitor / 100) * (this.autoDb < 0 ? Math.pow(10, -this.autoDb / 20) : 1);
  }
  constructor(private onDone: (blob: Blob) => void) {}
  private ensure() {
    if (this.cancelled) throw new Error('试听已取消。');
  }
  async start(settings: SessionSettings) {
    if (!settings.inputFile) throw new Error('请先选择文件。');
    try {
      const encoded = await audioPcm(settings.inputFile);
      this.ensure();
      const pcm = new Float32Array(Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0)).buffer);
      const lufs = measureLUFS(pcm, 16000);
      this.fileLUFS = lufs;
      const autoDb = (this.autoDb = Number.isFinite(lufs) ? settings.fileTarget - lufs : 0);
      const gain = Math.pow(10, autoDb / 20);
      pcm.forEach((value, i) => (pcm[i] = value * gain));
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          ...(settings.micDevice ? { deviceId: { exact: settings.micDevice } } : {}),
        },
      });
      if (this.cancelled) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      this.stream = stream;
      const context = (this.context = new AudioContext({ sampleRate: 16000 }));
      await context.audioWorklet.addModule(workletUrl);
      this.ensure();
      const capture = (this.node = new AudioWorkletNode(context, 'capture-processor', {
        processorOptions: { chunkSize: 16000 },
      }));
      const micGain = context.createGain();
      micGain.gain.value = Math.pow(
        10,
        (settings.micTarget - settings.micMeasured + settings.micTrim) / 20,
      );
      context.createMediaStreamSource(stream).connect(micGain).connect(capture);
      const fileGain = context.createGain();
      fileGain.gain.value = Math.pow(10, settings.fileTrim / 20);
      const file = (this.source = context.createBufferSource());
      file.buffer = context.createBuffer(1, pcm.length, 16000);
      file.buffer.getChannelData(0).set(pcm);
      file.connect(fileGain).connect(capture);
      const monitor = context.createGain();
      monitor.gain.value = (settings.monitor / 100) * (autoDb < 0 ? Math.pow(10, -autoDb / 20) : 1);
      fileGain.connect(monitor).connect(context.destination);
      this.gains = { mic: micGain, file: fileGain, monitor };
      for (const [key, source] of Object.entries({ mic: micGain, file: fileGain, mix: capture })) {
        const analyser = context.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        this.nodes[key] = analyser;
      }
      if (settings.speakerDevice && (context as any).setSinkId)
        await (context as any).setSinkId(settings.speakerDevice);
      this.ensure();
      const silent = context.createGain();
      silent.gain.value = 0;
      capture.connect(silent).connect(context.destination);
      capture.port.onmessage = (event) => {
        if (!this.cancelled && event.data.type === 'chunk')
          this.chunks.push(event.data.audio.slice());
      };
      capture.port.postMessage({ command: 'start' });
      // Keep collecting the microphone until the user ends the mixer preview.
      await context.resume();
      this.ensure();
      file.start();
    } catch (error) {
      this.dispose();
      throw error;
    }
  }
  async finish() {
    if (this.done || this.cancelled) return;
    this.done = true;
    this.node?.port.postMessage({ command: 'stop' });
    await new Promise((resolve) => setTimeout(resolve, 100));
    if (this.cancelled) return;
    const pcm = new Float32Array(this.chunks.reduce((n, c) => n + c.length, 0));
    let pos = 0;
    for (const c of this.chunks) {
      pcm.set(c, pos);
      pos += c.length;
    }
    this.dispose();
    this.onDone(encodeWav(pcm, 16000));
  }
  dispose() {
    this.cancelled = true;
    this.nodes = {};
    this.gains = {};
    this.source?.disconnect();
    try {
      this.source?.stop();
    } catch {}
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    void this.context?.close().catch(() => {});
  }
}
