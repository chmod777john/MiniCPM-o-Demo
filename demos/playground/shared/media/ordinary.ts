import workletUrl from '../../../../static/duplex/lib/capture-processor.js?url';
import { FileAudioProvider } from './ordinary-audio-file.js';
import { FileMediaProvider } from './ordinary-video-file.js';
import type { SessionSettings } from '../session/types';

export type MediaChunk = { audio: Float32Array; frameBase64?: string };
export interface InputSource {
  start(): Promise<void>;
  stop(): void;
  setMuted?(muted: boolean): void;
  pause?(): void;
  resume?(): void;
  updateMixer?(settings: SessionSettings): void;
  setSpeaker?(deviceId: string): Promise<void>;
  meterNodes?(): Record<string, AnalyserNode | undefined>;
  fileLUFS?(): number | undefined;
  getAudioOutputStream?(): MediaStream | null;
  getFullAudioBuffer?(): Float32Array | null;
  flip?(): Promise<void>;
  toggleMirror?(): void;
}

/** Ordinary duplex keeps native camera resolution; FC uses its own locked-size capture. */
export class OrdinaryLive implements InputSource {
  private stream: MediaStream | null = null;
  private camera: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private recordingStream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private cancelled = false;
  private canvas = document.createElement('canvas');
  private flipping = false;
  private front: boolean;
  private mirror: boolean;
  constructor(
    private settings: SessionSettings,
    private video: HTMLVideoElement | null,
    private chunk: (c: MediaChunk) => void,
    private level: (value: number) => void,
    private onError: (error: unknown) => void,
  ) {
    this.front = settings.frontCamera;
    this.mirror = settings.mirror;
  }
  private ensure() {
    if (this.cancelled) throw new Error('输入启动已取消。');
  }
  async start() {
    if (this.video) await this.openCamera();
    this.ensure();
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        ...(this.video ? { channelCount: 1 } : {}),
        ...(this.settings.micDevice ? { deviceId: { exact: this.settings.micDevice } } : {}),
      },
      video: false,
    });
    if (this.cancelled) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    this.stream = stream;
    const ctx = (this.ctx = new AudioContext({ sampleRate: 16000 }));
    await ctx.audioWorklet.addModule(workletUrl);
    this.ensure();
    this.node = new AudioWorkletNode(ctx, 'capture-processor', {
      processorOptions: { chunkSize: 16000 },
    });
    this.node.port.postMessage({ command: 'start' });
    this.node.port.onmessage = (event) => {
      if (this.cancelled || event.data.type !== 'chunk') return;
      try {
        const audio = event.data.audio;
        this.level(
          Math.sqrt(audio.reduce((sum: number, v: number) => sum + v * v, 0) / audio.length),
        );
        this.chunk({ audio, ...(this.video ? { frameBase64: this.frame() } : {}) });
      } catch (error) {
        this.onError(error);
      }
    };
    ctx.createMediaStreamSource(stream).connect(this.node);
    const silent = ctx.createGain();
    silent.gain.value = 0;
    this.node.connect(silent).connect(ctx.destination);
    const recording = ctx.createMediaStreamDestination();
    this.node.connect(recording);
    this.recordingStream = recording.stream;
    await ctx.resume();
    this.ensure();
  }
  private async openCamera() {
    if (!this.video) return;
    this.camera?.getTracks().forEach((track) => track.stop());
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: this.front ? 'user' : 'environment' },
    });
    if (this.cancelled) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    this.camera = stream;
    this.video.removeAttribute('src');
    this.video.style.display = '';
    this.video.srcObject = stream;
    this.video.muted = true;
    this.video.style.transform = this.front !== this.mirror ? 'scaleX(-1)' : 'none';
    await this.video.play();
    this.ensure();
  }
  private frame() {
    const video = this.video!;
    this.canvas.width = video.videoWidth;
    this.canvas.height = video.videoHeight;
    if (!video.videoWidth) throw new Error('摄像头没有可用画面。');
    const ctx = this.canvas.getContext('2d')!;
    ctx.save();
    if (this.mirror) {
      ctx.translate(this.canvas.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(video, 0, 0);
    ctx.restore();
    return this.canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
  }
  getAudioOutputStream() {
    return this.recordingStream;
  }
  async flip() {
    if (this.flipping) throw new Error('摄像头正在切换，请稍候。');
    this.flipping = true;
    const previous = this.front;
    this.front = !previous;
    try {
      await this.openCamera();
    } catch (error) {
      this.front = previous;
      if (!this.cancelled) await this.openCamera().catch((recovery) => this.onError(recovery));
      throw error;
    } finally {
      this.flipping = false;
    }
  }
  toggleMirror() {
    this.mirror = !this.mirror;
    if (this.video) this.video.style.transform = this.front !== this.mirror ? 'scaleX(-1)' : 'none';
  }
  setMuted(muted: boolean) {
    this.stream?.getAudioTracks().forEach((track) => {
      track.enabled = !muted;
    });
  }
  stop() {
    this.cancelled = true;
    this.node?.port.postMessage({ command: 'stop' });
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.camera?.getTracks().forEach((t) => t.stop());
    void this.ctx?.close().catch(() => {});
    if (this.video) this.video.srcObject = null;
  }
}

/** Same original file algorithms, with scoped DOM dependencies instead of page-global IDs. */
export function ordinaryFile(
  settings: SessionSettings,
  video: HTMLVideoElement | null,
  chunk: (c: MediaChunk) => void,
  onEnd: () => void,
  log: (s: string) => void,
): InputSource {
  if (!settings.inputFile) throw new Error('请先选择输入文件。');
  const audio = document.createElement('audio');
  const fields: Record<string, any> = {
    fileAudioEl: audio,
    videoEl: video,
    frameCanvas: document.createElement('canvas'),
    filePlaybackVol: { value: settings.fileVolume * 100 },
    mxMicTarget: { value: settings.micTarget },
    mxMicTrim: { value: settings.micTrim },
    mxFileTarget: { value: settings.fileTarget },
    mxFileTrim: { value: settings.fileTrim },
    mxMonitor: { value: settings.monitor },
    mxFileMeasured: {},
    mxFileAuto: {},
  };
  let cancelled = false;
  const env = {
    settings,
    log,
    field: (id: string) => fields[id],
    ensure: () => {
      if (cancelled) throw new Error('文件准备已取消。');
    },
  };
  const options = {
    audioMode: video && settings.fileAudioMode === 'file' ? 'video' : settings.fileAudioMode,
    padBeforeSec: settings.padBefore,
    padAfterSec: settings.padAfter,
    frameOffset: settings.frameOffset,
  };
  const Provider = (video ? FileMediaProvider : FileAudioProvider) as any;
  const provider = new Provider(settings.inputFile, options, env);
  provider.onChunk = (c: MediaChunk) => {
    if (!cancelled) chunk(c);
  };
  provider.onEnd = () => {
    if (!cancelled) onEnd();
  };
  return {
    async start() {
      try {
        env.ensure();
        if (settings.speakerDevice) {
          const element = (video || audio) as any;
          try {
            if (!element.setSinkId) throw new Error('浏览器不支持选择扬声器，将使用系统默认。');
            await element.setSinkId(settings.speakerDevice);
          } catch (error) {
            log(String(error));
          }
          env.ensure();
        }
        await provider.start();
        if (!cancelled && settings.speakerDevice && provider._micCtx?.setSinkId)
          await provider._micCtx
            .setSinkId(settings.speakerDevice)
            .catch((error: unknown) => log(String(error)));
        if (cancelled) provider.stop();
        else if (!video && settings.fileAudioMode === 'file') audio.volume = settings.fileVolume;
      } catch (error) {
        provider.stop();
        throw error;
      }
    },
    stop() {
      cancelled = true;
      provider.stop();
    },
    async setSpeaker(deviceId: string) {
      const element = (video || audio) as HTMLMediaElement & {
        setSinkId?: (id: string) => Promise<void>;
      };
      if (provider._micCtx?.setSinkId) await provider._micCtx.setSinkId(deviceId || '');
      if (element.setSinkId) await element.setSinkId(deviceId || '');
      else if (deviceId) throw new Error('浏览器不支持选择扬声器，请使用系统默认。');
    },
    meterNodes() {
      const n = provider.mixerNodes;
      return { mic: n.micAnalyser, file: n.fileAnalyser, mix: n.mixAnalyser };
    },
    fileLUFS() {
      return provider.measuredLUFS;
    },
    setMuted(muted: boolean) {
      provider._micStream?.getAudioTracks().forEach((track: MediaStreamTrack) => {
        track.enabled = !muted;
      });
    },
    updateMixer(next: SessionSettings) {
      const nodes = provider.mixerNodes;
      if (nodes.micGain)
        nodes.micGain.gain.value = Math.pow(
          10,
          (next.micTarget - next.micMeasured + next.micTrim) / 20,
        );
      if (
        nodes.monitorEl &&
        (settings.fileAudioMode !== 'mixed' || next.fileVolume !== settings.fileVolume)
      )
        nodes.monitorEl.volume = next.fileVolume;
      if (nodes.fileGain) nodes.fileGain.gain.value = Math.pow(10, next.fileTrim / 20);
      if (nodes.monitorGain) nodes.monitorGain.gain.value = next.monitor / 100;
      if (
        nodes.monitorEl &&
        settings.fileAudioMode === 'mixed' &&
        next.monitor !== settings.monitor
      )
        nodes.monitorEl.volume = next.monitor / 100;
      settings = next;
      for (const [id, key] of Object.entries({
        mxMicTarget: 'micTarget',
        mxMicTrim: 'micTrim',
        mxFileTarget: 'fileTarget',
        mxFileTrim: 'fileTrim',
        mxMonitor: 'monitor',
      }))
        fields[id].value = (next as any)[key];
    },
    getAudioOutputStream() {
      return provider.getAudioOutputStream?.() || null;
    },
    getFullAudioBuffer() {
      return provider.getFullAudioBuffer?.() || null;
    },
    pause() {
      provider.pause();
    },
    resume() {
      provider.resume();
    },
  };
}
