// @ts-nocheck
// Ported from static/omni/omni-app.js @ a3e7c70.
// Preserve frame timing, padding, LUFS normalization, and AudioWorklet mixing.
// DOM/service dependencies are injected; the original page is untouched.
import { measureLUFS } from '../../../../static/duplex/lib/lufs.js';
import captureWorklet from '../../../../static/duplex/lib/capture-processor.js?url';
const SAMPLE_RATE_IN = 16000;
const CHUNK_MS = 1000;
class FileMediaProvider {
  /**
   * @param {File} file
   * @param {{audioMode: 'video'|'mic'|'mixed', frameOffset: number,
   *          padBeforeSec: number, padAfterSec: number}} opts
   */
  constructor(file, opts = {}, env) {
    this.env = env;

    this._file = file;
    this._audioMode = opts.audioMode || 'video';
    this._frameOffset = opts.frameOffset ?? 0.5;
    this._padBefore = Math.max(0, Math.floor(opts.padBeforeSec ?? 0));
    this._padAfter = Math.max(0, Math.floor(opts.padAfterSec ?? 2));

    // Pre-processed data (unpadded)
    this._videoAudioChunks = [];
    this._frames = [];
    this._mainChunks = 0; // video content chunk count

    // Padded arrays (video audio mode only)
    this._allAudio = [];
    this._allFrames = [];

    // Phase tracking
    this._chunkIdx = 0;
    this._mainStart = 0; // first main chunk index
    this._mainEnd = 0; // first trailing-pad chunk index
    this._grandTotal = 0; // total chunks including pads
    this._firstFrame = null;
    this._lastFrame = null;
    this._timer = null;

    // Mic pipeline (modes: mic, mixed — AudioWorklet graph)
    this._micStream = null;
    this._micCtx = null;
    this._micSource = null;
    this._captureNode = null;
    this._micGainNode = null;
    this._fileGainNode = null;
    this._monitorGainNode = null;
    this._micAnalyserNode = null;
    this._mixAnalyserNode = null;
    this._fileSrcNode = null;
    this._graphConnected = false;

    this._videoEl = this.env.field('videoEl');
    this._canvas = this.env.field('frameCanvas');
    this._ctx2d = this._canvas.getContext('2d');
    this._objectUrl = null;
    this._padBeforeTimer = null;
    this.paused = false;
  }

  async start() {
    this.env.log('Processing video file...');

    // 1. Get duration
    const duration = await this._getVideoDuration();
    this.env.ensure();
    if (!Number.isFinite(duration)) throw new Error('Invalid video duration');
    this._mainChunks = Math.floor(duration);
    if (this._mainChunks === 0) throw new Error('Video too short');

    // 2. Pre-extract frames (with offset k)
    this._frames = await this._extractFrames(this._mainChunks, this._frameOffset);
    this.env.ensure();
    this._firstFrame = this._frames[0] || null;
    this._lastFrame = this._frames[this._frames.length - 1] || null;

    // 3. Decode video audio (modes: video, mixed)
    if (this._audioMode === 'video' || this._audioMode === 'mixed') {
      await this._decodeVideoAudio(duration);
      this.env.ensure();
    }

    // 4. Setup mic but don't connect yet (modes: mic, mixed)
    if (this._audioMode === 'mic' || this._audioMode === 'mixed') {
      await this._setupMic();
      this.env.ensure();
    }

    // 5. Compute phase boundaries
    this._mainStart = this._padBefore;
    this._mainEnd = this._padBefore + this._mainChunks;
    this._grandTotal = this._padBefore + this._mainChunks + this._padAfter;

    const parts = [];
    if (this._padBefore > 0) parts.push(`${this._padBefore}s pad`);
    parts.push(`${this._mainChunks}s video`);
    if (this._padAfter > 0) parts.push(`${this._padAfter}s pad`);
    this.env.log(
      `Ready: [${parts.join(' + ')}] = ${this._grandTotal} chunks, audio=${this._audioMode}`,
    );

    // 6. Prepare video element (don't play yet — playback timing controlled by phases)
    this._objectUrl = URL.createObjectURL(this._file);
    this._videoEl.autoplay = false; // override HTML autoplay; we control play timing
    this._videoEl.src = this._objectUrl;
    this._videoEl.pause(); // ensure autoplay doesn't start
    this._videoEl.style.display = 'block';
    // mic-only: mute video; video-only/mixed: play original audio through speaker
    this._videoEl.muted = this._audioMode === 'mic';
    if (this._audioMode !== 'mic') {
      const volPct = parseInt(this.env.field('filePlaybackVol')?.value) ?? 30;
      this._videoEl.volume = volPct / 100;
    }

    // 7. Start chunk feeding
    this._chunkIdx = 0;
    this.running = true;

    if (this._audioMode === 'video') {
      this._buildPaddedArrays();
      this._feedNext();
    } else {
      // mic/mixed: mic runs for entire padded duration; file buffer + frames include padding
      // During padding, file audio = zero → mix = mic only; frames = first/last frame
      this._paddedFrames = [
        ...Array(this._padBefore).fill(this._firstFrame),
        ...this._frames,
        ...Array(this._padAfter).fill(this._lastFrame),
      ];
      this._startMainMicPhase();
    }
  }

  // ==================== Video audio mode: unified timer ====================

  _buildPaddedArrays() {
    const silence = () => new Float32Array(SAMPLE_RATE_IN);
    this._allAudio = [
      ...Array.from({ length: this._padBefore }, silence),
      ...this._videoAudioChunks,
      ...Array.from({ length: this._padAfter }, silence),
    ];
    this._allFrames = [
      ...Array(this._padBefore).fill(this._firstFrame),
      ...this._frames,
      ...Array(this._padAfter).fill(this._lastFrame),
    ];
  }

  _feedNext() {
    if (!this.running || this.paused) return;
    if (this._chunkIdx >= this._grandTotal) {
      this.running = false;
      if (this.onEnd) this.onEnd();
      return;
    }
    // Start video playback when entering main phase
    if (this._chunkIdx === this._mainStart) {
      this._videoEl.play();
    }
    const t0 = performance.now();
    const audio = this._allAudio[this._chunkIdx];
    const frame = this._allFrames[this._chunkIdx] || null;
    this._chunkIdx++;
    if (this.onChunk) this.onChunk({ audio, frameBase64: frame });
    const elapsed = performance.now() - t0;
    this._timer = setTimeout(() => this._feedNext(), Math.max(0, CHUNK_MS - elapsed));
  }

  // ==================== Mic/Mixed modes: phased feeding ====================

  async _setupMic() {
    const _omniMicId = this.env.settings.micDevice;
    this._micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, ...(_omniMicId ? { deviceId: { exact: _omniMicId } } : {}) },
      video: false,
    });

    if (this.env) {
      try {
        this.env.ensure();
      } catch (error) {
        this._micStream?.getTracks().forEach((track) => track.stop());
        throw error;
      }
    }
    this._micCtx = new AudioContext({ sampleRate: SAMPLE_RATE_IN });
    if (this._micCtx.state === 'suspended') await this._micCtx.resume();

    await this._micCtx.audioWorklet.addModule(captureWorklet);
    this.env.ensure();

    this._micSource = this._micCtx.createMediaStreamSource(this._micStream);

    // LUFS-based gain model: effective = auto + trim
    const micTarget = parseFloat(this.env.field('mxMicTarget')?.value) || -23;
    const micAutoGainDb = micTarget - (this.env.settings.micMeasured ?? -23);
    const micTrimDb = parseInt(this.env.field('mxMicTrim')?.value) || 0;
    const fileTrimDb = parseInt(this.env.field('mxFileTrim')?.value) || 0;
    const monPct = parseInt(this.env.field('mxMonitor')?.value) || 50;

    this._micGainNode = this._micCtx.createGain();
    this._micGainNode.gain.value = Math.pow(10, (micAutoGainDb + micTrimDb) / 20);

    this._fileGainNode = this._micCtx.createGain();
    this._fileGainNode.gain.value = Math.pow(10, fileTrimDb / 20);

    this._captureNode = new AudioWorkletNode(this._micCtx, 'capture-processor', {
      processorOptions: { chunkSize: SAMPLE_RATE_IN },
    });

    this._micAnalyserNode = this._micCtx.createAnalyser();
    this._micAnalyserNode.fftSize = 2048;

    this._mixAnalyserNode = this._micCtx.createAnalyser();
    this._mixAnalyserNode.fftSize = 2048;

    this._fileAnalyserNode = this._micCtx.createAnalyser();
    this._fileAnalyserNode.fftSize = 2048;

    this._monitorGainNode = this._micCtx.createGain();
    this._monitorGainNode.gain.value = monPct / 100;

    this._micChunkCount = 0;
    this.env.log(
      `Mic ready: AudioWorklet @${SAMPLE_RATE_IN}Hz, micAutoGain=${micAutoGainDb.toFixed(1)}dB, micTrim=${micTrimDb}dB, fileTrim=${fileTrimDb}dB, monitor=${monPct}%`,
    );
  }

  _connectMic() {
    // mic → micGain → captureNode
    this._micSource.connect(this._micGainNode);
    this._micGainNode.connect(this._captureNode);
    this._micGainNode.connect(this._micAnalyserNode);

    // mixed mode: connect padded video audio source [silence × padBefore] + video + [silence × padAfter]
    if (this._audioMode === 'mixed') {
      const silence = () => new Float32Array(SAMPLE_RATE_IN);
      const paddedChunks = [
        ...Array.from({ length: this._padBefore }, silence),
        ...this._videoAudioChunks,
        ...Array.from({ length: this._padAfter }, silence),
      ];
      const totalSamples = paddedChunks.reduce((a, c) => a + c.length, 0);
      const audioBuf = this._micCtx.createBuffer(1, totalSamples, SAMPLE_RATE_IN);
      const ch = audioBuf.getChannelData(0);
      let pos = 0;
      for (const chunk of paddedChunks) {
        ch.set(chunk, pos);
        pos += chunk.length;
      }
      this._fileSrcNode = this._micCtx.createBufferSource();
      this._fileSrcNode.buffer = audioBuf;

      this._fileSrcNode.connect(this._fileGainNode);
      this._fileGainNode.connect(this._captureNode);
      this._fileGainNode.connect(this._fileAnalyserNode);
      // Speaker output uses HTML video element (original quality); Web Audio monitor disconnected
      this._fileSrcNode.start();
      this._fileSrcNode.onended = () => this.env.log('Video audio in graph completed');
    }

    this._captureNode.connect(this._mixAnalyserNode);

    // Real-time audio output for recording (CaptureProcessor pass-through, zero delay)
    this._recDest = this._micCtx.createMediaStreamDestination();
    this._captureNode.connect(this._recDest);

    this._captureNode.port.onmessage = (e) => {
      if (e.data.type === 'chunk') {
        this._handleMicChunk(e.data.audio);
      }
    };
    this._captureNode.port.postMessage({ command: 'start' });
    this._graphConnected = true;
    this.env.log(`Mic connected — AudioWorklet graph (${this._audioMode})`);
  }

  _disconnectMic() {
    this._graphConnected = false;
    if (this._captureNode) {
      this._captureNode.port.postMessage({ command: 'stop' });
      try {
        this._captureNode.disconnect();
      } catch (_) {}
    }
    if (this._fileSrcNode) {
      try {
        this._fileSrcNode.stop();
      } catch (_) {}
      try {
        this._fileSrcNode.disconnect();
      } catch (_) {}
      this._fileSrcNode = null;
    }
    if (this._fileGainNode)
      try {
        this._fileGainNode.disconnect();
      } catch (_) {}
    if (this._micGainNode)
      try {
        this._micGainNode.disconnect();
      } catch (_) {}
    if (this._micSource)
      try {
        this._micSource.disconnect();
      } catch (_) {}
    if (this._monitorGainNode)
      try {
        this._monitorGainNode.disconnect();
      } catch (_) {}
    if (this._micAnalyserNode)
      try {
        this._micAnalyserNode.disconnect();
      } catch (_) {}
    if (this._mixAnalyserNode)
      try {
        this._mixAnalyserNode.disconnect();
      } catch (_) {}
  }

  _startMainMicPhase() {
    // Delay video playback by padBefore — graph plays silence during leading padding
    if (this._padBefore > 0) {
      this._padBeforeTimer = setTimeout(() => {
        if (this.running) this._videoEl.play();
      }, this._padBefore * CHUNK_MS);
    } else {
      this._videoEl.play();
    }
    this._connectMic();
  }

  _handleMicChunk(mixedAudio) {
    // During padding: file audio = zero → mix = mic only; frame = first/last
    // During main:    file audio = real → mix = mic + file; frame from video
    if (this._chunkIdx >= this._grandTotal) {
      this._disconnectMic();
      this._videoEl.pause();
      this.env.log(`Mic phase done: ${this._micChunkCount} chunks via AudioWorklet`);
      this.running = false;
      if (this.onEnd) this.onEnd();
      return;
    }
    const frame = this._paddedFrames ? this._paddedFrames[this._chunkIdx] : null;
    this._micChunkCount++;
    this._chunkIdx++;
    if (this.onChunk) this.onChunk({ audio: mixedAudio, frameBase64: frame });
  }

  // ==================== Recording audio accessors ====================

  /** Real-time audio output stream for recording (mic/mixed modes, zero delay) */
  getAudioOutputStream() {
    return this._recDest?.stream || null;
  }

  /** Concatenated padded audio buffer for recording (video-only mode) */
  getFullAudioBuffer() {
    if (!this._allAudio || this._allAudio.length === 0) return null;
    const total = this._allAudio.reduce((a, c) => a + c.length, 0);
    if (total === 0) return null;
    const buf = new Float32Array(total);
    let pos = 0;
    for (const chunk of this._allAudio) {
      buf.set(chunk, pos);
      pos += chunk.length;
    }
    return buf;
  }

  // ==================== Audio graph accessors ====================

  get mixerNodes() {
    return {
      micGain: this._micGainNode,
      fileGain: this._fileGainNode,
      monitorGain: this._monitorGainNode,
      monitorEl: this._videoEl,
      micAnalyser: this._micAnalyserNode,
      fileAnalyser: this._fileAnalyserNode,
      mixAnalyser: this._mixAnalyserNode,
      captureNode: this._captureNode,
      connected: this._graphConnected,
    };
  }

  // ==================== Audio processing helpers ====================

  async _decodeVideoAudio(duration) {
    let decodeContext;
    try {
      const arrayBuffer = await this._file.arrayBuffer();
      const audioCtx = (decodeContext = new AudioContext());
      const audioBuf = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
      const targetSamples = Math.ceil(duration * SAMPLE_RATE_IN);
      const offCtx = new OfflineAudioContext(1, targetSamples, SAMPLE_RATE_IN);
      const src = offCtx.createBufferSource();
      src.buffer = audioBuf;
      src.connect(offCtx.destination);
      src.start();
      const resampled = await offCtx.startRendering();
      const pcm = resampled.getChannelData(0);
      await audioCtx.close();

      // LUFS normalization
      const fileTargetEl = this.env.field('mxFileTarget');
      const targetLUFS = fileTargetEl ? parseFloat(fileTargetEl.value) || -33 : -33;
      const srcLUFS = measureLUFS(pcm, SAMPLE_RATE_IN);
      this.measuredLUFS = srcLUFS;

      let videoNormGain;
      if (this._audioMode === 'mixed' && isFinite(srcLUFS)) {
        const autoGainDb = targetLUFS - srcLUFS;
        videoNormGain = Math.pow(10, autoGainDb / 20);
        this.env.log(
          `Video audio: ${srcLUFS.toFixed(1)} LUFS → target ${targetLUFS} LUFS (auto ${autoGainDb.toFixed(1)} dB)`,
        );
      } else {
        const foTarget = -28;
        const autoDb = isFinite(srcLUFS) ? foTarget - srcLUFS : 0;
        videoNormGain = Math.pow(10, autoDb / 20);
        this.env.log(
          `Video audio: ${isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—'} LUFS → ${foTarget} LUFS (gain ${autoDb.toFixed(1)} dB)`,
        );
      }

      this._videoAudioChunks = [];
      for (let i = 0; i < pcm.length; i += SAMPLE_RATE_IN) {
        const chunk = pcm.slice(i, Math.min(i + SAMPLE_RATE_IN, pcm.length));
        for (let j = 0; j < chunk.length; j++) chunk[j] *= videoNormGain;
        this._videoAudioChunks.push(chunk);
      }

      // Update Mixer display
      const measEl = this.env.field('mxFileMeasured');
      if (measEl) measEl.textContent = isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—';
      const autoEl = this.env.field('mxFileAuto');
      if (autoEl && this._audioMode === 'mixed' && isFinite(srcLUFS)) {
        autoEl.textContent = (targetLUFS - srcLUFS).toFixed(1);
      }
    } catch (err) {
      this.env.log(`No audio track in video, using silence: ${err.message}`);
      this._videoAudioChunks = [];
      for (let i = 0; i < this._mainChunks; i++) {
        this._videoAudioChunks.push(new Float32Array(SAMPLE_RATE_IN));
      }
    } finally {
      if (decodeContext?.state !== 'closed') await decodeContext?.close().catch(() => {});
    }
  }

  async _getVideoDuration() {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      video.preload = 'metadata';
      const url = URL.createObjectURL(this._file);
      video.src = url;
      video.onloadedmetadata = () => {
        const d = video.duration;
        URL.revokeObjectURL(url);
        resolve(d);
      };
      video.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Failed to load video metadata'));
      };
    });
  }

  async _extractFrames(count, offset) {
    return new Promise((resolve) => {
      const video = document.createElement('video');
      video.muted = true;
      video.preload = 'auto';
      const url = URL.createObjectURL(this._file);
      video.src = url;
      const frames = [];
      let idx = 0;
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      video.onloadedmetadata = () => {
        const cw = video.videoWidth;
        const ch = video.videoHeight;
        canvas.width = cw;
        canvas.height = ch;
        const seekNext = () => {
          const seekTime = idx + offset;
          if (idx >= count || seekTime >= video.duration) {
            URL.revokeObjectURL(url);
            resolve(frames);
            return;
          }
          video.currentTime = seekTime;
        };
        video.onseeked = () => {
          ctx.drawImage(video, 0, 0, cw, ch);
          frames.push(canvas.toDataURL('image/jpeg', 0.7).split(',')[1]);
          idx++;
          seekNext();
        };
        seekNext();
      };
      video.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(frames);
      };
    });
  }

  pause() {
    if (!this.running || this.paused) return;
    this.paused = true;
    // Video-only: stop the timer
    if (this._timer) {
      clearTimeout(this._timer);
      this._timer = null;
    }
    // Cancel pending padBefore timer (delayed video play)
    if (this._padBeforeTimer) {
      clearTimeout(this._padBeforeTimer);
      this._padBeforeTimer = null;
    }
    // Mic/Mixed: suspend AudioContext (freezes entire graph — mic, file buffer, capture)
    if (this._micCtx && this._micCtx.state === 'running') {
      this._micCtx.suspend();
    }
    // Pause speaker output
    if (!this._videoEl.paused) this._videoEl.pause();
    this.env.log('File provider paused');
  }

  resume() {
    if (!this.running || !this.paused) return;
    this.paused = false;
    if (this._audioMode === 'video') {
      // Video-only: restart the timer and video element
      if (this._chunkIdx >= this._mainStart && this._chunkIdx < this._grandTotal) {
        this._videoEl.play();
      }
      this._feedNext();
    } else {
      // Mic/Mixed: resume AudioContext (unfreezes entire graph)
      if (this._micCtx && this._micCtx.state === 'suspended') {
        this._micCtx.resume();
      }
      // Resume or schedule video playback
      if (this._chunkIdx < this._mainStart) {
        // Still in leading padding — schedule delayed play for remaining pad
        const remainingPad = this._mainStart - this._chunkIdx;
        this._padBeforeTimer = setTimeout(() => {
          if (this.running && !this.paused) this._videoEl.play();
        }, remainingPad * CHUNK_MS);
      } else if (this._chunkIdx < this._mainEnd) {
        // In main phase — resume playback immediately
        this._videoEl.play();
      }
    }
    this.env.log('File provider resumed');
  }

  stop() {
    this.running = false;
    this.paused = false;
    if (this._timer) {
      clearTimeout(this._timer);
      this._timer = null;
    }
    if (this._padBeforeTimer) {
      clearTimeout(this._padBeforeTimer);
      this._padBeforeTimer = null;
    }
    this._disconnectMic();
    if (this._micCtx) {
      this._micCtx.close().catch(() => {});
      this._micCtx = null;
    }
    if (this._micStream) {
      this._micStream.getTracks().forEach((t) => t.stop());
      this._micStream = null;
    }
    this._captureNode = null;
    this._micGainNode = null;
    this._fileGainNode = null;
    this._monitorGainNode = null;
    this._micAnalyserNode = null;
    this._mixAnalyserNode = null;
    this._fileSrcNode = null;
    this._micSource = null;
    this._graphConnected = false;
    this._paddedFrames = null;
    this._videoEl.autoplay = true; // restore for LiveMediaProvider camera preview
    this._videoEl.muted = false;
    this._videoEl.volume = 1.0;
    this._videoEl.pause();
    this._videoEl.src = '';
    this._videoEl.style.display = 'none';
    if (this._objectUrl) {
      URL.revokeObjectURL(this._objectUrl);
      this._objectUrl = null;
    }
  }
}
export { FileMediaProvider };
