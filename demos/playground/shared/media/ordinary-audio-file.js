// @ts-nocheck
// Ported from static/audio-duplex/audio-duplex-app.js @ a3e7c70.
// Preserve frame timing, padding, LUFS normalization, and AudioWorklet mixing.
// DOM/service dependencies are injected; the original page is untouched.
import { measureLUFS } from '../../../../static/duplex/lib/lufs.js';
import captureWorklet from '../../../../static/duplex/lib/capture-processor.js?url';
const SAMPLE_RATE_IN = 16000;
const CHUNK_MS = 1000;
class FileAudioProvider {
  /**
   * @param {File} file
   * @param {{audioMode: 'file'|'mixed', padBeforeSec: number, padAfterSec: number}} opts
   */
  constructor(file, opts = {}, env) {
    this.env = env;
    this._file = file;
    this._audioMode = opts.audioMode || 'file';
    this._padBefore = Math.max(0, Math.floor(opts.padBeforeSec ?? 0));
    this._padAfter = Math.max(0, Math.floor(opts.padAfterSec ?? 2));

    this._fileAudioChunks = []; // decoded + normalized file audio
    this._mainChunks = 0;

    // Padded array (file-only mode)
    this._allAudio = [];

    // Phase tracking
    this._chunkIdx = 0;
    this._mainStart = 0;
    this._mainEnd = 0;
    this._grandTotal = 0;
    this._timer = null;

    // Mic pipeline (mixed mode — AudioWorklet graph)
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

    // Playback element
    this._audioEl = this.env.field('fileAudioEl');
    this._objectUrl = null;

    this.onChunk = null;
    this.onEnd = null;
    this.running = false;
    this.paused = false;
    this._padBeforeTimer = null;
  }

  async start() {
    this.env.log('Processing audio file...');

    // 1. Get duration
    const duration = await this._getAudioDuration();
    this.env.ensure();
    if (!Number.isFinite(duration)) throw new Error('Invalid audio duration');
    this._mainChunks = Math.floor(duration);
    if (this._mainChunks === 0) throw new Error('Audio too short');

    // 2. Decode file audio and normalize
    await this._decodeFileAudio(duration);
    this.env.ensure();

    // 3. Setup mic (mixed mode)
    if (this._audioMode === 'mixed') {
      await this._setupMic();
      this.env.ensure();
    }

    // 4. Phase boundaries
    this._mainStart = this._padBefore;
    this._mainEnd = this._padBefore + this._mainChunks;
    this._grandTotal = this._padBefore + this._mainChunks + this._padAfter;

    const parts = [];
    if (this._padBefore > 0) parts.push(`${this._padBefore}s pad`);
    parts.push(`${this._mainChunks}s audio`);
    if (this._padAfter > 0) parts.push(`${this._padAfter}s pad`);
    this.env.log(
      `Ready: [${parts.join(' + ')}] = ${this._grandTotal} chunks, mode=${this._audioMode}`,
    );

    // 5. Prepare playback element (user hears the original file audio)
    this._objectUrl = URL.createObjectURL(this._file);
    this._audioEl.src = this._objectUrl;

    // 6. Start feeding
    this._chunkIdx = 0;
    this.running = true;

    if (this._audioMode === 'file') {
      this._buildPaddedArrays();
      this._feedNext();
    } else {
      // mixed: mic runs for entire padded duration; file buffer includes silence padding
      // During padding, file audio = zero → mix = mic only; during main, mix = mic + file
      this._startMainMicPhase();
    }
  }

  // ==================== File-only mode: unified timer ====================

  _buildPaddedArrays() {
    const silence = () => new Float32Array(SAMPLE_RATE_IN);
    this._allAudio = [
      ...Array.from({ length: this._padBefore }, silence),
      ...this._fileAudioChunks,
      ...Array.from({ length: this._padAfter }, silence),
    ];
  }

  _feedNext() {
    if (!this.running || this.paused) return;
    if (this._chunkIdx >= this._grandTotal) {
      this.running = false;
      this._audioEl.pause();
      if (this.onEnd) this.onEnd();
      return;
    }
    // Start playback when entering main phase
    if (this._chunkIdx === this._mainStart) {
      this._audioEl.play();
    }
    const t0 = performance.now();
    const audio = this._allAudio[this._chunkIdx];
    this._chunkIdx++;
    if (this.onChunk) this.onChunk({ audio });
    const elapsed = performance.now() - t0;
    this._timer = setTimeout(() => this._feedNext(), Math.max(0, CHUNK_MS - elapsed));
  }

  // ==================== Mixed mode: phased feeding ====================

  async _setupMic() {
    const _micId = this.env.settings.micDevice;
    this._micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, ...(_micId ? { deviceId: { exact: _micId } } : {}) },
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
    this._fileGainNode.gain.value = Math.pow(10, fileTrimDb / 20); // LUFS norm already applied to PCM

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
    // Build padded file AudioBuffer: [silence × padBefore] + file + [silence × padAfter]
    // Padding is transparent to the graph — mic captures the entire duration
    const silence = () => new Float32Array(SAMPLE_RATE_IN);
    const paddedChunks = [
      ...Array.from({ length: this._padBefore }, silence),
      ...this._fileAudioChunks,
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

    // Connect graph:
    //   mic → micGain ──→ captureNode → mixAnalyser (waveform)
    //   file → fileGain ─┘
    //   fileGain → monitorGain → speaker (file only, no echo)
    //   micGain → micAnalyser (mic-only meter)
    this._micSource.connect(this._micGainNode);
    this._micGainNode.connect(this._captureNode);
    this._micGainNode.connect(this._micAnalyserNode);

    this._fileSrcNode.connect(this._fileGainNode);
    this._fileGainNode.connect(this._captureNode);
    this._fileGainNode.connect(this._fileAnalyserNode);
    // Speaker output uses HTML audio element (original quality); Web Audio monitor disconnected

    this._captureNode.connect(this._mixAnalyserNode);

    // Wire chunk handler
    this._captureNode.port.onmessage = (e) => {
      if (e.data.type === 'chunk') {
        this._handleMicChunk(e.data.audio);
      }
    };

    this._captureNode.port.postMessage({ command: 'start' });
    this._fileSrcNode.start();
    this._fileSrcNode.onended = () => this.env.log('File audio in graph completed');

    this._graphConnected = true;

    // Waveform visualization uses the mix analyser
    // The React mixer reads meterNodes; no legacy global waveform canvas.
    this.env.log('Mic connected — AudioWorklet graph mixing');
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
    // Use native HTML element for speaker output (original quality, not LUFS-normalized 16kHz)
    const monPct = parseInt(this.env.field('mxMonitor')?.value) || 50;
    this._audioEl.muted = false;
    this._audioEl.volume = monPct / 100;
    // Delay file playback by padBefore — graph plays silence during leading padding
    if (this._padBefore > 0) {
      this._padBeforeTimer = setTimeout(() => {
        if (this.running) this._audioEl.play();
      }, this._padBefore * CHUNK_MS);
    } else {
      this._audioEl.play();
    }
    this._connectMic();
  }

  _handleMicChunk(mixedAudio) {
    // During padding: file audio = zero → mix = mic only
    // During main:    file audio = real → mix = mic + file
    if (this._chunkIdx >= this._grandTotal) {
      this._disconnectMic();
      this._audioEl.pause();
      this.env.log(`Mixed mode done: ${this._micChunkCount} chunks via AudioWorklet`);
      this.running = false;
      if (this.onEnd) this.onEnd();
      return;
    }
    this._micChunkCount++;
    this._chunkIdx++;
    if (this.onChunk) this.onChunk({ audio: mixedAudio });
  }

  // ==================== Audio graph accessors ====================

  /** Expose AudioWorklet graph nodes for mixer panel and recording */
  get mixerNodes() {
    return {
      micGain: this._micGainNode,
      fileGain: this._fileGainNode,
      monitorGain: this._monitorGainNode,
      monitorEl: this._audioEl,
      micAnalyser: this._micAnalyserNode,
      fileAnalyser: this._fileAnalyserNode,
      mixAnalyser: this._mixAnalyserNode,
      captureNode: this._captureNode,
      connected: this._graphConnected,
    };
  }

  // ==================== Audio processing helpers ====================

  async _decodeFileAudio(duration) {
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

      // LUFS normalization: measure → compute gain → apply
      const fileTargetEl = this.env.field('mxFileTarget');
      const targetLUFS = fileTargetEl ? parseFloat(fileTargetEl.value) || -33 : -33;
      const srcLUFS = measureLUFS(pcm, SAMPLE_RATE_IN);
      this.measuredLUFS = srcLUFS;

      let fileNormGain;
      if (this._audioMode === 'mixed' && isFinite(srcLUFS)) {
        const autoGainDb = targetLUFS - srcLUFS;
        fileNormGain = Math.pow(10, autoGainDb / 20);
        this.env.log(
          `File audio: ${srcLUFS.toFixed(1)} LUFS → target ${targetLUFS} LUFS (auto ${autoGainDb.toFixed(1)} dB)`,
        );
      } else {
        // file-only: normalize to -28 LUFS
        const foTarget = -28;
        const autoDb = isFinite(srcLUFS) ? foTarget - srcLUFS : 0;
        fileNormGain = Math.pow(10, autoDb / 20);
        this.env.log(
          `File audio: ${isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—'} LUFS → ${foTarget} LUFS (gain ${autoDb.toFixed(1)} dB)`,
        );
      }

      this._fileAudioChunks = [];
      for (let i = 0; i < pcm.length; i += SAMPLE_RATE_IN) {
        const chunk = pcm.slice(i, Math.min(i + SAMPLE_RATE_IN, pcm.length));
        for (let j = 0; j < chunk.length; j++) chunk[j] *= fileNormGain;
        this._fileAudioChunks.push(chunk);
      }

      // Update Mixer display
      const measEl = this.env.field('mxFileMeasured');
      if (measEl) measEl.textContent = isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—';
      const autoEl = this.env.field('mxFileAuto');
      if (autoEl && this._audioMode === 'mixed' && isFinite(srcLUFS)) {
        autoEl.textContent = (targetLUFS - srcLUFS).toFixed(1);
      }
    } catch (err) {
      this.env.log(`Failed to decode audio: ${err.message}`);
      throw err;
    } finally {
      if (decodeContext?.state !== 'closed') await decodeContext?.close().catch(() => {});
    }
  }

  async _getAudioDuration() {
    return new Promise((resolve, reject) => {
      const audio = document.createElement('audio');
      audio.preload = 'metadata';
      const url = URL.createObjectURL(this._file);
      audio.src = url;
      audio.onloadedmetadata = () => {
        const d = audio.duration;
        URL.revokeObjectURL(url);
        resolve(d);
      };
      audio.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Failed to load audio metadata'));
      };
    });
  }

  pause() {
    if (!this.running || this.paused) return;
    this.paused = true;
    // File-only: stop the timer
    if (this._timer) {
      clearTimeout(this._timer);
      this._timer = null;
    }
    // Cancel pending padBefore timer (delayed HTML play)
    if (this._padBeforeTimer) {
      clearTimeout(this._padBeforeTimer);
      this._padBeforeTimer = null;
    }
    // Mixed: suspend AudioContext (freezes entire graph — mic, file buffer, capture)
    if (this._micCtx && this._micCtx.state === 'running') {
      this._micCtx.suspend();
    }
    // Pause speaker output
    if (!this._audioEl.paused) this._audioEl.pause();
    this.env.log('File provider paused');
  }

  resume() {
    if (!this.running || !this.paused) return;
    this.paused = false;
    if (this._audioMode === 'file') {
      // File-only: restart the timer and HTML element
      if (this._chunkIdx >= this._mainStart && this._chunkIdx < this._grandTotal) {
        this._audioEl.play();
      }
      this._feedNext();
    } else {
      // Mixed: resume AudioContext (unfreezes entire graph)
      if (this._micCtx && this._micCtx.state === 'suspended') {
        this._micCtx.resume();
      }
      // Resume or schedule HTML audio playback
      if (this._chunkIdx < this._mainStart) {
        // Still in leading padding — schedule delayed play for remaining pad
        const remainingPad = this._mainStart - this._chunkIdx;
        this._padBeforeTimer = setTimeout(() => {
          if (this.running && !this.paused) this._audioEl.play();
        }, remainingPad * CHUNK_MS);
      } else if (this._chunkIdx < this._mainEnd) {
        // In main phase — resume playback immediately
        this._audioEl.play();
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
    this._audioEl.muted = false;
    this._audioEl.volume = 1.0;
    this._audioEl.pause();
    this._audioEl.src = '';
    if (this._objectUrl) {
      URL.revokeObjectURL(this._objectUrl);
      this._objectUrl = null;
    }
  }
}
export { FileAudioProvider };
