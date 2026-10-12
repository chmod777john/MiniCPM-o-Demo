export class FcRealtimeClient {
  constructor({ onEvent, onStatus, onSend } = {}) {
    this.onEvent = onEvent;
    this.onStatus = onStatus;
    this.onSend = onSend;
    this.ws = null;
    this.ready = false;
    this.queued = false;
    this._pendingSessionFrame = null;
    this.history = [];
    this.resumeIdentity = null;
    this._inputSeq = 0;
  }

  /**
   * @param {string} url  所选 API 端点的 realtime 地址（wss://host:port/e/<端点>/v1/realtime?mode=audio）。
   *   Demo 与 API 是两个服务，不再假设 API 与页面同源。
   */
  async connect(url) {
    if (!url) throw new Error('No API endpoint selected');
    this.ws = new WebSocket(url);
    this.ws.onmessage = (message) => this._handleMessage(message);
    this.ws.onclose = () => {
      this.ready = false;
      this.onStatus?.('WebSocket closed');
    };
    await new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = () => reject(new Error(`WebSocket connection failed: ${url}`));
    });
  }

  initSession(payload) {
    const frame = { type: 'session.init', payload };
    if (this.queued) {
      this._send(frame);
    } else {
      this._pendingSessionFrame = frame;
    }
  }

  resumeSession(payload) {
    const priorHistory = Array.isArray(payload?.history) ? payload.history : [];
    this.history = priorHistory.map((frame) => ({
      dir: (
        frame?.type === 'session.init'
        || frame?.type === 'input.append'
        || String(frame?.type || '').startsWith('input.')
      ) ? 'up' : 'down',
      frame: JSON.parse(JSON.stringify(frame)),
    }));
    this.resumeIdentity = {
      protocol_version: payload.protocol_version,
      model: payload.model,
      tokenizer_target: payload.tokenizer_target,
      tokenizer_fingerprint: payload.tokenizer_fingerprint,
      system_audio_sha256: payload.system_audio_sha256,
      tts_prompt_audio_sha256: payload.tts_prompt_audio_sha256,
    };
    this._inputSeq = priorHistory.filter(
      (frame) => frame?.type === 'input.append',
    ).length;
    const frame = { type: 'session.resume', payload };
    if (this.queued) {
      this._send(frame);
    } else {
      this._pendingSessionFrame = frame;
    }
  }

  buildResumePayload(throughUnitIndex) {
    const checkpointIndex = this.history.findIndex((item) => (
      item.dir === 'down'
      && item.frame.type === 'response.unit.committed'
      && item.frame.unit_index === throughUnitIndex
    ));
    if (checkpointIndex < 0) {
      throw new Error(`Missing Unit checkpoint: ${throughUnitIndex}`);
    }
    const checkpoint = this.history[checkpointIndex].frame;
    if (checkpoint.resume?.status !== 'available') {
      throw new Error(
        `Unit ${throughUnitIndex} is not resumable: ${checkpoint.resume?.reason || 'unknown'}`,
      );
    }
    if (!this.resumeIdentity) {
      throw new Error('Missing session resume identity');
    }
    return {
      ...this.resumeIdentity,
      through_unit_index: throughUnitIndex,
      history: this.history
        .slice(0, checkpointIndex + 1)
        .map((item) => item.frame),
    };
  }

  appendAudio({ audioBase64, sampleRate = 16000, inputId = null }) {
    if (!this.ready) {
      throw new Error('Session is not ready for audio input');
    }
    const resolvedInputId = inputId || `input_${String(this._inputSeq).padStart(6, '0')}`;
    this._inputSeq += 1;
    this._send({
      type: 'input.append',
      input: {
        input_id: resolvedInputId,
        audio_base64: audioBase64,
        sample_rate: sampleRate,
      },
    });
  }

  sendToolResult({ toolCallId, content }) {
    this._send({
      type: 'input.tool_result',
      tool_call_id: toolCallId,
      content,
    });
  }

  // 不关联 tool_call 的外部事件（如 subagent 的进度汇报），渲染成 <standalone_event>。
  // 与工具结果共用一个队列，进入下一个被处理的 Unit；eventId 可选，回执在
  // response.unit.started.standalone_events 里。content 必须是非空字符串。
  sendStandaloneEvent({ content, eventId = null }) {
    if (typeof content !== 'string' || !content) {
      throw new Error('standalone event content must be a non-empty string');
    }
    const frame = { type: 'input.standalone_event', content };
    if (eventId !== null && eventId !== undefined) frame.event_id = eventId;
    this._send(frame);
  }

  close(reason = 'client_closed') {
    if (!this.ws) return;
    if (this.ws.readyState === WebSocket.OPEN) {
      this._send({ type: 'session.close', reason });
    } else {
      this.ws.close();
    }
  }

  disconnect(reason = 'network_reconnect') {
    if (!this.ws) return;
    this.ws.close(1000, reason);
  }

  _send(message) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    this.history.push({
      dir: 'up',
      frame: JSON.parse(JSON.stringify(message)),
    });
    this.onSend?.(message);
    this.ws.send(JSON.stringify(message));
  }

  _handleMessage(message) {
    const event = JSON.parse(message.data);
    if (
      event.type === 'session.created'
      || event.type === 'response.unit.committed'
      || event.type === 'response.unit.started'
      || event.type.startsWith('response.think.')
      || event.type.startsWith('response.tool_call.')
      || event.type.startsWith('response.spoken.')
      || event.type === 'response.warning'
    ) {
      this.history.push({
        dir: 'down',
        frame: JSON.parse(JSON.stringify(event)),
      });
    }
    if (event.type === 'session.queue_done') {
      this.queued = true;
      if (this._pendingSessionFrame) {
        const frame = this._pendingSessionFrame;
        this._pendingSessionFrame = null;
        this._send(frame);
      }
    }
    if (event.type === 'session.created' || event.type === 'session.resumed') {
      this.ready = true;
    }
    if (event.type === 'session.created' && event.resume) {
      this.resumeIdentity = { ...event.resume };
    }
    this.onEvent?.(event);
  }
}
