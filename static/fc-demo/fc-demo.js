const $ = (id) => document.getElementById(id);

const presets = {
  binary: {
    systemPrompt: '',
    tools: [
      {
        type: 'function',
        function: {
          name: 'convert_decimal_to_binary',
          description: 'Convert a decimal number to its binary representation.',
          parameters: {
            type: 'object',
            properties: {
              decimal_number: { type: 'number', description: 'The decimal number to be converted.' },
            },
            required: ['decimal_number'],
          },
        },
      },
    ],
  },
  garden: {
    systemPrompt: '',
    tools: [
      {
        type: 'function',
        function: {
          name: 'fetch_container',
          description: 'Retrieve a container to hold the specified amount of liquid.',
          parameters: { type: 'object', properties: { volume: { type: 'number' } }, required: ['volume'] },
        },
      },
      {
        type: 'function',
        function: {
          name: 'fill_container_with_liquid',
          description: 'Fill the specified container with a given volume of liquid.',
          parameters: {
            type: 'object',
            properties: { container_id: { type: 'string' }, volume: { type: 'number' } },
            required: ['container_id', 'volume'],
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'water_plant',
          description: 'Water a plant with the liquid from the specified container.',
          parameters: {
            type: 'object',
            properties: { plant_id: { type: 'string' }, container_id: { type: 'string' } },
            required: ['plant_id', 'container_id'],
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'measure_soil_moisture',
          description: 'Measure the soil moisture level of a specified plant.',
          parameters: { type: 'object', properties: { plant_id: { type: 'string' } }, required: ['plant_id'] },
        },
      },
    ],
  },
};

let ws = null;
let stream = null;
let audioCtx = null;
let sourceNode = null;
let processorNode = null;
let captureBuffer = [];
let chunkCount = 0;
let toolCount = 0;
let audioPlayCtx = null;
let playAt = 0;
let queuedInit = false;
let inputMode = 'mic';
let inputStarted = false;
let caseSilenceActive = false;

function setState(text) { $('connState').textContent = text; }
function logEvent(text, cls = '') { append($('eventLog'), text, cls); }
function logTool(text, cls = '') { append($('toolLog'), text, cls); }
function appendNonSpoken(text) {
  if (!text) return;
  $('nonSpoken').textContent += text;
  $('nonSpoken').scrollTop = $('nonSpoken').scrollHeight;
}
function append(parent, text, cls = '') {
  const div = document.createElement('div');
  div.className = `entry ${cls}`.trim();
  div.textContent = text;
  parent.appendChild(div);
  parent.scrollTop = parent.scrollHeight;
}

function wsUrl() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}/v1/realtime?mode=audio`;
}

function selectedPreset() {
  return presets[$('toolPreset').value] || presets.binary;
}

function initPayload() {
  return {
    mode: 'full_duplex',
    fc_duplex: true,
    system_prompt: selectedPreset().systemPrompt,
    tools: selectedPreset().tools,
    generate_audio: $('generateAudio').checked,
    config: {
      sample_rate: 16000,
      unit_sec: 1.0,
      decode_mode: 'greedy',
      non_spoken_budget_per_unit: Number($('budget').value || 30),
      max_spoken_tokens: Number($('spokenTokens').value || 24),
      non_spoken_scheduling: 'quality',
    },
  };
}

async function start(mode = 'mic') {
  inputMode = mode;
  if (inputMode === 'case') $('toolPreset').value = 'binary';
  $('reply').textContent = '';
  $('nonSpoken').textContent = '';
  $('toolLog').textContent = '';
  $('eventLog').textContent = '';
  chunkCount = 0;
  toolCount = 0;
  queuedInit = false;
  inputStarted = false;
  caseSilenceActive = false;
  $('chunkCount').textContent = '0';
  $('toolCount').textContent = '0';
  setState('connecting');

  ws = new WebSocket(wsUrl());
  ws.onmessage = (message) => handleEvent(JSON.parse(message.data)).catch((err) => logEvent(String(err?.message || err), 'error'));
  ws.onclose = () => { setState('closed'); stopCaptureOnly(); };
  ws.onerror = () => logEvent('WebSocket error', 'error');
  await new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = reject;
  });

  $('startBtn').disabled = true;
  $('caseBtn').disabled = true;
  $('stopBtn').disabled = false;
  setState('waiting worker');
}

function stop() {
  caseSilenceActive = false;
  stopCaptureOnly();
  if (ws && ws.readyState === WebSocket.OPEN) send({ type: 'session.close', reason: 'user_stop' });
  ws = null;
  $('startBtn').disabled = false;
  $('caseBtn').disabled = false;
  $('stopBtn').disabled = true;
  setState('stopped');
}

function stopCaptureOnly() {
  processorNode?.disconnect();
  sourceNode?.disconnect();
  stream?.getTracks().forEach((track) => track.stop());
  processorNode = null;
  sourceNode = null;
  stream = null;
  captureBuffer = [];
}

async function startMic() {
  stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  audioCtx = new AudioContext();
  sourceNode = audioCtx.createMediaStreamSource(stream);
  processorNode = audioCtx.createScriptProcessor(4096, 1, 1);
  processorNode.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const down = resample(input, audioCtx.sampleRate, 16000);
    captureBuffer.push(...down);
    while (captureBuffer.length >= 16000) {
      const chunk = captureBuffer.splice(0, 16000);
      sendAudio(new Float32Array(chunk));
    }
  };
  sourceNode.connect(processorNode);
  processorNode.connect(audioCtx.destination);
}

function sendAudio(samples) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const inputId = `unit_${String(chunkCount).padStart(3, '0')}`;
  send({
    type: 'input.append',
    input: { type: 'audio', input_id: inputId, audio: float32ToBase64(samples), sample_rate: 16000 },
  });
  chunkCount += 1;
  $('chunkCount').textContent = String(chunkCount);
  $('lastUnit').textContent = inputId;
}

async function handleEvent(event) {
  if (event.type === 'session.queue_done' && !queuedInit) {
    queuedInit = true;
    send({ type: 'session.init', payload: initPayload() });
  }
  if (event.type === 'session.created') {
    setState('ready');
    await startInputOnce();
  }
  if (event.type === 'session.closed') setState(`closed: ${event.reason || ''}`);
  if (event.type === 'response.output.delta') handleOutput(event);
  if (event.type === 'debug.fc_non_spoken.delta') handleNonSpoken(event);
  if (event.type === 'response.think.delta') appendNonSpoken(event.delta || '');
  if (event.type === 'response.tool_call.args.delta') appendNonSpoken(event.delta || '');
  if (event.type === 'response.tool_call.args.raw') handleToolCall(event);
  if (event.type === 'response.debug' && event.debug?.unit_timing_ms) {
    const t = event.debug.unit_timing_ms;
    logEvent(`${event.input_id} total=${Math.round(t.total)}ms prefill=${Math.round(t.prefill)} spoken=${Math.round(t.spoken)} non=${Math.round(t.non_spoken)}`);
    return;
  }
  if (event.type && !event.type.startsWith('response.output.delta')) logEvent(event.type);
}

async function startInputOnce() {
  if (inputStarted) return;
  inputStarted = true;
  if (inputMode === 'case') {
    setState('running case');
    sendCaseAudio().catch((err) => logEvent(String(err?.message || err), 'error'));
  } else {
    await startMic();
    setState('running');
  }
}

async function sendCaseAudio() {
  logEvent('sending tauvoice_01029_user.wav');
  const response = await fetch('/static/fc-demo/cases/tauvoice_01029_user.wav');
  if (!response.ok) throw new Error(`case audio fetch failed: ${response.status}`);
  const arrayBuffer = await response.arrayBuffer();
  const ctx = new AudioContext();
  const playbackBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
  const src = ctx.createBufferSource();
  src.buffer = playbackBuffer;
  src.connect(ctx.destination);
  src.start();
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
  const channel = audioBuffer.getChannelData(0);
  const samples = new Float32Array(resample(channel, audioBuffer.sampleRate, 16000));
  const chunkSize = 16000;
  for (let offset = 0; offset < samples.length; offset += chunkSize) {
    const chunk = samples.slice(offset, Math.min(samples.length, offset + chunkSize));
    sendAudio(padChunk(chunk, chunkSize));
    await delay(1000);
  }
  logEvent('case audio done; streaming silence until stop');
  caseSilenceActive = true;
  while (caseSilenceActive && ws && ws.readyState === WebSocket.OPEN) {
    sendAudio(new Float32Array(chunkSize));
    await delay(1000);
  }
  logEvent('case silence stopped');
  src.onended = () => ctx.close().catch(() => {});
}

function padChunk(samples, size) {
  if (samples.length === size) return samples;
  const padded = new Float32Array(size);
  padded.set(samples);
  return padded;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function handleNonSpoken(event) {
  const text = event.text || '';
  if (text) appendNonSpoken(text);
  if (!text && Array.isArray(event.token_strs) && event.token_strs.length) {
    appendNonSpoken(`[${event.token_strs.join(' ')}]`);
  }
}

function handleOutput(event) {
  if (event.kind === 'text' && event.text) {
    $('reply').textContent += event.text;
    $('reply').scrollTop = $('reply').scrollHeight;
  } else if (event.kind === 'audio' && event.audio) {
    playAudio(event.audio, event.sample_rate || 24000);
  } else if (event.kind === 'listen') {
    $('lastUnit').textContent = event.input_id || $('lastUnit').textContent;
  }
}

function handleToolCall(event) {
  const raw = event.raw || {};
  const args = parseArgs(raw.arguments);
  const result = executeTool(raw.name, args);
  toolCount += 1;
  $('toolCount').textContent = String(toolCount);
  logTool(`${raw.name}(${JSON.stringify(args)})\n→ ${result}`);
  send({ type: 'input.append', input: { type: 'tool_result', tool_call_id: event.tool_call_id, contents: result } });
}

function executeTool(name, args) {
  if (name === 'convert_decimal_to_binary') {
    const number = Number(args.decimal_number || 0);
    return JSON.stringify({ result: [{ name, arguments: { decimal_number: number }, results: { binary_representation: Math.trunc(number).toString(2) } }] });
  }
  if (name === 'fetch_container') {
    return JSON.stringify({ result: [{ name, arguments: args, results: { container_id: 'container_12345' } }] });
  }
  if (name === 'fill_container_with_liquid') {
    return JSON.stringify({ result: [{ name, arguments: args, results: { filled_container_id: args.container_id || 'container_12345' } }] });
  }
  if (name === 'water_plant') {
    return JSON.stringify({ result: [{ name, arguments: args, results: { watering_status: `Plant ${args.plant_id || 'herb_garden_001'} has been successfully watered with liquid from container ${args.container_id || 'container_12345'}.` } }] });
  }
  if (name === 'measure_soil_moisture') {
    return JSON.stringify({ result: [{ name, arguments: args, results: { soil_moisture_level: 45.3 } }] });
  }
  return JSON.stringify({ result: [{ name, arguments: args, results: { ok: true } }] });
}

function send(message) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify(message));
}

function parseArgs(value) {
  if (!value) return {};
  if (typeof value === 'object') return value;
  try { return JSON.parse(value); } catch { return {}; }
}

function resample(input, fromRate, toRate) {
  if (fromRate === toRate) return Array.from(input);
  const ratio = fromRate / toRate;
  const length = Math.floor(input.length / ratio);
  const output = new Array(length);
  for (let i = 0; i < length; i++) {
    const idx = i * ratio;
    const lo = Math.floor(idx);
    const hi = Math.min(input.length - 1, lo + 1);
    const frac = idx - lo;
    output[i] = input[lo] * (1 - frac) + input[hi] * frac;
  }
  return output;
}

function float32ToBase64(samples) {
  return btoa(String.fromCharCode(...new Uint8Array(samples.buffer)));
}

function base64ToFloat32(base64) {
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

async function playAudio(base64, sampleRate) {
  audioPlayCtx ||= new AudioContext({ sampleRate });
  const data = base64ToFloat32(base64);
  const buffer = audioPlayCtx.createBuffer(1, data.length, sampleRate);
  buffer.copyToChannel(data, 0);
  const src = audioPlayCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(audioPlayCtx.destination);
  const now = audioPlayCtx.currentTime;
  playAt = Math.max(playAt, now + 0.05);
  src.start(playAt);
  playAt += buffer.duration;
}

$('startBtn').addEventListener('click', () => start('mic').catch((err) => { logEvent(String(err?.message || err), 'error'); stop(); }));
$('caseBtn').addEventListener('click', () => start('case').catch((err) => { logEvent(String(err?.message || err), 'error'); stop(); }));
$('stopBtn').addEventListener('click', stop);
