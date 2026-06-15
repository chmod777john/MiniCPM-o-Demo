const state = {
  sessionId: null,
  events: null,
  running: false,
  previewUrl: null,
  ready: false,
};

const els = {
  sessionId: document.getElementById("sessionId"),
  codexBin: document.getElementById("codexBin"),
  sandboxMode: document.getElementById("sandboxMode"),
  eventList: document.getElementById("eventList"),
  form: document.getElementById("messageForm"),
  input: document.getElementById("messageInput"),
  sendBtn: document.getElementById("sendBtn"),
  resetBtn: document.getElementById("resetBtn"),
  newSessionBtn: document.getElementById("newSessionBtn"),
  previewFrame: document.getElementById("previewFrame"),
  previewVersion: document.getElementById("previewVersion"),
  openPreview: document.getElementById("openPreview"),
};

function formatTime(ts) {
  const date = ts ? new Date(ts * 1000) : new Date();
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function isNearBottom(element) {
  return element.scrollHeight - element.clientHeight - element.scrollTop < 80;
}

function appendEvent(kind, body, options = {}) {
  const shouldFollow = isNearBottom(els.eventList);
  const item = document.createElement("article");
  item.className = `event ${options.tone || ""}`.trim();

  const row = document.createElement("div");
  row.className = "row";
  const kindEl = document.createElement("div");
  kindEl.className = "kind";
  kindEl.textContent = kind;
  const time = document.createElement("div");
  time.className = "time";
  time.textContent = formatTime(options.ts);
  row.append(kindEl, time);

  const bodyEl = document.createElement("pre");
  bodyEl.className = "body";
  bodyEl.textContent = body;

  item.append(row, bodyEl);
  els.eventList.append(item);
  if (shouldFollow || options.forceScroll) {
    els.eventList.scrollTop = els.eventList.scrollHeight;
  }
}

function setRunning(value) {
  state.running = value;
  els.sendBtn.disabled = value || !state.ready;
  els.resetBtn.disabled = value || !state.ready || !state.sessionId;
  els.input.disabled = value;
}

function setReady(value) {
  state.ready = value;
  els.newSessionBtn.disabled = !value;
  setRunning(state.running);
}

function updatePreview(url, version) {
  state.previewUrl = url;
  els.previewFrame.src = url;
  els.openPreview.href = url;
  if (version !== undefined && version !== null) {
    els.previewVersion.textContent = `v${version}`;
  }
}

function compactPayload(payload) {
  if (!payload) return "";
  const text = JSON.stringify(payload, null, 2);
  return text.length > 1400 ? `${text.slice(0, 1400)}\n...` : text;
}

function handleEvent(event) {
  switch (event.type) {
    case "hello":
      els.sessionId.textContent = event.session_id;
      els.codexBin.textContent = event.codex_bin || "-";
      els.sandboxMode.textContent = event.sandbox_mode || "-";
      if (event.preview_url) updatePreview(event.preview_url);
      appendEvent("connected", "事件流已连接。", { ts: event.ts });
      if (event.sandbox_warning) {
        appendEvent("sandbox", event.sandbox_warning, { tone: "error", ts: event.ts });
      }
      break;
    case "session.created":
      appendEvent("session", `创建 session: ${event.session_id}`, { ts: event.ts });
      break;
    case "session.reset":
      appendEvent("session", "已重置 index.html。", { ts: event.ts });
      break;
    case "chat.message":
      appendEvent("o45", event.text || "", { tone: "o45", ts: event.ts });
      break;
    case "tool.call":
      appendEvent("tool call", `${event.name}\n${compactPayload(event.args)}`, {
        tone: "tool",
        ts: event.ts,
      });
      if (event.sandbox_mode) {
        appendEvent("sandbox", event.sandbox_warning || `mode: ${event.sandbox_mode}`, {
          tone: event.sandbox_warning ? "error" : "tool",
          ts: event.ts,
        });
      }
      setRunning(true);
      break;
    case "task.started":
      appendEvent("task", `Codex task started: ${event.task_id}`, { tone: "tool", ts: event.ts });
      setRunning(true);
      break;
    case "codex.turn.started":
      appendEvent("codex", `turn_id: ${event.turn_id}`, { tone: "codex", ts: event.ts });
      break;
    case "codex.text.delta":
      appendEvent("codex text", event.text || "", { tone: "codex", ts: event.ts });
      break;
    case "codex.reasoning.delta":
      appendEvent("reasoning", event.text || "", { tone: "codex", ts: event.ts });
      break;
    case "codex.command.output":
      appendEvent("command output", event.text || "", { tone: "tool", ts: event.ts });
      break;
    case "codex.item.started":
      appendEvent("item started", compactPayload(event.payload), { tone: "tool", ts: event.ts });
      break;
    case "codex.item.completed":
      appendEvent("item completed", compactPayload(event.payload), { tone: "tool", ts: event.ts });
      break;
    case "codex.agent_message.completed":
      if (event.text) {
        appendEvent("codex final", event.text, { tone: "codex", ts: event.ts });
      }
      break;
    case "codex.diff.updated":
      appendEvent("diff", compactPayload(event.payload), { tone: "tool", ts: event.ts });
      break;
    case "codex.file.patch":
      appendEvent("file patch", compactPayload(event.payload), { tone: "tool", ts: event.ts });
      break;
    case "preview.updated":
      updatePreview(event.url, event.version);
      appendEvent("preview", `${event.path} updated -> v${event.version}`, {
        tone: "tool",
        ts: event.ts,
      });
      break;
    case "tool.result":
      appendEvent(`result ${event.status}`, event.summary || "", {
        tone: event.status === "failed" ? "error" : "tool",
        ts: event.ts,
      });
      setRunning(false);
      break;
    case "task.idle":
      setRunning(false);
      break;
    case "codex.error":
    case "preview.error":
      appendEvent("error", compactPayload(event.payload || event.message), {
        tone: "error",
        ts: event.ts,
      });
      break;
    default:
      appendEvent(event.type || "event", compactPayload(event.payload || event), { ts: event.ts });
      break;
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function connectEvents(sessionId) {
  if (state.events) {
    state.events.close();
  }
  const source = new EventSource(`/api/sessions/${sessionId}/events`);
  source.onmessage = (message) => {
    try {
      handleEvent(JSON.parse(message.data));
    } catch (error) {
      appendEvent("error", `Failed to parse event: ${error.message}`, { tone: "error" });
    }
  };
  source.onerror = () => {
    appendEvent("connection", "事件流断开，浏览器会自动重连。", { tone: "error" });
  };
  state.events = source;
}

async function createSession() {
  if (!state.ready) return;
  setRunning(false);
  els.eventList.innerHTML = "";
  const session = await api("/api/sessions", { method: "POST", body: "{}" });
  state.sessionId = session.session_id;
  els.sessionId.textContent = session.session_id;
  els.codexBin.textContent = session.codex_bin || "-";
  els.sandboxMode.textContent = session.sandbox_mode || "-";
  updatePreview(session.preview_url, 0);
  connectEvents(session.session_id);
}

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = els.input.value.trim();
  if (!message || !state.sessionId || state.running) return;
  try {
    setRunning(true);
    await api(`/api/sessions/${state.sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    els.input.value = "";
  } catch (error) {
    appendEvent("error", error.message, { tone: "error" });
    setRunning(false);
  }
});

els.resetBtn.addEventListener("click", async () => {
  if (!state.sessionId || state.running) return;
  try {
    const result = await api(`/api/sessions/${state.sessionId}/reset`, {
      method: "POST",
      body: "{}",
    });
    updatePreview(result.preview_url, 0);
  } catch (error) {
    appendEvent("error", error.message, { tone: "error" });
  }
});

els.newSessionBtn.addEventListener("click", () => {
  createSession().catch((error) => appendEvent("error", error.message, { tone: "error" }));
});

async function boot() {
  setReady(false);
  const status = await api("/api/status");
  els.codexBin.textContent = status.codex_bin || "-";
  els.sandboxMode.textContent = status.sandbox_mode || "-";
  if (!status.ready) {
    const message = [
      "本地 Codex binary 尚不可用；harness 不会使用 PATH 里的全局 codex。",
      "",
      status.error || "Unknown startup error",
      "",
      "构建命令：",
      "/user/weihongliang/MiniCPM-o-Demo-wt-agent-harness-2026-06-09/agent-harness/codex-harness-demo/scripts/build-local-codex.sh",
      "",
      "也可以显式指定：",
      "CODEX_BIN=/abs/path/to/codex python3 server.py",
    ].join("\n");
    appendEvent("startup", message, { tone: "error" });
    return;
  }
  setReady(true);
  appendEvent("startup", `使用本地 Codex: ${status.codex_bin}`, { tone: "tool" });
  appendEvent("sandbox", status.sandbox_warning || `mode: ${status.sandbox_mode}`, {
    tone: status.sandbox_warning ? "error" : "tool",
  });
  await createSession();
}

boot().catch((error) => {
  setReady(false);
  appendEvent("startup error", error.message, { tone: "error" });
});
