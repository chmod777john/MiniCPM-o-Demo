#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
DEFAULT_CODEX_REPO = Path("/user/weihongliang/codex")
DEFAULT_CODEX_WORKTREE = Path("/user/weihongliang/codex-wt-harness-local-2026-06-10")
DEFAULT_CODEX_SDK_SRC = DEFAULT_CODEX_WORKTREE / "sdk/python/src"
DEFAULT_CODEX_BINS = (
    DEFAULT_CODEX_WORKTREE / "codex-rs/target/debug/codex",
    DEFAULT_CODEX_WORKTREE / "codex-rs/target/release/codex",
    DEFAULT_CODEX_REPO / "target/debug/codex",
    DEFAULT_CODEX_REPO / "target/release/codex",
    DEFAULT_CODEX_REPO / "codex-rs/target/debug/codex",
    DEFAULT_CODEX_REPO / "codex-rs/target/release/codex",
)
DATA_ROOT = Path(os.environ.get("HARNESS_DATA_ROOT", ROOT / ".data")).resolve()
PORT = int(os.environ.get("PORT", "8765"))
SANDBOX_ENV = os.environ.get("HARNESS_CODEX_SANDBOX", "auto").strip().lower()


def import_codex_sdk() -> None:
    sdk_src = Path(os.environ.get("CODEX_SDK_SRC", DEFAULT_CODEX_SDK_SRC)).resolve()
    if not sdk_src.exists():
        raise RuntimeError(
            f"Codex Python SDK source not found at {sdk_src}. "
            "Set CODEX_SDK_SRC to the local sdk/python/src directory."
        )
    sys.path.insert(0, str(sdk_src))


def resolve_codex_bin() -> Path:
    explicit = os.environ.get("CODEX_BIN")
    candidates = [Path(explicit).expanduser().resolve()] if explicit else list(DEFAULT_CODEX_BINS)
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise RuntimeError(
        "No local Codex binary found. This demo intentionally does not use a global PATH codex.\n"
        f"Searched:\n{searched}\n\n"
        "Build one from the local repo, for example:\n"
        "  /user/weihongliang/MiniCPM-o-Demo-wt-agent-harness-2026-06-09/agent-harness/codex-harness-demo/scripts/build-local-codex.sh\n"
        "Then start with:\n"
        "  CODEX_BIN=/user/weihongliang/codex-wt-harness-local-2026-06-10/codex-rs/target/debug/codex "
        "python3 /user/weihongliang/MiniCPM-o-Demo-wt-agent-harness-2026-06-09/agent-harness/codex-harness-demo/server.py"
    )


def probe_workspace_write_sandbox() -> tuple[bool, str | None]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        return False, "bubblewrap executable bwrap not found"
    command = [
        bwrap,
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--unshare-pid",
        "--die-with-parent",
        "true",
        "/bin/true",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return False, f"bubblewrap probe failed: {exc}"
    if result.returncode == 0:
        return True, None
    detail = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
    return False, f"bubblewrap probe failed: {detail}"


def write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    raw_len = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_len)
    except ValueError:
        length = 0
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def safe_rel_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("path must be relative and stay inside the session workspace")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def initial_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Codex Harness Preview</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f7fb;
      color: #1f2937;
    }
    body {
      min-height: 100vh;
      margin: 0;
      display: grid;
      place-items: center;
    }
    main {
      width: min(760px, calc(100vw - 40px));
      padding: 42px;
      background: white;
      border: 1px solid #d9e2ef;
      border-radius: 8px;
      box-shadow: 0 20px 50px rgba(31, 41, 55, 0.08);
    }
    h1 {
      margin: 0 0 14px;
      font-size: clamp(32px, 6vw, 58px);
      line-height: 1;
    }
    p {
      margin: 0;
      color: #4b5563;
      font-size: 18px;
      line-height: 1.7;
    }
    .actions {
      display: flex;
      gap: 12px;
      margin-top: 28px;
      flex-wrap: wrap;
    }
    a {
      color: white;
      background: #2563eb;
      padding: 11px 16px;
      border-radius: 6px;
      text-decoration: none;
      font-weight: 700;
    }
  </style>
</head>
<body>
  <main>
    <h1>Blank canvas</h1>
    <p>This is the session-local index.html. Ask the left panel to transform it.</p>
    <div class="actions">
      <a href="#">Primary action</a>
    </div>
  </main>
</body>
</html>
"""


@dataclass
class Session:
    id: str
    workspace: Path
    target_file: str = "index.html"
    created_at: float = field(default_factory=time.time)
    version: int = 0
    codex_thread: Any | None = None
    codex_client: Any | None = None
    running: bool = False
    last_hash: str | None = None
    subscribers: list[queue.Queue[dict[str, Any]]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    task_queue: queue.Queue[tuple[str, str]] = field(default_factory=queue.Queue)

    @property
    def target_path(self) -> Path:
        return self.workspace / self.target_file


class HarnessApp:
    def __init__(self) -> None:
        import_codex_sdk()
        from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

        self.ApprovalMode = ApprovalMode
        self.Codex = Codex
        self.CodexConfig = CodexConfig
        self.Sandbox = Sandbox
        self.codex_bin = resolve_codex_bin()
        self.sandbox, self.sandbox_mode, self.sandbox_warning = self._resolve_sandbox()
        self.sessions: dict[str, Session] = {}
        self.lock = threading.RLock()
        DATA_ROOT.mkdir(parents=True, exist_ok=True)

    def _resolve_sandbox(self) -> tuple[Any, str, str | None]:
        requested = SANDBOX_ENV
        if requested in ("workspace-write", "workspace_write"):
            return self.Sandbox.workspace_write, "workspace-write", None
        if requested in ("full-access", "full_access", "danger-full-access", "danger_full_access"):
            return (
                self.Sandbox.full_access,
                "full-access",
                "HARNESS_CODEX_SANDBOX explicitly requested full-access; filesystem access is not isolated.",
            )
        if requested not in ("", "auto"):
            raise RuntimeError(
                "HARNESS_CODEX_SANDBOX must be one of auto, workspace-write, or full-access"
            )

        ok, warning = probe_workspace_write_sandbox()
        if ok:
            return self.Sandbox.workspace_write, "workspace-write", None
        return (
            self.Sandbox.full_access,
            "full-access",
            f"workspace-write unavailable; falling back to full-access for this local demo. {warning}",
        )

    def runtime_info(self) -> dict[str, Any]:
        return {
            "codex_bin": str(self.codex_bin),
            "sandbox_mode": self.sandbox_mode,
            "sandbox_warning": self.sandbox_warning,
        }

    def create_session(self) -> Session:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        workspace = DATA_ROOT / "sessions" / session_id / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        target = workspace / "index.html"
        target.write_text(initial_html(), encoding="utf-8")
        session = Session(id=session_id, workspace=workspace)
        session.last_hash = sha256_file(target)
        with self.lock:
            self.sessions[session_id] = session
        self.publish(
            session,
            {
                "type": "session.created",
                "session_id": session.id,
                "preview_url": self.preview_url(session),
                "workspace": str(session.workspace),
                "codex_bin": str(self.codex_bin),
                "sandbox_mode": self.sandbox_mode,
                "sandbox_warning": self.sandbox_warning,
            },
        )
        self.publish_preview_update(session, force=True)
        return session

    def reset_session(self, session: Session) -> None:
        with session.lock:
            if session.running:
                raise RuntimeError("cannot reset while Codex is running")
            session.target_path.write_text(initial_html(), encoding="utf-8")
            session.last_hash = None
            session.version = 0
        self.publish(session, {"type": "session.reset", "session_id": session.id})
        self.publish_preview_update(session, force=True)

    def get_session(self, session_id: str) -> Session:
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        with self.lock:
            sessions = list(self.sessions.values())
        return [
            {
                "id": s.id,
                "created_at": s.created_at,
                "running": s.running,
                "preview_url": self.preview_url(s),
                "workspace": str(s.workspace),
                "sandbox_mode": self.sandbox_mode,
                "sandbox_warning": self.sandbox_warning,
            }
            for s in sessions
        ]

    def subscribe(self, session: Session) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=500)
        with session.lock:
            session.subscribers.append(q)
        q.put(
            {
                "type": "hello",
                "session_id": session.id,
                "preview_url": self.preview_url(session),
                "workspace": str(session.workspace),
                "codex_bin": str(self.codex_bin),
                "sandbox_mode": self.sandbox_mode,
                "sandbox_warning": self.sandbox_warning,
            }
        )
        return q

    def unsubscribe(self, session: Session, q: queue.Queue[dict[str, Any]]) -> None:
        with session.lock:
            if q in session.subscribers:
                session.subscribers.remove(q)

    def publish(self, session: Session, event: dict[str, Any]) -> None:
        event.setdefault("session_id", session.id)
        event.setdefault("ts", time.time())
        dead: list[queue.Queue[dict[str, Any]]] = []
        with session.lock:
            subscribers = list(session.subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        if dead:
            with session.lock:
                for q in dead:
                    if q in session.subscribers:
                        session.subscribers.remove(q)

    def preview_url(self, session: Session) -> str:
        return f"/preview/{session.id}/{session.target_file}?v={session.version}"

    def publish_preview_update(self, session: Session, *, force: bool = False) -> None:
        path = session.target_path
        if not path.exists():
            return
        digest = sha256_file(path)
        with session.lock:
            if not force and digest == session.last_hash:
                return
            session.last_hash = digest
            session.version += 1
            version = session.version
        self.publish(
            session,
            {
                "type": "preview.updated",
                "path": session.target_file,
                "version": version,
                "sha256": digest,
                "url": self.preview_url(session),
            },
        )

    def start_task(self, session: Session, instruction: str) -> str:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("instruction is required")
        task_id = f"task_{uuid.uuid4().hex[:10]}"
        with session.lock:
            if session.running:
                raise RuntimeError("Codex is already running for this session")
            session.running = True
        self.publish(session, {"type": "chat.message", "role": "o45", "text": instruction})
        self.publish(
            session,
            {
                "type": "tool.call",
                "task_id": task_id,
                "name": "codex.edit_html",
                "args": {"target": session.target_file, "instruction": instruction},
                "sandbox_mode": self.sandbox_mode,
                "sandbox_warning": self.sandbox_warning,
            },
        )
        thread = threading.Thread(
            target=self._run_task_thread,
            args=(session, task_id, instruction),
            name=f"codex-task-{session.id}",
            daemon=True,
        )
        thread.start()
        return task_id

    def _ensure_codex_thread(self, session: Session) -> Any:
        with session.lock:
            if session.codex_thread is not None:
                return session.codex_thread
            config_overrides = (
                'web_search="disabled"',
                'sandbox_workspace_write.network_access=false',
            )
            config = self.CodexConfig(
                codex_bin=str(self.codex_bin),
                cwd=str(session.workspace),
                config_overrides=config_overrides,
                client_name="o45_harness_demo",
                client_title="O45 Harness Demo",
                client_version="0.1.0",
                experimental_api=True,
            )
            codex = self.Codex(config=config)
            thread = codex.thread_start(
                cwd=str(session.workspace),
                sandbox=self.sandbox,
                approval_mode=self.ApprovalMode.deny_all,
                config={
                    "web_search": "disabled",
                    "sandbox_workspace_write": {"network_access": False},
                },
            )
            session.codex_client = codex
            session.codex_thread = thread
            return thread

    def _run_task_thread(self, session: Session, task_id: str, instruction: str) -> None:
        watcher_stop = threading.Event()
        watcher = threading.Thread(
            target=self._watch_preview,
            args=(session, watcher_stop),
            name=f"preview-watch-{session.id}",
            daemon=True,
        )
        watcher.start()
        try:
            self._run_codex_turn(session, task_id, instruction)
        except Exception as exc:
            self.publish(
                session,
                {
                    "type": "tool.result",
                    "task_id": task_id,
                    "status": "failed",
                    "summary": str(exc),
                },
            )
        finally:
            watcher_stop.set()
            watcher.join(timeout=1)
            self.publish_preview_update(session)
            with session.lock:
                session.running = False
            self.publish(session, {"type": "task.idle", "task_id": task_id})

    def _watch_preview(self, session: Session, stop: threading.Event) -> None:
        while not stop.wait(0.35):
            try:
                self.publish_preview_update(session)
            except Exception as exc:
                self.publish(session, {"type": "preview.error", "message": str(exc)})

    def _run_codex_turn(self, session: Session, task_id: str, instruction: str) -> None:
        thread = self._ensure_codex_thread(session)
        prompt = (
            "你是被 o45 调度的 Codex worker。当前工作目录里有且只有一个目标文件 index.html。\n"
            "请根据 o45 的需求直接编辑 index.html，并尽量保持它是一个完整、可直接在浏览器渲染的单文件 HTML。\n"
            "约束：\n"
            "- 只修改 index.html。\n"
            "- 不要联网。\n"
            "- 不要创建额外文件。\n"
            "- 如果需要运行命令，只运行用于检查当前文件内容的安全本地命令。\n"
            "- 完成后用中文简要说明你改了什么。\n\n"
            f"o45 需求：{instruction}"
        )
        self.publish(session, {"type": "task.started", "task_id": task_id})
        turn = thread.turn(
            prompt,
            approval_mode=self.ApprovalMode.deny_all,
            cwd=str(session.workspace),
            sandbox=self.sandbox,
        )
        self.publish(session, {"type": "codex.turn.started", "task_id": task_id, "turn_id": turn.id})
        stream = turn.stream()
        final_text = ""
        try:
            for notification in stream:
                event = self._map_codex_notification(notification, task_id)
                if event:
                    if event["type"] == "codex.agent_message.completed":
                        final_text = event.get("text") or final_text
                    self.publish(session, event)
        finally:
            stream.close()
        self.publish_preview_update(session)
        self.publish(
            session,
            {
                "type": "tool.result",
                "task_id": task_id,
                "status": "completed",
                "summary": final_text or "Codex 已完成本轮修改。",
                "preview_url": self.preview_url(session),
                "version": session.version,
            },
        )

    def _map_codex_notification(self, notification: Any, task_id: str) -> dict[str, Any] | None:
        method = getattr(notification, "method", "")
        payload = getattr(notification, "payload", None)
        base = {"task_id": task_id, "codex_method": method}

        if method == "item/agentMessage/delta":
            return {**base, "type": "codex.text.delta", "text": getattr(payload, "delta", "")}
        if method == "item/reasoning/summaryTextDelta":
            return {**base, "type": "codex.reasoning.delta", "text": getattr(payload, "delta", "")}
        if method == "item/commandExecution/outputDelta":
            return {
                **base,
                "type": "codex.command.output",
                "stream": getattr(payload, "stream", None),
                "text": getattr(payload, "delta", ""),
            }
        if method == "item/fileChange/patchUpdated":
            return {**base, "type": "codex.file.patch", "payload": self._jsonable(payload)}
        if method == "turn/plan/updated":
            return {**base, "type": "codex.plan.updated", "payload": self._jsonable(payload)}
        if method == "turn/diff/updated":
            return {**base, "type": "codex.diff.updated", "payload": self._jsonable(payload)}
        if method == "item/started":
            return {**base, "type": "codex.item.started", "payload": self._jsonable(payload)}
        if method == "item/completed":
            data = self._jsonable(payload)
            item = data.get("item", {}) if isinstance(data, dict) else {}
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type == "agentMessage":
                return {
                    **base,
                    "type": "codex.agent_message.completed",
                    "text": item.get("text", ""),
                    "payload": data,
                }
            return {**base, "type": "codex.item.completed", "payload": data}
        if method == "turn/completed":
            return {**base, "type": "codex.turn.completed", "payload": self._jsonable(payload)}
        if method == "error":
            return {**base, "type": "codex.error", "payload": self._jsonable(payload)}
        return {**base, "type": "codex.event", "payload": self._jsonable(payload)}

    def _jsonable(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump(by_alias=True, mode="json", exclude_none=True)
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        return value


APP: HarnessApp | None = None
APP_LOCK = threading.RLock()


def get_app() -> HarnessApp:
    global APP
    with APP_LOCK:
        if APP is None:
            APP = HarnessApp()
        return APP


def app_status() -> dict[str, Any]:
    try:
        app = get_app()
    except Exception as exc:
        return {
            "ready": False,
            "error": str(exc),
            "data_root": str(DATA_ROOT),
            "uses_global_codex": False,
            "codex_bin": None,
            "sandbox_mode": None,
            "sandbox_warning": None,
        }
    return {
        "ready": True,
        "error": None,
        "data_root": str(DATA_ROOT),
        "uses_global_codex": False,
        **app.runtime_info(),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "CodexHarnessDemo/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:
        try:
            self._do_GET()
        except Exception as exc:
            write_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self) -> None:
        try:
            self._do_POST()
        except ValueError as exc:
            write_json(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except KeyError:
            write_json(self, HTTPStatus.NOT_FOUND, {"error": "session not found"})
        except RuntimeError as exc:
            write_json(self, HTTPStatus.CONFLICT, {"error": str(exc)})
        except Exception as exc:
            write_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._serve_static(STATIC_ROOT / "index.html")
            return
        if path.startswith("/static/"):
            rel = safe_rel_path(path.removeprefix("/static/"))
            self._serve_static(STATIC_ROOT / rel)
            return
        if path == "/api/status":
            write_json(self, HTTPStatus.OK, app_status())
            return
        if path == "/api/sessions":
            app = get_app()
            write_json(self, HTTPStatus.OK, {"sessions": app.list_sessions()})
            return
        if path.startswith("/api/sessions/") and path.endswith("/events"):
            session_id = path.split("/")[3]
            app = get_app()
            self._serve_events(app.get_session(session_id))
            return
        if path.startswith("/preview/"):
            self._serve_preview(path)
            return
        write_json(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/sessions":
            app = get_app()
            session = app.create_session()
            write_json(
                self,
                HTTPStatus.OK,
                {
                    "session_id": session.id,
                    "preview_url": app.preview_url(session),
                    "workspace": str(session.workspace),
                    "codex_bin": str(app.codex_bin),
                    "sandbox_mode": app.sandbox_mode,
                    "sandbox_warning": app.sandbox_warning,
                },
            )
            return
        if path.startswith("/api/sessions/") and path.endswith("/messages"):
            session_id = path.split("/")[3]
            app = get_app()
            session = app.get_session(session_id)
            payload = read_json(self)
            instruction = str(payload.get("message", ""))
            task_id = app.start_task(session, instruction)
            write_json(self, HTTPStatus.OK, {"task_id": task_id})
            return
        if path.startswith("/api/sessions/") and path.endswith("/reset"):
            session_id = path.split("/")[3]
            app = get_app()
            session = app.get_session(session_id)
            app.reset_session(session)
            write_json(self, HTTPStatus.OK, {"ok": True, "preview_url": app.preview_url(session)})
            return
        write_json(self, HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _serve_static(self, path: Path) -> None:
        path = path.resolve()
        if STATIC_ROOT not in path.parents and path != STATIC_ROOT:
            write_json(self, HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        self._serve_file(path, cache=False)

    def _serve_preview(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) < 4:
            write_json(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        app = get_app()
        session = app.get_session(parts[2])
        rel = safe_rel_path("/".join(parts[3:]))
        target = (session.workspace / rel).resolve()
        if session.workspace.resolve() not in target.parents and target != session.workspace.resolve():
            write_json(self, HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            return
        self._serve_file(target, cache=False)

    def _serve_file(self, path: Path, *, cache: bool) -> None:
        if not path.exists() or not path.is_file():
            write_json(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=60" if cache else "no-store")
        if content_type == "text/html":
            self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline' data: blob:")
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self, session: Session) -> None:
        app = get_app()
        q = app.subscribe(session)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                try:
                    event = q.get(timeout=15)
                    payload = json.dumps(event, ensure_ascii=False)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            app.unsubscribe(session, q)


def main() -> None:
    status = app_status()
    print("Codex Harness Demo")
    print(f"  URL:       http://127.0.0.1:{PORT}")
    print(f"  Data root: {DATA_ROOT}")
    print(f"  Codex bin: {status.get('codex_bin') or 'not ready'}")
    print(f"  Sandbox:  {status.get('sandbox_mode') or 'not ready'}")
    if status.get("sandbox_warning"):
        print("  Sandbox warning:")
        print(f"    {status['sandbox_warning']}")
    print("  Note: this demo does not use PATH/global codex.")
    if not status["ready"]:
        print("  Startup warning:")
        print(f"    {status['error']}")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
