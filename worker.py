"""modeling.o5 推理 Worker

每个 Worker 占用一张 GPU，持有一个 UnifiedProcessor 实例，
提供 Chat (HTTP) / Streaming (WebSocket) / Duplex (WebSocket) 三种推理 API。

启动方式：
    cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/base/bin/python worker.py \\
        --port 10031 \\
        --model-path /path/to/base_model \\
        --pt-path /path/to/custom.pt \\
        --ref-audio-path /path/to/ref.wav
"""

import json
import time
import asyncio
import argparse
import copy
import logging
import base64
import os
import threading
from typing import Optional, List, Dict, Any

import numpy as np
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from worker_state import WorkerState, WorkerStatus
from runtime.protocol import DEFAULT_WORKER_CAPABILITIES
from runtime.session import BackendRuntimeSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")


def _ws_debug_enabled() -> bool:
    return os.environ.get("MCPMO_WS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _uvicorn_log_config(*, debug: bool) -> Dict[str, Any]:
    config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    fmt = "%(asctime)s [%(levelprefix)s] %(name)s: %(message)s"
    config["formatters"]["default"]["fmt"] = fmt
    config["formatters"]["access"]["fmt"] = (
        '%(asctime)s [%(levelprefix)s] %(name)s: %(client_addr)s - "%(request_line)s" %(status_code)s'
    )
    if debug:
        config["loggers"]["uvicorn.error"]["level"] = "DEBUG"
        config["loggers"]["websockets"] = {"handlers": ["default"], "level": "DEBUG", "propagate": False}
        config["loggers"]["websockets.client"] = {"handlers": ["default"], "level": "DEBUG", "propagate": False}
        config["loggers"]["websockets.server"] = {"handlers": ["default"], "level": "DEBUG", "propagate": False}
    return config


def _enable_ws_debug_logging() -> bool:
    enabled = _ws_debug_enabled()
    if not enabled:
        return False
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
    for name in ("websockets", "websockets.client", "websockets.server", "uvicorn.error"):
        logging.getLogger(name).setLevel(logging.DEBUG)
    logger.info("WebSocket protocol debug logging enabled")
    return True

# ============ 请求/响应模型 ============

class WorkerHealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    worker_status: WorkerStatus
    gpu_id: int
    model_loaded: bool
    current_ticket_id: Optional[str] = None
    total_requests: int = 0
    avg_inference_time_ms: float = 0.0
    kv_cache_length: int = 0  # 当前 LLM KV cache token 总数
    capabilities: List[str] = Field(default_factory=list)


# ============ FastAPI 应用 ============

worker: Optional[Any] = None

# 启动参数（通过 main() 传入）
WORKER_CONFIG: Dict[str, Any] = {}


class RemoteBackendWorker:
    """Worker host used when inference lives in backend_server.py."""

    def __init__(self, *, backend_server_url: str, gpu_id: int = 0) -> None:
        self.backend_server_url = backend_server_url
        self.gpu_id = gpu_id
        self.processor = None
        self.state = WorkerState(status=WorkerStatus.IDLE)

    def metrics(self) -> Dict[str, Any]:
        return {"backend": "backend_server", "backend_server_url": self.backend_server_url}

    def shutdown(self) -> None:
        return None


def _backend_server_url() -> Optional[str]:
    value = WORKER_CONFIG.get("backend_server_url")
    return str(value).rstrip("/") if value else None


def _input_payload(message: Dict[str, Any]) -> Dict[str, Any]:
    value = message.get("input")
    if isinstance(value, dict):
        return value
    raise RuntimeError("input.append must carry an object `input`")


def _init_payload(message: Dict[str, Any]) -> Dict[str, Any]:
    value = message.get("payload")
    if isinstance(value, dict):
        return value
    raise RuntimeError("session.init must carry an object `payload`")


def _event_payload(event: Any) -> Dict[str, Any]:
    payload = dict(getattr(event, "payload", {}) or {})
    raw_event = payload.get("event")
    if isinstance(raw_event, dict):
        return raw_event
    return payload


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型"""
    global worker
    config = WORKER_CONFIG

    backend_server_url = _backend_server_url()
    if backend_server_url:
        worker = RemoteBackendWorker(
            backend_server_url=backend_server_url,
            gpu_id=int(config.get("gpu_id", 0) or 0),
        )
        logger.info("Worker running as backend-server runtime host: %s", backend_server_url)
    else:
        from core.processors.backend_factory import create_backend

        worker = create_backend(config)

        # 模型加载是同步操作（~15s），在线程中执行避免阻塞
        await asyncio.to_thread(worker.load_model)

    try:
        yield
    finally:
        logger.info("Worker shutting down")
        if worker is not None:
            await asyncio.to_thread(worker.shutdown)


app = FastAPI(title="modeling.o5 Worker", lifespan=lifespan)


# ========== 健康检查 ==========

@app.get("/health", response_model=WorkerHealthResponse)
async def health():
    """健康检查"""
    if worker is None:
        return WorkerHealthResponse(
            status="initializing",
            worker_status=WorkerStatus.LOADING,
            gpu_id=0,
            model_loaded=False,
            capabilities=[],
        )

    avg_time = 0.0
    if worker.state.total_requests > 0:
        avg_time = worker.state.total_inference_time_ms / worker.state.total_requests

    worker_metrics = worker.metrics()
    kv_len = int(worker_metrics.get("kv_cache_length", 0) or 0)
    remote_backend_url = _backend_server_url()
    model_loaded = bool(remote_backend_url) or worker.processor is not None or bool(getattr(worker.state, "is_idle", False))
    return WorkerHealthResponse(
        status="healthy" if model_loaded else "error",
        worker_status=worker.state.status,
        gpu_id=worker.gpu_id,
        model_loaded=model_loaded,
        current_ticket_id=worker.state.current_ticket_id,
        total_requests=worker.state.total_requests,
        avg_inference_time_ms=avg_time,
        kv_cache_length=kv_len,
        capabilities=DEFAULT_WORKER_CAPABILITIES,
    )



_TURN_BASED_MODES = frozenset({"chat", "turn_based", "turn-based"})
_REALTIME_ADMISSION_TIMEOUT_S = 30.0
_REALTIME_ADMISSION_LOCK = asyncio.Lock()
_FULL_DUPLEX_MODES = frozenset({"audio", "video", "omni", "duplex", "full_duplex", "full-duplex"})


def _normalize_runtime_mode(value: Any, *, source: str) -> str:
    """Map public mode spellings to the backend runtime's two native modes."""
    mode = str(value or "").strip().lower()
    if mode in _TURN_BASED_MODES:
        return "turn_based"
    if mode in _FULL_DUPLEX_MODES:
        return "full_duplex"
    raise RuntimeError(f"unsupported {source} mode: {value!r}")


def _resolve_runtime_mode(first: Dict[str, Any], query_mode: Optional[str]) -> str:
    """Resolve one Session's runtime mode entirely inside the Worker."""
    if str(first.get("type") or "") != "session.init":
        raise RuntimeError(
            f"first message must be session.init, got: {first.get('type')!r}"
        )
    payload = _init_payload(first)
    candidates = (
        ("payload.mode", payload.get("mode")),
        ("payload.input_mode", payload.get("input_mode")),
        ("query mode", query_mode),
    )
    resolved = {
        _normalize_runtime_mode(value, source=source)
        for source, value in candidates
        if value
    }
    if len(resolved) > 1:
        raise RuntimeError(
            "conflicting session modes: "
            + ", ".join(f"{source}={value!r}" for source, value in candidates if value)
        )
    return next(iter(resolved), "full_duplex")


def _runtime_status(mode: str) -> WorkerStatus:
    return WorkerStatus.BUSY_CHAT if mode == "turn_based" else WorkerStatus.DUPLEX_ACTIVE


async def _handle_remote_backend_runtime_ws(
    ws: WebSocket,
    *,
    query_mode: Optional[str],
) -> None:
    """Bridge the single Worker realtime endpoint to backend_server.py."""

    backend_url = _backend_server_url()
    if backend_url is None:
        await ws.close(code=1013, reason="Backend server URL is not configured")
        return
    if worker is None:
        await ws.close(code=1013, reason="Worker not ready")
        return
    # This Worker accepts one stateful realtime session at a time. Reserve the
    # slot before the first frame arrives so concurrent connects cannot both
    # observe IDLE while the accepted client is still choosing its mode.
    await _REALTIME_ADMISSION_LOCK.acquire()
    first: Dict[str, Any]
    mode: str
    try:
        if not worker.state.is_idle:
            await ws.close(code=1013, reason=f"Worker busy: {worker.state.status.value}")
            return
        await ws.accept()
        try:
            first_raw = await asyncio.wait_for(
                ws.receive_text(), timeout=_REALTIME_ADMISSION_TIMEOUT_S
            )
            first = json.loads(first_raw)
            if not isinstance(first, dict):
                raise RuntimeError("first message must be a JSON object")
            mode = _resolve_runtime_mode(first, query_mode)
            worker.state.status = _runtime_status(mode)
        except Exception as exc:
            logger.error("Realtime session admission failed: %s", exc)
            try:
                await ws.send_json({
                    "type": "session.closed",
                    "session_id": None,
                    "reason": "backend_error",
                    "diagnostic": {"message": str(exc)},
                })
            except Exception:
                pass
            try:
                await ws.close(code=1008, reason="invalid_session_init")
            except Exception:
                pass
            return
    finally:
        _REALTIME_ADMISSION_LOCK.release()

    runtime: Optional[BackendRuntimeSession] = None
    backend_closed = False

    async def _send_runtime_event(event: Any) -> Dict[str, Any]:
        payload = _event_payload(event)
        await ws.send_json(payload)
        return payload

    try:
        runtime = BackendRuntimeSession(
            backend_base_url=backend_url,
            mode=mode,
        )

        init_params = dict(_init_payload(first))
        # The Worker owns mode normalization. Backend Server only sees its two
        # native modes, regardless of whether the client used chat/audio/video/omni.
        init_params["mode"] = mode
        init_event = await _send_runtime_event(await runtime.init(init_params))
        if init_event.get("type") == "session.closed":
            backend_closed = True
            return

        async def client_to_backend() -> None:
            nonlocal backend_closed
            async for raw in ws.iter_text():
                msg = json.loads(raw)
                msg_type = str(msg.get("type") or "")

                if msg_type == "input.append":
                    await runtime.push(_input_payload(msg))
                    continue

                if msg_type in {"input.tool_result", "input.tool_result.delta", "input.tool_result.done", "input.standalone_event"}:
                    payload = msg.get("payload")
                    if not isinstance(payload, dict):
                        payload = {k: v for k, v in msg.items() if k != "type"}
                    payload["type"] = msg_type.replace("input.", "")
                    await runtime.push(payload)
                    continue

                if msg_type == "session.close":
                    close_event = await runtime.unary("close", {"reason": str(msg.get("reason") or "client_closed")})
                    backend_closed = True
                    close_payload = _event_payload(close_event)
                    if close_payload.get("type") != "session.closed":
                        close_payload = {
                            "type": "session.closed",
                            "session_id": runtime.session_id,
                            "reason": msg.get("reason", "client_closed"),
                        }
                    await ws.send_json(close_payload)
                    await ws.close(code=1000, reason="client_closed")
                    return

                raise RuntimeError(f"unsupported runtime message type: {msg_type}")

        async def backend_to_client() -> None:
            nonlocal backend_closed
            while not backend_closed:
                event = await runtime.pull()
                payload = await _send_runtime_event(event)
                if payload.get("type") == "session.closed":
                    backend_closed = True
                    return

        tasks = [
            asyncio.create_task(client_to_backend()),
            asyncio.create_task(backend_to_client()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            task.result()

    except WebSocketDisconnect:
        logger.info("Remote backend runtime WebSocket disconnected")
    except Exception as exc:
        logger.error("Remote backend runtime failed: error=%s", exc, exc_info=True)
        try:
            await ws.send_json({
                "type": "session.closed",
                "session_id": runtime.session_id if runtime is not None else None,
                "reason": "backend_error",
                "diagnostic": {"message": str(exc)},
            })
        except Exception:
            pass
    finally:
        if runtime is not None:
            try:
                if not backend_closed:
                    await runtime.unary("close", {"reason": "worker_disconnected"})
            except Exception:
                logger.exception("Remote backend runtime cleanup failed")
        worker.state.status = WorkerStatus.IDLE
        worker.state.current_ticket_id = None
        try:
            await ws.close()
        except Exception:
            pass


@app.websocket("/v1/realtime")
async def worker_realtime_runtime_ws(ws: WebSocket):
    """Single realtime endpoint; Worker selects turn-based/duplex internally."""
    await _handle_remote_backend_runtime_ws(
        ws,
        query_mode=ws.query_params.get("mode"),
    )


# ============ 缓存状态查询 ==========

@app.get("/cache_info")
async def cache_info():
    """查询当前 Worker 的 KV Cache 状态"""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not ready")

    return {
        "status": worker.state.status.value,
        "note": "KV cache state is now tracked by Gateway (cached_hash on WorkerConnection)",
    }


@app.post("/clear_cache")
async def clear_cache():
    """手动清除 KV Cache（重置 Streaming 模型 session）"""
    if worker is None:
        raise HTTPException(status_code=503, detail="Worker not ready")

    worker.reset_half_duplex_session()
    return {"success": True, "message": "Cache cleared"}


# ============ 入口 ============

def main():
    from config import get_config
    ws_debug = _enable_ws_debug_logging()
    cfg = get_config()

    parser = argparse.ArgumentParser(description="modeling.o5 Worker")
    parser.add_argument("--port", type=int, default=None, help=f"Worker port (default: from config, base={cfg.worker_base_port})")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--model-path", type=str, default=None, help="Base model path")
    parser.add_argument("--pt-path", type=str, default=None, help="Custom weights path (.pt)")
    parser.add_argument("--ref-audio-path", type=str, default=None, help="Default ref audio path")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU ID (inferred from port if not set)")
    parser.add_argument("--worker-index", type=int, default=0, help="Worker index (0, 1, 2, ...)")
    parser.add_argument("--duplex-pause-timeout", type=float, default=None, help="Duplex pause timeout (s)")
    parser.add_argument("--backend-server-url", type=str, default=None, help="Remote backend_server.py base URL")
    args = parser.parse_args()

    port = args.port or cfg.worker_port(args.worker_index)
    gpu_id = args.gpu_id if args.gpu_id is not None else args.worker_index

    WORKER_CONFIG.update({
        "model_path": args.model_path or cfg.model.model_path,
        "gpu_id": gpu_id,
        "pt_path": args.pt_path or cfg.model.pt_path,
        "ref_audio_path": args.ref_audio_path or cfg.ref_audio_path,
        "duplex_pause_timeout": args.duplex_pause_timeout or cfg.duplex_pause_timeout,
        "backend_server_url": args.backend_server_url,
        "compile": cfg.compile,
        "chat_vocoder": cfg.chat_vocoder,
        "attn_implementation": cfg.attn_implementation,
    })

    logger.info(f"Starting Worker on port {port}, GPU {gpu_id}")
    # Bump WS max payload from uvicorn's 16 MiB default to 128 MiB so that
    # base64-encoded video attachments (commonly 30-60 MiB after inflation)
    # can be received without the connection being torn down with code 1009.
    uvicorn.run(
        app,
        host=args.host,
        port=port,
        ws="websockets" if ws_debug else "auto",
        log_level="debug" if ws_debug else "info",
        log_config=_uvicorn_log_config(debug=ws_debug),
        ws_max_size=128 * 1024 * 1024,
        # 与 py_backend/server.py 同理：本端忙于把输入转给 backend 时读不到对端的 pong，
        # 默认 20s 窗口会把健康会话误判为 keepalive 超时（2026-10-12 在 Gateway↔Worker 上实际发生）。
        ws_ping_interval=360.0,
        ws_ping_timeout=360.0,
    )


if __name__ == "__main__":
    main()
