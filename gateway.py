"""modeling.o5 推理 Gateway

请求分发网关，不加载模型，负责：
- 路由 Chat/Streaming/Duplex 请求到 Worker
- 会话映射和 KV Cache LRU 命中路由
- 统一 FIFO 请求排队（容量 1000，位置追踪 + ETA 估算）
- Worker 健康检查

启动方式：
    cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
    PYTHONPATH=. .venv/base/bin/python gateway.py \\
        --port 10024 --internal-port 10025
"""

import os
import re
import json
import asyncio
import argparse
import copy
import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager
from urllib.parse import urlencode

import zipfile
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Body
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, Response, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gateway_modules.models import (
    GatewayWorkerStatus,
    ServiceStatus,
    WorkersResponse,
    WorkerRegistrationRequest,
    WorkerDrainRequest,
    QueueStatus,
    EtaConfig,
    EtaStatus,
)
from gateway_modules.worker_pool import WorkerPool, WorkerConnection
from gateway_modules.endpoints import DEFAULT_ENDPOINT, EndpointError, EndpointRegistry
from gateway_modules.model_catalog import ModelCatalogError, catalog_entries, pool_model_id
from gateway_modules.live_sessions import LiveSession, LiveSessionRegistry
from runtime.protocol import API_INPUT_MODES, API_PROTOCOL_VERSION
from control_plane.routes import docs_router as control_docs_router
from control_plane.routes import public_router as control_public_router
from control_plane.routes import router as control_router
from control_plane.routes import set_control_plane
from gateway_modules.ref_audio_registry import (
    RefAudioRegistry,
    RefAudioListResponse,
    UploadRefAudioRequest,
    RefAudioResponse,
)
from gateway_modules.app_registry import (
    AppRegistry,
    AppToggleRequest,
    AppsPublicResponse,
    AppsAdminResponse,
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gateway")



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



_SESSION_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]+$')


def _sanitize_session_id(session_id: str) -> str:
    """校验 session_id 只含安全字符，防止 path traversal"""
    if not _SESSION_ID_RE.match(session_id):
        safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', session_id)
        return safe
    return session_id


def _sessions_root() -> str:
    """录制根目录：每个 Session 落在 <root>/<session_id>/。

    ``recording.dir`` 指定时用它（离线评测按 run 指到评测输出目录下）；
    否则沿用 ``<data_dir>/sessions``。读取接口与写入端共用这一个函数，保证两边一致。
    """
    from config import get_config
    cfg = get_config()
    if cfg.recording.dir:
        return os.path.realpath(os.path.join(_BASE_DIR, cfg.recording.dir))
    return os.path.realpath(os.path.join(_BASE_DIR, cfg.data_dir, "sessions"))


def _client_ip_from_ws(ws: WebSocket) -> Optional[str]:
    xff = ws.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return ws.headers.get("x-real-ip") or (ws.client.host if ws.client else None)


def _identity_dict(
    ws: WebSocket,
    *,
    source_channel: str,
    source_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect all client/page/source identity meta from the incoming WS.

    单一数据源:既用于透传给 worker(urlencode),也直接作为录制 meta。
    前端新增任何 identity 字段,只要在这里收集一次,即同时流入两处。
    """
    client_id = (
        ws.query_params.get("client_id")
        or ws.headers.get("x-client-id")
        or ws.cookies.get("client_id")
    )
    page_session_id = (
        ws.query_params.get("page_session_id")
        or ws.headers.get("x-page-session-id")
        or ws.cookies.get("page_session_id")
    )
    return {
        "client_id": client_id,
        "page_session_id": page_session_id,
        "client_ip": _client_ip_from_ws(ws),
        "user_agent": ws.headers.get("user-agent"),
        "origin": ws.headers.get("origin"),
        "source_channel": source_channel,
        "source_mode": source_mode,
        "source_path": ws.url.path,
        "page_route": ws.query_params.get("page_route"),
        "client_surface": ws.query_params.get("client_surface"),
    }


# ============ 全局变量 ============

worker_pool: Optional[WorkerPool] = None
"""默认内部池；公开会话通过 model ID 解析池，不使用隐式 default 路由。"""
endpoint_registry: Optional[EndpointRegistry] = None
live_sessions = LiveSessionRegistry()
_model_registration_lock = asyncio.Lock()
ref_audio_registry: Optional[RefAudioRegistry] = None
app_registry: AppRegistry = AppRegistry()

# 配置（通过 main() 传入）
GATEWAY_CONFIG: Dict[str, Any] = {}


# ============ 应用初始化 ============

_cleanup_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    global worker_pool, endpoint_registry, ref_audio_registry, _cleanup_task

    workers = GATEWAY_CONFIG.get("workers", [])
    max_queue = GATEWAY_CONFIG.get("max_queue_size", 1000)
    timeout = GATEWAY_CONFIG.get("timeout", 300.0)

    # 从 config 读取 ETA 参数
    eta_config_data = GATEWAY_CONFIG.get("eta_config")
    eta_config = EtaConfig(**eta_config_data) if eta_config_data else EtaConfig()

    def pool_factory(worker_addresses: Optional[List[str]] = None) -> WorkerPool:
        return WorkerPool(
            worker_addresses=worker_addresses or [],
            max_queue_size=max_queue,
            request_timeout=timeout,
            eta_config=eta_config.model_copy(),
            ema_alpha=GATEWAY_CONFIG.get("eta_ema_alpha", 0.3),
            ema_min_samples=GATEWAY_CONFIG.get("eta_ema_min_samples", 3),
            placement=GATEWAY_CONFIG.get("placement", "spread"),
        )

    endpoint_registry = EndpointRegistry(pool_factory)
    worker_pool = await endpoint_registry.start(pool_factory(workers))

    control_plane = None
    if GATEWAY_CONFIG.get("control_plane_root"):
        # 控制面：管理端点、Worker 模板与 Worker job 生命周期（cctl），与上面的端点池共用。
        from control_plane.cctl import CctlRunner
        from control_plane.manager import ControlPlane
        from scripts.run_o5_api import load_o5_api_runtime_settings

        control_plane = ControlPlane(
            root=Path(GATEWAY_CONFIG["control_plane_root"]),
            registry=endpoint_registry,
            live_sessions=live_sessions,
            jobs=CctlRunner(),
            repo_dir=str(Path(__file__).resolve().parent),
            python=str(load_o5_api_runtime_settings().runtime_python),
        )
        await control_plane.start()
        set_control_plane(control_plane)
        logger.info("Control plane enabled: root=%s", GATEWAY_CONFIG["control_plane_root"])

    # 初始化参考音频注册表
    data_dir = os.path.join(os.path.dirname(__file__), "data", "assets", "ref_audio")
    ref_audio_registry = RefAudioRegistry(storage_dir=data_dir)

    # 启动 session 清理后台任务（每天一次）
    _cleanup_task = asyncio.create_task(_session_cleanup_loop())

    logger.info(f"Gateway started, {len(worker_pool.workers)} workers, {ref_audio_registry.count} ref audios")

    yield

    if _cleanup_task:
        _cleanup_task.cancel()
    if control_plane is not None:
        set_control_plane(None)
        await control_plane.stop()
    await endpoint_registry.stop()
    logger.info("Gateway stopped")


async def _session_cleanup_loop() -> None:
    """每天执行一次 session 清理（retention_days 和 max_storage_gb 都为 -1 时不执行）"""
    from session_cleanup import cleanup_sessions
    from config import get_config

    await asyncio.sleep(60)
    while True:
        try:
            cfg = get_config()
            days = cfg.recording.session_retention_days
            gb = cfg.recording.max_storage_gb
            if days < 0 and gb < 0:
                logger.info("[Cleanup] Disabled (retention_days=-1, max_storage_gb=-1), sleeping")
            else:
                report = await asyncio.to_thread(
                    cleanup_sessions, cfg.data_dir, days, gb,
                )
                logger.info(f"[Cleanup] {report}")
        except Exception as e:
            logger.error(f"[Cleanup] Failed: {e}", exc_info=True)
        await asyncio.sleep(86400)


app = FastAPI(
    title="modeling.o5 Gateway",
    description="modeling.o5 多模态推理网关",
    version="1.0.0-alpha.2",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

internal_app = FastAPI(
    title="modeling.o5 Gateway Internal",
    description="Internal worker registration API",
    version="1.0.0-alpha.2",
    docs_url=None,
    redoc_url=None,
)


_PUBLIC_ADMIN_ENABLED = os.getenv("ENABLE_PUBLIC_ADMIN", "").lower() in {"1", "true", "yes", "on"}
_BLOCKED_ADMIN_PATHS = {"/admin", "/admin/"}
_BLOCKED_ADMIN_STATIC_PATHS = {"/static/admin.html", "/static/admin.html/"}


# API-only 部署（对外公开的 API Gateway）：只放行 API、文档、控制面板与公开状态页，其余页面与接口一律 404。
# 老页面（O45 时期的 /、/omni、/admin…）和会修改服务状态的旧接口（DELETE /api/queue、
# PUT /api/config/eta、参考音频上传、会话录音）都不对外。评测本地网关不设这个开关，行为不变。
_API_ONLY = os.getenv("GATEWAY_API_ONLY", "").lower() in {"1", "true", "yes", "on"}
_API_ONLY_PREFIXES = ("/api-docs", "/control", "/overview")
_API_ONLY_EXACT = {"/health", "/v1/models"}


def api_only_allows(path: str) -> bool:
    return path in _API_ONLY_EXACT or path.startswith(_API_ONLY_PREFIXES)


@app.middleware("http")
async def restrict_api_only(request: Request, call_next):
    if _API_ONLY and not api_only_allows(request.url.path):
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return await call_next(request)


@app.middleware("http")
async def hide_admin_routes(request: Request, call_next):
    """Keep admin-only surfaces unavailable on public deployments by default."""
    if not _PUBLIC_ADMIN_ENABLED:
        path = request.url.path
        if (
            path in _BLOCKED_ADMIN_PATHS
            or path.startswith("/api/admin/")
            or path in _BLOCKED_ADMIN_STATIC_PATHS
        ):
            return JSONResponse({"detail": "Not found"}, status_code=404)
    return await call_next(request)


# ============ 健康检查 ============

@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


# ============ 前端诊断日志 (用于排查录音故障) ============

_DEBUG_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".run-logs")
_DEBUG_LOG_PATH = os.path.join(_DEBUG_LOG_DIR, "mobile-record-trace.jsonl")
_DEBUG_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB rolling cap


def _append_debug_trace(payload: Dict[str, Any]) -> None:
    """Append a single JSON line to the rolling debug log."""
    try:
        os.makedirs(_DEBUG_LOG_DIR, exist_ok=True)
        # Roll over if oversized.
        try:
            if os.path.exists(_DEBUG_LOG_PATH) and os.path.getsize(_DEBUG_LOG_PATH) > _DEBUG_LOG_MAX_BYTES:
                rolled = _DEBUG_LOG_PATH + ".1"
                if os.path.exists(rolled):
                    os.remove(rolled)
                os.rename(_DEBUG_LOG_PATH, rolled)
        except OSError:
            pass
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            **payload,
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("failed to write mobile record trace: %s", exc)


@app.post("/api/_debug/record_trace")
async def post_record_trace(payload: Dict[str, Any]):
    """Receive a recording-session trace from the mobile frontend.

    The frontend posts one record per press-to-talk attempt with the
    full event timeline so we can diagnose failures (overlay shown but
    no audio captured) without asking users to copy console logs.
    """
    _append_debug_trace(payload)
    return {"ok": True}


@app.get("/status", response_model=ServiceStatus)
async def status():
    """服务状态"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return ServiceStatus(
        gateway_healthy=True,
        total_workers=len(worker_pool.workers),
        idle_workers=worker_pool.idle_count,
        busy_workers=worker_pool.busy_count,
        duplex_workers=worker_pool.duplex_count,
        loading_workers=worker_pool.loading_count,
        error_workers=worker_pool.error_count,
        offline_workers=worker_pool.offline_count,
        queue_length=worker_pool.queue_length,
        max_queue_size=worker_pool.max_queue_size,
        running_tasks=worker_pool._get_running_tasks(),
    )


@app.get("/workers", response_model=WorkersResponse)
async def list_workers():
    """Worker 列表"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return WorkersResponse(
        total=len(worker_pool.workers),
        workers=worker_pool.get_all_workers(),
    )


@internal_app.get("/health")
async def internal_health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@internal_app.put("/internal/workers/{worker_id}")
async def register_worker(worker_id: str, payload: WorkerRegistrationRequest):
    """Register or update a worker endpoint from the internal control plane."""
    if endpoint_registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    async with _model_registration_lock:
        target = payload.endpoint_name or DEFAULT_ENDPOINT
        pool = endpoint_registry.get(target)
        if pool is None:
            raise HTTPException(status_code=404, detail=f"unknown endpoint {target}")
        found = endpoint_registry.find_worker(worker_id)
        if found is not None and found[0] != target:
            raise HTTPException(status_code=409, detail=f"worker {worker_id} 已属于端点 {found[0]}")
        model_id = (payload.model_id or getattr(pool, "public_model_id", None)
                    or payload.model_info.get("model_id") or payload.model_info.get("profile_id"))
        if model_id:
            bound = getattr(pool, "public_model_id", None)
            if bound and bound != model_id:
                raise HTTPException(status_code=409, detail="Model pool already bound to a different model ID")
            for name, other in endpoint_registry.items():
                if name != target and (getattr(other, "public_model_id", None) == model_id or pool_model_id(other) == model_id):
                    raise HTTPException(status_code=409, detail="Model ID already belongs to another pool")
        try:
            worker = await pool.register_worker(
                worker_id=worker_id,
                endpoint=payload.endpoint,
                gpu_group=payload.gpu_group,
                labels=payload.labels,
                model_info=payload.model_info or None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if model_id:
            pool.public_model_id = model_id
        return {
            "ok": True,
            "worker": worker.to_info(),
        }


@internal_app.post("/internal/workers/{worker_id}/drain")
async def drain_worker(worker_id: str, payload: WorkerDrainRequest):
    """摘流 / 恢复一个 Worker：摘流后不再接新会话，手上的会话照常跑完。

    离线评测的弹性缩容用它：控制器先摘流某台扩容主机的全部 Worker，确认都空闲后再让该主机退出，
    这样退出时不会掐断会话，也不会在退出前一刻又被分到新会话。
    """
    if endpoint_registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    found = endpoint_registry.find_worker(worker_id)
    worker = found[1].set_draining(worker_id, payload.draining) if found else None
    if worker is None:
        raise HTTPException(status_code=404, detail=f"unknown worker {worker_id}")
    return {"ok": True, "worker": worker.to_info()}



# ============ Chat WebSocket 代理 ============

async def _api_worker_passthrough_ws(
    ws: WebSocket,
    *,
    request_type: str,
    worker_status: GatewayWorkerStatus,
    worker_path: str,
    source_channel: str,
    source_mode: Optional[str],
    max_duration_s: Optional[float] = None,
    pool: Optional[WorkerPool] = None,
    endpoint_name: str = DEFAULT_ENDPOINT,
    input_mode: Optional[str] = None,
) -> None:
    """Queue, assign a worker, then pass API-shaped events through unchanged.

    ``input_mode``(v5 端点会话)不为 None 时,把它写进客户端 ``session.init`` 的
    ``payload.input_mode`` 再转发;客户端自己写了且与 URL 不一致则直接拒绝。其它帧原样转发。
    """

    pool = pool or worker_pool
    if pool is None:
        await ws.close(code=1013, reason="Service not ready")
        return

    await ws.accept()

    # 客户端连接从接受起就持续读取，进入队列，排队期间也不停：uvicorn（sansio 实现）收到一条消息后
    # 会暂停读 socket，直到应用 receive()；若排队时不读，客户端的 ping 得不到 pong，排队超过约 40 秒
    # 就会被客户端判为 keepalive 超时断开。排队期间提前发来的消息（如 session.init）按序缓存，
    # 分到 Worker 后先转发它们。读到 None 表示客户端已断开。
    client_frames: "asyncio.Queue[Optional[tuple]]" = asyncio.Queue()
    client_gone = asyncio.Event()

    async def read_client() -> None:
        try:
            async for raw in ws.iter_text():
                await client_frames.put((time.time(), raw))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            client_gone.set()
            await client_frames.put(None)

    client_reader = asyncio.create_task(read_client())

    # ---- API 往返数据录制(旁路,fail-safe:任何异常都不影响转发主路径)----
    # 网关只认 session_id,不知道这个 Session 拿来干什么。从连接建立起就开始缓冲,
    # 包括排队阶段网关自己发的帧;拿到 worker 在 session.created 里返回的 session_id
    # 后才建 <recording_root>/<session_id>/ 并补写缓冲。始终没拿到 session_id 的连接
    # (排队满、排队中断开)不落盘。
    recording_enabled = False
    recorder_cls = None
    recorder_sessions_dir: Optional[str] = None
    try:
        from config import get_config
        _cfg = get_config()
        if _cfg.recording.enabled:
            from gateway_modules.session_recording import SessionRecorder
            recorder_cls = SessionRecorder
            recorder_sessions_dir = _sessions_root()
            recording_enabled = True
    except Exception:
        recording_enabled = False
        recorder_cls = None
        recorder_sessions_dir = None
    recorder_mode = "turn_based" if request_type == "chat" else "full_duplex"
    pending_record_frames: List[tuple] = []
    recorder = None

    def record_or_buffer(
        direction: str,
        frame: Dict[str, Any],
        *,
        recv_ts: float,
        send_ts: Optional[float],
        src: str,
    ) -> None:
        if not recording_enabled:
            return
        if recorder is None:
            pending_record_frames.append((direction, frame, recv_ts, send_ts, src))
            return
        try:
            recorder.record(direction, frame, recv_ts=recv_ts, send_ts=send_ts, src=src)
        except Exception:
            pass

    async def gateway_send(frame: Dict[str, Any]) -> None:
        """网关自己产生的帧(排队/错误/超时):发给客户端并录制。"""
        produced = time.time()
        sent: Optional[float] = None
        try:
            await ws.send_json(frame)
            sent = time.time()
        finally:
            record_or_buffer("down", frame, recv_ts=produced, send_ts=sent, src="gateway")

    async def acquire_worker() -> Optional[tuple]:
        """排队直到分到一个 Worker；失败时已经通知客户端并关闭，返回 None。"""

        try:
            ticket, future = pool.enqueue(request_type)
        except WorkerPool.QueueFullError:
            await gateway_send({
                "type": "error",
                "error": {"code": "queue_full", "message": "Queue full", "type": "server_error"},
            })
            await ws.close(code=1013, reason="Queue full")
            return None

        worker: Optional[WorkerConnection] = None
        if future.done():
            worker = future.result()
        else:
            try:
                await gateway_send({
                    "type": "session.queued",
                    "position": ticket.position,
                    "estimated_wait_s": ticket.estimated_wait_s,
                    "ticket_id": ticket.ticket_id,
                    "queue_length": pool.queue_length,
                })
                while not future.done():
                    try:
                        worker = await asyncio.wait_for(asyncio.shield(future), timeout=3.0)
                        break
                    except asyncio.TimeoutError:
                        if client_gone.is_set():
                            logger.info("API WS disconnected during queue: ticket=%s", ticket.ticket_id)
                            pool.cancel(ticket.ticket_id)
                            return None
                        updated = pool.get_ticket(ticket.ticket_id)
                        if updated:
                            await gateway_send({
                                "type": "session.queue_update",
                                "position": updated.position,
                                "estimated_wait_s": updated.estimated_wait_s,
                                "queue_length": pool.queue_length,
                            })
                    except asyncio.CancelledError:
                        pool.cancel(ticket.ticket_id)
                        return None
            except (WebSocketDisconnect, Exception) as exc:
                logger.info("API WS disconnected during queue: ticket=%s (%s)", ticket.ticket_id, exc)
                pool.cancel(ticket.ticket_id)
                return None
            if worker is None and future.done():
                worker = future.result()

        if worker is None:
            await gateway_send({
                "type": "error",
                "error": {"code": "worker_busy", "message": "No worker available", "type": "server_error"},
            })
            await ws.close(code=1013, reason="No worker available")
            return None
        return ticket, worker

    import websockets
    identity = _identity_dict(
        ws,
        source_channel=source_channel,
        source_mode=source_mode,
    )
    identity["endpoint"] = endpoint_name
    identity["model_id"] = ws.query_params.get("model")
    identity_qs = urlencode({k: v for k, v in identity.items() if v})

    # 分到的 Worker 连不上（例如刚挂掉、健康检查还没发现）：标为离线并重新排队一次，
    # 而不是直接让这次会话失败。
    worker_ws = None
    for attempt in range(2):
        acquired = await acquire_worker()
        if acquired is None:
            client_reader.cancel()
            return
        ticket, worker = acquired
        worker.mark_busy(worker_status, request_type, ticket_id=ticket.ticket_id)
        ws_url = f"ws://{worker.host}:{worker.port}{worker_path}?{identity_qs}"
        try:
            worker_ws = await websockets.connect(
                ws_url,
                open_timeout=5,
                max_size=128 * 1024 * 1024,
                # 内部链路不做 keepalive ping（Worker 侧窗口放宽到 360s，与 worker→backend 一致）：
                # 对端忙于处理时读不到 pong 会被误判超时；存活由每 10 秒的健康检查负责。
                ping_interval=None,
            )
            break
        except Exception as exc:
            logger.warning(
                "Worker %s unreachable (%s), marking offline; attempt=%d ticket=%s",
                worker.worker_id, exc, attempt + 1, ticket.ticket_id,
            )
            pool.release_worker(worker, failed=True)
    if worker_ws is None:
        await gateway_send({
            "type": "error",
            "error": {"code": "session_failed", "message": "Assigned worker is unreachable", "type": "server_error"},
        })
        await ws.close(code=1011, reason="Worker unreachable")
        client_reader.cancel()
        return

    task_start = datetime.now()
    session_closed = asyncio.Event()
    live = LiveSession(worker_id=worker.worker_id, ticket_id=ticket.ticket_id, endpoint=endpoint_name)
    live_sessions.register(live)
    recorder_worker = {"host": worker.host, "port": worker.port, "gpu_id": getattr(worker, "gpu_id", None)}

    try:
        # Include the handoff notification in the worker-release scope: an
        # ordinary client can leave while its connection is being assigned.
        await gateway_send({"type": "session.queue_done", "worker_id": worker.worker_id})

        def ensure_recorder(session_id: Optional[str]) -> None:
            nonlocal recorder
            if (
                not recording_enabled
                or recorder is not None
                or recorder_cls is None
                or recorder_sessions_dir is None
                or not session_id
            ):
                return
            safe_session_id = _sanitize_session_id(session_id)
            recording_identity = dict(identity)
            recording_identity["session_id"] = safe_session_id
            recording_identity["ticket_id"] = ticket.ticket_id
            try:
                recorder = recorder_cls(
                    safe_session_id,
                    recorder_mode,
                    sessions_dir=recorder_sessions_dir,
                    identity=recording_identity,
                    worker=recorder_worker,
                )
            except Exception:
                recorder = None
                return
            buffered = list(pending_record_frames)
            pending_record_frames.clear()
            for direction, frame, recv_ts, send_ts, src in buffered:
                try:
                    recorder.record(direction, frame, recv_ts=recv_ts, send_ts=send_ts, src=src)
                except Exception:
                    pass

        def _as_frame(raw: Any) -> Dict[str, Any]:
            """解析失败的帧也原样留下,不丢。"""
            if isinstance(raw, (bytes, bytearray)):
                import base64 as _b64
                return {"_raw_bytes_base64": _b64.b64encode(bytes(raw)).decode("ascii")}
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {"_raw_json": parsed}
            except Exception:
                return {"_raw_text": raw}

        async def client_to_worker() -> None:
            while True:
                item = await client_frames.get()
                if item is None:
                    return                      # 客户端断开
                recv, raw = item
                if input_mode is not None:
                    raw = _stamp_input_mode(raw, input_mode)
                sent: Optional[float] = None
                try:
                    await worker_ws.send(raw)
                    sent = time.time()
                finally:
                    record_or_buffer("up", _as_frame(raw), recv_ts=recv, send_ts=sent, src="client")

        async def worker_to_client() -> None:
            async for raw in worker_ws:
                recv = time.time()
                frame = _as_frame(raw)
                if frame.get("type") in {"session.created", "session.resumed"}:
                    frame["model_id"] = ws.query_params.get("model")
                    raw = json.dumps(frame, ensure_ascii=False)
                if frame.get("type") == "session.created":
                    ensure_recorder(frame.get("session_id"))
                sent: Optional[float] = None
                try:
                    await ws.send_text(raw if isinstance(raw, str) else raw.decode("utf-8"))
                    sent = time.time()
                finally:
                    record_or_buffer("down", frame, recv_ts=recv, send_ts=sent, src="worker")
                if frame.get("type") == "session.closed":
                    session_closed.set()
                    return

        tasks = [
            asyncio.create_task(client_to_worker()),
            asyncio.create_task(worker_to_client()),
        ]

        if max_duration_s is not None:
            async def session_timeout_watchdog() -> None:
                await asyncio.sleep(max_duration_s)
                if session_closed.is_set():
                    return
                logger.info("API session timeout (%ss): ticket=%s", max_duration_s, ticket.ticket_id)
                await gateway_send({"type": "session.closed", "reason": "timeout"})
                session_closed.set()

            tasks.append(asyncio.create_task(session_timeout_watchdog()))

        async def forced_close_watcher() -> None:
            # 控制面强制关闭该 Worker：告诉客户端会话被服务端终止，然后收尾。
            await live.close_event.wait()
            logger.info("API session force-closed (%s): worker=%s ticket=%s",
                        live.close_reason, worker.worker_id, ticket.ticket_id)
            await gateway_send({"type": "session.closed", "reason": live.close_reason or "worker_removed"})
            session_closed.set()

        tasks.append(asyncio.create_task(forced_close_watcher()))

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            task.result()

    except Exception as exc:
        logger.error("API worker passthrough failed: ticket=%s error=%s", ticket.ticket_id, exc, exc_info=True)
        try:
            await gateway_send({
                "type": "error",
                "error": {"code": "session_failed", "message": str(exc), "type": "server_error"},
            })
        except Exception:
            pass
    finally:
        if recorder is not None:
            try:
                recorder.close(reason="session_end" if session_closed.is_set() else "disconnected")
            except Exception:
                pass
        if worker_ws:
            try:
                await worker_ws.close()
            except Exception:
                pass
        live_sessions.unregister(live)
        client_reader.cancel()
        duration = (datetime.now() - task_start).total_seconds()
        pool.release_worker(worker, request_type=request_type, duration_s=duration)
        try:
            await ws.close()
        except Exception:
            pass

# ============ 默认 Ref Audio 分发 ============

@app.get("/api/frontend_defaults")
async def get_frontend_defaults():
    """返回前端页面需要的默认配置

    前端页面加载时调用此接口获取 playback_delay_ms 等可配置的默认值，
    避免前端硬编码。返回值来自 config.json。
    若 gateway 启动时指定了 --lang，此处也会包含 default_lang 字段。
    """
    from config import get_config
    defaults = get_config().frontend_defaults()
    mode = os.environ.get("O5_FRONTEND_SLIDING_WINDOW_MODE", "basic")
    if mode not in {"off", "basic", "kv_surgery"}:
        raise ValueError("Invalid O5_FRONTEND_SLIDING_WINDOW_MODE")
    defaults.update(sliding_window_mode=mode, max_kv_tokens=262144)
    for field, env in (("basic_window_high_tokens", "O5_BASIC_WINDOW_HIGH_TOKENS"),
                       ("basic_window_low_tokens", "O5_BASIC_WINDOW_LOW_TOKENS")):
        if os.environ.get(env):
            defaults[field] = int(os.environ[env])
    allowed = os.environ.get("O5_FRONTEND_SLIDING_WINDOW_MODES")
    if allowed:
        defaults["sliding_window_modes"] = [value.strip() for value in allowed.split(",")]

    server_lang = GATEWAY_CONFIG.get("default_lang")
    if server_lang:
        defaults["default_lang"] = server_lang
    return defaults


# ============ System Prompt 预设 ============

_presets_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None


def _get_audio_meta(rel_path: str, project_root: str) -> Dict[str, Any]:
    """获取音频文件的元数据（不加载 base64），用于预设列表"""
    import librosa

    if not rel_path:
        return {"name": "", "duration": 0}

    abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(project_root, rel_path)
    name = os.path.basename(abs_path)

    if not os.path.exists(abs_path):
        return {"name": name, "duration": 0}

    try:
        audio, sr = librosa.load(abs_path, sr=16000, mono=True)
        return {"name": name, "duration": round(len(audio) / sr, 1)}
    except Exception:
        return {"name": name, "duration": 0}


def _load_audio_base64(rel_path: str, project_root: str) -> Optional[Dict[str, Any]]:
    """加载音频文件为 base64（按需调用）"""
    import librosa

    if not rel_path:
        return None

    abs_path = rel_path if os.path.isabs(rel_path) else os.path.join(project_root, rel_path)
    if not os.path.exists(abs_path):
        return None

    try:
        audio, sr = librosa.load(abs_path, sr=16000, mono=True)
        audio_bytes = audio.astype(np.float32).tobytes()
        import base64 as b64mod
        return {
            "data": b64mod.b64encode(audio_bytes).decode("ascii"),
            "name": os.path.basename(abs_path),
            "duration": round(len(audio) / sr, 1),
        }
    except Exception as e:
        logger.error(f"Failed to load audio {abs_path}: {e}")
        return None


def _load_presets_from_dir(project_root: str) -> Dict[str, List[Dict[str, Any]]]:
    """扫描 assets/presets/<mode>/*.yaml，返回元数据（不含音频 base64）"""
    import yaml

    presets_root = os.path.join(project_root, "assets", "presets")
    result: Dict[str, List[Dict[str, Any]]] = {}

    if not os.path.isdir(presets_root):
        return result

    for mode_dir in sorted(os.listdir(presets_root)):
        mode_path = os.path.join(presets_root, mode_dir)
        if not os.path.isdir(mode_path):
            continue

        mode_presets = []
        for fname in sorted(os.listdir(mode_path)):
            if not fname.endswith((".yaml", ".yml")):
                continue
            fpath = os.path.join(mode_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    preset = yaml.safe_load(f)
                if not preset or not isinstance(preset, dict):
                    continue

                if "system_content" in preset:
                    resolved = []
                    for item in preset["system_content"]:
                        if item.get("type") == "audio" and item.get("path"):
                            meta = _get_audio_meta(item["path"], project_root)
                            resolved.append({
                                "type": "audio",
                                "data": None,
                                "path": item["path"],
                                "name": meta["name"],
                                "duration": meta["duration"],
                            })
                        else:
                            resolved.append(item)
                    preset["system_content"] = resolved

                if "ref_audio_path" in preset:
                    meta = _get_audio_meta(preset["ref_audio_path"], project_root)
                    preset["ref_audio"] = {
                        "data": None,
                        "path": preset["ref_audio_path"],
                        "name": meta["name"],
                        "duration": meta["duration"],
                    }
                    del preset["ref_audio_path"]

                mode_presets.append(preset)
            except Exception as e:
                logger.error(f"Failed to load preset {fpath}: {e}")

        if mode_presets:
            mode_presets.sort(key=lambda p: p.get("order", 999))
            result[mode_dir] = mode_presets

    total = sum(len(v) for v in result.values())
    logger.info(f"Loaded {total} presets (metadata only) across {len(result)} modes")
    return result


@app.get("/api/presets")
async def get_presets():
    """返回预设元数据（不含音频 base64，音频通过 /api/presets/{mode}/{id}/audio 按需加载）"""
    global _presets_cache

    if _presets_cache is not None:
        return _presets_cache

    project_root = os.path.dirname(__file__)
    _presets_cache = _load_presets_from_dir(project_root)
    return _presets_cache


@app.get("/api/presets/{mode}/{preset_id}/audio")
async def get_preset_audio(mode: str, preset_id: str):
    """按需加载单个 preset 的音频数据"""
    global _presets_cache
    if _presets_cache is None:
        project_root = os.path.dirname(__file__)
        _presets_cache = _load_presets_from_dir(project_root)

    mode_presets = _presets_cache.get(mode, [])
    preset = next((p for p in mode_presets if p.get("id") == preset_id), None)
    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset not found: {mode}/{preset_id}")

    project_root = os.path.dirname(__file__)
    result: Dict[str, Any] = {}

    if "system_content" in preset:
        audio_items = []
        for item in preset["system_content"]:
            if item.get("type") == "audio" and item.get("path"):
                loaded = _load_audio_base64(item["path"], project_root)
                audio_items.append(loaded or {"data": None, "name": item.get("name", ""), "duration": 0})
        result["system_content_audio"] = audio_items

    if preset.get("ref_audio") and preset["ref_audio"].get("path"):
        loaded = _load_audio_base64(preset["ref_audio"]["path"], project_root)
        result["ref_audio"] = loaded or {"data": None, "name": preset["ref_audio"].get("name", ""), "duration": 0}

    return result


# 缓存：启动后首次请求时加载，之后直接返回
_default_ref_audio_cache: Optional[Dict[str, Any]] = None


@app.get("/api/default_ref_audio")
async def get_default_ref_audio():
    """返回默认参考音频（PCM float32 16kHz mono base64）

    前端页面加载时调用此接口获取默认 ref audio，
    之后所有请求统一通过 ref_audio_base64 传递音频数据。
    """
    global _default_ref_audio_cache

    if _default_ref_audio_cache is not None:
        return _default_ref_audio_cache

    from config import get_config
    cfg = get_config()

    if not cfg.ref_audio_path:
        raise HTTPException(status_code=404, detail="No default ref audio configured")

    # 解析路径（支持相对路径，相对于 minicpmo45_service/）
    ref_path = cfg.ref_audio_path
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(os.path.dirname(__file__), ref_path)

    if not os.path.exists(ref_path):
        raise HTTPException(status_code=404, detail=f"Default ref audio not found: {cfg.ref_audio_path}")

    try:
        import base64
        import librosa
        import numpy as np

        # 加载并重采样为 16kHz mono float32（与前端上传格式一致）
        audio, sr = librosa.load(ref_path, sr=16000, mono=True)
        duration = len(audio) / 16000

        # 转换为 base64（PCM float32）
        audio_bytes = audio.astype(np.float32).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        _default_ref_audio_cache = {
            "name": os.path.basename(cfg.ref_audio_path),
            "duration": round(duration, 1),
            "sample_rate": 16000,
            "samples": len(audio),
            "base64": audio_b64,
        }
        logger.info(
            f"Default ref audio loaded: {_default_ref_audio_cache['name']} "
            f"({duration:.1f}s, {len(audio)} samples)"
        )
        return _default_ref_audio_cache

    except Exception as e:
        logger.error(f"Failed to load default ref audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load ref audio: {e}")


# ============ 素材管理 API ============

@app.get("/api/assets/ref_audio", response_model=RefAudioListResponse)
async def list_ref_audios():
    """列出参考音频"""
    if ref_audio_registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return RefAudioListResponse(
        total=ref_audio_registry.count,
        ref_audios=ref_audio_registry.list_all(),
    )


@app.post("/api/assets/ref_audio", response_model=RefAudioResponse)
async def upload_ref_audio(request: UploadRefAudioRequest):
    """上传参考音频"""
    if ref_audio_registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        info = ref_audio_registry.upload(
            name=request.name,
            audio_base64=request.audio_base64,
        )
        return RefAudioResponse(
            success=True,
            id=info.id,
            name=info.name,
            message=f"Uploaded successfully, duration={info.duration_ms}ms",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload ref audio failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/assets/ref_audio/{ref_id}", response_model=RefAudioResponse)
async def delete_ref_audio(ref_id: str):
    """删除参考音频"""
    if ref_audio_registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if not ref_audio_registry.exists(ref_id):
        raise HTTPException(status_code=404, detail=f"Ref audio not found: {ref_id}")

    success = ref_audio_registry.delete(ref_id)
    return RefAudioResponse(
        success=success,
        id=ref_id,
        message="Deleted" if success else "Failed to delete",
    )


# ============ 队列状态 API ============

@app.get("/api/queue", response_model=QueueStatus)
async def get_queue():
    """获取当前队列状态"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return worker_pool.get_queue_status()


@app.get("/api/queue/{ticket_id}")
async def get_queue_ticket(ticket_id: str):
    """获取指定排队项的状态（前端轮询用）"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    ticket = worker_pool.get_ticket(ticket_id)
    if ticket is None:
        return {"found": False, "message": "Ticket not in queue (may have been assigned or cancelled)"}
    return {"found": True, "ticket": ticket.model_dump()}


@app.delete("/api/queue/{ticket_id}")
async def cancel_queue_item(ticket_id: str):
    """取消排队项（Admin 用）"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    ok = worker_pool.cancel(ticket_id)
    return {"success": ok}


# ============ ETA 配置 API ============

@app.get("/api/config/eta", response_model=EtaStatus)
async def get_eta_config():
    """获取 ETA 配置和 EMA 状态"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return worker_pool.eta_tracker.get_status()


@app.put("/api/config/eta", response_model=EtaStatus)
async def update_eta_config(new_config: EtaConfig):
    """更新 ETA 基准配置（运行时生效，无需重启）"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    worker_pool.eta_tracker.update_config(new_config)
    alpha_str = f", ema_alpha={new_config.ema_alpha}" if new_config.ema_alpha is not None else ""
    logger.info(
        f"ETA config updated: chat={new_config.eta_chat_s}s, "
        f"half_duplex={new_config.eta_half_duplex_s}s, "
        f"audio_duplex={new_config.eta_audio_duplex_s}s, "
        f"omni_duplex={new_config.eta_omni_duplex_s}s{alpha_str}"
    )
    return worker_pool.eta_tracker.get_status()


# ============ 缓存状态 API ============

@app.get("/cache")
async def list_cache():
    """查看各 Worker 的 KV cache 状态"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {
        "workers": [
            {
                "worker_id": w.worker_id,
                "status": w.status.value,
            }
            for w in worker_pool.workers.values()
        ],
    }


# ============ Session API ============

_BASE_DIR = os.path.dirname(__file__)


def _list_active_sessions_payload() -> Dict[str, Any]:
    """Gateway 当前占用 Worker 的请求列表（Admin / 测试共用）。"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    sessions: List[Dict[str, Any]] = []
    for w in worker_pool.workers.values():
        if not w.current_ticket_id:
            continue
        last_active = w.last_heartbeat or w.task_started_at or datetime.now()
        sessions.append(
            {
                "ticket_id": w.current_ticket_id,
                "worker_id": w.worker_id,
                "messages_hash": getattr(w, "cached_hash", "") or "",
                "last_active": last_active.isoformat()
                if hasattr(last_active, "isoformat")
                else str(last_active),
            }
        )
    return {"total": len(sessions), "sessions": sessions}


@app.get("/sessions")
async def list_active_sessions():
    """列出活跃会话（与 admin.html `GET /sessions` 一致）。"""
    return _list_active_sessions_payload()


@app.get("/api/sessions")
async def list_active_sessions_api():
    """同 list_active_sessions，REST 风格路径便于与 `/api/sessions/{id}` 并列。"""
    return _list_active_sessions_payload()


def _session_dir(session_id: str) -> str:
    """获取 session 目录的绝对路径（含路径安全校验）"""
    safe_id = _sanitize_session_id(session_id)
    sessions_root = _sessions_root()
    base = os.path.join(sessions_root, safe_id)
    resolved = os.path.realpath(base)
    if os.path.commonpath([sessions_root, resolved]) != sessions_root:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    return resolved


@app.get("/api/sessions/{session_id}")
async def get_session_meta(session_id: str):
    """获取 session 元数据 (meta.json)"""
    sdir = _session_dir(session_id)
    meta_path = os.path.join(sdir, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/sessions/{session_id}/recording")
async def get_session_recording(session_id: str):
    """获取录制事件流 (stream.jsonl + meta.json)。

    返回忠实事件流:{meta, events:[{seq, ts, dir, frame}, ...]}。
    frame 为协议帧原样;音视频二进制以 "@blob/NNN.ext" 指针引用,前端去掉 '@'
    后经 /assets/blob/NNN.ext 取实际文件。
    """
    sdir = _session_dir(session_id)
    stream_path = os.path.join(sdir, "stream.jsonl")
    meta_path = os.path.join(sdir, "meta.json")
    if not os.path.exists(stream_path):
        raise HTTPException(status_code=404, detail=f"Recording not found for session: {session_id}")

    events = []
    with open(stream_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    return {"session_id": session_id, "meta": meta, "events": events}


_MIME_MAP = {
    ".wav": "audio/wav",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webm": "video/webm",
    ".mp4": "video/mp4",
    ".mp3": "audio/mpeg",
}


@app.api_route("/api/sessions/{session_id}/assets/{asset_path:path}", methods=["GET", "HEAD"])
async def get_session_asset(session_id: str, asset_path: str):
    """获取 session 资源文件（音频 WAV / 图片）

    音频已在录制时存为 16-bit PCM WAV，直接 serve（支持 Range 请求）。
    """
    sdir = _session_dir(session_id)

    if ".." in asset_path or asset_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid asset path")

    full_path = os.path.realpath(os.path.join(sdir, asset_path))
    if not full_path.startswith(os.path.realpath(sdir)):
        raise HTTPException(status_code=400, detail="Path traversal detected")

    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")

    ext = os.path.splitext(full_path)[1].lower()
    mime = _MIME_MAP.get(ext)
    if mime:
        return FileResponse(full_path, media_type=mime)
    return FileResponse(full_path)


@app.get("/api/sessions/{session_id}/download")
async def download_session(session_id: str):
    """打包下载整个 session (zip)"""
    sdir = _session_dir(session_id)
    if not os.path.isdir(sdir):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    def _build_zip() -> BytesIO:
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(sdir):
                for fname in files:
                    abs_path = os.path.join(root, fname)
                    arc_name = os.path.join(session_id, os.path.relpath(abs_path, sdir))
                    zf.write(abs_path, arc_name)
        buf.seek(0)
        return buf

    zip_buf = await asyncio.to_thread(_build_zip)
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={session_id}.zip"},
    )


_UPLOAD_MAX_MB = 500
_UPLOAD_MAX_BYTES = _UPLOAD_MAX_MB * 1024 * 1024
_ALLOWED_UPLOAD_TYPES = {"video/webm", "video/mp4", "audio/wav", "audio/webm"}
_EXT_FROM_MIME = {"video/webm": ".webm", "video/mp4": ".mp4", "audio/wav": ".wav", "audio/webm": ".webm"}


@app.post("/api/sessions/{session_id}/upload-recording")
async def upload_session_recording(session_id: str, file: UploadFile = File(...)):
    """上传前端录制文件（视频/音频）到 session 目录

    存储为 data/sessions/{session_id}/frontend_replay.{ext}
    """
    sdir = _session_dir(session_id)
    os.makedirs(sdir, exist_ok=True)

    content_type = (file.content_type or "").split(";")[0].strip()
    if content_type not in _ALLOWED_UPLOAD_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    ext = _EXT_FROM_MIME.get(content_type, ".bin")
    dest = os.path.join(sdir, f"frontend_replay{ext}")

    total = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _UPLOAD_MAX_BYTES:
                f.close()
                os.remove(dest)
                raise HTTPException(status_code=413, detail=f"File too large (max {_UPLOAD_MAX_MB}MB)")
            f.write(chunk)

    logger.info(f"[Session] uploaded frontend recording: {dest} ({total / 1024 / 1024:.1f} MB)")
    return {"status": "ok", "path": f"frontend_replay{ext}", "size_bytes": total}


_COMMENT_MAX_CHARS = 2000


@app.post("/api/sessions/{session_id}/comment")
async def save_session_comment(session_id: str, payload: Dict[str, Any] = Body(...)):
    """保存用户对该 session 的评语，写入 data/sessions/{session_id}/comment.txt"""
    sdir = _session_dir(session_id)
    os.makedirs(sdir, exist_ok=True)

    raw = payload.get("comment")
    if raw is None:
        raw = ""
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="comment must be a string")
    comment = raw.strip()
    if len(comment) > _COMMENT_MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Comment too long (max {_COMMENT_MAX_CHARS} chars)",
        )

    dest = os.path.join(sdir, "comment.txt")
    if comment:
        with open(dest, "w", encoding="utf-8") as f:
            f.write(comment)
    else:
        # Empty comment ⇒ remove any prior comment file so it doesn't linger.
        try:
            os.remove(dest)
        except FileNotFoundError:
            pass

    logger.info(f"[Session] comment saved: {session_id} ({len(comment)} chars)")
    return {"status": "ok", "len": len(comment)}


@app.get("/api/sessions/{session_id}/comment")
async def get_session_comment(session_id: str):
    """读取 session 的评语；不存在则返回空字符串。"""
    sdir = _session_dir(session_id)
    if not os.path.isdir(sdir):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    dest = os.path.join(sdir, "comment.txt")
    if not os.path.exists(dest):
        return {"comment": ""}
    try:
        with open(dest, "r", encoding="utf-8") as f:
            return {"comment": f.read()}
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/s/{session_id}", response_class=HTMLResponse)
async def session_viewer(session_id: str):
    """Session 回看页面"""
    sdir = _session_dir(session_id)
    meta_path = os.path.join(sdir, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    viewer_path = os.path.join(os.path.dirname(__file__), "static", "session-viewer.html")
    if os.path.exists(viewer_path):
        return FileResponse(viewer_path)
    return HTMLResponse(f"<h1>Session {session_id}</h1><p>Viewer page not found</p>")


# ============ APP 管理 ============

@app.get("/api/apps", response_model=AppsPublicResponse)
async def get_enabled_apps():
    """返回当前启用的 APP 列表（前端导航栏和主页卡片使用）"""
    return AppsPublicResponse(apps=app_registry.get_enabled_apps())


@app.get("/api/admin/apps", response_model=AppsAdminResponse)
async def get_all_apps():
    """返回所有 APP 列表（含 enabled 状态，Admin 页面使用）"""
    return AppsAdminResponse(apps=app_registry.get_all_apps())


@app.put("/api/admin/apps/{app_id}")
async def toggle_app(app_id: str, req: AppToggleRequest):
    """切换 APP 启用/禁用状态（Admin 操作，运行时生效）"""
    result = app_registry.set_enabled(app_id, req.enabled)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown app: {app_id}")
    action = "enabled" if req.enabled else "disabled"
    logger.info(f"[AppRegistry] App '{app_id}' {action}")
    return {"success": True, "app": result}


# ============ 控制面板与公开文档 ============
# /control（需密码）、/overview（公开只读状态页）与 /api-docs（公开）。控制面只在 --control-plane-root 启动时启用，否则相关接口 503。
app.include_router(control_router)
app.include_router(control_docs_router)
app.include_router(control_public_router)

# ============ 静态文件 ============

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Demo 页面不在 API Gateway 里：Demo 是独立服务（demo_server.py），只通过协议访问 API。

@app.get("/", response_class=HTMLResponse)
async def index():
    """首页：模式选择（Turn-based / Omni Duplex / Audio Duplex）"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(
        "<h1>modeling.o5 Service</h1>"
        "<p>API docs: <a href='/docs'>/docs</a></p>"
    )


@app.get("/turnbased", response_class=HTMLResponse)
async def turnbased():
    """Turn-based Chat Demo 页面"""
    if not app_registry.is_enabled("turnbased"):
        return RedirectResponse(url="/", status_code=302)
    page_path = os.path.join(static_dir, "turnbased.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    return HTMLResponse("<h1>Turn-based Chat</h1><p>Page not found</p>")


@app.get("/omni", response_class=HTMLResponse)
async def omni():
    """Omni Duplex Demo 页面"""
    if not app_registry.is_enabled("omni"):
        return RedirectResponse(url="/", status_code=302)
    omni_path = os.path.join(static_dir, "omni", "omni.html")
    if os.path.exists(omni_path):
        return FileResponse(omni_path)
    return HTMLResponse("<h1>Omni</h1><p>Omni page not found</p>")


@app.get("/half_duplex", response_class=HTMLResponse)
async def half_duplex():
    """Half-Duplex Audio Demo 页面"""
    if not app_registry.is_enabled("half_duplex_audio"):
        return RedirectResponse(url="/", status_code=302)
    page_path = os.path.join(static_dir, "half-duplex", "half_duplex.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    return HTMLResponse("<h1>Half-Duplex Audio</h1><p>Page not found</p>")


@app.get("/audio_duplex", response_class=HTMLResponse)
async def audio_duplex():
    """语音双工 Demo 页面（简化版 Omni，无视频）"""
    if not app_registry.is_enabled("audio_duplex"):
        return RedirectResponse(url="/", status_code=302)
    page_path = os.path.join(static_dir, "audio-duplex", "audio_duplex.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    return HTMLResponse("<h1>Audio Duplex</h1><p>Page not found</p>")


@app.get("/realtime", response_class=HTMLResponse)
async def realtime_page():
    """Realtime API Demo 页面（OpenAI Realtime Protocol 风格双工）"""
    page_path = os.path.join(static_dir, "realtime", "realtime.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    return HTMLResponse("<h1>Realtime API</h1><p>Page not found</p>")


@app.get("/fc-demo", response_class=HTMLResponse)
async def fc_demo_page():
    """TauVoice FC audio duplex demo."""
    page_path = os.path.join(static_dir, "fc-demo", "fc-demo.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    return HTMLResponse("<h1>FC Demo</h1><p>Page not found</p>")


docs_static_dir = os.path.join(static_dir, "docs")


@app.api_route("/docs", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def docs_index():
    """Docs entry: Fumadocs static site."""
    return RedirectResponse(url="/docs/zh/", status_code=302)


@app.api_route("/docs/overview", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def docs_legacy_overview():
    """Compatibility redirect for the previous runtime docs route."""
    return RedirectResponse(url="/docs/zh/realtime-api/overview/", status_code=302)


@app.api_route("/docs/video", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def docs_legacy_video():
    """Compatibility redirect for the previous runtime docs route."""
    return RedirectResponse(url="/docs/zh/realtime-api/video/", status_code=302)


@app.api_route("/docs/audio", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def docs_legacy_audio():
    """Compatibility redirect for the previous runtime docs route."""
    return RedirectResponse(url="/docs/zh/realtime-api/audio/", status_code=302)


@app.api_route("/docs/en", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def docs_en_index():
    """English docs entry."""
    return RedirectResponse(url="/docs/en/", status_code=302)


@app.api_route("/docs/en/overview", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def docs_en_legacy_overview():
    """Compatibility redirect for the previous English runtime docs route."""
    return RedirectResponse(url="/docs/en/realtime-api/overview/", status_code=302)


@app.api_route("/docs/en/video", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def docs_en_legacy_video():
    """Compatibility redirect for the previous English runtime docs route."""
    return RedirectResponse(url="/docs/en/realtime-api/video/", status_code=302)


@app.api_route("/docs/en/audio", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def docs_en_legacy_audio():
    """Compatibility redirect for the previous English runtime docs route."""
    return RedirectResponse(url="/docs/en/realtime-api/audio/", status_code=302)


app.mount("/docs", StaticFiles(directory=docs_static_dir, html=True, check_dir=False), name="docs")


# ============ Realtime API (OpenAI Realtime Protocol) ============

@app.get("/v1/models")
async def list_models():
    """Public catalog: model IDs, loaded identity/capabilities and recommended defaults."""
    try:
        data = [entry for _, _, entry in catalog_entries(endpoint_registry)]
    except ModelCatalogError as exc:
        return JSONResponse({"error": {"code": "model_catalog_unavailable", "message": str(exc)}},
                            status_code=503, headers=_PUBLIC_CORS_HEADERS)
    return JSONResponse({"object": "list", "data": data}, headers=_PUBLIC_CORS_HEADERS)


@app.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket):
    """Resolve one public model ID to its pool; instances stay an internal concern."""
    model_id = ws.query_params.get("model")
    mode = ws.query_params.get("mode")
    async def reject(code: str, message: str, close_code: int):
        await ws.accept()
        await ws.send_json({"type": "error", "error": {"code": code, "message": message}})
        await ws.close(code=close_code)
    if not model_id:
        await reject("model_required", "The model query parameter is required", 1008)
        return
    try:
        selected = next((item for item in catalog_entries(endpoint_registry) if item[2]["id"] == model_id), None)
    except ModelCatalogError as exc:
        await reject("model_catalog_unavailable", str(exc), 1013)
        return
    if selected is None:
        await reject("model_not_found", f"Unknown model: {model_id}", 4404)
        return
    pool_name, pool, model = selected
    if mode not in model["capabilities"]["modes"]:
        await reject("unsupported_mode", f"Unsupported mode for this model: {mode}", 1008)
        return
    await _realtime_entry(ws, pool=pool, endpoint_name=pool_name, allow_chat=True,
                          input_mode=mode if mode in API_INPUT_MODES else None)


async def _realtime_entry(
    ws: WebSocket, *, pool: Optional[WorkerPool], endpoint_name: str, allow_chat: bool,
    input_mode: Optional[str] = None,
) -> None:
    """One selected model pool supports chat, ordinary audio/video and FC audio/omni."""

    mode = input_mode or ws.query_params.get("mode", "video")

    if mode == "chat" and allow_chat:
        if not app_registry.is_enabled("turnbased"):
            await ws.close(code=1008, reason="Turn-based Chat is currently disabled")
            return
        await _api_worker_passthrough_ws(
            ws,
            request_type="chat",
            worker_status=GatewayWorkerStatus.BUSY_CHAT,
            worker_path="/v1/worker/chat",
            source_channel="realtime_api",
            source_mode=mode,
            pool=pool,
            endpoint_name=endpoint_name,
        )
        return

    if mode not in {"video", "audio", "omni"}:
        await ws.close(code=1008, reason=f"Unsupported realtime mode: {mode}")
        return

    # The ordinary desktop pages opt in to long conversations; preserve other
    # API clients' existing watchdog policy unless explicitly overridden.
    max_duration_s: Optional[float] = (
        None if ws.query_params.get("ordinary_reconnect") == "1"
        else (1200 if mode in ("video", "omni") else 2400)
    )
    # 离线评测按 Unit 锁步驱动，墙钟时长与对话时长无关；副本池部署用 0 关闭该看门狗。
    override = os.environ.get("FC_API_SESSION_MAX_DURATION_S")
    if override is not None:
        max_duration_s = float(override) or None
    request_type = "omni_duplex" if mode in ("video", "omni") else "audio_duplex"

    if ws.query_params.get("ordinary_reconnect") == "1":
        from gateway_modules.ordinary_reconnect import serve
        async def run_retained(retained):
            await _api_worker_passthrough_ws(
                retained,
                request_type=request_type,
                worker_status=GatewayWorkerStatus.DUPLEX_ACTIVE,
                worker_path="/v1/worker/duplex",
                source_channel="realtime_api",
                source_mode=mode,
                max_duration_s=max_duration_s,
                pool=pool,
                endpoint_name=endpoint_name,
                input_mode=input_mode,
            )
        await serve(ws, identity=(endpoint_name, mode), run=run_retained)
        return

    await _api_worker_passthrough_ws(
        ws,
        request_type=request_type,
        worker_status=GatewayWorkerStatus.DUPLEX_ACTIVE,
        worker_path="/v1/worker/duplex",
        source_channel="realtime_api",
        source_mode=mode,
        max_duration_s=max_duration_s,
        pool=pool,
        endpoint_name=endpoint_name,
        input_mode=input_mode,
    )


_PUBLIC_CORS_HEADERS = {"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"}


def _stamp_input_mode(raw: Any, input_mode: str) -> Any:
    """给 ``session.init`` 写上 URL 决定的 ``payload.input_mode``;其它帧原样返回。

    客户端没写:补上。写了且一致:原样。写了但不一致:写成 ``input_mode_conflict``,
    由 Worker 走统一的 ``session.closed`` 报错路径返回给客户端,网关不另起一套关闭逻辑。
    """

    if not isinstance(raw, str) or '"session.init"' not in raw:
        return raw
    try:
        frame = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(frame, dict) or frame.get("type") != "session.init":
        return raw
    payload = frame.get("payload")
    if not isinstance(payload, dict):
        return raw
    declared = payload.get("input_mode")
    if declared is not None and declared != input_mode:
        payload = dict(payload, input_mode_conflict={"url": input_mode, "payload": declared})
    else:
        payload = dict(payload, input_mode=input_mode)
    return json.dumps(dict(frame, payload=payload), ensure_ascii=False)


def build_endpoint_info(endpoint: str, pool: WorkerPool) -> Dict[str, Any]:
    """公开的端点描述。不含 Worker 的内网地址。"""

    workers = []
    models: Dict[str, Dict[str, Any]] = {}
    for w in pool.workers.values():
        key = json.dumps(w.model_info, sort_keys=True, ensure_ascii=False)
        models.setdefault(key, {"model": w.model_info, "workers": 0, "available": 0})
        models[key]["workers"] += 1
        if w.is_idle:
            models[key]["available"] += 1
        workers.append({
            "worker_id": w.worker_id,
            "status": w.status.value,
            "draining": w.draining,
            "profile_id": w.model_info.get("profile_id"),
        })
    return {
        "endpoint": endpoint,
        "protocol_version": API_PROTOCOL_VERSION,
        "model_id": pool_model_id(pool),
        "realtime_path": "/v1/realtime",
        "realtime_paths": {m: "/v1/realtime?" + urlencode({"model": pool_model_id(pool) or "", "mode": m}) for m in ("chat", "audio", "video", "omni")},
        "workers": workers,
        "available_workers": pool.idle_count,
        "queue_length": pool.queue_length,
        "models": list(models.values()),
    }


@app.get("/mobile-omni", include_in_schema=False)
async def mobile_omni_redirect():
    """移动版 Omni 入口重定向到带尾斜杠版本"""
    return RedirectResponse(url="/mobile-omni/", status_code=302)


@app.get("/mobile-omni/", response_class=HTMLResponse)
async def mobile_omni():
    """移动版全双工页：复用桌面 omni-app.js，外壳为 static/mobile-omni/"""
    if not app_registry.is_enabled("omni"):
        return RedirectResponse(url="/", status_code=302)
    page_path = os.path.join(static_dir, "mobile-omni", "index.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    return HTMLResponse("<h1>Mobile Omni</h1><p>Page not found</p>", status_code=404)


@app.get("/mobile", include_in_schema=False)
async def mobile_redirect():
    """移动端入口重定向到带尾斜杠版本，确保相对资源路径正确解析"""
    return RedirectResponse(url="/mobile/", status_code=302)


@app.get("/mobile/", response_class=HTMLResponse)
async def mobile():
    """移动端 React 预览页面"""
    page_path = os.path.join(static_dir, "mobile", "index.html")
    if os.path.exists(page_path):
        return FileResponse(page_path)
    return HTMLResponse(
        "<h1>Mobile Preview</h1>"
        "<p>Build output not found. Run "
        "<code>cd frontend/mobile && bun run --bun build:static</code> "
        "or <code>npm run build:static</code>.</p>",
        status_code=404,
    )


@app.api_route("/mobile/{asset_path:path}", methods=["GET", "HEAD"])
async def mobile_asset(asset_path: str):
    """移动端构建产物静态资源"""
    mobile_root = os.path.realpath(os.path.join(static_dir, "mobile"))
    full_path = os.path.realpath(os.path.join(mobile_root, asset_path))

    if not full_path.startswith(mobile_root + os.sep):
        raise HTTPException(status_code=400, detail="Path traversal detected")
    if not os.path.exists(full_path) or os.path.isdir(full_path):
        raise HTTPException(status_code=404, detail=f"Mobile asset not found: {asset_path}")

    return FileResponse(full_path)


@app.get("/admin", response_class=HTMLResponse)
async def admin():
    """Admin Dashboard"""
    admin_path = os.path.join(static_dir, "admin.html")
    if os.path.exists(admin_path):
        return FileResponse(admin_path)
    return HTMLResponse("<h1>Admin</h1><p>Admin page not found</p>")


# ============ 入口 ============

def main():
    from config import get_config
    ws_debug = _enable_ws_debug_logging()
    cfg = get_config()

    parser = argparse.ArgumentParser(description="modeling.o5 Gateway")
    parser.add_argument("--port", type=int, default=None, help=f"Gateway port (default: {cfg.gateway_port})")
    parser.add_argument("--internal-port", type=int, default=8007, help="Internal worker registration port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--control-plane-root", type=str, default=None,
                        help="启用控制面：状态与 Worker 运行目录的根（须在共享存储上，Worker job 要读写它）")
    parser.add_argument("--internal-host", type=str, default=None,
                        help="内部注册接口绑定地址（默认同 --host；对外部署设 127.0.0.1，注册接口没有鉴权）")
    parser.add_argument("--max-queue-size", type=int, default=None, help="Max queue size")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout (s)")
    # 协议选择：默认 HTTPS，--http 可降级为 HTTP
    proto_group = parser.add_mutually_exclusive_group()
    proto_group.add_argument("--https", action="store_true", default=True,
                             help="启用 HTTPS（默认，自签名证书）")
    proto_group.add_argument("--http", action="store_true",
                             help="降级为 HTTP（不推荐，麦克风等浏览器 API 需要 HTTPS）")
    parser.add_argument("--ssl-certfile", type=str, default="certs/cert.pem", help="SSL cert file path")
    parser.add_argument("--ssl-keyfile", type=str, default="certs/key.pem", help="SSL key file path")
    parser.add_argument("--lang", type=str, default=None, choices=["zh", "en"],
                         help="Default UI language (zh/en). Overrides config.json. "
                              "Clients can still switch; this sets the server-side default.")
    args = parser.parse_args()

    # --http 被指定时，关闭 HTTPS
    use_https = not args.http

    port = args.port or cfg.gateway_port

    if args.lang:
        GATEWAY_CONFIG["default_lang"] = args.lang

    GATEWAY_CONFIG.update({
        "control_plane_root": args.control_plane_root,
        "workers": [],
        "max_queue_size": args.max_queue_size or cfg.max_queue_size,
        "timeout": args.timeout or cfg.request_timeout,
        "eta_config": {
            "eta_chat_s": cfg.eta_chat_s,
            "eta_half_duplex_s": cfg.eta_half_duplex_s,
            "eta_audio_duplex_s": cfg.eta_audio_duplex_s,
            "eta_omni_duplex_s": cfg.eta_omni_duplex_s,
        },
        "eta_ema_alpha": cfg.eta_ema_alpha,
        "eta_ema_min_samples": cfg.eta_ema_min_samples,
        "placement": cfg.placement,
    })

    proto_label = "HTTPS" if use_https else "HTTP"
    logger.info(f"Starting Gateway on port {port} ({proto_label})")
    logger.info(f"Starting internal registry on port {args.internal_port} (HTTP)")
    logger.info("Workers: dynamic registration only")

    ssl_kwargs = {}
    if use_https:
        cert = args.ssl_certfile
        key = args.ssl_keyfile
        if not os.path.exists(cert) or not os.path.exists(key):
            logger.error(f"SSL cert/key not found: {cert}, {key}")
            logger.error("Generate with: openssl req -x509 -newkey rsa:2048 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes -subj '/CN=dev'")
            logger.error("Or use --http to start without HTTPS")
            return
        ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
        logger.info(f"HTTPS enabled: cert={cert}, key={key}")
    else:
        logger.warning("Running in HTTP mode (no TLS). Browser microphone/camera APIs may not work.")

    # Bump WS max payload from uvicorn's 16 MiB default to 128 MiB so that
    # base64-encoded video attachments coming in from the browser can be
    # proxied to a worker without being rejected with code 1009.
    async def serve() -> None:
        public_config = uvicorn.Config(
            app,
            host=args.host,
            port=port,
            ws="websockets" if ws_debug else "auto",
            log_level="debug" if ws_debug else "info",
            log_config=_uvicorn_log_config(debug=ws_debug),
            ws_max_size=128 * 1024 * 1024,
            # 对外客户端：仍用 ping 发现死连接，但窗口放宽 —— 本端忙于转发时读 pong 会滞后，
            # 默认的 20s/20s 会把健康的会话误判为超时。
            ws_ping_interval=30.0,
            ws_ping_timeout=120.0,
            **ssl_kwargs,
        )
        internal_config = uvicorn.Config(
            internal_app,
            host=args.internal_host or args.host,
            port=args.internal_port,
            log_level="info",
            log_config=_uvicorn_log_config(debug=ws_debug),
        )
        public_server = uvicorn.Server(public_config)
        internal_server = uvicorn.Server(internal_config)
        await asyncio.gather(public_server.serve(), internal_server.serve())

    asyncio.run(serve())


if __name__ == "__main__":
    main()
