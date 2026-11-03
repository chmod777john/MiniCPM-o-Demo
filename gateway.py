"""modeling.o5 Realtime API Gateway.

The Gateway owns only model discovery, Worker registration, FIFO scheduling,
health/lifecycle and transparent WebSocket relay. It does not load a model or
serve application pages. Chat, ordinary duplex and FC duplex all enter the same
Worker transport at ``/v1/worker/realtime``.

启动方式：
    cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
    PYTHONPATH=. .venv/base/bin/python gateway.py \\
        --port 10024 --internal-port 10025
"""

import os
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

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse

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
    """Collect client/page identity metadata forwarded to the selected Worker."""
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

# 配置（通过 main() 传入）
GATEWAY_CONFIG: Dict[str, Any] = {}


# ============ 应用初始化 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    global worker_pool, endpoint_registry

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

    logger.info("Gateway started, %d workers", len(worker_pool.workers))

    yield

    await endpoint_registry.stop()
    logger.info("Gateway stopped")


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


# Public deployments expose only health, model discovery and named realtime
# endpoints. Application, recording and administrative APIs are served elsewhere.
_API_ONLY = os.getenv("GATEWAY_API_ONLY", "").lower() in {"1", "true", "yes", "on"}
_API_ONLY_PREFIXES = ("/e/",)
_API_ONLY_EXACT = {"/health", "/v1/models"}
_WORKER_REALTIME_PATH = "/v1/worker/realtime"


def api_only_allows(path: str) -> bool:
    return path in _API_ONLY_EXACT or path.startswith(_API_ONLY_PREFIXES)


@app.middleware("http")
async def restrict_api_only(request: Request, call_next):
    if _API_ONLY and not api_only_allows(request.url.path):
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
    source_channel: str,
    source_mode: Optional[str],
    max_duration_s: Optional[float] = None,
    pool: Optional[WorkerPool] = None,
    endpoint_name: str = DEFAULT_ENDPOINT,
) -> None:
    """Queue one session, then transparently relay raw client/worker messages.

    Gateway owns scheduling and lifecycle only. Chat, ordinary duplex and FC
    duplex all use the same internal realtime WebSocket and event schema.
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

    async def gateway_send(frame: Dict[str, Any]) -> None:
        """Send a Gateway-owned scheduling or lifecycle event."""
        await ws.send_json(frame)

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
    forwarded_query = list(ws.query_params.multi_items())
    present = {key for key, _ in forwarded_query}
    forwarded_query.extend((key, value) for key, value in identity.items()
                           if value and key not in present)
    identity_qs = urlencode(forwarded_query)

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
        ws_url = f"ws://{worker.host}:{worker.port}{_WORKER_REALTIME_PATH}?{identity_qs}"
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
    try:
        # Include the handoff notification in the worker-release scope: an
        # ordinary client can leave while its connection is being assigned.
        await gateway_send({"type": "session.queue_done", "worker_id": worker.worker_id})

        async def client_to_worker() -> None:
            while True:
                item = await client_frames.get()
                if item is None:
                    return                      # 客户端断开
                _recv_ts, raw = item
                await worker_ws.send(raw)

        async def worker_to_client() -> None:
            async for raw in worker_ws:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    frame = {}
                await ws.send_text(raw if isinstance(raw, str) else raw.decode("utf-8"))
                if isinstance(frame, dict) and frame.get("type") == "session.closed":
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


@app.get("/", include_in_schema=False)
async def index():
    """Gateway is API-only; application pages are served by the demo service."""
    return {
        "service": "o5-realtime-gateway",
        "models": "/v1/models",
        "realtime": "/v1/realtime",
    }


# ============ Realtime API (OpenAI Realtime Protocol) ============

_PUBLIC_CORS_HEADERS = {"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"}

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
    await _realtime_entry(ws, pool=pool, endpoint_name=pool_name, allow_chat=True)


@app.websocket("/e/{endpoint}/v1/realtime")
async def endpoint_realtime_ws(ws: WebSocket, endpoint: str):
    """Route a named API endpoint to its own Worker pool."""
    pool = endpoint_registry.get(endpoint) if endpoint_registry is not None else None
    if pool is None:
        await ws.close(code=4404, reason=f"Unknown endpoint: {endpoint}")
        return
    mode = ws.query_params.get("mode")
    if mode not in API_INPUT_MODES:
        await ws.close(code=1008, reason=f"mode must be one of {API_INPUT_MODES}, got {mode!r}")
        return
    await _realtime_entry(ws, pool=pool, endpoint_name=endpoint, allow_chat=False)


async def _realtime_entry(
    ws: WebSocket, *, pool: Optional[WorkerPool], endpoint_name: str, allow_chat: bool,
) -> None:
    """Schedule one session and relay it through the common Worker transport."""
    mode = ws.query_params.get("mode")

    if mode == "chat":
        if not allow_chat:
            await ws.close(code=1008, reason="chat is not exposed by named endpoints")
            return
        await _api_worker_passthrough_ws(
            ws,
            request_type="chat",
            worker_status=GatewayWorkerStatus.BUSY_CHAT,
            source_channel="realtime_api",
            source_mode=mode,
            pool=pool,
            endpoint_name=endpoint_name,
        )
        return

    if mode not in {"video", "audio", "omni"}:
        await ws.close(code=1008, reason=f"Unsupported realtime mode: {mode}")
        return

    max_duration_s: Optional[float] = (
        None if ws.query_params.get("ordinary_reconnect") == "1"
        else (1200 if mode in ("video", "omni") else 2400)
    )
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
                source_channel="realtime_api",
                source_mode=mode,
                max_duration_s=max_duration_s,
                pool=pool,
                endpoint_name=endpoint_name,
            )

        await serve(ws, identity=(endpoint_name, mode), run=run_retained)
        return

    await _api_worker_passthrough_ws(
        ws,
        request_type=request_type,
        worker_status=GatewayWorkerStatus.DUPLEX_ACTIVE,
        source_channel="realtime_api",
        source_mode=mode,
        max_duration_s=max_duration_s,
        pool=pool,
        endpoint_name=endpoint_name,
    )


@app.get("/e/{endpoint}/v1/info")
async def endpoint_info(endpoint: str):
    """Public endpoint metadata without Worker addresses."""
    pool = endpoint_registry.get(endpoint) if endpoint_registry is not None else None
    if pool is None:
        return JSONResponse({"detail": f"unknown endpoint {endpoint}"}, status_code=404,
                            headers=_PUBLIC_CORS_HEADERS)
    return JSONResponse(build_endpoint_info(endpoint, pool), headers=_PUBLIC_CORS_HEADERS)


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
        "realtime_path": f"/e/{endpoint}/v1/realtime?mode=audio",
        "realtime_paths": {m: f"/e/{endpoint}/v1/realtime?mode={m}" for m in API_INPUT_MODES},
        "workers": workers,
        "available_workers": pool.idle_count,
        "queue_length": pool.queue_length,
        "models": list(models.values()),
    }


# ============ 入口 ============

def main():
    from config import get_config
    ws_debug = _enable_ws_debug_logging()
    cfg = get_config()

    parser = argparse.ArgumentParser(description="modeling.o5 Gateway")
    parser.add_argument("--port", type=int, default=None, help=f"Gateway port (default: {cfg.gateway_port})")
    parser.add_argument("--internal-port", type=int, default=8007, help="Internal worker registration port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
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
    args = parser.parse_args()

    # --http 被指定时，关闭 HTTPS
    use_https = not args.http

    port = args.port or cfg.gateway_port

    GATEWAY_CONFIG.update({
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
