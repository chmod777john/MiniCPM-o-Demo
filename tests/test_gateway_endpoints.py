"""多端点路由：每个端点独立的 Worker 组与队列、强制关闭、Worker 连不上时重新排队、API-only 模式。

端到端部分起真实的 Gateway 应用（TestClient）+ 不带模型的假 Worker（uvicorn 线程），
不碰 GPU、不碰集群。
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import Iterator, List

import pytest
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient

import config
import gateway
from gateway_modules.endpoints import EndpointError, EndpointRegistry, validate_endpoint_name
from gateway_modules.models import GatewayWorkerStatus
from gateway_modules.worker_pool import WorkerPool


# ---------------------------------------------------------------- 纯逻辑


def _pool() -> WorkerPool:
    return WorkerPool(worker_addresses=[])


def _add_idle(pool: WorkerPool, worker_id: str) -> None:
    from gateway_modules.worker_pool import WorkerConnection

    pool.workers[worker_id] = WorkerConnection(
        worker_id=worker_id, host="127.0.0.1", port=1, gpu_id=0, status=GatewayWorkerStatus.IDLE,
    )


def test_endpoint_names_are_validated() -> None:
    assert validate_endpoint_name("f9t4-rejection") == "f9t4-rejection"
    for bad in ("", "Upper", "-lead", "has space", "a/b", "x" * 41):
        with pytest.raises(EndpointError):
            validate_endpoint_name(bad)


def test_registry_create_delete_and_find() -> None:
    async def go() -> None:
        reg = EndpointRegistry(_pool)
        await reg.start()
        alpha = await reg.create("alpha")
        _add_idle(alpha, "w1")
        assert reg.find_worker("w1")[0] == "alpha"
        with pytest.raises(EndpointError):
            await reg.create("alpha")
        with pytest.raises(EndpointError, match="还有 1 个 Worker"):
            await reg.delete("alpha")
        with pytest.raises(EndpointError):
            await reg.delete("default")
        alpha.remove_worker("w1")
        await reg.delete("alpha")
        assert reg.names() == ["default"]
        await reg.stop()

    asyncio.run(go())


def test_removed_busy_worker_is_not_resurrected_on_release() -> None:
    async def go() -> None:
        pool = _pool()
        _add_idle(pool, "w1")
        _, future = pool.enqueue("audio_duplex")
        worker = future.result()
        assert pool.remove_worker("w1") is worker
        pool.release_worker(worker, request_type="audio_duplex", duration_s=3.0)
        assert "w1" not in pool.workers
        _, waiting = pool.enqueue("audio_duplex")     # 池里已经没有 Worker：只能排队
        assert not waiting.done()

    asyncio.run(go())


def test_failed_release_marks_offline_instead_of_idle() -> None:
    async def go() -> None:
        pool = _pool()
        _add_idle(pool, "w1")
        _, future = pool.enqueue("audio_duplex")
        worker = future.result()
        pool.release_worker(worker, failed=True)
        assert worker.status == GatewayWorkerStatus.OFFLINE and not worker.is_idle

    asyncio.run(go())


# ---------------------------------------------------------------- 端到端（假 Worker）


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fake_worker_app(name: str) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"worker_status": "idle"}

    @app.websocket("/v1/worker/duplex")
    async def duplex(ws: WebSocket):
        await ws.accept()
        try:
            async for raw in ws.iter_text():
                frame = json.loads(raw)
                if frame.get("type") == "session.init":
                    await ws.send_json({"type": "session.created", "session_id": f"s_{name}_{time.time_ns()}",
                                        "worker": name})
                elif frame.get("type") == "session.close":
                    await ws.send_json({"type": "session.closed", "reason": "client"})
                    return
                else:
                    await ws.send_json({"type": "echo", "worker": name})
        except WebSocketDisconnect:
            pass

    return app


class _FakeWorker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.port = _free_port()
        self.server = uvicorn.Server(uvicorn.Config(_fake_worker_app(name), host="127.0.0.1",
                                                    port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "_FakeWorker":
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("fake worker did not start")

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


@pytest.fixture
def gw(monkeypatch) -> Iterator[TestClient]:
    for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(config.get_config().recording, "enabled", False)    # 不往仓库里写会话录音
    monkeypatch.setitem(gateway.GATEWAY_CONFIG, "workers", [])
    with TestClient(gateway.app) as client:
        yield client


def _register(client: TestClient, endpoint: str, worker_id: str, port: int, **model) -> None:
    async def go() -> None:
        pool = gateway.endpoint_registry.get(endpoint) or await gateway.endpoint_registry.create(endpoint)
        await pool.register_worker(worker_id=worker_id, endpoint=f"127.0.0.1:{port}", model_info=model)

    client.portal.call(go)


def _open(client: TestClient, endpoint: str):
    ws = client.websocket_connect(f"/e/{endpoint}/v1/realtime?mode=audio")
    return ws.__enter__(), ws


def _until(ws, wanted: str) -> List[dict]:
    seen = []
    while True:
        frame = ws.receive_json()
        seen.append(frame)
        if frame.get("type") == wanted:
            return seen


def test_each_endpoint_routes_to_its_own_workers(gw: TestClient) -> None:
    with _FakeWorker("A") as a, _FakeWorker("B") as b:
        _register(gw, "alpha", "wa", a.port, profile_id="model-a")
        _register(gw, "beta", "wb", b.port, profile_id="model-b")
        for endpoint, expected in (("alpha", "A"), ("beta", "B")):
            ws, cm = _open(gw, endpoint)
            ws.send_json({"type": "session.init"})
            assert _until(ws, "session.created")[-1]["worker"] == expected
            ws.send_json({"type": "session.close"})
            _until(ws, "session.closed")
            cm.__exit__(None, None, None)


def test_second_session_queues_until_the_worker_is_free(gw: TestClient) -> None:
    with _FakeWorker("A") as a:
        _register(gw, "alpha", "wa", a.port)
        first, cm1 = _open(gw, "alpha")
        first.send_json({"type": "session.init"})
        _until(first, "session.created")

        second, cm2 = _open(gw, "alpha")
        assert second.receive_json()["type"] == "session.queued"

        first.send_json({"type": "session.close"})
        _until(first, "session.closed")
        cm1.__exit__(None, None, None)

        assert _until(second, "session.queue_done")
        second.send_json({"type": "session.init"})
        assert _until(second, "session.created")[-1]["worker"] == "A"
        cm2.__exit__(None, None, None)


def test_force_close_ends_the_live_session(gw: TestClient) -> None:
    with _FakeWorker("A") as a:
        _register(gw, "alpha", "wa", a.port)
        ws, cm = _open(gw, "alpha")
        ws.send_json({"type": "session.init"})
        _until(ws, "session.created")
        assert gw.portal.call(lambda: _close_async("wa"))
        closed = _until(ws, "session.closed")[-1]
        assert closed["reason"] == "worker_removed"
        cm.__exit__(None, None, None)


async def _close_async(worker_id: str) -> bool:
    return gateway.live_sessions.request_close(worker_id, "worker_removed")


def test_unreachable_worker_is_marked_offline_and_session_requeued(gw: TestClient) -> None:
    with _FakeWorker("B") as b:
        _register(gw, "alpha", "a-dead", _free_port())      # 没人监听：注册时健康检查就会判离线
        _register(gw, "alpha", "b-live", b.port)

        async def pretend_dead_is_idle() -> None:            # 模拟「刚挂、健康检查还没发现」
            gateway.endpoint_registry.get("alpha").workers["a-dead"].status = GatewayWorkerStatus.IDLE

        gw.portal.call(pretend_dead_is_idle)
        ws, cm = _open(gw, "alpha")
        ws.send_json({"type": "session.init"})
        assert _until(ws, "session.created")[-1]["worker"] == "B"
        dead = gateway.endpoint_registry.get("alpha").workers["a-dead"]
        assert dead.status == GatewayWorkerStatus.OFFLINE
        cm.__exit__(None, None, None)


def test_unknown_endpoint_is_rejected(gw: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect as StarletteDisconnect

    with pytest.raises(StarletteDisconnect) as info:
        with gw.websocket_connect("/e/nope/v1/realtime?mode=audio") as ws:
            ws.receive_json()
    assert info.value.code == 4404


def test_info_is_public_and_hides_worker_addresses(gw: TestClient) -> None:
    with _FakeWorker("A") as a:
        _register(gw, "alpha", "wa", a.port, profile_id="model-a", tts_pad_protocol="revised")
        r = gw.get("/e/alpha/v1/info")
        assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "*"
        body = r.json()
        assert body["protocol_version"] == "4" and body["available_workers"] == 1
        assert body["models"] == [{"model": {"profile_id": "model-a", "tts_pad_protocol": "revised"},
                                   "workers": 1, "available": 1}]
        assert "127.0.0.1" not in r.text and str(a.port) not in r.text
    assert gw.get("/e/nope/v1/info").status_code == 404


def test_api_only_mode_hides_everything_else(gw: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(gateway, "_API_ONLY", True)
    gw.portal.call(gateway.endpoint_registry.create, "alpha")
    assert gw.get("/health").status_code == 200
    assert gw.get("/e/alpha/v1/info").status_code == 200
    for path in ("/status", "/workers", "/api/queue", "/sessions", "/", "/admin", "/api/config/eta"):
        assert gw.get(path).status_code == 404, path


def test_messages_sent_while_queued_are_forwarded_after_dispatch(gw: TestClient) -> None:
    """排队期间客户端连接也一直在读（否则 uvicorn 暂停读 socket、ping 得不到 pong）；提前发来的消息按序缓存转发。"""

    with _FakeWorker("A") as a:
        _register(gw, "alpha", "wa", a.port)
        first, cm1 = _open(gw, "alpha")
        first.send_json({"type": "session.init"})
        _until(first, "session.created")

        second, cm2 = _open(gw, "alpha")
        second.send_json({"type": "session.init"})              # 还在排队就发
        second.send_json({"type": "ping-me"})
        assert second.receive_json()["type"] == "session.queued"

        first.send_json({"type": "session.close"})
        _until(first, "session.closed")
        cm1.__exit__(None, None, None)

        frames = _until(second, "echo")
        kinds = [f["type"] for f in frames]
        assert kinds.index("session.queue_done") < kinds.index("session.created") < kinds.index("echo")
        cm2.__exit__(None, None, None)


def test_disconnect_while_queued_releases_the_ticket(gw: TestClient) -> None:
    with _FakeWorker("A") as a:
        _register(gw, "alpha", "wa", a.port)
        first, cm1 = _open(gw, "alpha")
        first.send_json({"type": "session.init"})
        _until(first, "session.created")
        second, cm2 = _open(gw, "alpha")
        assert second.receive_json()["type"] == "session.queued"
        assert gateway.endpoint_registry.get("alpha").queue_length == 1
        cm2.__exit__(None, None, None)                           # 排队中断开
        for _ in range(50):
            if gateway.endpoint_registry.get("alpha").queue_length == 0:
                break
            time.sleep(0.1)
        assert gateway.endpoint_registry.get("alpha").queue_length == 0
        cm1.__exit__(None, None, None)
