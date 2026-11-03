"""Worker exposes one realtime API and owns public-mode normalization."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import worker as worker_module
from runtime.events import RuntimeEvent


class _FakeRuntimeSession:
    instances: list["_FakeRuntimeSession"] = []

    def __init__(self, *, backend_base_url: str, mode: str, session_id=None) -> None:
        self.backend_base_url = backend_base_url
        self.mode = mode
        self.session_id = session_id
        self.init_params = None
        self.pushed = []
        self._stop = asyncio.Event()
        self.__class__.instances.append(self)

    async def init(self, params=None):
        self.init_params = dict(params or {})
        self.session_id = "sess_fake"
        return RuntimeEvent(
            channel="session",
            payload={
                "event": {
                    "type": "session.created",
                    "session_id": self.session_id,
                    "mode": self.mode,
                }
            },
        )

    async def push(self, payload):
        self.pushed.append(payload)

    async def pull(self):
        await self._stop.wait()
        return RuntimeEvent(
            channel="session",
            payload={
                "event": {
                    "type": "session.closed",
                    "session_id": self.session_id,
                    "reason": "closed",
                }
            },
        )

    async def unary(self, method, payload=None):
        assert method == "close"
        self._stop.set()
        return RuntimeEvent(
            channel="session",
            payload={
                "event": {
                    "type": "session.closed",
                    "session_id": self.session_id,
                    "reason": (payload or {}).get("reason", "client_closed"),
                }
            },
        )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("chat", "turn_based"),
        ("turn_based", "turn_based"),
        ("audio", "full_duplex"),
        ("video", "full_duplex"),
        ("omni", "full_duplex"),
        ("full_duplex", "full_duplex"),
    ],
)
def test_public_modes_normalize(mode: str, expected: str) -> None:
    assert worker_module._normalize_runtime_mode(mode, source="test") == expected


def test_session_mode_payload_precedence_and_default() -> None:
    init = {"type": "session.init", "payload": {"mode": "turn_based"}}
    assert worker_module._resolve_runtime_mode(init, "chat") == "turn_based"
    assert worker_module._resolve_runtime_mode(
        {"type": "session.init", "payload": {"input_mode": "omni"}}, None
    ) == "full_duplex"
    assert worker_module._resolve_runtime_mode(
        {"type": "session.init", "payload": {}}, None
    ) == "full_duplex"


def test_conflicting_modes_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="conflicting session modes"):
        worker_module._resolve_runtime_mode(
            {"type": "session.init", "payload": {"mode": "turn_based"}},
            "audio",
        )


def test_worker_route_table_has_only_unified_realtime_endpoint() -> None:
    paths = {getattr(route, "path", None) for route in worker_module.app.routes}
    assert "/v1/realtime" in paths
    assert "/v1/worker/chat" not in paths
    assert "/v1/worker/duplex" not in paths


def test_worker_dispatches_init_mode_without_gateway_path_split(monkeypatch) -> None:
    _FakeRuntimeSession.instances.clear()
    monkeypatch.setattr(worker_module, "BackendRuntimeSession", _FakeRuntimeSession)
    monkeypatch.setitem(worker_module.WORKER_CONFIG, "backend_server_url", "http://backend.test")

    with TestClient(worker_module.app) as client:
        with client.websocket_connect("/v1/realtime?mode=chat") as ws:
            ws.send_json(
                {
                    "type": "session.init",
                    "payload": {"mode": "turn_based", "client": "unit-test"},
                }
            )
            created = ws.receive_json()
            assert created["mode"] == "turn_based"
            ws.send_json({"type": "session.close", "reason": "test_done"})
            closed = ws.receive_json()
            assert closed["reason"] == "test_done"

    runtime = _FakeRuntimeSession.instances[-1]
    assert runtime.mode == "turn_based"
    assert runtime.init_params["mode"] == "turn_based"
    assert runtime.init_params["client"] == "unit-test"
    assert worker_module.worker.state.status.value == "idle"
