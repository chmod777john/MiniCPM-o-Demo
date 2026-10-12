"""控制面板 HTTP 接口：密码门禁、登录 cookie、操作接口、公开文档；跑在真实 Gateway 应用上。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

import config
import gateway
from control_plane.manager import ControlPlane
from control_plane.routes import set_control_plane
from tests.control_plane.test_manager import TEMPLATE, FakeJobs


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setattr(config.get_config().recording, "enabled", False)
    monkeypatch.setitem(gateway.GATEWAY_CONFIG, "workers", [])
    monkeypatch.setenv("CONTROL_PANEL_PASSWORD", "pw-123")
    with TestClient(gateway.app, base_url="https://testserver") as c:
        async def make() -> ControlPlane:
            plane = ControlPlane(root=tmp_path, registry=gateway.endpoint_registry,
                                 live_sessions=gateway.live_sessions, jobs=FakeJobs(),
                                 repo_dir="/repo", python="/venv/bin/python", job_poll_s=0.0)
            await plane.start(run_loop=False)
            return plane

        set_control_plane(c.portal.call(make))
        yield c
        set_control_plane(None)


def _login(client: TestClient) -> None:
    assert client.post("/control/api/login", json={"password": "pw-123"}).status_code == 200


def test_panel_requires_password(client: TestClient, monkeypatch) -> None:
    assert client.get("/control/api/state").status_code == 401
    assert client.post("/control/api/login", json={"password": "wrong"}).status_code == 401
    assert client.post("/control/api/endpoints", json={"name": "x"}).status_code == 401
    _login(client)
    assert client.get("/control/api/state").status_code == 200
    client.post("/control/api/logout")
    client.cookies.clear()
    assert client.get("/control/api/state").status_code == 401

    monkeypatch.delenv("CONTROL_PANEL_PASSWORD")
    assert client.get("/control/api/state").status_code == 503          # 没配密码：整体禁用
    assert client.post("/control/api/login", json={"password": ""}).status_code == 503


def test_forged_cookie_is_rejected(client: TestClient) -> None:
    client.cookies.set("o5_control", "9999999999.deadbeef", path="/control")
    assert client.get("/control/api/state").status_code == 401


def test_endpoint_template_worker_flow(client: TestClient) -> None:
    _login(client)
    assert client.post("/control/api/endpoints", json={"name": "alpha", "description": "测试"}).status_code == 200
    template = TEMPLATE.model_dump(mode="json")
    assert client.put(f"/control/api/templates/{TEMPLATE.name}", json=template).status_code == 200
    bad = {**template, "tts_pad_protocol": "auto"}
    assert client.put(f"/control/api/templates/{TEMPLATE.name}", json=bad).status_code == 422
    r = client.post("/control/api/workers", json={"endpoint": "alpha", "template": TEMPLATE.name})
    assert r.status_code == 200 and r.json()["worker_id"] == "alpha-1"
    state = client.get("/control/api/state").json()
    assert state["public_base"] == "https://testserver"
    (endpoint,) = state["endpoints"]
    assert endpoint["name"] == "alpha" and endpoint["workers"][0]["worker_id"] == "alpha-1"
    assert client.post("/control/api/workers/alpha-1/scale").json()["worker_id"] == "alpha-2"
    assert client.delete("/control/api/endpoints/alpha").status_code == 400     # 还有 Worker
    assert client.post("/control/api/workers/nope/drain").status_code == 400
    # 端点建好后就能被公开查询
    assert client.get("/e/alpha/v1/info").status_code == 200


def test_panel_page_and_public_docs(client: TestClient) -> None:
    assert client.get("/control").status_code == 200
    assert client.get("/control/static/panel.js").status_code == 200
    assert client.get("/control/static/../routes.py").status_code == 404
    docs = client.get("/api-docs")
    assert docs.status_code == 200 and "Realtime API v4" in docs.text and "<table>" in docs.text
