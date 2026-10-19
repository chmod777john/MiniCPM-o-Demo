"""Demo 侧连接 API：端点地址规范化、/v1/info 摘要、本机音频读成 v4 base64（白名单）。"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tests.conftest import needs_node, repo_module_url, run_node


@needs_node
def test_endpoint_addresses_are_normalized() -> None:
    got = run_node(f"""
const m = await import({json.dumps(repo_module_url('static/demo-gallery/core/api-endpoint.js'))});
const out = {{}};
for (const raw of ['https://h:7029/e/f9t4', 'https://h:7029/e/f9t4/', 'wss://h:7029/e/f9t4/v1/realtime?mode=audio',
                   'https://h:7029/e/f9t4/v1/info']) out[raw] = m.normalizeEndpoint(raw);
let bad = [];
for (const raw of ['', 'h:7029/e/f9t4', 'ftp://x']) {{ try {{ m.normalizeEndpoint(raw); }} catch (e) {{ bad.push(raw); }} }}
console.log(JSON.stringify({{ out, bad }}));
""")
    expected = {"base": "https://h:7029/e/f9t4", "info": "https://h:7029/e/f9t4/v1/info",
                "realtime": "wss://h:7029/e/f9t4/v1/realtime?mode=audio"}
    assert all(value == expected for value in got["out"].values()), got["out"]
    assert got["bad"] == ["", "h:7029/e/f9t4", "ftp://x"]


@needs_node
def test_endpoint_info_summary() -> None:
    info = {"workers": [{"worker_id": "f9t4-1"}], "available_workers": 1, "queue_length": 0, "models": [{
        "model": {"profile_id": "p", "session_defaults": {
            "tts_pad_protocol": "legacy", "non_spoken_scheduling": "quality",
            "unit_policy": {"unit_sec": 1.0, "non_spoken_budgets_while_listening": [45],
                            "non_spoken_budgets_while_speaking": [25]}}}, "workers": 1, "available": 1}]}
    got = run_node(f"""
const m = await import({json.dumps(repo_module_url('static/demo-gallery/core/api-endpoint.js'))});
const s = m.summarizeEndpointInfo({json.dumps(info)});
const empty = m.summarizeEndpointInfo({{ workers: [], models: [], available_workers: 0, queue_length: 0 }});
const created = m.summarizeSessionConfig({{ tts_pad_protocol: 'revised', non_spoken_scheduling: 'latency',
  unit_policy: {{ unit_sec: 1.0, non_spoken_budgets_while_listening: [30], non_spoken_budgets_while_speaking: [10] }} }});
console.log(JSON.stringify({{ s: {{ ...s, model: undefined }}, empty: empty.statusText, created }}));
""")
    assert got["s"] == {"profileId": "p", "ttsPad": "legacy", "scheduling": "quality", "listening": "45",
                        "speaking": "25", "unitSec": 1.0, "statusText": "1 个 Worker · 可用 1 · 排队 0"}
    assert "没有 Worker" in got["empty"]
    # 建会话后以 session.created.session_config（实际生效值）为准
    assert got["created"] == {"ttsPad": "revised", "scheduling": "latency", "listening": "30",
                              "speaking": "10", "unitSec": 1.0}


@needs_node
def test_endpoints_discovered_from_overview_merge_with_configured() -> None:
    overview = {"public_base": "https://pub:7029", "endpoints": [
        {"name": "m1", "model": "tpl-a", "available": 2, "queue_length": 1, "workers": [{}, {}, {}]},
        {"name": "empty", "model": None, "available": 0, "queue_length": 0, "workers": []},
    ]}
    got = run_node(f"""
const m = await import({json.dumps(repo_module_url('static/demo-gallery/core/api-endpoint.js'))});
const found = m.endpointsFromOverview({json.dumps(overview)}, 'https://ignored:1');
const noBase = m.endpointsFromOverview({{ endpoints: [{{ name: 'x', workers: [] }}] }}, 'https://svc:2/');
const merged = m.mergeEndpointOptions(found, [{{ name: 'm1-old', url: 'https://pub:7029/e/m1/' }},
                                             {{ name: 'manual', url: 'https://other:1/e/z' }}, {{ name: 'bad', url: 'nope' }}]);
console.log(JSON.stringify({{ found, noBase, merged }}));
""")
    assert got["found"] == [
        {"name": "m1", "url": "https://pub:7029/e/m1", "label": "m1 · tpl-a · 可用 2/3 · 排队 1", "available": 2},
        {"name": "empty", "url": "https://pub:7029/e/empty", "label": "empty · 未部署模型 · 没有 Worker", "available": 0},
    ]
    assert got["noBase"][0]["url"] == "https://svc:2/e/x"
    # 发现到的在前；同一地址（规范化后）只留发现到的那条；不合法的地址丢掉
    assert [(o["name"], o["url"]) for o in got["merged"]] == [
        ("m1", "https://pub:7029/e/m1"), ("empty", "https://pub:7029/e/empty"), ("manual", "https://other:1/e/z")]
    assert got["merged"][2]["available"] is None


def test_default_api_service(monkeypatch) -> None:
    from demo_gallery.runtime import gallery_runtime

    monkeypatch.setenv("DEMO_API_ENDPOINTS", json.dumps([{"name": "a", "url": "https://h:1/e/a"}]))
    monkeypatch.delenv("DEMO_API_SERVICE", raising=False)
    monkeypatch.setenv("DEMO_DEFAULT_API", "https://g:2/e/c")
    assert gallery_runtime()["default_api_service"] == "https://g:2"
    monkeypatch.delenv("DEMO_DEFAULT_API")
    assert gallery_runtime()["default_api_service"] == "https://h:1"
    monkeypatch.setenv("DEMO_API_SERVICE", "https://x:3/")
    assert gallery_runtime()["default_api_service"] == "https://x:3"


@needs_node
def test_api_service_address_is_parsed() -> None:
    got = run_node(f"""
const m = await import({json.dumps(repo_module_url('static/demo-gallery/core/api-endpoint.js'))});
const out = {{}};
for (const raw of ['https://h:7029', 'https://h:7029/', 'https://h:7029/overview', 'https://h:7029/e/m1',
                   'wss://h:7029/e/m1/v1/realtime?mode=audio', 'https://h:7029/e/m1/v1/info']) out[raw] = m.parseApiService(raw);
let bad = [];
for (const raw of ['', 'h:7029']) {{ try {{ m.parseApiService(raw); }} catch (e) {{ bad.push(raw); }} }}
console.log(JSON.stringify({{ out, bad }}));
""")
    plain = {"service": "https://h:7029", "endpoint": None}
    with_ep = {"service": "https://h:7029", "endpoint": "https://h:7029/e/m1"}
    assert got["out"] == {"https://h:7029": plain, "https://h:7029/": plain, "https://h:7029/overview": plain,
                          "https://h:7029/e/m1": with_ep, "wss://h:7029/e/m1/v1/realtime?mode=audio": with_ep,
                          "https://h:7029/e/m1/v1/info": with_ep}
    assert got["bad"] == ["", "h:7029"]


def _wav(path: Path, *, sr: int = 16000, channels: int = 1, fmt: str = "WAV") -> bytes:
    data = np.zeros((1600, channels) if channels > 1 else 1600, dtype=np.float32)
    sf.write(path, data, sr, format=fmt)
    return path.read_bytes()


@pytest.fixture
def demo_client(monkeypatch, tmp_path: Path) -> TestClient:
    import demo_server

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("DEMO_AUDIO_ROOTS", str(allowed))
    return TestClient(demo_server.app)


def test_audio_inside_allowed_root_becomes_v4_base64(demo_client: TestClient, tmp_path: Path) -> None:
    raw = _wav(tmp_path / "allowed" / "ref.wav")
    r = demo_client.get("/api/demos/_audio", params={"path": str(tmp_path / "allowed" / "ref.wav")})
    assert r.status_code == 200
    body = r.json()
    assert (body["source"], body["format"]) == ("base64", "wav") and base64.b64decode(body["data"]) == raw


def test_audio_resolver_refuses_outside_paths_and_bad_formats(demo_client: TestClient, tmp_path: Path, monkeypatch) -> None:
    outside = tmp_path / "outside.wav"
    _wav(outside)
    escape = tmp_path / "allowed" / ".." / "outside.wav"
    for path in (str(outside), str(escape), "relative.wav", "/etc/passwd"):
        assert demo_client.get("/api/demos/_audio", params={"path": path}).status_code == 400, path
    _wav(tmp_path / "allowed" / "hi.wav", sr=24000)
    _wav(tmp_path / "allowed" / "stereo.wav", channels=2)
    _wav(tmp_path / "allowed" / "ref.flac", fmt="FLAC")
    for name, match in (("hi.wav", "16 kHz"), ("stereo.wav", "单声道"), ("ref.flac", "只支持 WAV")):
        r = demo_client.get("/api/demos/_audio", params={"path": str(tmp_path / "allowed" / name)})
        assert r.status_code == 400 and match in r.json()["detail"], name

    monkeypatch.delenv("DEMO_AUDIO_ROOTS")                 # 未配置白名单：一律拒绝
    r = demo_client.get("/api/demos/_audio", params={"path": str(tmp_path / "allowed" / "hi.wav")})
    assert r.status_code == 400 and "DEMO_AUDIO_ROOTS" in r.json()["detail"]


def test_demo_server_is_independent_of_the_api(demo_client: TestClient) -> None:
    assert demo_client.get("/demos").status_code == 200
    assert demo_client.get("/static/demo-gallery/shell/app.js").status_code == 200
    # Demo 服务不提供任何 API 端点
    assert demo_client.get("/v1/info").status_code == 404
    assert demo_client.get("/e/f9t4/v1/info").status_code == 404
