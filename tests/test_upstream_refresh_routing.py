"""CPU regressions for the external-HF adapter's upstream merge boundaries."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import gateway
from core.fc_duplex.model_adapter import O5FcDuplexModelAdapter
from core.processors.unified import FcDuplexView
from gateway_modules import ordinary_reconnect
from py_backend.server import BackendProtocolSession


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["audio", "video", "omni"])
@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("override", [None, "12"])
async def test_realtime_mode_and_timeout_survive_retained_transport(
    monkeypatch, mode, retained, override
):
    """Both branches pass the same mode; ordinary retention keeps its timeout opt-in."""
    if override is None:
        monkeypatch.delenv("FC_API_SESSION_MAX_DURATION_S", raising=False)
    else:
        monkeypatch.setenv("FC_API_SESSION_MAX_DURATION_S", override)
    ws = SimpleNamespace(query_params={"mode": mode})
    if retained:
        ws.query_params["ordinary_reconnect"] = "1"
    captured = {}
    retained_socket = object()
    pool = object()

    async def passthrough(socket, **kwargs):
        captured.update(socket=socket, **kwargs)

    async def serve(socket, *, identity, run):
        assert socket is ws
        captured["identity"] = identity
        await run(retained_socket)

    monkeypatch.setattr(gateway, "_api_worker_passthrough_ws", passthrough)
    monkeypatch.setattr(ordinary_reconnect, "serve", serve)
    await gateway._realtime_entry(
        ws, pool=pool, endpoint_name="candidate", allow_chat=False,
    )
    assert captured["request_type"] == (
        "audio_duplex" if mode == "audio" else "omni_duplex"
    )
    assert captured["source_mode"] == mode
    assert captured["socket"] is (retained_socket if retained else ws)
    expected_duration = (
        float(override) if override is not None
        else None if retained
        else 2400 if mode == "audio" else 1200
    )
    assert captured["max_duration_s"] == expected_duration
    assert captured["pool"] is pool
    if retained:
        assert captured["identity"] == ("candidate", mode)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["audio", "video"])
async def test_default_ordinary_route_uses_common_transport(monkeypatch, mode):
    captured = {}

    async def passthrough(socket, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(gateway, "_api_worker_passthrough_ws", passthrough)
    await gateway._realtime_entry(
        SimpleNamespace(query_params={"mode": mode}),
        pool=object(), endpoint_name="default", allow_chat=True,
    )
    assert "worker_path" not in captured
    assert captured["source_mode"] == mode


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol_version", [None, "5"])
async def test_ordinary_backend_prepare_is_not_forced_into_fc(monkeypatch, protocol_version):
    calls = {}

    class Backend:
        def set_duplex_config(self, config):
            calls["config"] = config

        def duplex_prepare(self, **kwargs):
            calls["prepare"] = kwargs

    def unexpected_fc(*args, **kwargs):
        pytest.fail("ordinary request entered FC runtime")

    monkeypatch.setattr("py_backend.server.FcDuplexSessionRuntime", unexpected_fc)
    session = BackendProtocolSession(
        session_id="ordinary-candidate", mode="full_duplex", backend=Backend(),
        ws=object(), state=object(),
    )
    params = {"system_prompt": "ordinary", "config": {"sliding_window_mode": "basic"}}
    if protocol_version is not None:
        params["protocol_version"] = protocol_version
    await session._init_duplex(params)
    assert session._fc_runtime is None
    assert calls["prepare"]["system_prompt_text"] == "ordinary"
    assert calls["config"]["sliding_window_mode"] == "basic"


@pytest.mark.parametrize("mode", ["audio", "omni"])
def test_o5_adapter_forwards_new_mode_and_frame_metadata(mode):
    calls = {}
    frame = object()

    class Model:
        def fc_duplex_prepare(self, **kwargs):
            calls["prepare"] = kwargs
            return {}

        def fc_duplex_streaming_prefill(self, **kwargs):
            calls["prefill"] = kwargs
            return {"unit": 3, "cost_video": 0.012, "n_video_tokens": 64}

    adapter = O5FcDuplexModelAdapter(Model())
    adapter.prepare(
        system_content=object(), tts_prompt_audio_path=None,
        generate_audio=False, input_mode=mode,
    )
    assert calls["prepare"].get("input_mode", "audio") == mode
    result = adapter.streaming_prefill(frame_list=[frame], audio_waveform=None)
    assert calls["prefill"]["frame_list"][0] is frame
    projected = FcDuplexView._prefill_result(result)
    assert projected.unit_index == 3
    assert projected.cost_video_ms == pytest.approx(12)
    assert projected.n_video_tokens == 64


# Reuse the existing loopback-only fake-worker fixture, while exercising the
# real gateway and retained transport rather than replacing the serve function.
from tests.test_gateway_endpoints import _FakeWorker, _register, _until, gw  # noqa: E402,F401


@pytest.mark.parametrize("mode", ["audio", "omni"])
def test_named_mode_survives_real_retained_gateway_reconnect(gw, mode):
    with _FakeWorker("retained") as worker:
        _register(gw, "candidate", "retained-worker", worker.port)
        address = f"/e/candidate/v1/realtime?mode={mode}&ordinary_reconnect=1"
        with gw.websocket_connect(address) as first:
            first.send_json({"type": "session.init", "payload": {"protocol_version": "5"}})
            created = _until(first, "session.created")[-1]
            assert created["transport_mode"] == mode
            token = created["reconnect_token"]
            assert token in ordinary_reconnect._sessions
        with gw.websocket_connect(f"{address}&reconnect_token={token}") as resumed:
            _until(resumed, "session.transport_resumed")
            resumed.send_json({"type": "input.append", "input": {"input_id": "u0"}})
            assert _until(resumed, "echo")[-1]["worker"] == "retained"
            resumed.send_json({"type": "session.close"})
            _until(resumed, "session.closed")
