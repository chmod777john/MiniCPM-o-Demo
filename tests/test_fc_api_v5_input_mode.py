"""协议 v5:端点会话分 audio / omni 两种输入模式。

- 模式由连接地址 ``?mode=audio|omni`` 决定,网关写进 ``session.init.payload.input_mode``;
- audio:每个 input.append 不得带视频帧;omni:每个 input.append 必须恰好 1 帧;不合规直接让会话报错结束;
- v4 / v3 客户端(评测)只有 audio。

只用假 backend,不加载模型、不连真实服务。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from gateway import _stamp_input_mode
from py_backend.fc_duplex_runtime import (
    FcDuplexSessionRuntime,
    InputModeError,
    _resolve_input_mode,
    _validate_unit_frames,
)
from py_backend.server import _fatal_reason


class _Backend:
    def __init__(self) -> None:
        self.prepare_fields: dict[str, Any] = {}

    def fc_duplex_prepare(self, **fields: Any) -> None:
        self.prepare_fields = fields


def _runtime(backend: _Backend) -> FcDuplexSessionRuntime:
    async def send(_: str, **__: Any) -> None:
        return

    return FcDuplexSessionRuntime(session_id="v5_test", backend=backend, send=send)


def _payload(**extra: Any) -> dict[str, Any]:
    return {
        "system": {"segments": [], "tools": []},
        "generate_audio": False,
        "checkpoint_profile_id": "profile_test",
        "config": {"non_spoken_scheduling": "quality", "non_spoken_budget_while_listening": 30,
                   "non_spoken_budget_while_speaking": 15},
        **extra,
    }


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for name in ("FC_API_ACCEPT_V3", "FC_DUPLEX_TTS_PAD_PROTOCOL", "CHECKPOINT_PROFILE_ID",
                 "FC_DUPLEX_UNIT_POLICY_JSON", "FC_DUPLEX_NON_SPOKEN_SCHEDULING",
                 "FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", "FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING",
                 "FC_DUPLEX_UNIT_SEC"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FC_AUDIO_CACHE_DIR", str(tmp_path / "audio_cache"))


# ---------------------------------------------------------------- 会话模式解析


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["audio", "omni"])
async def test_v5_input_mode_reaches_backend_and_session_config(mode: str) -> None:
    backend = _Backend()
    runtime = _runtime(backend)
    await runtime.prepare(_payload(protocol_version="5", input_mode=mode))
    assert backend.prepare_fields["input_mode"] == mode
    assert runtime.session_config["input_mode"] == mode


@pytest.mark.asyncio
async def test_v5_requires_input_mode() -> None:
    with pytest.raises(InputModeError, match="input_mode"):
        await _runtime(_Backend()).prepare(_payload(protocol_version="5"))


@pytest.mark.asyncio
async def test_v4_clients_are_audio_only() -> None:
    backend = _Backend()
    runtime = _runtime(backend)
    await runtime.prepare(_payload(protocol_version="4"))
    assert backend.prepare_fields["input_mode"] == "audio"
    with pytest.raises(InputModeError, match="只支持 audio"):
        _resolve_input_mode({"input_mode": "omni"}, protocol_version="4")


def test_url_and_payload_conflict_is_an_error() -> None:
    with pytest.raises(InputModeError, match="不一致"):
        _resolve_input_mode({"input_mode_conflict": {"url": "omni", "payload": "audio"}}, protocol_version="5")


# ---------------------------------------------------------------- 每个 input.append 的帧数


@pytest.mark.parametrize(("mode", "frames", "ok"), [
    ("audio", None, True),
    ("audio", [], True),
    ("audio", ["jpg"], False),
    ("omni", ["jpg"], True),
    ("omni", None, False),
    ("omni", [], False),
    ("omni", ["a", "b"], False),
])
def test_frame_count_must_match_mode(mode, frames, ok) -> None:
    if ok:
        _validate_unit_frames(mode, frames)
    else:
        with pytest.raises(InputModeError):
            _validate_unit_frames(mode, frames)


@pytest.mark.asyncio
async def test_omni_input_without_frame_fails_before_queueing() -> None:
    runtime = _runtime(_Backend())
    await runtime.prepare(_payload(protocol_version="5", input_mode="omni"))
    with pytest.raises(InputModeError, match="恰好带 1 帧"):
        await runtime.enqueue_audio_input({"input_id": "u0", "audio_base64": "AAAA"})
    assert runtime._input_queue.empty()


@pytest.mark.asyncio
async def test_audio_input_with_frame_fails_before_queueing() -> None:
    runtime = _runtime(_Backend())
    await runtime.prepare(_payload(protocol_version="5", input_mode="audio"))
    with pytest.raises(InputModeError, match="不接受视频帧"):
        await runtime.enqueue_audio_input({"input_id": "u0", "audio_base64": "AAAA", "video_frames": ["/9j/"]})
    assert runtime._input_queue.empty()


def test_input_mode_error_maps_to_its_own_close_reason() -> None:
    reason, diagnostic = _fatal_reason(InputModeError("x"))
    assert reason == "input_mode_error" and diagnostic == {"message": "x"}


# ---------------------------------------------------------------- 网关按 URL 写入模式


def test_gateway_stamps_url_mode_into_session_init() -> None:
    raw = json.dumps({"type": "session.init", "payload": {"protocol_version": "5"}})
    assert json.loads(_stamp_input_mode(raw, "omni"))["payload"]["input_mode"] == "omni"


def test_gateway_keeps_matching_mode_and_flags_conflict() -> None:
    same = json.dumps({"type": "session.init", "payload": {"input_mode": "audio"}})
    assert json.loads(_stamp_input_mode(same, "audio"))["payload"] == {"input_mode": "audio"}
    other = json.dumps({"type": "session.init", "payload": {"input_mode": "audio"}})
    stamped = json.loads(_stamp_input_mode(other, "omni"))["payload"]
    assert stamped["input_mode_conflict"] == {"url": "omni", "payload": "audio"}


def test_gateway_leaves_other_frames_untouched() -> None:
    raw = json.dumps({"type": "input.append", "input": {"input_id": "u0", "video_frames": ["x"]}})
    assert _stamp_input_mode(raw, "omni") is raw
