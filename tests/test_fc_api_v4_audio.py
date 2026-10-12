"""协议 v4：音频只收 base64 WAV；v3 路径输入默认拒绝；tts_pad 由部署绑定。

只用 tmp_path 与假 backend，不加载模型。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from core.fc_duplex.profiles import apply_fc_deployment_profile_environment
from core.fc_duplex.system_input import (
    FcAudioBase64Input,
    FcAudioPathInput,
    FcSystemContentV4Input,
)
from gateway_modules.session_recording import SessionRecorder
from py_backend.fc_duplex_runtime import FcDuplexSessionRuntime


def _wav_bytes(*, sr: int = 16000, channels: int = 1, seconds: float = 0.25, fmt: str = "WAV") -> bytes:
    frames = int(sr * seconds)
    data = np.zeros((frames, channels), dtype=np.float32) if channels > 1 else np.zeros(frames, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format=fmt, subtype="FLOAT" if fmt == "WAV" else None)
    return buf.getvalue()


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _audio(raw: bytes) -> dict[str, str]:
    return {"source": "base64", "format": "wav", "data": _b64(raw)}


# ---------------------------------------------------------------- DTO


def test_base64_wav_materializes_to_content_addressed_file(tmp_path: Path) -> None:
    raw = _wav_bytes()
    audio = FcAudioBase64Input.model_validate(_audio(raw))
    first = audio.materialize(tmp_path)
    second = FcAudioBase64Input.model_validate(_audio(raw)).materialize(tmp_path)
    assert isinstance(first, FcAudioPathInput) and first.file_path == second.file_path
    assert Path(first.file_path).name == f"{hashlib.sha256(raw).hexdigest()}.wav"
    assert Path(first.file_path).read_bytes() == raw          # 文件字节原样 → tts_prompt_audio_sha256 与 v3 一致


@pytest.mark.parametrize("payload,match", [
    ({"source": "base64", "format": "wav", "data": "!!not-base64!!"}, "base64"),
    ({"source": "base64", "format": "wav", "data": _b64(b"RIFF....garbage")}, "WAV"),
    ({"source": "base64", "format": "wav", "data": _b64(_wav_bytes(sr=24000))}, "16 kHz"),
    ({"source": "base64", "format": "wav", "data": _b64(_wav_bytes(channels=2))}, "单声道"),
    ({"source": "base64", "format": "wav", "data": _b64(_wav_bytes(seconds=0))}, "为空|不能为空|WAV"),
    ({"source": "path", "file_path": "/etc/hostname"}, "base64"),
])
def test_bad_audio_is_rejected(payload: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FcAudioBase64Input.model_validate(payload)


def test_v4_system_keeps_segment_order_and_tools(tmp_path: Path) -> None:
    raw = _wav_bytes()
    system = FcSystemContentV4Input.model_validate({
        "segments": [{"kind": "text", "text": "前缀"}, {"kind": "audio", "audio": _audio(raw)},
                     {"kind": "text", "text": "正文"}],
        "tools": [{"type": "function", "function": {"name": "f", "description": "d",
                                                     "parameters": {"type": "object", "properties": {}}}}],
    }).materialize(tmp_path)
    assert [segment.kind for segment in system.segments] == ["text", "audio", "text"]
    assert Path(system.segments[1].audio.file_path).parent == tmp_path
    assert system.tools[0].function.name == "f"


# ---------------------------------------------------------------- Runtime


class _Backend:
    def __init__(self) -> None:
        self.prepare_fields: dict[str, Any] = {}

    def fc_duplex_prepare(self, **fields: Any) -> None:
        self.prepare_fields = fields


def _runtime(backend: _Backend) -> FcDuplexSessionRuntime:
    async def send(_: str, **__: Any) -> None:
        return

    return FcDuplexSessionRuntime(session_id="v4_test", backend=backend, send=send)


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


@pytest.mark.asyncio
async def test_v3_is_rejected_unless_explicitly_enabled(monkeypatch, tmp_path: Path) -> None:
    wav = tmp_path / "ref.wav"
    wav.write_bytes(_wav_bytes())
    v3 = _payload(protocol_version="3", generate_audio=True,
                  tts_prompt_audio={"source": "path", "file_path": str(wav)})
    with pytest.raises(RuntimeError, match="protocol_version='4'"):
        await _runtime(_Backend()).prepare(v3)

    monkeypatch.setenv("FC_API_ACCEPT_V3", "1")         # 评测副本池
    backend = _Backend()
    await _runtime(backend).prepare(v3)
    assert backend.prepare_fields["tts_prompt_audio"].file_path == str(wav)


@pytest.mark.asyncio
async def test_v4_tts_prompt_reaches_backend_as_cached_file(tmp_path: Path) -> None:
    raw = _wav_bytes()
    backend = _Backend()
    runtime = _runtime(backend)
    await runtime.prepare(_payload(protocol_version="4", generate_audio=True, tts_prompt_audio=_audio(raw)))
    path = Path(backend.prepare_fields["tts_prompt_audio"].file_path)
    assert path.parent == tmp_path / "audio_cache" and path.read_bytes() == raw
    assert runtime.resume_identity["protocol_version"] == "4"


@pytest.mark.asyncio
async def test_v4_rejects_server_path_audio() -> None:
    with pytest.raises(ValueError):
        await _runtime(_Backend()).prepare(_payload(
            protocol_version="4", generate_audio=True,
            tts_prompt_audio={"source": "path", "file_path": "/etc/hostname"},
        ))


@pytest.mark.asyncio
@pytest.mark.parametrize("registered,requested,expected", [
    (None, None, "legacy"),
    (None, "revised", "revised"),
    ("revised", None, "revised"),
    ("revised", "revised", "revised"),
    ("revised", "legacy", RuntimeError),
])
async def test_tts_pad_is_bound_by_deployment(monkeypatch, registered, requested, expected) -> None:
    if registered:
        monkeypatch.setenv("FC_DUPLEX_TTS_PAD_PROTOCOL", registered)
    config = {"non_spoken_scheduling": "quality", "non_spoken_budget_while_listening": 30,
              "non_spoken_budget_while_speaking": 15}
    if requested:
        config["tts_pad_protocol"] = requested
    runtime = _runtime(_Backend())
    if expected is RuntimeError:
        with pytest.raises(RuntimeError, match="tts_pad_protocol"):
            await runtime.prepare(_payload(protocol_version="4", config=config))
        return
    await runtime.prepare(_payload(protocol_version="4", config=config))
    assert runtime.resume_identity["tts_pad_protocol"] == expected


def test_profile_projects_tts_pad(monkeypatch, tmp_path: Path) -> None:
    from core.fc_duplex.profiles import O5FcDeploymentProfile

    model = tmp_path / "model"
    model.mkdir()
    pt = tmp_path / "model.pt"
    pt.write_bytes(b"x")
    base = dict(profile_id="p", model_path=str(model), pt_path=str(pt), unit_policy=None,
                deployment_mode="single_eager", tp_size=1)
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"FC_DUPLEX_TTS_PAD_PROTOCOL": "stale"}):   # 投影会直接写 os.environ，测完还原
        apply_fc_deployment_profile_environment(O5FcDeploymentProfile(**base))
        assert "FC_DUPLEX_TTS_PAD_PROTOCOL" not in os.environ          # 不绑定 → 清掉残留，由 Session 给
        apply_fc_deployment_profile_environment(O5FcDeploymentProfile(**base, tts_pad_protocol="revised"))
        assert os.environ["FC_DUPLEX_TTS_PAD_PROTOCOL"] == "revised"


# ---------------------------------------------------------------- 录音


def test_recording_stores_v4_audio_as_original_wav(tmp_path: Path) -> None:
    raw = _wav_bytes()
    frame = {"type": "session.init", "payload": {"tts_prompt_audio": _audio(raw), "system": {
        "segments": [{"kind": "audio", "audio": _audio(raw)}]}}}
    rec = SessionRecorder("s_v4", "full_duplex", sessions_dir=str(tmp_path))
    rec.record("up", frame, recv_ts=1.0, send_ts=1.0, src="client")
    rec.close(reason="test")
    rec.wait_closed()
    with open(f"{rec.directory}/stream.jsonl", encoding="utf-8") as handle:
        recorded = json.loads(handle.readline())["frame"]["payload"]
    for audio in (recorded["tts_prompt_audio"], recorded["system"]["segments"][0]["audio"]):
        assert audio["data"].startswith("@blob/") and audio["source"] == "base64"
        assert Path(rec.directory, audio["data"][1:]).read_bytes() == raw


def test_client_converts_path_session_init_to_v4(tmp_path: Path) -> None:
    from clients.v4_payload import to_v4_session_init

    raw = _wav_bytes()
    wav = tmp_path / "ref.wav"
    wav.write_bytes(raw)
    v3 = {"type": "session.init", "payload": {
        "protocol_version": "3",
        "system": {"segments": [{"kind": "audio", "audio": {"source": "path", "file_path": str(wav)}}], "tools": []},
        "tts_prompt_audio": {"source": "path", "file_path": str(wav)},
    }}
    v4 = to_v4_session_init(v3)
    assert v4["payload"]["protocol_version"] == "4"
    for audio in (v4["payload"]["tts_prompt_audio"], v4["payload"]["system"]["segments"][0]["audio"]):
        assert audio == {"source": "base64", "format": "wav", "data": _b64(raw)}
        FcAudioBase64Input.model_validate(audio)                 # 服务端能收
    assert v3["payload"]["protocol_version"] == "3"               # 不改原对象
