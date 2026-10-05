"""锁步参考客户端的协议契约测试（不连真实服务）。"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

import clients.o5_fc_lockstep_client as client
from clients.o5_fc_lockstep_client import (
    O5FcLockstepSession,
    O5FcSessionError,
    O5FcSessionPreset,
    build_system,
)

POLICY = {
    "unit_sec": 1.0,
    "non_spoken_budgets_while_listening": [45],
    "non_spoken_budgets_while_speaking": [25],
}


class _FakeWebSocket:
    """按脚本回放服务端事件，并记录客户端上行帧。"""

    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)
        self.sent: list[dict] = []

    def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def recv(self, timeout: float) -> str:
        return json.dumps(self.script.pop(0))

    def close(self) -> None:
        pass


def _preset(tmp_path: Path, **overrides: object) -> Path:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    data = {
        "prefix_text": "Speak using the voice in the following audio.\n",
        "system_audio_path": str(audio),
        "tts_prompt_audio_path": str(audio),
        "unit_policy": POLICY,
        "non_spoken_scheduling": "quality",
        "decode_mode": "greedy",
        "tts_pad_protocol": "legacy",
        **overrides,
    }
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_build_system_orders_prefix_audio_task() -> None:
    """system 段顺序固定为 prefix → 参考音频 → 任务 prompt；无 prefix 时省略该段。"""

    system = build_system(
        system_audio_path="/a.wav", task_prompt="TASK", tools=[{"x": 1}], prefix_text="P\n"
    )
    assert [seg["kind"] for seg in system["segments"]] == ["text", "audio", "text"]
    assert system["segments"][0]["text"] == "P\n"
    assert system["segments"][2]["text"] == "TASK"
    assert system["tools"] == [{"x": 1}]

    no_prefix = build_system(system_audio_path="/a.wav", task_prompt="TASK", prefix_text=None)
    assert [seg["kind"] for seg in no_prefix["segments"]] == ["audio", "text"]


def test_preset_requires_every_field(tmp_path: Path) -> None:
    """preset 缺字段或多字段都必须报错，不允许隐式默认值。"""

    path = _preset(tmp_path)
    data = json.loads(path.read_text())
    del data["decode_mode"]
    data["budget"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="缺字段"):
        O5FcSessionPreset.load(str(path))


def test_preset_requires_explicit_tts_pad_protocol(tmp_path: Path) -> None:
    """tts_pad_protocol 故意不给默认值：新 ckpt 漏写时必须报错，不能静默按 legacy 跑。"""

    path = _preset(tmp_path)
    data = json.loads(path.read_text())
    del data["tts_pad_protocol"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="缺字段.*tts_pad_protocol"):
        O5FcSessionPreset.load(str(path))


@pytest.mark.parametrize("value", ["revised", "legacy"])
def test_tts_pad_protocol_reaches_session_init(tmp_path: Path, value: str) -> None:
    """preset 里写什么，session.init.config 里就发什么（两档都要通）。"""

    path = _preset(tmp_path)
    data = json.loads(path.read_text())
    data["tts_pad_protocol"] = value
    path.write_text(json.dumps(data))
    spec = O5FcSessionPreset.load(str(path)).to_spec(server_url="http://h:1", task_prompt="T", tools=[])
    assert spec.session_init_frame()["payload"]["config"]["tts_pad_protocol"] == value


def test_preset_rejects_unknown_tts_pad_protocol(tmp_path: Path) -> None:
    """写错值（例如旧名 speak_first / bare）在 load 时就拦下，不拖到服务端。"""

    path = _preset(tmp_path)
    data = json.loads(path.read_text())
    data["tts_pad_protocol"] = "speak_first"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="legacy / revised"):
        O5FcSessionPreset.load(str(path))


def test_session_init_carries_policy_and_scheduling(tmp_path: Path) -> None:
    """session.init 必须显式携带 UnitPolicy、调度策略和 TTS prompt。"""

    spec = O5FcSessionPreset.load(str(_preset(tmp_path))).to_spec(
        server_url="http://h:1", task_prompt="TASK", tools=[]
    )
    payload = spec.session_init_frame()["payload"]
    assert payload["protocol_version"] == "3"
    assert payload["unit_policy"] == POLICY
    assert payload["config"]["non_spoken_scheduling"] == "quality"
    assert payload["config"]["decode_mode"] == "greedy"
    assert payload["config"]["tts_pad_protocol"] == "legacy"
    assert payload["generate_audio"] is True
    assert payload["tts_prompt_audio"]["file_path"].endswith("ref.wav")


def test_step_aggregates_one_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """一个 Unit 的 spoken/think/tool 事件被聚合，tool result 先于 input.append 上行。"""

    spec = O5FcSessionPreset.load(str(_preset(tmp_path))).to_spec(
        server_url="http://h:1", task_prompt="TASK", tools=[]
    )
    audio_out = np.arange(4, dtype=np.float32)
    fake = _FakeWebSocket(
        [
            {"type": "session.queue_done"},
            {"type": "session.created", "session_id": "s", "system_prefill": {"token_count": 9}},
            {"type": "response.unit.started", "unit_index": 0,
             "tool_events": [{"type": "tool_result", "tool_call_id": "tc_1"}]},
            {"type": "response.think.begin", "unit_index": 0},
            {"type": "response.think.delta", "unit_index": 0, "steps": [{"kind": "text", "text": "hm"}]},
            {"type": "response.think.end", "unit_index": 0, "full_text": "hm"},
            {"type": "response.tool_call.begin", "tool_call_id": "tc_2", "unit_index": 0},
            {"type": "response.tool_call.done", "tool_call_id": "tc_2", "unit_index": 0,
             "full_text": "<f/>", "call": {"name": "f", "arguments": {"a": 1}}},
            {"type": "response.spoken.delta", "unit_index": 0,
             "steps": [{"kind": "text", "text": "Hi"}, {"kind": "pending"}],
             "audio": base64.b64encode(audio_out.tobytes()).decode(), "sample_rate": 24000},
            {"type": "response.spoken.end", "unit_index": 0, "reason": "turn_eos"},
            {"type": "response.non_spoken.end", "unit_index": 0, "reason": "eos"},
            {"type": "response.unit.committed", "unit_index": 0},
        ]
    )
    monkeypatch.setattr(client, "connect", lambda *a, **k: fake)

    session = O5FcLockstepSession(spec)
    created = session.open()
    unit = session.step(np.zeros(16000, np.float32), [("tc_1", {"ok": True})])

    assert created["system_prefill"]["token_count"] == 9
    kinds = [frame["type"] for frame in fake.sent]
    assert kinds == ["session.init", "input.tool_result", "input.append"]
    assert unit.consumed_tool_call_ids == ["tc_1"]
    assert unit.spoken_text == "Hi"
    assert unit.think_text == "hm"
    assert unit.non_spoken_render == "<think>hm</think><tool_call></tool_call>"
    assert unit.end_of_turn and unit.is_speaking and not unit.is_listen
    assert unit.non_spoken_end_reason == "eos"
    assert [(c.name, c.arguments) for c in unit.tool_calls] == [("f", {"a": 1})]
    np.testing.assert_array_equal(unit.audio, audio_out)
    assert unit.audio_sample_rate == 24000


def test_step_raises_on_server_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """服务端 error 事件必须变成异常，不能被当成空 Unit 吞掉。"""

    spec = O5FcSessionPreset.load(str(_preset(tmp_path))).to_spec(
        server_url="http://h:1", task_prompt="TASK", tools=[]
    )
    fake = _FakeWebSocket(
        [
            {"type": "session.queue_done"},
            {"type": "session.created", "session_id": "s"},
            {"type": "error", "message": "cache full"},
        ]
    )
    monkeypatch.setattr(client, "connect", lambda *a, **k: fake)
    session = O5FcLockstepSession(spec)
    session.open()
    with pytest.raises(O5FcSessionError, match="cache full"):
        session.step(np.zeros(16000, np.float32))


@pytest.mark.parametrize("policy", ["raise_protocol_error", "constrained_decoding"])
def test_decoding_policy_reaches_session_init(tmp_path: Path, policy: str) -> None:
    """decoding_policy 由调用方按 benchmark 给（不在 preset 里），原样进 session.init.config。"""

    spec = O5FcSessionPreset.load(str(_preset(tmp_path))).to_spec(
        server_url="http://h:1", task_prompt="T", tools=[], decoding_policy=policy
    )
    assert spec.session_init_frame()["payload"]["config"]["decoding_policy"] == policy


def test_decoding_policy_defaults_to_raise_and_rejects_typos(tmp_path: Path) -> None:
    preset = O5FcSessionPreset.load(str(_preset(tmp_path)))
    spec = preset.to_spec(server_url="http://h:1", task_prompt="T", tools=[])
    assert spec.session_init_frame()["payload"]["config"]["decoding_policy"] == "raise_protocol_error"
    with pytest.raises(ValueError, match="decoding_policy"):
        preset.to_spec(server_url="http://h:1", task_prompt="T", tools=[], decoding_policy="constrained")
