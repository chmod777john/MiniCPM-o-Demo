"""网关 API 往返数据录制(gateway_modules.session_recording)的单元测试。

只用 tmp_path,不起网关、不连 worker、不碰任何外部系统。
覆盖用户提出的硬要求:原封不动(可逐字节还原)、按 session_id 隔离目录、
每帧有收发时间、行序 = seq、两代协议的音频字段都覆盖。
"""

from __future__ import annotations

import base64
import json
import struct

import numpy as np
import pytest

from gateway_modules.session_recording import SessionRecorder, read_wav_data_bytes


def _b64_f32(n: int, seed: int = 0) -> tuple[str, bytes]:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(n).astype(np.float32).tobytes()
    return base64.b64encode(raw).decode("ascii"), raw


def _read_stream(rec: SessionRecorder) -> list[dict]:
    with open(f"{rec.directory}/stream.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _restore(rec: SessionRecorder, value: str) -> str:
    """按 docstring 的还原规则把 @blob 指针还原成原 base64 字段值。"""
    if isinstance(value, str) and value.startswith("@blob/"):
        return base64.b64encode(read_wav_data_bytes(f"{rec.directory}/{value[1:]}")).decode("ascii")
    return value


def _finish(rec: SessionRecorder) -> None:
    rec.close(reason="test")
    rec.wait_closed()


def test_layout_is_sessions_dir_slash_session_id(tmp_path):
    rec = SessionRecorder("sess_abc123", "full_duplex", sessions_dir=str(tmp_path / "api_trace"))
    rec.record("down", {"type": "session.created", "session_id": "sess_abc123"},
               recv_ts=1.0, send_ts=1.1, src="worker")
    _finish(rec)
    sdir = tmp_path / "api_trace" / "sess_abc123"
    assert (sdir / "stream.jsonl").is_file()
    meta = json.loads((sdir / "meta.json").read_text())
    assert meta["session_id"] == "sess_abc123"
    assert meta["close_reason"] == "test"
    assert meta["frames"] == 1


def test_default_layout_keeps_data_dir_sessions(tmp_path):
    rec = SessionRecorder("sess_x", "full_duplex", data_dir=str(tmp_path))
    _finish(rec)
    assert (tmp_path / "sessions" / "sess_x" / "meta.json").is_file()


def test_v3_input_audio_base64_is_lossless(tmp_path):
    b64, raw = _b64_f32(16000, seed=1)
    frame = {"type": "input.append",
             "input": {"input_id": "unit_000000", "audio_base64": b64, "sample_rate": 16000}}
    rec = SessionRecorder("s1", "full_duplex", sessions_dir=str(tmp_path))
    rec.record("up", frame, recv_ts=10.0, send_ts=10.002, src="client")
    _finish(rec)

    (line,) = _read_stream(rec)
    ptr = line["frame"]["input"]["audio_base64"]
    assert ptr.startswith("@blob/") and ptr.endswith(".wav")
    assert read_wav_data_bytes(f"{rec.directory}/{ptr[1:]}") == raw
    assert _restore(rec, ptr) == b64
    # 其余字段原样
    assert line["frame"]["input"]["input_id"] == "unit_000000"
    assert line["frame"]["input"]["sample_rate"] == 16000
    # 原 frame 对象未被修改
    assert frame["input"]["audio_base64"] == b64


def test_wav_is_float32_with_frame_sample_rate(tmp_path):
    b64, _ = _b64_f32(24000, seed=2)
    rec = SessionRecorder("s2", "full_duplex", sessions_dir=str(tmp_path))
    rec.record("down", {"type": "response.spoken.delta", "unit_index": 3,
                        "audio": b64, "sample_rate": 24000},
               recv_ts=1.0, send_ts=1.0, src="worker")
    _finish(rec)
    (line,) = _read_stream(rec)
    with open(f"{rec.directory}/{line['frame']['audio'][1:]}", "rb") as f:
        head = f.read(44)
    fmt_tag, channels, sr = struct.unpack("<HHI", head[20:28])
    bits = struct.unpack("<H", head[34:36])[0]
    assert (fmt_tag, channels, sr, bits) == (3, 1, 24000, 32)


def test_legacy_protocol_audio_fields_also_lossless(tmp_path):
    b64_up, raw_up = _b64_f32(800, seed=3)
    b64_dn, raw_dn = _b64_f32(1200, seed=4)
    rec = SessionRecorder("s3", "full_duplex", sessions_dir=str(tmp_path))
    rec.record("up", {"type": "input.append", "input": {"audio": b64_up}},
               recv_ts=1.0, send_ts=1.0, src="client")
    rec.record("down", {"type": "response.output.delta", "kind": "audio", "audio": b64_dn},
               recv_ts=2.0, send_ts=2.0, src="worker")
    _finish(rec)
    up, dn = _read_stream(rec)
    assert read_wav_data_bytes(f"{rec.directory}/{up['frame']['input']['audio'][1:]}") == raw_up
    assert read_wav_data_bytes(f"{rec.directory}/{dn['frame']['audio'][1:]}") == raw_dn


@pytest.mark.parametrize("bad", [
    "not-base64!!" * 20,                                  # 非法字符
    base64.b64encode(b"\x00" * 99).decode() + "\n",      # 带换行(非标准)
    base64.b64encode(b"\x01\x02\x03" * 30).decode(),     # 字节数不是 4 的倍数
])
def test_unrestorable_audio_stays_inline(tmp_path, bad):
    rec = SessionRecorder("s4", "full_duplex", sessions_dir=str(tmp_path))
    rec.record("down", {"type": "response.spoken.delta", "audio": bad},
               recv_ts=1.0, send_ts=1.0, src="worker")
    _finish(rec)
    (line,) = _read_stream(rec)
    assert line["frame"]["audio"] == bad


def test_non_audio_frames_are_verbatim(tmp_path):
    frame = {"type": "response.think.delta", "unit_index": 7,
             "steps": [{"kind": "text", "text": " user"}, {"kind": "pending"}],
             "server_send_ts": 1759.25}
    rec = SessionRecorder("s5", "full_duplex", sessions_dir=str(tmp_path))
    rec.record("down", frame, recv_ts=1.0, send_ts=1.5, src="worker")
    _finish(rec)
    (line,) = _read_stream(rec)
    assert line["frame"] == frame


def test_timestamps_direction_and_source(tmp_path):
    rec = SessionRecorder("s6", "full_duplex", sessions_dir=str(tmp_path))
    rec.record("down", {"type": "session.queued", "position": 2},
               recv_ts=5.0, send_ts=5.01, src="gateway")
    rec.record("up", {"type": "session.init"}, recv_ts=6.0, send_ts=None, src="client")
    _finish(rec)
    a, b = _read_stream(rec)
    assert (a["dir"], a["src"], a["recv_ts"], a["send_ts"], a["ts"]) == ("down", "gateway", 5.0, 5.01, 5.0)
    assert (b["dir"], b["src"], b["recv_ts"], b["send_ts"]) == ("up", "client", 6.0, None)


def test_file_order_equals_seq_under_load(tmp_path):
    b64, _ = _b64_f32(4000, seed=5)
    rec = SessionRecorder("s7", "full_duplex", sessions_dir=str(tmp_path))
    for i in range(300):
        frame = ({"type": "response.spoken.delta", "unit_index": i, "audio": b64}
                 if i % 2 else {"type": "response.unit.committed", "unit_index": i})
        rec.record("down", frame, recv_ts=float(i), send_ts=float(i), src="worker")
    _finish(rec)
    lines = _read_stream(rec)
    assert [x["seq"] for x in lines] == list(range(300))
    assert [x["frame"]["unit_index"] for x in lines] == list(range(300))
    blobs = [x["frame"]["audio"] for x in lines if "audio" in x["frame"]]
    assert len(set(blobs)) == 150                       # 每段音频一个独立 blob


def test_record_after_close_is_ignored(tmp_path):
    rec = SessionRecorder("s8", "full_duplex", sessions_dir=str(tmp_path))
    rec.record("down", {"type": "a"}, recv_ts=1.0, send_ts=1.0, src="worker")
    rec.close(reason="x")
    rec.record("down", {"type": "late"}, recv_ts=2.0, send_ts=2.0, src="worker")
    rec.wait_closed()
    assert [x["frame"]["type"] for x in _read_stream(rec)] == ["a"]
