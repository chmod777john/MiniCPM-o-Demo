"""Read and materialize exact realtime session inputs for offline replay."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from core.schemas.duplex import DuplexConfig


INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            rows.append(row)
    return rows


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _blob_path(session_dir: Path, pointer: str) -> Path:
    if not pointer.startswith("@blob/"):
        raise ValueError(f"expected @blob pointer, got {pointer!r}")
    root = session_dir.resolve()
    path = (root / pointer[1:]).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"blob pointer escapes session directory: {pointer}")
    return path


def _read_blob(session_dir: Path, pointer: str, expected_sha: Optional[str]) -> bytes:
    raw = _blob_path(session_dir, pointer).read_bytes()
    if expected_sha and _sha256(raw) != expected_sha:
        raise ValueError(f"sha256 mismatch for {pointer}")
    return raw


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return dict(value or {})


def _frame_payload(row: dict[str, Any]) -> dict[str, Any]:
    frame = row.get("frame")
    return frame if isinstance(frame, dict) else {}


def _input_payload(row: dict[str, Any]) -> dict[str, Any]:
    value = _frame_payload(row).get("input")
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class SessionUnit:
    index: int
    input_id: str
    audio: np.ndarray
    frames: list[Image.Image]
    force_listen: bool
    max_slice_nums: int


class RecordedSession:
    """An exact, ordered view of one gateway session bundle."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        stream_path = self.root / "stream.jsonl"
        if not stream_path.is_file():
            raise FileNotFoundError(f"stream.jsonl not found: {stream_path}")
        self.events = read_jsonl(stream_path)
        self.init_payload = self._session_init_payload()
        self.replay_manifest = self._replay_manifest()
        self.config = self._resolved_config()
        self.seed = self._resolved_seed()

    def _session_init_payload(self) -> dict[str, Any]:
        for row in self.events:
            frame = _frame_payload(row)
            if frame.get("type") == "session.init" and isinstance(frame.get("payload"), dict):
                return dict(frame["payload"])
        raise ValueError("recorded session has no session.init payload")

    def _replay_manifest(self) -> dict[str, Any]:
        for row in self.events:
            frame = _frame_payload(row)
            value = frame.get("replay_manifest")
            if frame.get("type") == "session.created" and isinstance(value, dict):
                return dict(value)
        return {}

    def _resolved_config(self) -> dict[str, Any]:
        resolved = self.replay_manifest.get("resolved")
        if isinstance(resolved, dict) and isinstance(resolved.get("duplex_config"), dict):
            return dict(resolved["duplex_config"])
        config = self.init_payload.get("config")
        values = dict(config) if isinstance(config, dict) else {}
        if "use_tts" in self.init_payload:
            values["generate_audio"] = bool(self.init_payload["use_tts"])
        return _model_dump(DuplexConfig(**values))

    def _resolved_seed(self) -> int:
        resolved = self.replay_manifest.get("resolved")
        resolved = resolved if isinstance(resolved, dict) else {}
        raw_config = self.init_payload.get("config")
        raw_config = raw_config if isinstance(raw_config, dict) else {}
        for value in (
            resolved.get("llm_seed"),
            resolved.get("seed"),
            self.init_payload.get("seed"),
            raw_config.get("seed"),
            self.config.get("seed"),
        ):
            if value is not None:
                return int(value)
        return 0

    @property
    def system_prompt(self) -> str:
        return str(self.init_payload.get("system_prompt") or "Streaming Omni Conversation.")

    def restore_ref_audio(self, out_dir: Path) -> str:
        path = self.init_payload.get("ref_audio_path")
        if path and Path(str(path)).is_file():
            return str(Path(str(path)).resolve())
        encoded = self.init_payload.get("ref_audio_base64")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("session.init needs a readable ref_audio_path or ref_audio_base64")
        samples = np.frombuffer(base64.b64decode(encoded), dtype="<f4")
        out_path = Path(out_dir) / "ref_audio_from_session.wav"
        write_wav(out_path, samples, INPUT_SAMPLE_RATE)
        return str(out_path)

    def input_rows(self, max_units: int = 0) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.events
            if row.get("dir") == "up" and _frame_payload(row).get("type") == "input.append"
        ]
        if not rows:
            raise ValueError("recorded session has no upstream input.append events")
        return rows[:max_units] if max_units > 0 else rows

    def units(self, max_units: int = 0) -> list[SessionUnit]:
        units: list[SessionUnit] = []
        for index, row in enumerate(self.input_rows(max_units)):
            trace = row.get("payload_trace")
            trace = trace if isinstance(trace, dict) else {}
            audio_meta = trace.get("audio")
            if not isinstance(audio_meta, dict) or not audio_meta.get("blob_f32"):
                raise ValueError(
                    f"input seq={row.get('seq')} has no payload_trace.audio.blob_f32"
                )
            if int(audio_meta.get("sample_rate", INPUT_SAMPLE_RATE)) != INPUT_SAMPLE_RATE:
                raise ValueError(f"input seq={row.get('seq')} is not 16 kHz")
            audio_raw = _read_blob(
                self.root,
                str(audio_meta["blob_f32"]),
                audio_meta.get("sha256"),
            )
            audio = np.frombuffer(audio_raw, dtype="<f4").astype(np.float32, copy=True)
            frames: list[Image.Image] = []
            for frame_meta in trace.get("video_frames") or []:
                if not isinstance(frame_meta, dict) or not frame_meta.get("blob_jpg"):
                    raise ValueError(f"input seq={row.get('seq')} has malformed video frame metadata")
                raw = _read_blob(
                    self.root,
                    str(frame_meta["blob_jpg"]),
                    frame_meta.get("sha256"),
                )
                frames.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            payload = _input_payload(row)
            units.append(SessionUnit(
                index=index,
                input_id=str(payload.get("input_id") or f"unit_{index:06d}"),
                audio=audio,
                frames=frames,
                force_listen=bool(payload.get("force_listen", trace.get("force_listen", False))),
                max_slice_nums=int(payload.get("max_slice_nums", trace.get("max_slice_nums", 1)) or 1),
            ))
        return units


def materialize_session_inputs(source: Path, destination: Path) -> None:
    """Copy immutable replay inputs into a newly produced trace bundle."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("stream.jsonl", "meta.json"):
        src = source / name
        if src.is_file():
            shutil.copy2(src, destination / name)
    src_blob = source / "blob"
    dst_blob = destination / "blob"
    if src_blob.is_dir():
        if dst_blob.exists():
            shutil.rmtree(dst_blob)
        shutil.copytree(src_blob, dst_blob, copy_function=shutil.copy2)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = OUTPUT_SAMPLE_RATE) -> None:
    data = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2", copy=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
