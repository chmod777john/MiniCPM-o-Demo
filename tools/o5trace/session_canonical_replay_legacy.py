#!/usr/bin/env python3
"""Replay older realtime sessions that store input media as @blob WAV/JPEG.

This wrapper intentionally does not modify `session_canonical_replay.py`.
It reuses the canonical replay driver and only swaps the old-session input
decoders:

- audio comes from `frame.input.audio = "@blob/xxxx.wav"`
- video frames come from `frame.input.video_frames = ["@blob/yyyy.jpg"]`

Old sessions usually lack `payload_trace` and `session.created.replay_manifest`,
so this is offline re-inference from recorded media, not exact demo reproduction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

import session_canonical_replay as base
from thin_duplex_video_probe import INPUT_SAMPLE_RATE


def _input_payload(event: dict[str, Any]) -> dict[str, Any]:
    frame = event.get("frame") if isinstance(event.get("frame"), dict) else {}
    payload = frame.get("input") if isinstance(frame.get("input"), dict) else {}
    return payload


def _blob_rel(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("@blob/"):
        raise ValueError(f"legacy input {field} must be an @blob pointer, got: {value!r}")
    return value


def _read_wav_blob(session_dir: Path, rel: str) -> np.ndarray:
    path = base.blob_path(session_dir, rel)
    if path.suffix.lower() != ".wav":
        raise ValueError(f"legacy audio replay expects WAV input blobs, got: {path}")
    audio = base.load_ref_audio(path)
    if audio.ndim != 1:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return audio.astype(np.float32, copy=False)


def event_audio(event: dict[str, Any], session_dir: Path) -> np.ndarray:
    payload = _input_payload(event)
    rel = _blob_rel(payload.get("audio"), field="input.audio")
    audio = _read_wav_blob(session_dir, rel)
    expected = INPUT_SAMPLE_RATE
    if audio.size == 0:
        raise ValueError(f"legacy input event seq={event.get('seq')} has empty audio blob: {rel}")
    # Old recorder stored exactly one second of 16 kHz float/int16 audio per
    # unit. The canonical model can handle shorter final chunks, so do not pad.
    if audio.size > expected * 2:
        raise ValueError(
            f"legacy input event seq={event.get('seq')} audio looks too long: "
            f"{audio.size} samples from {rel}"
        )
    return audio


def event_frames(event: dict[str, Any], session_dir: Path) -> list[Image.Image]:
    payload = _input_payload(event)
    values = payload.get("video_frames")
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"legacy input video_frames must be a list, got: {type(values).__name__}")
    frames: list[Image.Image] = []
    for value in values:
        if value is None:
            continue
        rel = _blob_rel(value, field="input.video_frames[]")
        frames.append(base.read_image_blob(session_dir, rel, expected_sha=None))
    return frames


def main() -> int:
    base.__doc__ = __doc__
    base.event_audio = event_audio
    base.event_frames = event_frames
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
