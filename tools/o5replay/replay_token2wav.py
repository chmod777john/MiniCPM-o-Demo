#!/usr/bin/env python3
"""Replay one Token2Wav chunk from a recorded O5 session."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.o5replay.session_io import RecordedSession  # noqa: E402


OUTPUT_SAMPLE_RATE = 24000


def _frame(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("frame")
    return value if isinstance(value, dict) else {}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().to("cpu").contiguous().numpy().copy()


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _cuda_rng_state() -> np.ndarray:
    return _tensor_numpy(torch.cuda.get_rng_state())


def _flow_rand_noise_buffers(tokenizer: Any) -> list[tuple[str, torch.Tensor]]:
    return [
        (name, buffer)
        for name, buffer in tokenizer.flow.named_buffers()
        if "rand_noise" in name
    ]


def _copy_flow_rand_noise(tokenizer: Any, source_dir: Path) -> None:
    manifest_path = source_dir / "capture.json"
    noise_path = source_dir / "flow_rand_noise.npz"
    if not manifest_path.is_file() or not noise_path.is_file():
        raise FileNotFoundError(
            f"flow rand_noise capture is incomplete: expected {manifest_path} and {noise_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved = manifest.get("flow_rand_noise_buffers") or []
    source_arrays = np.load(noise_path)
    current = dict(_flow_rand_noise_buffers(tokenizer))
    with torch.no_grad():
        for item in saved:
            name = str(item["name"])
            if name not in current:
                raise KeyError(f"captured rand_noise buffer is absent in current Token2wav: {name}")
            value = torch.from_numpy(np.asarray(source_arrays[str(item["key"])]))
            target = current[name]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(f"rand_noise shape mismatch for {name}: {value.shape} vs {target.shape}")
            target.copy_(value.to(device=target.device, dtype=target.dtype))


class _Token2WavCapture:
    """Non-invasive hooks for the Flow and HiFT boundaries inside Token2wav.stream."""

    def __init__(self, tokenizer: Any) -> None:
        self.latest: dict[str, np.ndarray] = {}
        self._flow_inference_chunk = tokenizer.flow.inference_chunk
        self._hift_forward = tokenizer.hift.forward

        def flow_inference_chunk(*args: Any, **kwargs: Any) -> Any:
            result = self._flow_inference_chunk(*args, **kwargs)
            self.latest["chunk_mel"] = _tensor_numpy(result[0])
            return result

        def hift_forward(*args: Any, **kwargs: Any) -> Any:
            if args:
                self.latest["hift_input_mel"] = _tensor_numpy(args[0])
            elif "speech_feat" in kwargs:
                self.latest["hift_input_mel"] = _tensor_numpy(kwargs["speech_feat"])
            result = self._hift_forward(*args, **kwargs)
            self.latest["hift_speech"] = _tensor_numpy(result[0])
            self.latest["hift_source"] = _tensor_numpy(result[1])
            return result

        # These are instance-local method replacements. The installed model files are
        # untouched, and the hooks disappear with this Token2wav instance.
        tokenizer.flow.inference_chunk = flow_inference_chunk
        tokenizer.hift.forward = hift_forward

    def reset(self) -> None:
        self.latest = {}


def _write_capture(
    capture_dir: Path,
    *,
    session: RecordedSession,
    rows: list[dict[str, Any]],
    target_index: int,
    target: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
    calls: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    flow_rand_noise: list[tuple[str, torch.Tensor]],
) -> None:
    capture_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(capture_dir / "arrays.npz", **arrays)

    noise_arrays: dict[str, np.ndarray] = {}
    noise_manifest = []
    for index, (name, buffer) in enumerate(flow_rand_noise):
        key = f"buffer_{index}"
        value = _tensor_numpy(buffer)
        noise_arrays[key] = value
        noise_manifest.append({
            "name": name,
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _array_hash(value),
        })
    np.savez_compressed(capture_dir / "flow_rand_noise.npz", **noise_arrays)

    manifest = {
        "schema_version": 1,
        "session": str(session.root),
        "target_stream_seq": rows[target_index].get("seq"),
        "target_turn_id": target.get("turn_id"),
        "target_input_range": target.get("input_range"),
        "target_output_sample_range": target.get("output_sample_range"),
        "replayed_t2w_calls": len(calls),
        "target_token_count_including_lookahead": len(target.get("input_token_ids", [])),
        "seed": seed,
        "n_timesteps": args.n_timesteps,
        "float16": bool(args.float16),
        "perturb_token_index": args.perturb_token_index,
        "perturb_token_id": args.perturb_token_id if args.perturb_token_index is not None else None,
        "flow_rand_noise_buffers": noise_manifest,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_hash(value),
            }
            for name, value in arrays.items()
        },
        "calls": calls,
    }
    (capture_dir / "capture.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _load_stream_rng(source_dir: Path, call_number: int) -> None:
    path = source_dir / f"call_{call_number:03d}_cuda_rng_before.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing captured CUDA RNG state: {path}")
    torch.cuda.set_rng_state(torch.from_numpy(np.load(path)))


def _find_target_audio_index(
    rows: list[dict[str, Any]],
    *,
    target_audio: str | None,
    target_text: str | None,
) -> int:
    if target_audio:
        expected = Path(target_audio.removeprefix("@blob/")).name
        for index, row in enumerate(rows):
            frame = _frame(row)
            pointer = str(frame.get("audio") or "")
            if frame.get("kind") == "audio" and Path(pointer.removeprefix("@blob/")).name == expected:
                return index
        raise ValueError(f"audio chunk not found in session: {target_audio}")

    assert target_text
    needle = target_text.casefold()
    for text_index, row in enumerate(rows):
        frame = _frame(row)
        if frame.get("kind") != "text" or needle not in str(frame.get("text") or "").casefold():
            continue
        response_id = frame.get("response_id")
        for audio_index in range(text_index + 1, len(rows)):
            candidate = _frame(rows[audio_index])
            if candidate.get("kind") == "text":
                break
            if candidate.get("kind") == "audio" and candidate.get("response_id") == response_id:
                return audio_index
        raise ValueError(f"text matched but no following audio chunk was found: {target_text!r}")
    raise ValueError(f"text not found in session: {target_text!r}")


def _find_t2w_index(
    rows: list[dict[str, Any]],
    *,
    audio_index: int | None,
    target_t2w_seq: int | None,
) -> int:
    if target_t2w_seq is not None:
        for index, row in enumerate(rows):
            if int(row.get("seq", -1)) == target_t2w_seq and _frame(row).get("kind") in {
                "t2w.chunk",
                "token2wav.call",
            }:
                return index
        raise ValueError(f"T2W event not found at stream seq {target_t2w_seq}")

    assert audio_index is not None
    for index in range(audio_index - 1, -1, -1):
        if _frame(rows[index]).get("kind") in {"t2w.chunk", "token2wav.call"}:
            return index
    raise ValueError("no T2W event found before the target audio chunk")


def _turn_calls(rows: list[dict[str, Any]], target_index: int) -> list[tuple[int, dict[str, Any]]]:
    target = _frame(rows[target_index])
    turn_id = target.get("turn_id")
    calls = []
    for index, row in enumerate(rows[: target_index + 1]):
        frame = _frame(row)
        if frame.get("kind") not in {"t2w.chunk", "token2wav.call"}:
            continue
        if frame.get("turn_id") == turn_id:
            calls.append((index, frame))
    if not calls:
        raise ValueError(f"no T2W calls found for turn {turn_id}")
    return calls


def _write_pcm16(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(OUTPUT_SAMPLE_RATE)
        handle.writeframes(pcm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load only Token2Wav and reconstruct its streaming state from the start of the "
            "target turn before writing one selected audio chunk."
        )
    )
    parser.add_argument("--session-dir", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-audio", help="Audio blob name or pointer, for example 0039.wav")
    target.add_argument("--target-text", help="Text substring whose following audio chunk is selected")
    target.add_argument("--target-t2w-seq", type=int, help="Exact stream.jsonl seq of a T2W event")
    parser.add_argument("--output", required=True)
    parser.add_argument("--token2wav-dir", default="/user/weihongliang/o5_model_assets/token2wav")
    parser.add_argument("--ref-audio", default=None, help="Override the reference voice WAV from session.init")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-timesteps", type=int, default=10)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument(
        "--capture-dir",
        default=None,
        help="Save per-call Flow/HiFT tensors, RNG states, and flow rand_noise here",
    )
    parser.add_argument(
        "--flow-rand-noise-from",
        default=None,
        help="Reuse flow rand_noise captured by a previous --capture-dir run",
    )
    parser.add_argument(
        "--stream-rng-from",
        default=None,
        help="Restore each call's CUDA RNG state from a previous --capture-dir run",
    )
    parser.add_argument(
        "--perturb-token-index",
        type=int,
        default=None,
        help="Replace this target-call token index before T2W (0-based, includes lookahead)",
    )
    parser.add_argument(
        "--perturb-token-id",
        type=int,
        default=4218,
        help="Replacement token for --perturb-token-index (default: silence 4218)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Token2Wav replay requires a CUDA GPU")

    session = RecordedSession(Path(args.session_dir))
    rows = session.events
    audio_index = None
    if args.target_t2w_seq is None:
        audio_index = _find_target_audio_index(
            rows,
            target_audio=args.target_audio,
            target_text=args.target_text,
        )
    target_index = _find_t2w_index(
        rows,
        audio_index=audio_index,
        target_t2w_seq=args.target_t2w_seq,
    )
    calls = _turn_calls(rows, target_index)
    target = _frame(rows[target_index])

    seed = session.seed if args.seed is None else args.seed
    _seed_all(seed)
    from stepaudio2 import Token2wav

    tokenizer = Token2wav(
        str(Path(args.token2wav_dir).resolve()),
        float16=args.float16,
        n_timesteps=args.n_timesteps,
    )
    if args.flow_rand_noise_from:
        _copy_flow_rand_noise(tokenizer, Path(args.flow_rand_noise_from).resolve())

    capture = _Token2WavCapture(tokenizer) if args.capture_dir else None
    arrays: dict[str, np.ndarray] = {}
    captured_calls: list[dict[str, Any]] = []
    flow_rand_noise = _flow_rand_noise_buffers(tokenizer)
    if args.capture_dir:
        Path(args.capture_dir).resolve().mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="o5-t2w-replay-") as temp_dir:
        ref_audio = args.ref_audio or session.restore_ref_audio(Path(temp_dir))
        flow_cache, hift_cache = tokenizer.set_stream_cache(ref_audio)
        tokenizer.stream_cache = flow_cache
        tokenizer.hift_cache_dict = hift_cache

        target_pcm = b""
        for call_number, (row_index, call) in enumerate(calls):
            original_tokens = [int(token) for token in call.get("input_token_ids", [])]
            tokens = list(original_tokens)
            if not tokens:
                raise ValueError(f"T2W event has no input_token_ids: seq={rows[row_index].get('seq')}")
            if row_index == target_index and args.perturb_token_index is not None:
                if not 0 <= args.perturb_token_index < len(tokens):
                    raise ValueError(
                        f"perturb token index {args.perturb_token_index} is outside target input "
                        f"range of {len(tokens)} tokens"
                    )
                tokens[args.perturb_token_index] = args.perturb_token_id
            if args.stream_rng_from:
                _load_stream_rng(Path(args.stream_rng_from).resolve(), call_number)
            if capture:
                capture.reset()
            rng_before = _cuda_rng_state()
            output = tokenizer.stream(
                tokens,
                prompt_wav=ref_audio,
                last_chunk=bool(call.get("last_chunk", False)),
            )
            rng_after = _cuda_rng_state()
            if row_index == target_index:
                target_pcm = bytes(output)

            if capture:
                output_pcm = np.frombuffer(output, dtype="<i2").copy()
                prefix = f"call_{call_number:03d}"
                arrays[f"{prefix}_output_pcm"] = output_pcm
                for name, value in capture.latest.items():
                    arrays[f"{prefix}_{name}"] = value
                np.save(Path(args.capture_dir) / f"{prefix}_cuda_rng_before.npy", rng_before)
                np.save(Path(args.capture_dir) / f"{prefix}_cuda_rng_after.npy", rng_after)
                captured_calls.append({
                    "call_number": call_number,
                    "stream_seq": rows[row_index].get("seq"),
                    "last_chunk": bool(call.get("last_chunk", False)),
                    "input_token_ids": tokens,
                    "original_input_token_ids": original_tokens,
                    "input_range": call.get("input_range"),
                    "output_sample_range": call.get("output_sample_range"),
                    "cuda_rng_before_sha256": _array_hash(rng_before),
                    "cuda_rng_after_sha256": _array_hash(rng_after),
                    "captured_keys": sorted(capture.latest),
                })

    if not target_pcm:
        raise RuntimeError("target Token2Wav call returned no PCM data")

    output_path = Path(args.output).resolve()
    _write_pcm16(output_path, target_pcm)
    if args.capture_dir:
        _write_capture(
            Path(args.capture_dir).resolve(),
            session=session,
            rows=rows,
            target_index=target_index,
            target=target,
            seed=seed,
            args=args,
            calls=captured_calls,
            arrays=arrays,
            flow_rand_noise=flow_rand_noise,
        )
    summary = {
        "session": str(session.root),
        "target_stream_seq": rows[target_index].get("seq"),
        "target_turn_id": target.get("turn_id"),
        "target_input_range": target.get("input_range"),
        "target_output_sample_range": target.get("output_sample_range"),
        "replayed_t2w_calls": len(calls),
        "target_token_count_including_lookahead": len(target.get("input_token_ids", [])),
        "seed": seed,
        "output": str(output_path),
        "output_samples": len(target_pcm) // 2,
        "capture_dir": str(Path(args.capture_dir).resolve()) if args.capture_dir else None,
        "flow_rand_noise_buffers": [name for name, _ in flow_rand_noise],
        "perturb_token_index": args.perturb_token_index,
        "perturb_token_id": args.perturb_token_id if args.perturb_token_index is not None else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
