#!/usr/bin/env python3
"""Record or replay one exact session through Canonical, Demo single, or Demo TP2."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tracing import (  # noqa: E402
    DuplexTraceController,
    ForcingPolicy,
    MemoryTraceSink,
    ReplayReference,
    SessionBundleWriter,
)
from tools.o5replay.loaders import load_runtime, seed_all  # noqa: E402
from tools.o5replay.session_io import (  # noqa: E402
    OUTPUT_SAMPLE_RATE,
    RecordedSession,
    materialize_session_inputs,
    write_wav,
)


DEFAULT_CANONICAL_ROOT = (
    "/user/weihongliang/worktrees/"
    "moe-35b-a3b-canonical-tts-drift-investigation-2026-08-17"
)
DEFAULT_CKPT = (
    "/user/weihongliang/o5_weights/"
    "chenmoye_minicpm_5o_moe_omni_long_context_sft_stage2_sft2_8k_"
    "audio_online_process_on_online_audio_process_v2_iter_100.pt"
)


def _bool_override(value: bool | None, fallback: Any) -> bool:
    return bool(fallback) if value is None else value


def _sampling(args: argparse.Namespace, session: RecordedSession) -> dict[str, Any]:
    config = session.config

    def value(name: str, default: Any) -> Any:
        current = getattr(args, name, None)
        if current is not None:
            return current
        resolved = config.get(name)
        return default if resolved is None else resolved

    return {
        "generate_audio": _bool_override(args.generate_audio, config.get("generate_audio", True)),
        "ls_mode": str(value("ls_mode", "explicit")),
        "force_listen_count": int(value("force_listen_count", 0)),
        "max_new_speak_tokens_per_chunk": int(value("max_new_speak_tokens_per_chunk", 20)),
        "decode_mode": str(value("decode_mode", "sampling")),
        "temperature": float(value("temperature", 0.7)),
        "top_k": int(value("top_k", 100)),
        "top_p": float(value("top_p", 0.8)),
        "listen_prob_scale": float(value("listen_prob_scale", 1.0)),
        "text_repetition_penalty": float(value("text_repetition_penalty", 1.05)),
        "text_repetition_window_size": int(value("text_repetition_window_size", 512)),
        "tts_temperature": float(value("tts_temperature", 0.8)),
        "tts_repetition_penalty": float(value("tts_repetition_penalty", 1.05)),
        "n_timesteps": int(value("n_timesteps", 10)),
        "strategy_hd": _bool_override(args.strategy_hd, config.get("strategy_hd", False)),
        "strategy_hd_max_slice_nums": int(value("strategy_hd_max_slice_nums", 4)),
    }


def _result_value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _result_audio(result: Any) -> np.ndarray | None:
    waveform = _result_value(result, "audio_waveform")
    if waveform is not None:
        return np.asarray(waveform, dtype=np.float32).reshape(-1)
    encoded = _result_value(result, "audio_data")
    if not encoded:
        return None
    return np.frombuffer(base64.b64decode(encoded), dtype="<f4").astype(np.float32, copy=True)


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _prepare_out_dir(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir).resolve()
    reference = Path(args.reference_session).resolve() if args.reference_session else None
    if reference is not None and reference == out_dir:
        raise ValueError("--out-dir must differ from --reference-session")
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    return out_dir


def _is_tp2_worker(args: argparse.Namespace) -> bool:
    return args.target == "demo-tp2" and int(os.environ.get("RANK", "0")) != 0


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["O5_LAYER_TRACE"] = "1" if args.capture_layers else "0"

    session = RecordedSession(Path(args.session_dir))
    sampling = _sampling(args, session)
    forcing = ForcingPolicy.parse(args.forcing)
    if forcing.names() and not args.reference_session:
        raise ValueError("non-empty --forcing requires --reference-session")

    is_tp2_worker = _is_tp2_worker(args)
    out_dir = Path(args.out_dir).resolve()
    reference = None
    ref_audio_path = ""
    if not is_tp2_worker:
        out_dir = _prepare_out_dir(args)
        materialize_session_inputs(session.root, out_dir)
        ref_audio_path = session.restore_ref_audio(out_dir)
        reference = ReplayReference.load(Path(args.reference_session)) if forcing.names() else None

    seed = session.seed if args.seed is None else args.seed
    seed_all(seed)
    runtime = load_runtime(args, sampling)
    seed_all(seed)
    resolved_weights = (
        getattr(runtime.backend, "weights_dir", None)
        if args.target != "canonical"
        else None
    )
    controller = DuplexTraceController(
        sink=MemoryTraceSink(),
        capture_mode=args.capture_mode,
        reference=reference,
        forcing=forcing,
        tp_driver=args.target == "demo-tp2",
        capture_layers=args.capture_layers,
    ).install(runtime.duplex)
    session_id = args.session_id or f"replay-{args.target}-{int(time.time())}"
    controller.set_session(session_id)
    writer = SessionBundleWriter(
        out_dir,
        source_implementation=args.target,
        manifest_extra={
            "session_id": session_id,
            "capture_mode": args.capture_mode,
            "capture_layers": args.capture_layers,
            "reference_session": str(Path(args.reference_session).resolve()) if args.reference_session else None,
            "forcing": forcing.names(),
            "input_session": str(session.root),
            "seed": seed,
            "target": args.target,
            "canonical_root": str(Path(args.canonical_root).resolve()) if args.target == "canonical" else None,
            "checkpoint": str(Path(args.ckpt_path).resolve()) if args.target == "canonical" else None,
            "weights_dir": str(Path(resolved_weights).resolve()) if resolved_weights else None,
            "assets_dir": str(Path(args.assets_dir).resolve()) if args.assets_dir else None,
            "sampling": sampling,
        },
    )
    writer.append(controller.drain())

    outputs: list[dict[str, Any]] = []
    waveforms: list[np.ndarray] = []
    speak_waveforms: list[np.ndarray] = []
    output_log = out_dir / "replay_outputs.jsonl"
    units = session.units(args.max_units)
    completed = False
    try:
        runtime.prepare(
            system_prompt=session.system_prompt,
            ref_audio_path=ref_audio_path,
            seed=seed,
            sampling=sampling,
        )
        writer.append(controller.drain())
        with output_log.open("w", encoding="utf-8") as handle:
            for unit in units:
                controller.set_unit(unit.input_id, unit.index)
                started = time.perf_counter()
                prefill = runtime.prefill(
                    audio=unit.audio,
                    frames=unit.frames,
                    max_slice_nums=unit.max_slice_nums,
                )
                result = runtime.generate(
                    force_listen=unit.force_listen,
                    prompt_wav_path=ref_audio_path,
                    sampling=sampling,
                )
                runtime.finalize_unit()
                events = controller.drain(unit.input_id)
                writer.append(events)
                waveform = _result_audio(result)
                audio_path = None
                if waveform is not None and waveform.size:
                    audio_path = out_dir / "audio" / f"unit_{unit.index:06d}.wav"
                    write_wav(audio_path, waveform, OUTPUT_SAMPLE_RATE)
                    waveforms.append(waveform)
                    if not bool(_result_value(result, "is_listen", False)):
                        speak_waveforms.append(waveform)
                row = {
                    "unit_index": unit.index,
                    "input_id": unit.input_id,
                    "force_listen": unit.force_listen,
                    "prefill": _jsonable(prefill),
                    "output": {
                        "is_listen": bool(_result_value(result, "is_listen", False)),
                        "text": str(_result_value(result, "text", "") or ""),
                        "end_of_turn": bool(_result_value(result, "end_of_turn", False)),
                        "n_tokens": int(_result_value(result, "n_tokens", 0) or 0),
                        "n_tts_tokens": int(_result_value(result, "n_tts_tokens", 0) or 0),
                        "audio_path": audio_path.relative_to(out_dir).as_posix() if audio_path else None,
                        "audio_samples": int(waveform.size) if waveform is not None else 0,
                    },
                    "trace_event_count": len(events),
                    "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
                }
                outputs.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({"event": "unit", **row}, ensure_ascii=False), flush=True)
        if waveforms:
            write_wav(out_dir / "output_audio.wav", np.concatenate(waveforms), OUTPUT_SAMPLE_RATE)
        if speak_waveforms:
            write_wav(out_dir / "output_speak_audio.wav", np.concatenate(speak_waveforms), OUTPUT_SAMPLE_RATE)
        summary = {
            "schema": "o5.session-replay-result.v1",
            "target": args.target,
            "units": len(outputs),
            "text": "".join(row["output"]["text"] for row in outputs),
            "trace_events": sum(row["trace_event_count"] for row in outputs),
            "forcing": forcing.names(),
            "capture_mode": args.capture_mode,
            "reference_session": str(Path(args.reference_session).resolve()) if args.reference_session else None,
            "output_audio": "output_audio.wav" if waveforms else None,
            "output_speak_audio": "output_speak_audio.wav" if speak_waveforms else None,
        }
        (out_dir / "replay_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"event": "complete", "out_dir": str(out_dir), **summary}, ensure_ascii=False), flush=True)
        completed = True
        return 0
    finally:
        writer.append(controller.drain())
        writer.close(completed=completed)
        controller.uninstall()
        runtime.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target", choices=("canonical", "demo-single", "demo-tp2"), required=True)
    parser.add_argument("--reference-session")
    parser.add_argument("--forcing", default="none", help="none, all, or comma-separated llm,tts-condition,tts-token,vocoder")
    parser.add_argument("--capture-mode", choices=("tokens", "replay"), default="replay")
    parser.add_argument("--capture-layers", action="store_true")
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--session-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--canonical-root", default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--token2wav-dir", default="/user/weihongliang/o5_model_assets/token2wav")
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT)
    parser.add_argument("--weights-dir")
    parser.add_argument("--assets-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")

    parser.add_argument("--generate-audio", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ls-mode")
    parser.add_argument("--force-listen-count", type=int)
    parser.add_argument("--max-new-speak-tokens-per-chunk", type=int)
    parser.add_argument("--decode-mode", choices=("sampling", "greedy"))
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--listen-prob-scale", type=float)
    parser.add_argument("--text-repetition-penalty", type=float)
    parser.add_argument("--text-repetition-window-size", type=int)
    parser.add_argument("--tts-temperature", type=float)
    parser.add_argument("--tts-repetition-penalty", type=float)
    parser.add_argument("--n-timesteps", type=int)

    parser.add_argument(
        "--experts-implementation",
        choices=("eager", "batched_mm", "grouped_mm", "hybrid"),
        default="batched_mm",
    )
    parser.add_argument("--grouped-prefill-min-tokens", type=int, default=100)
    parser.add_argument("--strategy-hd", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--strategy-hd-max-slice-nums", type=int, default=None)
    parser.add_argument("--demo-single-mode", choices=("single_eager", "single_opt"), default="single_opt")
    parser.add_argument("--llm-cache", type=int, default=32768)
    parser.add_argument("--llm-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tts-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vocoder-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tts-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lmhead", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse-vision-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-vision-feed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
