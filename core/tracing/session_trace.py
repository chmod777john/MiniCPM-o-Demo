"""Implementation-neutral tracing and teacher-forced replay for O5 duplex.

The controller patches a concrete ``MiniCPMODuplex`` instance.  It deliberately
does not import either the Demo or Canonical modeling package, so both loaders
can share the exact same recorder and forcing semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch


TRACE_SCHEMA = "o5.session-trace.v1"
_TENSOR_POINTER = "@trace-tensor/"


def _flat_ints(value: Any) -> list[int]:
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().reshape(-1).cpu().tolist()]
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in np.asarray(value).reshape(-1).tolist()]
    return [int(value)]


def _tensor_meta(value: torch.Tensor, *, keep_value: bool) -> dict[str, Any]:
    tensor = value.detach().contiguous().cpu()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    record: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if keep_value:
        record["_tensor"] = tensor.clone()
    return record


def _token_meta(value: Any) -> dict[str, Any]:
    if torch.is_tensor(value):
        shape = list(value.shape)
    else:
        shape = list(np.asarray(value).shape)
    tokens = _flat_ints(value)
    raw = np.asarray(tokens, dtype="<i8").tobytes()
    return {
        "shape": shape,
        "numel": len(tokens),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "tokens": tokens,
    }


def _result_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def _safe_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text[:120] or "unknown"


class MemoryTraceSink:
    """Thread-safe event queue used by the API and small offline runners."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    def drain(self, input_id: Optional[str] = None) -> list[dict[str, Any]]:
        with self._lock:
            if input_id is None:
                events, self._events = self._events, []
                return events
            selected: list[dict[str, Any]] = []
            retained: list[dict[str, Any]] = []
            for event in self._events:
                (selected if event.get("input_id") == input_id else retained).append(event)
            self._events = retained
            return selected

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class SessionBundleWriter:
    """Append trace events and externalize tensor payloads under a session dir."""

    def __init__(
        self,
        root: Path,
        *,
        source_implementation: str,
        manifest_extra: Optional[dict[str, Any]] = None,
    ) -> None:
        self.root = Path(root)
        self.tensor_dir = self.root / "trace_tensors"
        self.events_path = self.root / "model_trace.jsonl"
        self.manifest_path = self.root / "trace_manifest.json"
        self.source_implementation = source_implementation
        self.manifest_extra = dict(manifest_extra or {})
        self._lock = threading.Lock()
        self._tensor_index = 0
        self.root.mkdir(parents=True, exist_ok=True)
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(completed=False)

    def _write_manifest(self, *, completed: bool) -> None:
        payload = {
            "schema": TRACE_SCHEMA,
            "source_implementation": self.source_implementation,
            "completed": completed,
            "updated_at": time.time(),
            **self.manifest_extra,
        }
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    def _externalize(self, value: Any, *, stem: str) -> Any:
        if torch.is_tensor(value):
            name = f"{self._tensor_index:08d}_{_safe_component(stem)}.pt"
            self._tensor_index += 1
            torch.save(value.detach().contiguous().cpu(), self.tensor_dir / name)
            return f"{_TENSOR_POINTER}{name}"
        if isinstance(value, Mapping):
            return {
                str(key): self._externalize(item, stem=f"{stem}_{key}")
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._externalize(item, stem=f"{stem}_{index}") for index, item in enumerate(value)]
        if isinstance(value, np.generic):
            return value.item()
        return value

    def append(self, events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        with self._lock:
            for event in events:
                item = self._externalize(event, stem=f"event_{event.get('event_id', 'unknown')}")
                serialized.append(item)
            if serialized:
                with self.events_path.open("a", encoding="utf-8") as handle:
                    for item in serialized:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    handle.flush()
        return serialized

    def close(self, *, completed: bool = True) -> None:
        with self._lock:
            self._write_manifest(completed=completed)


@dataclass(frozen=True)
class ForcingPolicy:
    llm_tokens: bool = False
    tts_condition: bool = False
    tts_tokens: bool = False
    vocoder_state: bool = False

    @classmethod
    def parse(cls, value: str | Iterable[str] | None) -> "ForcingPolicy":
        if value is None:
            names: set[str] = set()
        elif isinstance(value, str):
            names = {item.strip().replace("_", "-") for item in value.split(",") if item.strip()}
        else:
            names = {str(item).strip().replace("_", "-") for item in value if str(item).strip()}
        if "none" in names:
            names.clear()
        if "all" in names:
            names = {"llm", "tts-condition", "tts-token", "vocoder"}
        known = {"llm", "tts-condition", "tts-token", "vocoder"}
        unknown = names - known
        if unknown:
            raise ValueError(f"unknown forcing stages: {sorted(unknown)}")
        return cls(
            llm_tokens="llm" in names,
            tts_condition="tts-condition" in names,
            tts_tokens="tts-token" in names,
            vocoder_state="vocoder" in names,
        )

    def names(self) -> list[str]:
        out = []
        if self.llm_tokens:
            out.append("llm")
        if self.tts_condition:
            out.append("tts-condition")
        if self.tts_tokens:
            out.append("tts-token")
        if self.vocoder_state:
            out.append("vocoder")
        return out


def _load_pointer(root: Path, value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_TENSOR_POINTER):
        return torch.load(root / "trace_tensors" / value[len(_TENSOR_POINTER):], map_location="cpu", weights_only=True)
    if isinstance(value, Mapping):
        return {key: _load_pointer(root, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_load_pointer(root, item) for item in value]
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_session_trace_events(root: Path, *, load_tensors: bool = False) -> list[dict[str, Any]]:
    """Load native model trace, falling back to inline API response traces."""

    root = Path(root)
    events = _read_jsonl(root / "model_trace.jsonl")
    if not events:
        for row in _read_jsonl(root / "stream.jsonl"):
            frame = row.get("frame") if isinstance(row.get("frame"), dict) else {}
            trace = frame.get("trace") if isinstance(frame, dict) else None
            if not isinstance(trace, dict):
                continue
            for group in ("llm", "tts", "token2wav", "runtime"):
                values = trace.get(group)
                if isinstance(values, list):
                    events.extend(item for item in values if isinstance(item, dict))
    if load_tensors:
        return [_load_pointer(root, event) for event in events]
    return events


class ReplayReference:
    """Indexed forcing decisions and tensors loaded from a session bundle."""

    def __init__(self, events: Iterable[dict[str, Any]]) -> None:
        self.events = list(events)
        self._by_kind: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for event in self.events:
            kind = str(event.get("kind") or "")
            input_id = str(event.get("input_id") or "")
            self._by_kind.setdefault(kind, {}).setdefault(input_id, []).append(event)
        self._cursor: dict[tuple[str, str], int] = {}

    def reset(self) -> None:
        self._cursor.clear()

    @classmethod
    def load(cls, root: Path) -> "ReplayReference":
        return cls(load_session_trace_events(Path(root), load_tensors=True))

    def _next(self, kind: str, input_id: Optional[str]) -> dict[str, Any]:
        key = str(input_id or "")
        rows = self._by_kind.get(kind, {}).get(key, [])
        cursor_key = (kind, key)
        index = self._cursor.get(cursor_key, 0)
        if index >= len(rows):
            raise RuntimeError(f"reference exhausted: kind={kind} input_id={key} index={index}")
        self._cursor[cursor_key] = index + 1
        return rows[index]

    def next_llm_token(self, input_id: Optional[str]) -> int:
        return int(self._next("llm.decode", input_id)["selected_token_id"])

    def next_condition(self, input_id: Optional[str]) -> torch.Tensor:
        row = self._next("tts.condition", input_id)
        tensor = (row.get("used_condition") or {}).get("_tensor")
        if not torch.is_tensor(tensor):
            raise RuntimeError("reference TTS condition has no tensor payload; record with trace mode=replay")
        return tensor

    def next_tts_chunk_tokens(self, input_id: Optional[str]) -> torch.Tensor:
        row = self._next("tts.chunk", input_id)
        tensor = (row.get("new_tokens") or {}).get("_tensor")
        if torch.is_tensor(tensor):
            return tensor
        tokens = (row.get("new_tokens") or {}).get("tokens")
        shape = (row.get("new_tokens") or {}).get("shape")
        if tokens is None or shape is None:
            raise RuntimeError("reference TTS chunk has no token payload")
        return torch.tensor(tokens, dtype=torch.long).reshape(shape)

    def next_tts_sample_tokens(self, input_id: Optional[str], *, expected_step: int) -> list[int]:
        row = self._next("tts.sample", input_id)
        reference_step = row.get("step")
        if reference_step is not None and int(reference_step) != expected_step:
            raise RuntimeError(
                f"reference TTS sample step mismatch: reference={reference_step} actual={expected_step}"
            )
        tokens = row.get("selected_token_ids")
        if not isinstance(tokens, list) or not tokens:
            raise RuntimeError("reference TTS sample has no selected token IDs")
        return [int(token) for token in tokens]

    def vocoder_noise(self) -> Optional[torch.Tensor]:
        rows = self._by_kind.get("vocoder.state", {}).get("", [])
        if not rows:
            rows = [event for event in self.events if event.get("kind") == "vocoder.state"]
        if not rows:
            return None
        value = (rows[0].get("rand_noise") or {}).get("_tensor")
        return value if torch.is_tensor(value) else None


class DuplexTraceController:
    """Patch one live Duplex instance for trace capture and optional replay."""

    def __init__(
        self,
        *,
        sink: Optional[MemoryTraceSink] = None,
        capture_mode: str = "tokens",
        reference: Optional[ReplayReference] = None,
        forcing: Optional[ForcingPolicy] = None,
        tp_driver: bool = False,
        capture_layers: bool = False,
    ) -> None:
        if capture_mode not in {"tokens", "replay"}:
            raise ValueError("capture_mode must be 'tokens' or 'replay'")
        self.sink = sink or MemoryTraceSink()
        self.capture_mode = capture_mode
        self.reference = reference
        self.forcing = forcing or ForcingPolicy()
        self.tp_driver = tp_driver
        self.capture_layers = capture_layers
        self.session_id: Optional[str] = None
        self.input_id: Optional[str] = None
        self.unit_index: Optional[int] = None
        self.turn_id = 0
        self._event_id = 0
        self._originals: list[tuple[Any, str, Any]] = []
        self._original_multinomial: Any = None
        self._force_tts_samples = False
        self._active_tts_step = 0
        self._inside_tts_chunk = False
        self._vocoder_rand_noise: Optional[torch.Tensor] = None
        self._vocoder_forced = False
        self._t2w_token_pos = 0
        self._t2w_sample_pos = 0
        self._installed = False
        self._layer_modules: list[Any] = []
        self._layer_hooks: list[Any] = []
        self._layer_outputs: dict[int, torch.Tensor] = {}

    def set_session(self, session_id: Optional[str]) -> None:
        self.session_id = session_id
        if session_id is not None:
            if self.reference is not None:
                self.reference.reset()
            self.turn_id = 0
            self._t2w_token_pos = 0
            self._t2w_sample_pos = 0
            self.sink.clear()
            if self._vocoder_rand_noise is not None:
                self._emit_vocoder_state()

    def set_unit(self, input_id: Optional[str], unit_index: Optional[int] = None) -> None:
        self.input_id = input_id
        self.unit_index = unit_index

    def _identity(self, kind: str) -> dict[str, Any]:
        event_id = self._event_id
        self._event_id += 1
        return {
            "schema": TRACE_SCHEMA,
            "event_id": event_id,
            "kind": kind,
            "session_id": self.session_id,
            "input_id": self.input_id,
            "unit_index": self.unit_index,
            "turn_id": self.turn_id,
            "ts": time.time(),
        }

    def _emit(self, kind: str, **fields: Any) -> None:
        self.sink.emit({**self._identity(kind), **fields})

    def _save_original(self, owner: Any, name: str) -> Any:
        original = getattr(owner, name)
        self._originals.append((owner, name, original))
        return original

    @staticmethod
    def _first_tensor(output: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            return next((item for item in output if torch.is_tensor(item)), None)
        value = getattr(output, "last_hidden_state", None)
        return value if torch.is_tensor(value) else None

    @staticmethod
    def _find_layers(llm: Any) -> list[Any]:
        candidates = [llm, getattr(llm, "inner", None)]
        for parent in list(candidates):
            if parent is None:
                continue
            candidates.extend([
                getattr(parent, "model", None),
                getattr(getattr(parent, "inner", None), "model", None),
            ])
        for candidate in candidates:
            layers = getattr(candidate, "layers", None) if candidate is not None else None
            if layers is not None and len(layers) > 1:
                return list(layers)
        raise RuntimeError("could not locate decoder layers for trace capture")

    def _install_layer_hooks(self, duplex: Any) -> None:
        if not self.capture_layers:
            return
        self._layer_modules = self._find_layers(duplex.decoder.m)
        for index, layer in enumerate(self._layer_modules):
            def capture(_module, _inputs, output, index=index):
                tensor = self._first_tensor(output)
                if tensor is not None:
                    self._layer_outputs[index] = tensor.detach()

            self._layer_hooks.append(layer.register_forward_hook(capture))

    def _consume_layer_outputs(self, decoder: Any) -> tuple[list[torch.Tensor], str]:
        runner = getattr(decoder, "_llm_runner", None)
        graph_outputs = getattr(runner, "layer_outputs", None) if runner is not None else None
        if graph_outputs and len(graph_outputs) == len(self._layer_modules):
            self._layer_outputs.clear()
            return list(graph_outputs), "llm_graph"
        if len(self._layer_outputs) == len(self._layer_modules):
            outputs = [self._layer_outputs[index] for index in range(len(self._layer_modules))]
            self._layer_outputs.clear()
            return outputs, "forward_hooks"
        available = sorted(self._layer_outputs)
        self._layer_outputs.clear()
        raise RuntimeError(
            f"layer trace incomplete: expected={len(self._layer_modules)} available={available}"
        )

    def install(self, duplex: Any) -> "DuplexTraceController":
        if self._installed:
            raise RuntimeError("trace controller is already installed")
        self._installed = True
        self.duplex = duplex
        self._install_llm(duplex)
        self._install_condition(duplex)
        self._install_tts(duplex)
        self._install_token2wav(duplex)
        self._capture_or_force_vocoder(duplex)
        return self

    def _install_llm(self, duplex: Any) -> None:
        decoder = duplex.decoder
        original_feed = self._save_original(decoder, "feed") if self.capture_mode == "replay" else None
        original_decode = self._save_original(decoder, "decode")
        original_generate = self._save_original(duplex, "streaming_generate")
        trace = self
        self._install_layer_hooks(duplex)

        def traced_feed(decoder_self, embeds, return_logits=False):
            cache_before = int(decoder_self.get_cache_length()) if hasattr(decoder_self, "get_cache_length") else None
            trace._layer_outputs.clear()
            result = original_feed(embeds, return_logits=True)
            if result is None:
                raise RuntimeError("decoder.feed(return_logits=True) returned no result")
            logits, hidden = result
            event: dict[str, Any] = {
                "requested_return_logits": bool(return_logits),
                "cache_before": cache_before,
                "cache_after": (
                    int(decoder_self.get_cache_length())
                    if hasattr(decoder_self, "get_cache_length")
                    else None
                ),
                "embeds": _tensor_meta(embeds, keep_value=True),
                "hidden": _tensor_meta(hidden, keep_value=True),
                "logits": _tensor_meta(logits, keep_value=True),
            }
            if trace.capture_layers:
                layers, source = trace._consume_layer_outputs(decoder_self)
                event["layer_source"] = source
                event["layers"] = [_tensor_meta(layer, keep_value=True) for layer in layers]
            trace._emit("llm.feed", **event)
            return (logits, hidden) if return_logits else None

        def traced_decode(decoder_self, logits, *args, **kwargs):
            local_argmax = int(logits.detach().reshape(-1).argmax().item())
            forced = trace.forcing.llm_tokens
            if forced:
                if trace.reference is None:
                    raise RuntimeError("LLM forcing requested without a replay reference")
                selected = torch.tensor(
                    [trace.reference.next_llm_token(trace.input_id)],
                    device=logits.device,
                    dtype=torch.long,
                )
                if trace.tp_driver and torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.broadcast(selected, src=0)
            else:
                selected = original_decode(logits, *args, **kwargs)
            selected_id = int(selected.detach().reshape(-1)[0].item())
            trace._emit(
                "llm.decode",
                selected_token_id=selected_id,
                local_argmax_token_id=local_argmax,
                teacher_forced=forced,
                logits=_tensor_meta(logits, keep_value=trace.capture_mode == "replay"),
            )
            return selected

        def traced_generate(duplex_self, *args, **kwargs):
            before = len(getattr(duplex_self, "total_ids", []))
            result = original_generate(*args, **kwargs)
            accepted = list(getattr(duplex_self, "total_ids", []))[before:]
            trace._emit(
                "llm.accepted",
                token_ids=[int(item) for item in accepted],
                text=str(_result_value(result, "text", "") or ""),
                is_listen=bool(_result_value(result, "is_listen", False)),
                end_of_turn=bool(_result_value(result, "end_of_turn", False)),
            )
            if bool(_result_value(result, "end_of_turn", False)):
                trace.turn_id += 1
            return result

        if original_feed is not None:
            decoder.feed = types.MethodType(traced_feed, decoder)
        decoder.decode = types.MethodType(traced_decode, decoder)
        duplex.streaming_generate = types.MethodType(traced_generate, duplex)

    def _install_condition(self, duplex: Any) -> None:
        original_convert = self._save_original(duplex, "_convert_results_to_tts_input")
        trace = self

        def traced_convert(duplex_self, results):
            actual = original_convert(results)
            used = actual
            forced = trace.forcing.tts_condition
            if forced:
                if trace.reference is None:
                    raise RuntimeError("TTS condition forcing requested without a replay reference")
                used = trace.reference.next_condition(trace.input_id).to(device=actual.device, dtype=actual.dtype)
                if used.shape != actual.shape:
                    raise RuntimeError(
                        f"TTS condition shape mismatch: reference={tuple(used.shape)} actual={tuple(actual.shape)}"
                    )
            trace._emit(
                "tts.condition",
                llm_token_ids=[int(item[0]) for item in results],
                end_of_turn=[bool(item[2]) for item in results],
                actual_condition=_tensor_meta(actual, keep_value=trace.capture_mode == "replay"),
                used_condition=_tensor_meta(used, keep_value=trace.capture_mode == "replay"),
                teacher_forced=forced,
            )
            return used

        duplex._convert_results_to_tts_input = types.MethodType(traced_convert, duplex)

    def _install_tts(self, duplex: Any) -> None:
        tts = getattr(getattr(duplex, "model", None), "tts", None)
        if tts is None or not hasattr(tts, "generate_chunk"):
            return
        original_generate = self._save_original(tts, "generate_chunk")
        self._original_multinomial = torch.multinomial
        trace = self

        def traced_multinomial(input_tensor, num_samples, replacement=False, *, generator=None, out=None):
            if not trace._inside_tts_chunk or num_samples != 1:
                return trace._original_multinomial(
                    input_tensor,
                    num_samples,
                    replacement=replacement,
                    generator=generator,
                    out=out,
                )
            step = trace._active_tts_step
            if trace._force_tts_samples:
                if trace.reference is None:
                    raise RuntimeError("TTS token forcing requested without a replay reference")
                selected = torch.tensor(
                    trace.reference.next_tts_sample_tokens(trace.input_id, expected_step=step),
                    device=input_tensor.device,
                    dtype=torch.long,
                ).reshape(-1, 1)
                if selected.shape[0] != input_tensor.shape[0]:
                    raise RuntimeError(
                        f"TTS codebook mismatch: reference={selected.shape[0]} actual={input_tensor.shape[0]}"
                    )
            else:
                selected = trace._original_multinomial(
                    input_tensor,
                    num_samples,
                    replacement=replacement,
                    generator=generator,
                    out=out,
                )
            trace._active_tts_step += 1
            if out is not None and trace._force_tts_samples:
                out.copy_(selected)
                selected = out
            trace._emit(
                "tts.sample",
                step=step,
                selected_token_ids=_flat_ints(selected),
                teacher_forced=trace._force_tts_samples,
                probabilities=_tensor_meta(input_tensor, keep_value=trace.capture_mode == "replay"),
            )
            return selected

        torch.multinomial = traced_multinomial

        def traced_generate(tts_self, *args, **kwargs):
            forced = trace.forcing.tts_tokens
            trace._active_tts_step = 0
            if forced:
                if trace.reference is None:
                    raise RuntimeError("TTS token forcing requested without a replay reference")
            trace._force_tts_samples = forced
            trace._inside_tts_chunk = True
            try:
                new_tokens, cache = original_generate(*args, **kwargs)
            finally:
                trace._inside_tts_chunk = False
                trace._force_tts_samples = False
            trace._emit(
                "tts.chunk",
                new_tokens={
                    **_token_meta(new_tokens),
                    **({"_tensor": new_tokens.detach().contiguous().cpu().clone()} if trace.capture_mode == "replay" else {}),
                },
                teacher_forced=forced,
                text_start_pos=int(kwargs.get("text_start_pos") or 0),
                max_new_token=int(kwargs.get("max_new_token") or 0),
                min_new_tokens=int(kwargs.get("min_new_tokens") or 0),
            )
            return new_tokens, cache

        tts.generate_chunk = types.MethodType(traced_generate, tts)

    def _install_token2wav(self, duplex: Any) -> None:
        if not hasattr(duplex, "_generate_waveform_from_tokens"):
            return
        original_waveform = self._save_original(duplex, "_generate_waveform_from_tokens")
        audio_tokenizer = getattr(getattr(getattr(duplex, "model", None), "tts", None), "audio_tokenizer", None)
        trace = self

        def traced_waveform(duplex_self, new_tokens, *args, **kwargs):
            buffer_before = [int(item) for item in getattr(duplex_self, "token2wav_buffer", [])]
            output = original_waveform(new_tokens, *args, **kwargs)
            trace._emit(
                "tts.to_token2wav",
                new_tokens=_token_meta(new_tokens),
                buffer_before=buffer_before,
                buffer_after=[int(item) for item in getattr(duplex_self, "token2wav_buffer", [])],
                is_last_chunk=bool(kwargs.get("is_last_chunk", args[1] if len(args) > 1 else False)),
                force_flush=bool(kwargs.get("force_flush", False)),
            )
            return output

        duplex._generate_waveform_from_tokens = types.MethodType(traced_waveform, duplex)
        if audio_tokenizer is None or not hasattr(audio_tokenizer, "stream"):
            return
        original_stream = self._save_original(audio_tokenizer, "stream")

        def traced_stream(tokenizer_self, tokens, *args, **kwargs):
            token_ids = _flat_ints(tokens)
            last_chunk = bool(kwargs.get("last_chunk", False))
            prelook = int(getattr(duplex, "pre_lookahead", 0) or 0)
            committed = len(token_ids) if last_chunk else max(0, len(token_ids) - prelook)
            token_start = trace._t2w_token_pos
            output = original_stream(tokens, *args, **kwargs)
            if isinstance(output, (bytes, bytearray)):
                samples = len(output) // 2
            elif isinstance(output, np.ndarray):
                samples = int(output.size)
            else:
                samples = 0
            sample_start = trace._t2w_sample_pos
            trace._emit(
                "token2wav.call",
                input_token_ids=token_ids,
                input_range=[token_start, token_start + len(token_ids)],
                committed_range=[token_start, token_start + committed],
                lookahead_range=[token_start + committed, token_start + len(token_ids)],
                output_sample_range=[sample_start, sample_start + samples],
                last_chunk=last_chunk,
            )
            trace._t2w_token_pos += committed
            trace._t2w_sample_pos += samples
            return output

        audio_tokenizer.stream = types.MethodType(traced_stream, audio_tokenizer)

    def _capture_or_force_vocoder(self, duplex: Any) -> None:
        decoder = getattr(
            getattr(getattr(getattr(duplex, "model", None), "tts", None), "audio_tokenizer", None),
            "flow",
            None,
        )
        decoder = getattr(decoder, "decoder", None)
        rand_noise = getattr(decoder, "rand_noise", None)
        if not torch.is_tensor(rand_noise):
            return
        forced = self.forcing.vocoder_state
        if forced:
            if self.reference is None:
                raise RuntimeError("vocoder forcing requested without a replay reference")
            reference = self.reference.vocoder_noise()
            if reference is None:
                raise RuntimeError("reference has no vocoder rand_noise")
            if reference.shape != rand_noise.shape:
                raise RuntimeError(
                    f"vocoder rand_noise shape mismatch: reference={tuple(reference.shape)} actual={tuple(rand_noise.shape)}"
                )
            rand_noise.copy_(reference.to(device=rand_noise.device, dtype=rand_noise.dtype))
        self._vocoder_rand_noise = rand_noise
        self._vocoder_forced = forced
        if self.session_id is not None:
            self._emit_vocoder_state()

    def _emit_vocoder_state(self) -> None:
        assert self._vocoder_rand_noise is not None
        self._emit(
            "vocoder.state",
            rand_noise=_tensor_meta(self._vocoder_rand_noise, keep_value=self.capture_mode == "replay"),
            teacher_forced=self._vocoder_forced,
        )

    def drain(self, input_id: Optional[str] = None) -> list[dict[str, Any]]:
        return self.sink.drain(input_id)

    def uninstall(self) -> None:
        if not self._installed:
            return
        for owner, name, original in reversed(self._originals):
            setattr(owner, name, original)
        self._originals.clear()
        for hook in self._layer_hooks:
            hook.remove()
        self._layer_hooks.clear()
        if self._original_multinomial is not None:
            torch.multinomial = self._original_multinomial
            self._original_multinomial = None
        self._installed = False


def group_trace_events(events: Iterable[dict[str, Any]], *, input_id: Optional[str] = None) -> dict[str, Any]:
    grouped: dict[str, Any] = {
        "schema": TRACE_SCHEMA,
        "input_id": input_id,
        "llm": [],
        "tts": [],
        "token2wav": [],
        "runtime": [],
    }
    for event in events:
        kind = str(event.get("kind") or "")
        if kind.startswith("llm."):
            grouped["llm"].append(event)
        elif kind.startswith("tts."):
            grouped["tts"].append(event)
        elif kind.startswith("token2wav."):
            grouped["token2wav"].append(event)
        else:
            grouped["runtime"].append(event)
    return {key: value for key, value in grouped.items() if value not in (None, [], {})}
