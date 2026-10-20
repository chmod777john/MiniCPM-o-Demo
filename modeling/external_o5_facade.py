"""Retained O5 demo facade; all model primitives come from external HF."""
from __future__ import annotations
import io
import logging
import os
import tempfile
import time
import types
from copy import deepcopy
from dataclasses import fields, replace
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf
import torch
from core.external_o5 import canonical_module
from core.fc_duplex.protocol_errors import ConstrainedDecodingError, ModelProtocolError

_base = canonical_module("modeling_minicpmo")
BaseMiniCPMO = _base.MiniCPMO
DuplexCapability = _base.MiniCPMODuplex
TTSSamplingParams = canonical_module("utils").TTSSamplingParams
logger = logging.getLogger(__name__)

class ProcessorMode(Enum):
    CHAT = "chat"
    STREAMING = "streaming"
    DUPLEX = "duplex"
    FC_DUPLEX = "fc_duplex"

def _patch_token2wav_bytesio_save():
    try:
        import stepaudio2.token2wav as module
    except ImportError:
        return
    original = module.torchaudio.save
    if getattr(original, "_minicpmo_bytesio_patch", False):
        return
    def save(uri, src, sample_rate, *args, **kwargs):
        if isinstance(uri, io.BytesIO):
            audio = src.detach().cpu().numpy()
            sf.write(uri, audio.T if audio.ndim == 2 else audio, sample_rate, format="WAV")
            uri.seek(0)
            return
        return original(uri, src, sample_rate, *args, **kwargs)
    save._minicpmo_bytesio_patch = True
    module.torchaudio.save = save

def _load_pt_state_dict(path):
    from core.external_o5 import checkpoint_state
    return checkpoint_state(path)

def _plain_duplex_system_prompt(prefix):
    text = prefix or "Streaming Omni Conversation."
    if text.startswith("<|im_start|>system\n"):
        text = text[len("<|im_start|>system\n"):]
    for marker in ("<|audio_start|>", "<|audio_end|>", "<|im_end|>"):
        text = text.replace(marker, "")
    return text.strip()

class MiniCPMO(BaseMiniCPMO):
    """Demo runtime facade over external HF model primitives."""

    _external_hf_chat = True
    _tts_sampling_params_type = TTSSamplingParams

    def __init__(self, config):
        super().__init__(config)
        self._current_mode: Optional[ProcessorMode] = None
        self._unified_initialized = False
        self._compile_active = False
        self._compiled = False
        self._chat_vocoder = "token2wav"
        self.duplex: Optional[DuplexCapability] = None
        self.fc_duplex: Optional["FcDuplexCapability"] = None
        self._duplex_config: Dict[str, Any] = {
            "generate_audio": True,
            "ls_mode": "explicit",
            "max_new_speak_tokens_per_chunk": 20,
            "text_repetition_penalty": 1.05,
            "temperature": 0.7,
            "top_k": 20,
            "top_p": 0.8,
            "text_repetition_window_size": 512,
            "listen_prob_scale": 1.0,
            "force_listen_count": 0,
            "tts_temperature": 0.8,
            "tts_repetition_penalty": 1.05,
        }
        self._fc_duplex_config = {
            "temperature": 0.7,
            "tool_format": "minicpm4_xml",
            "default_unit_sec": 1.0,
            "max_spoken_tokens_per_unit": 24,
            "extra_response_units": 4,
            "generate_audio": False,
            "tts_temperature": 0.8,
            "tts_repetition_penalty": 1.05,
        }

    @property
    def current_mode(self) -> Optional[ProcessorMode]:
        return self._current_mode

    def init_tts(self, model_dir=None, enable_float16=False, n_timesteps=10, **kwargs):
        assets = model_dir or os.environ.get("O5_ASSETS_DIR")
        if assets and os.path.isdir(os.path.join(assets, "token2wav_causal_vocoder")):
            assets = os.path.join(assets, "token2wav_causal_vocoder")
        signature = (assets, bool(enable_float16), int(n_timesteps))
        existing = getattr(self.tts, "audio_tokenizer", None)
        if existing is not None and getattr(self, "_demo_t2w_signature", None) == signature:
            return existing
        result = super().init_tts(model_dir=assets, enable_float16=enable_float16,
                                  n_timesteps=n_timesteps, **kwargs)
        self._demo_t2w_signature = signature
        return result

    def init_token2wav(self, streaming=False, model_dir=None, enable_float16=False, n_timesteps=10):
        if not streaming:
            raise NotImplementedError("O5 unified facade only supports Token2wav streaming vocoder")
        _patch_token2wav_bytesio_save()
        return self.init_tts(model_dir=model_dir, enable_float16=enable_float16, n_timesteps=n_timesteps)

    def init_unified(
        self,
        pt_path: Optional[str] = None,
        preload_both_tts: bool = True,
        duplex_config: Optional[dict] = None,
        device: str = "cuda",
        chat_vocoder: str = "token2wav",
    ):
        self._chat_vocoder = chat_vocoder

        if pt_path is not None:
            logger.info("Loading extra weights: %s", pt_path)
            state_dict = _load_pt_state_dict(pt_path)
            info = self.load_state_dict(state_dict, strict=False)
            logger.info("Weights loaded - missing=%d unexpected=%d", len(info.missing_keys), len(info.unexpected_keys))
            del state_dict

        if duplex_config:
            self._duplex_config.update(duplex_config)
            fc_keys = set(self._fc_duplex_config)
            self._fc_duplex_config.update({k: v for k, v in duplex_config.items() if k in fc_keys})

        n_timesteps = getattr(self.config.tts_config, "s3_stream_n_timesteps", 10)
        self._duplex_config["n_timesteps"] = n_timesteps
        self.init_token2wav(
            streaming=True,
            n_timesteps=n_timesteps,
        )

        self.duplex = DuplexCapability.from_existing_model(
            model=self,
            device=device,
            **self._duplex_config,
        )
        self.fc_duplex = canonical_module("fc").attach(
            self, device=device, **self._fc_duplex_config,
        )
        self._unified_initialized = True
        self.set_mode(ProcessorMode.STREAMING)
        return self

    def set_mode(self, mode: ProcessorMode) -> None:
        if mode == self._current_mode:
            return
        self.reset_session(reset_token2wav_cache=True)
        if mode == ProcessorMode.DUPLEX and self.duplex is not None:
            self.duplex._reset_streaming_state()
            self.duplex.decoder.reset()
        self._current_mode = mode

    def apply_torch_compile(
        self,
        mode: str = "default",
        dynamic: bool = True,
        skip_modules: Optional[List[str]] = None,
    ) -> "MiniCPMO":
        raise ValueError("External O5 acceleration is owned by O5_ACCELERATION_PATH; disable compile=True")

    def set_compile_enabled(self, enabled: bool) -> None:
        if not getattr(self, "_compiled", False):
            return
        self._compile_active = bool(enabled)

    def warmup_compile(self, *args, **kwargs) -> None:
        logger.info("warmup_compile skipped by thin unified facade")

    def generate(self, *args, **kwargs):
        result, outputs = super().generate(*args, **kwargs)
        if getattr(self, "_collect_chat_metadata", False):
            # Observe native HF output; do not re-tokenize displayed text or change decoding.
            input_ids = kwargs["input_ids"]
            attention_mask = kwargs.get("attention_mask")
            prompt_tokens = (int(attention_mask[0].sum().item()) if attention_mask is not None
                             else int(input_ids.shape[-1]))
            sequence = outputs.sequences[0]
            generated_tokens = int(sequence.numel())
            terminators = {self.processor.tokenizer.convert_tokens_to_ids(t) for t in self.terminators}
            ended = generated_tokens > 0 and int(sequence[-1].item()) in terminators
            if not ended and generated_tokens < kwargs["max_new_tokens"]:
                raise RuntimeError("HF generation ended without EOS or reaching its token limit")
            self._last_chat_token_stats = {
                "cached_tokens": 0, "input_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "total_tokens": prompt_tokens + generated_tokens,
            }
            self._last_chat_finish_reason = "stop" if ended else "length"
        return result, outputs

    @torch.inference_mode()
    def chat(
        self,
        *args,
        generate_audio: bool = False,
        output_audio_path: Optional[str] = None,
        tts_ref_audio: Optional[np.ndarray] = None,
        return_prompt: bool = False,
        collect_chat_metadata: bool = False,
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
        **kwargs,
    ):
        if collect_chat_metadata and (kwargs.get("stream", False) or generate_audio):
            raise ValueError("Completion metadata currently requires nonstreaming text output")
        self._collect_chat_metadata = collect_chat_metadata
        self._last_chat_token_stats = {}
        self._last_chat_finish_reason = None
        temp_audio_path = None
        audio_path_for_base = output_audio_path
        if generate_audio and not kwargs.get("stream", False) and not audio_path_for_base:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="minicpmo_chat_", delete=False)
            temp_audio_path = tmp.name
            tmp.close()
            audio_path_for_base = temp_audio_path

        try:
            result = super().chat(
                *args, generate_audio=generate_audio,
                output_audio_path=audio_path_for_base,
                return_prompt=return_prompt, tts_ref_audio=tts_ref_audio,
                tts_sampling_params=tts_sampling_params or TTSSamplingParams(), **kwargs,
            )
        except BaseException:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise
        finally:
            self._collect_chat_metadata = False

        generated_waveform = None
        if temp_audio_path is not None:
            try:
                if os.path.getsize(temp_audio_path) > 0:
                    generated_waveform, _ = sf.read(temp_audio_path, dtype="float32")
            finally:
                try:
                    os.unlink(temp_audio_path)
                except FileNotFoundError:
                    pass

        if generated_waveform is None:
            return result

        if return_prompt:
            answer, prompt = result if isinstance(result, tuple) else (result, None)
            return answer, prompt, generated_waveform
        answer = result[0] if isinstance(result, tuple) else result
        return answer, generated_waveform

    def _observe_ordinary_windows(self):
        """Observe the installed HF/accel methods without changing their work."""
        decoder = self.duplex.decoder
        for name in ("enforce_window", "enforce_window_with_context", "enforce_window_with_kv_surgery"):
            native = getattr(decoder, name, None)
            if native is None or getattr(native, "_demo_window_observer", False):
                continue

            def observe(method):
                @wraps(method)
                def call(*args, **kwargs):
                    depth = getattr(self, "_ordinary_window_depth", 0)
                    self._ordinary_window_depth = depth + 1
                    before = decoder.get_cache_length()
                    count_before = decoder._sliding_event_count
                    started = time.perf_counter()
                    try:
                        result = method(*args, **kwargs)
                        if depth == 0 and decoder._sliding_event_count > count_before:
                            self._ordinary_window_events.append({
                                "count": decoder._sliding_event_count,
                                "before": before,
                                "after": decoder.get_cache_length(),
                                "mode": decoder._window_config.sliding_window_mode,
                                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                            })
                        return result
                    finally:
                        self._ordinary_window_depth = depth
                call._demo_window_observer = True
                return call

            setattr(decoder, name, observe(native))

    def ordinary_session_config(self):
        return dict(getattr(self, "_ordinary_effective_config", {}))

    def ordinary_unit_metrics(self):
        events = list(getattr(self, "_ordinary_window_events", []))
        self._ordinary_window_events = []
        return {
            "kv_cache_length": self.duplex.decoder.get_cache_length(),
            "window_event_count": self.duplex.decoder._sliding_event_count,
            "window_events": events,
        }

    def configure_duplex_session(self, config):
        """Validate and install external HF options before the retained View prepares."""
        from core.schemas.external_o5_duplex import ExternalO5DuplexConfig
        from core.schemas.duplex import DuplexConfig
        validated = ExternalO5DuplexConfig(**config)
        options = validated.model_dump(exclude_unset=True)
        window_class = canonical_module("utils").DuplexWindowConfig
        window_fields = {field.name for field in fields(window_class)}
        if not hasattr(self, "_ordinary_window_default"):
            self._ordinary_window_default = deepcopy(self.duplex.decoder._window_config)
            self._ordinary_tts_repetition_default = self.duplex.tts_repetition_penalty
        window = replace(self._ordinary_window_default, **{
            name: value for name, value in options.items() if name in window_fields
        })
        if window.basic_window_low_tokens > window.basic_window_high_tokens:
            raise ValueError("basic window requires low <= high")
        # Older API clients may still send length_penalty. HF ordinary generation
        # has no such argument; report it as inactive. Current pages omit it.
        if "length_penalty" in options:
            logger.warning("External HF ordinary generation has no length_penalty; "
                           "retained page value %s is inactive", validated.length_penalty)
        self._ordinary_session_options = options
        self._ordinary_window_events = []
        self._ordinary_window_depth = 0
        self._ordinary_next_slices = 1
        self._observe_ordinary_windows()
        self.duplex.decoder.set_window_config(window)
        self.duplex.decoder.set_window_enabled(window.sliding_window_mode != "off")
        for name, value in validated.model_dump(include=set(DuplexConfig.model_fields)).items():
            if hasattr(self.duplex, name):
                previous = getattr(self.duplex, name)
                if torch.is_tensor(previous):
                    value = torch.as_tensor(value, dtype=previous.dtype, device=previous.device).reshape(previous.shape)
                setattr(self.duplex, name, value)
        self.duplex.tts_repetition_penalty = options.get(
            "tts_repetition_penalty", self._ordinary_tts_repetition_default)
        # Current HF generate_chunk rebuilds its sampling processors using the
        # repetition_penalty argument on each call; no separate cache patch is needed.
        self.duplex.CHUNK_MS = self.CHUNK_MS = validated.chunk_ms
        self.duplex.SAMPLE_RATE = self.SAMPLE_RATE = validated.sample_rate
        self._ordinary_effective_config = validated.model_dump(exclude={"length_penalty"})
        self._ordinary_effective_config.update({field.name: getattr(window, field.name)
                                              for field in fields(window)})
        self._ordinary_effective_config["tts_repetition_penalty"] = self.duplex.tts_repetition_penalty
        self._ordinary_effective_config["unit_finalized_metrics"] = True
        self._ordinary_effective_config["inactive_options"] = (
            ["length_penalty"] if "length_penalty" in options else [])

    def duplex_prepare(
        self,
        prefix_system_prompt: Optional[str] = None,
        suffix_system_prompt: Optional[str] = None,
        ref_audio: Optional[np.ndarray] = None,
        prompt_wav_path: Optional[str] = None,
        context_previous_marker: str = "\n\nprevious: ",
        llm_seed: Optional[int] = None,
    ):
        if self.duplex is None:
            raise RuntimeError("Duplex is not initialized; call init_unified() first")
        result = self.duplex.prepare(
            prefix_system_prompt=_plain_duplex_system_prompt(prefix_system_prompt),
            ref_audio=ref_audio,
            prompt_wav_path=prompt_wav_path,
            context_previous_marker=context_previous_marker,
            llm_seed=llm_seed,
        )
        # HF prepare resets the LS policy from the environment. Apply session overrides
        # afterwards, including the decoder token-ID switch required by binary mode.
        options = getattr(self, "_ordinary_session_options", {})
        policy = self.duplex.ls_decision_config
        changes = {target: options[source] for source, target in (
            ("ls_decision_mode", "mode"), ("speak_threshold", "speak_threshold"),
            ("ls_strict_unit", "strict_unit"), ("ls_log_prob", "log_prob"),
        ) if source in options}
        if changes.get("mode") == "binary":
            changes["fix_enabled"] = True
        self.duplex.set_ls_decision_config(replace(policy, **changes))
        self._ordinary_next_slices = 1
        self._ordinary_window_events = []
        self._ordinary_effective_config.update({
            "ls_decision_mode": self.duplex.ls_decision_config.mode,
            "speak_threshold": self.duplex.ls_decision_config.speak_threshold,
            "ls_strict_unit": self.duplex.ls_decision_config.strict_unit,
            "ls_log_prob": self.duplex.ls_decision_config.log_prob,
        })
        logger.info("External HF ordinary session: window=%s ls=%s max_speak=%s",
                    self.duplex.decoder._window_config, self.duplex.ls_decision_config,
                    self.duplex.max_new_speak_tokens_per_chunk)
        return result

    def duplex_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[List] = None,
        text_list: Optional[List] = None,
        max_slice_nums: Optional[int] = None,
        batch_vision_feed: bool = False,
    ):
        if self.duplex is None:
            raise RuntimeError("Duplex is not initialized; call init_unified() first")
        options = getattr(self, "_ordinary_session_options", {})
        if options.get("strategy_hd", False):
            effective_slices = self._ordinary_next_slices
        else:
            effective_slices = max_slice_nums if max_slice_nums is not None else options.get("max_slice_nums", 1)
        if not 1 <= effective_slices <= 9:
            raise ValueError("max_slice_nums must be in [1, 9]")
        result = self.duplex.streaming_prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            text_list=text_list,
            max_slice_nums=effective_slices,
            batch_vision_feed=batch_vision_feed,
        )
        if isinstance(result, dict):
            result["effective_max_slice_nums"] = effective_slices
        return result

    def duplex_generate(
        self,
        decode_mode: str = "greedy",
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        listen_prob_scale: Optional[float] = None,
        listen_top_k: Optional[int] = None,
        text_repetition_penalty: Optional[float] = None,
        text_repetition_window_size: Optional[int] = None,
        length_penalty: float = 1.1,
        force_listen_override: bool = False,
    ):
        if self.duplex is None:
            raise RuntimeError("Duplex is not initialized; call init_unified() first")
        options = getattr(self, "_ordinary_session_options", {})
        values = {
            "decode_mode": decode_mode,
            "max_new_speak_tokens_per_chunk": self.duplex.max_new_speak_tokens_per_chunk,
            "listen_top_k": listen_top_k,
        }
        for name, value in (
            ("temperature", temperature), ("top_k", top_k), ("top_p", top_p),
            ("listen_prob_scale", listen_prob_scale),
            ("text_repetition_penalty", text_repetition_penalty),
            ("text_repetition_window_size", text_repetition_window_size),
        ):
            values[name] = value if value is not None else getattr(self.duplex, name)
        for name in ("ls_decision_mode", "speak_threshold", "ls_strict_unit", "ls_log_prob"):
            if name in options:
                values[name] = options[name]
        old_force = self.duplex.force_listen_count
        if force_listen_override:
            self.duplex.force_listen_count = self.duplex._streaming_generate_count + 1
        try:
            result = self.duplex.streaming_generate(**values)
            self._ordinary_next_slices = (
                options.get("strategy_hd_max_slice_nums", 4)
                if options.get("strategy_hd", False) and not result.get("is_listen", True)
                else 1
            )
            return result
        finally:
            self.duplex.force_listen_count = old_force

    def duplex_finalize(self):
        finalize = getattr(self.duplex, "finalize_unit", None) if self.duplex is not None else None
        if callable(finalize):
            finalize()

    def duplex_set_break(self):
        if self.duplex is not None:
            self.duplex.set_break_event()

    def duplex_clear_break(self):
        if self.duplex is not None:
            self.duplex.clear_break_event()

    def duplex_stop(self):
        if self.duplex is not None:
            self.duplex.set_session_stop()

    def duplex_is_break_set(self) -> bool:
        return bool(self.duplex and self.duplex.is_break_set())

    def duplex_is_stopped(self) -> bool:
        return bool(self.duplex and self.duplex.is_session_stop_set())

    # ==================== FC Duplex 透传方法 ====================

    def _fc_call(self, method, *args, **kwargs):
        if self.fc_duplex is None:
            raise RuntimeError("FC Duplex is not initialized")
        errors = canonical_module("fc.protocol_errors")
        try:
            return getattr(self.fc_duplex, method)(*args, **kwargs)
        except errors.ConstrainedDecodingError as exc:
            raise ConstrainedDecodingError(exc.kind, exc.detail, context=exc.context) from exc
        except errors.ModelProtocolError as exc:
            raise ModelProtocolError(exc.code, exc.detail, track=exc.track,
                                     token_id=exc.token_id, context=exc.context) from exc


def _flatten(method):
    def call(self, *args, **kwargs):
        return self._fc_call(method, *args, **kwargs)
    call.__name__ = "fc_duplex_" + method
    return call

for _method in (
    "prepare", "streaming_prefill", "streaming_spoken_generate",
    "streaming_non_spoken_generate", "finalize_unit", "decode_output_ids",
    "trace_snapshot", "dump_trace", "cleanup", "resume_boundary_status",
    "replay_completed_unit",
):
    setattr(MiniCPMO, "fc_duplex_" + _method, _flatten(_method))
