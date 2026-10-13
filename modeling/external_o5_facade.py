"""Retained O5 demo facade; all model primitives come from external HF."""
from __future__ import annotations
import io
import logging
import os
import tempfile
from enum import Enum
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

        self.init_token2wav(
            streaming=True,
            n_timesteps=getattr(self.config.tts_config, "s3_stream_n_timesteps", 10),
        )

        self.duplex = DuplexCapability.from_existing_model(
            model=self,
            device=device,
            **self._duplex_config,
        )
        self.fc_duplex = canonical_module("fc").attach(
            self,
            device=device,
            **self._fc_duplex_config,
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
        skip = set(skip_modules or [])
        compile_kwargs = dict(mode=mode, dynamic=dynamic)

        if hasattr(self, "llm") and "llm.model" not in skip:
            self.llm.model = torch.compile(self.llm.model, **compile_kwargs)
        if hasattr(self, "tts") and hasattr(self.tts, "model") and "tts.model" not in skip:
            self.tts.model = torch.compile(self.tts.model, **compile_kwargs)

        self._compiled = True
        self._compile_active = True
        return self

    def set_compile_enabled(self, enabled: bool) -> None:
        if not getattr(self, "_compiled", False):
            return
        self._compile_active = bool(enabled)

    def warmup_compile(self, *args, **kwargs) -> None:
        logger.info("warmup_compile skipped by thin unified facade")

    @torch.inference_mode()
    def chat(
        self,
        *args,
        generate_audio: bool = False,
        output_audio_path: Optional[str] = None,
        tts_ref_audio: Optional[np.ndarray] = None,
        return_prompt: bool = False,
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
        **kwargs,
    ):
        temp_audio_path = None
        audio_path_for_base = output_audio_path
        if generate_audio and not kwargs.get("stream", False) and not audio_path_for_base:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="minicpmo_chat_", delete=False)
            temp_audio_path = tmp.name
            tmp.close()
            audio_path_for_base = temp_audio_path

        sentinel = object()
        previous_tts_attr = self.__dict__.get("_generate_speech_non_streaming", sentinel)
        should_override_tts_prompt = tts_ref_audio is not None

        def _generate_speech_with_ref_audio(
            _self,
            outputs,
            tts_bound,
            tts_proj_layer,
            audio_prompt,
            spk_bound=None,
            output_tts_inputs_embeds_path=None,
            tts_sampling_params=TTSSamplingParams(),
        ):
            return BaseMiniCPMO._generate_speech_non_streaming(
                _self,
                outputs=outputs,
                tts_bound=tts_bound,
                tts_proj_layer=tts_proj_layer,
                audio_prompt=tts_ref_audio,
                spk_bound=spk_bound,
                output_tts_inputs_embeds_path=output_tts_inputs_embeds_path,
                tts_sampling_params=tts_sampling_params,
            )

        try:
            if should_override_tts_prompt:
                self._generate_speech_non_streaming = types.MethodType(_generate_speech_with_ref_audio, self)
            result = super().chat(
                *args,
                generate_audio=generate_audio,
                output_audio_path=audio_path_for_base,
                return_prompt=return_prompt,
                tts_sampling_params=tts_sampling_params,
                **kwargs,
            )
        finally:
            if should_override_tts_prompt:
                if previous_tts_attr is sentinel:
                    self.__dict__.pop("_generate_speech_non_streaming", None)
                else:
                    self._generate_speech_non_streaming = previous_tts_attr

        generated_waveform = None
        if (
            temp_audio_path is not None
            and os.path.exists(temp_audio_path)
            and os.path.getsize(temp_audio_path) > 0
        ):
            try:
                generated_waveform, _ = sf.read(temp_audio_path, dtype="float32")
            finally:
                try:
                    os.unlink(temp_audio_path)
                except OSError:
                    pass

        if generated_waveform is None:
            return result

        if return_prompt:
            answer, prompt = result if isinstance(result, tuple) else (result, None)
            return answer, prompt, generated_waveform
        answer = result[0] if isinstance(result, tuple) else result
        return answer, generated_waveform

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
        return self.duplex.prepare(
            prefix_system_prompt=_plain_duplex_system_prompt(prefix_system_prompt),
            ref_audio=ref_audio,
            prompt_wav_path=prompt_wav_path,
            context_previous_marker=context_previous_marker,
            llm_seed=llm_seed,
        )

    def duplex_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[List] = None,
        text_list: Optional[List] = None,
        max_slice_nums: int = 1,
        batch_vision_feed: bool = False,
    ):
        if self.duplex is None:
            raise RuntimeError("Duplex is not initialized; call init_unified() first")
        return self.duplex.streaming_prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            text_list=text_list,
            max_slice_nums=max_slice_nums,
            batch_vision_feed=batch_vision_feed,
        )

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
        if force_listen_override:
            old_force = self.duplex.force_listen_count
            old_count = getattr(self.duplex, "_streaming_generate_count", 0)
            self.duplex.force_listen_count = old_count + 1
            try:
                return self.duplex.streaming_generate(
                    decode_mode=decode_mode,
                    temperature=temperature if temperature is not None else self.duplex.temperature,
                    top_k=top_k if top_k is not None else self.duplex.top_k,
                    top_p=top_p if top_p is not None else self.duplex.top_p,
                    listen_prob_scale=listen_prob_scale if listen_prob_scale is not None else self.duplex.listen_prob_scale,
                    listen_top_k=listen_top_k,
                    text_repetition_penalty=(
                        text_repetition_penalty
                        if text_repetition_penalty is not None
                        else self.duplex.text_repetition_penalty
                    ),
                    text_repetition_window_size=(
                        text_repetition_window_size
                        if text_repetition_window_size is not None
                        else self.duplex.text_repetition_window_size
                    ),
                )
            finally:
                self.duplex.force_listen_count = old_force
        return self.duplex.streaming_generate(
            decode_mode=decode_mode,
            temperature=temperature if temperature is not None else self.duplex.temperature,
            top_k=top_k if top_k is not None else self.duplex.top_k,
            top_p=top_p if top_p is not None else self.duplex.top_p,
            listen_prob_scale=listen_prob_scale if listen_prob_scale is not None else self.duplex.listen_prob_scale,
            listen_top_k=listen_top_k,
            text_repetition_penalty=(
                text_repetition_penalty
                if text_repetition_penalty is not None
                else self.duplex.text_repetition_penalty
            ),
            text_repetition_window_size=(
                text_repetition_window_size
                if text_repetition_window_size is not None
                else self.duplex.text_repetition_window_size
            ),
        )

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
