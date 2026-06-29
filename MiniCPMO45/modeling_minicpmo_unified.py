#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified MiniCPM-o adapter for omni-dev.

This file intentionally keeps the model body in ``modeling_minicpmo.py``.
That file is synced from the model trusted code.  The adapter below carries
only the omni-dev mode switching surface that the service uses.
"""

import json
import logging
import os
import tempfile
import time
from copy import deepcopy
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch

from .configuration_minicpmo import MiniCPMOConfig
from .modeling_minicpmo import MiniCPMO as BaseMiniCPMO
from .modeling_minicpmo import MiniCPMODuplex
from .modeling_minicpmo import MiniCPMOPreTrainedModel
from .processing_minicpmo import MiniCPMOProcessor
from .utils import TTSSamplingParams
from .utils import normalize_content

logger = logging.getLogger(__name__)


class ProcessorMode(Enum):
    CHAT = "chat"
    STREAMING = "streaming"
    DUPLEX = "duplex"


def _as_content_list(content):
    normalized = normalize_content(content)
    if isinstance(normalized, list):
        return normalized
    return [normalized]


def _strip_duplex_system_prompt(prefix_system_prompt: Optional[str]) -> Optional[str]:
    """Accept both raw text and old unified's fully wrapped system prefix."""
    if prefix_system_prompt is None:
        return None

    text = prefix_system_prompt
    if text.startswith("<|im_start|>system\n"):
        text = text[len("<|im_start|>system\n") :]
    if text.endswith("<|audio_start|>"):
        text = text[: -len("<|audio_start|>")].rstrip()
    if text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")].rstrip()
    return text


def _strip_duplex_suffix_prompt(suffix_system_prompt: Optional[str], has_ref_audio: bool) -> Optional[str]:
    if suffix_system_prompt is None:
        return None
    if has_ref_audio:
        return suffix_system_prompt
    return suffix_system_prompt.replace("<|audio_end|>", "")


class MiniCPMO(BaseMiniCPMO):
    """Trusted-code MiniCPMO plus omni-dev runtime mode APIs."""

    def __init__(self, config):
        super().__init__(config)
        self._unified_initialized = False
        self._current_mode: Optional[ProcessorMode] = None
        self._duplex_config: Dict[str, object] = {}
        self.duplex: Optional[MiniCPMODuplex] = None
        self._compiled = False
        self._compile_active = False
        self._chat_vocoder = "token2wav"

    @classmethod
    def from_config_and_pt(
        cls,
        model_path: Union[str, os.PathLike],
        pt_path: Union[str, os.PathLike],
        _attn_implementation: Optional[str] = None,
        require_full: bool = True,
        map_location: str = "cpu",
    ) -> "MiniCPMO":
        """Instantiate from config and load a full ``.pt`` state dict.

        This skips HuggingFace ``from_pretrained`` weight shards.  It is meant
        for training checkpoints that fully cover the model state dict.
        """
        config = MiniCPMOConfig.from_pretrained(model_path, trust_remote_code=True)
        config._name_or_path = str(model_path)
        config.name_or_path = str(model_path)
        if _attn_implementation is not None:
            config._attn_implementation = _attn_implementation

        try:
            from accelerate import init_empty_weights
        except ImportError:
            init_empty_weights = None

        if init_empty_weights is None:
            logger.warning("accelerate is unavailable; falling back to normal CPU initialization")
            model = cls(config)
        else:
            with init_empty_weights():
                model = cls(config)

        logger.info("Loading full pt weights: %s", pt_path)
        try:
            state_dict = torch.load(pt_path, map_location=map_location, weights_only=True)
        except TypeError:
            state_dict = torch.load(pt_path, map_location=map_location)

        for wrapper_key in ("state_dict", "model", "module"):
            wrapped = state_dict.get(wrapper_key) if isinstance(state_dict, dict) else None
            if isinstance(wrapped, dict):
                state_dict = wrapped
                break

        try:
            try:
                info = model.load_state_dict(state_dict, strict=False, assign=True)
            except TypeError:
                info = model.load_state_dict(state_dict, strict=False)

            if require_full and info.missing_keys:
                raise RuntimeError(
                    f"{pt_path} is not a full checkpoint: missing={len(info.missing_keys)} "
                    f"unexpected={len(info.unexpected_keys)}"
                )

            logger.info(
                "Full pt weights loaded: missing=%d unexpected=%d",
                len(info.missing_keys),
                len(info.unexpected_keys),
            )
        finally:
            del state_dict

        return model

    def init_token2wav(self, streaming=False, model_dir=None, enable_float16=False, n_timesteps=10):
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
            state_dict = torch.load(pt_path, map_location="cpu")
            info = self.load_state_dict(state_dict, strict=False)
            logger.info(
                "Extra weights loaded: missing=%d unexpected=%d",
                len(info.missing_keys),
                len(info.unexpected_keys),
            )
            del state_dict

        if duplex_config:
            self._duplex_config.update(duplex_config)

        self.prepare_processor()

        if getattr(getattr(self, "tts", None), "audio_tokenizer", None) is None:
            self.init_tts(
                enable_float16=bool(self._duplex_config.get("enable_float16", False)),
                n_timesteps=int(self._duplex_config.get("n_timesteps", 10)),
            )

        self.duplex = self.as_duplex(device=device, **self._duplex_config)
        self._unified_initialized = True
        self.set_mode(ProcessorMode.STREAMING)
        return self

    def set_mode(self, mode: ProcessorMode) -> None:
        if not isinstance(mode, ProcessorMode):
            mode = ProcessorMode(mode)
        if mode == self._current_mode:
            return

        self.reset_session(reset_token2wav_cache=True)
        if mode == ProcessorMode.DUPLEX and self.duplex is not None:
            self.duplex._reset_streaming_state()
            self.duplex.decoder.reset()
        self._current_mode = mode

    @property
    def current_mode(self) -> Optional[ProcessorMode]:
        return self._current_mode

    def as_duplex(self, device: Optional[str] = None, **kwargs) -> MiniCPMODuplex:
        return MiniCPMODuplex.from_existing_model(model=self, device=device, **kwargs)

    def apply_torch_compile(
        self,
        mode: str = "default",
        dynamic: bool = True,
        skip_modules: Optional[List[str]] = None,
    ) -> "MiniCPMO":
        skip = set(skip_modules or [])
        compile_kwargs = dict(mode=mode, dynamic=dynamic)

        if hasattr(self, "llm") and hasattr(self.llm, "model") and "llm.model" not in skip:
            self.llm.model = torch.compile(self.llm.model, **compile_kwargs)
        if hasattr(self, "tts") and hasattr(self.tts, "model") and "tts.model" not in skip:
            self.tts.model = torch.compile(self.tts.model, **compile_kwargs)

        torch.set_float32_matmul_precision("high")
        self._compiled = True
        self._compile_active = True
        return self

    def set_compile_enabled(self, enabled: bool) -> None:
        if not getattr(self, "_compiled", False):
            return
        if enabled == getattr(self, "_compile_active", True):
            return

        for owner, attr in ((getattr(self, "llm", None), "model"), (getattr(self, "tts", None), "model")):
            if owner is None or not hasattr(owner, attr):
                continue
            cur = getattr(owner, attr)
            if enabled:
                compiled = getattr(cur, "_compiled_ref", None)
                if compiled is not None:
                    setattr(owner, attr, compiled)
            else:
                orig = getattr(cur, "_orig_mod", None)
                if orig is not None:
                    orig._compiled_ref = cur
                    setattr(owner, attr, orig)
        self._compile_active = enabled

    def warmup_compile(self, *args, **kwargs) -> None:
        logger.warning("warmup_compile is not implemented in the thin unified adapter")

    def benchmark(self, *args, **kwargs) -> dict:
        raise NotImplementedError("benchmark is not implemented in the thin unified adapter")

    @torch.inference_mode()
    def _generate_speech_non_streaming(self, *args, **kwargs):
        tts_ref_audio = getattr(self, "_chat_tts_ref_audio_override", None)
        if tts_ref_audio is not None:
            kwargs["audio_prompt"] = tts_ref_audio
        return super()._generate_speech_non_streaming(*args, **kwargs)

    @torch.inference_mode()
    def chat(self, *args, **kwargs):
        generate_audio = kwargs.get("generate_audio", False)
        output_audio_path = kwargs.get("output_audio_path")
        return_prompt = kwargs.get("return_prompt", False)
        tts_ref_audio = kwargs.pop("tts_ref_audio", None)

        tmp_path = None
        if generate_audio and not output_audio_path:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            kwargs["output_audio_path"] = tmp_path

        had_previous_tts_ref = hasattr(self, "_chat_tts_ref_audio_override")
        previous_tts_ref = getattr(self, "_chat_tts_ref_audio_override", None)
        if tts_ref_audio is not None:
            self._chat_tts_ref_audio_override = tts_ref_audio
        elif had_previous_tts_ref:
            delattr(self, "_chat_tts_ref_audio_override")

        try:
            result = super().chat(*args, **kwargs)
        finally:
            if had_previous_tts_ref:
                self._chat_tts_ref_audio_override = previous_tts_ref
            elif hasattr(self, "_chat_tts_ref_audio_override"):
                delattr(self, "_chat_tts_ref_audio_override")

        if tmp_path is None:
            return result

        waveform = None
        try:
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                import soundfile as sf

                waveform, _ = sf.read(tmp_path, dtype="float32")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if waveform is None:
            return result
        if return_prompt and isinstance(result, tuple):
            return result[0], result[1], waveform
        return result, waveform

    @torch.inference_mode()
    def streaming_prefill(self, session_id, msgs, *args, **kwargs):
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        msgs = deepcopy(msgs)
        for msg in msgs:
            if "content" in msg:
                msg["content"] = _as_content_list(msg["content"])
        return super().streaming_prefill(session_id=session_id, msgs=msgs, *args, **kwargs)

    @torch.inference_mode()
    def non_streaming_prefill(
        self,
        session_id,
        msgs,
        image=None,
        omni_mode=False,
        max_slice_nums=None,
        use_image_id=None,
        use_tts_template=False,
        enable_thinking=False,
        stream_input=False,
        max_inp_length=8192,
        merge_audio_from_same_content=True,
    ):
        assert session_id is not None, "session_id cannot be None"

        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        msgs = deepcopy(msgs)
        assert len(msgs) > 0, "msgs is empty"

        if image is not None and isinstance(msgs[0].get("content"), str):
            msgs[0]["content"] = [image, msgs[0]["content"]]

        self.reset_session(reset_token2wav_cache=False)
        self.prepare_processor()

        prompts = []
        for idx, msg in enumerate(msgs):
            current = deepcopy(msg)
            current["content"] = _as_content_list(current["content"])
            prompt = super().streaming_prefill(
                session_id=session_id,
                msgs=[current],
                omni_mode=omni_mode,
                max_slice_nums=max_slice_nums,
                use_tts_template=use_tts_template,
                enable_thinking=enable_thinking,
                is_last_chunk=idx == len(msgs) - 1,
                processor=self.processor,
            )
            prompts.append(prompt)

        return "".join(p for p in prompts if p)

    @torch.inference_mode()
    def non_streaming_generate(
        self,
        session_id,
        max_new_tokens=256,
        do_sample=True,
        min_new_tokens=0,
        generate_audio=False,
        use_tts_template=True,
        enable_thinking=False,
        tts_ref_audio=None,
        tts_sampling_params=None,
        output_audio_path=None,
        length_penalty=1.1,
        tts_proj_layer=-1,
    ):
        if generate_audio and getattr(self, "token2wav_cache", None) is None:
            if tts_ref_audio is None:
                raise RuntimeError("generate_audio=True requires a TTS reference audio cache")
            self.init_token2wav_cache(prompt_speech_16k=tts_ref_audio)

        kwargs = {
            "session_id": session_id,
            "generate_audio": generate_audio,
            "max_new_tokens": max_new_tokens,
            "enable_thinking": enable_thinking,
            "use_tts_template": use_tts_template,
            "do_sample": do_sample,
            "length_penalty": length_penalty,
        }
        if tts_sampling_params is not None:
            kwargs["tts_sampling_params"] = tts_sampling_params

        text_parts: List[str] = []
        audio_parts: List[np.ndarray] = []

        for item in self.streaming_generate(**kwargs):
            if item is None or not isinstance(item, (tuple, list)) or len(item) < 2:
                continue

            if generate_audio:
                waveform, text_delta = item[0], item[1]
                if isinstance(text_delta, str):
                    text_parts.append(text_delta)
                if waveform is not None:
                    if isinstance(waveform, torch.Tensor):
                        waveform = waveform.detach().cpu().numpy()
                    audio_parts.append(np.asarray(waveform, dtype=np.float32).reshape(-1))
            else:
                text_delta = item[0]
                if isinstance(text_delta, str):
                    text_parts.append(text_delta)

        text = "".join(text_parts)
        if not generate_audio:
            return text

        waveform = np.concatenate(audio_parts) if audio_parts else None
        if output_audio_path and waveform is not None:
            import soundfile as sf

            sf.write(output_audio_path, waveform, samplerate=24000)
        if waveform is not None:
            return text, waveform
        return text

    def duplex_prepare(
        self,
        prefix_system_prompt: Optional[str] = None,
        suffix_system_prompt: Optional[str] = None,
        ref_audio: Optional[np.ndarray] = None,
        prompt_wav_path: Optional[str] = None,
        context_previous_marker: str = "\n\nprevious: ",
    ):
        if self.duplex is None:
            raise RuntimeError("Duplex is not initialized; call init_unified() first")

        return self.duplex.prepare(
            prefix_system_prompt=prefix_system_prompt,
            suffix_system_prompt=suffix_system_prompt,
            ref_audio=ref_audio,
            prompt_wav_path=prompt_wav_path,
            context_previous_marker=context_previous_marker,
        )

    def duplex_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[List] = None,
        text_list: Optional[List[str]] = None,
        max_slice_nums: Union[int, List[int]] = 1,
    ):
        if self.duplex is None:
            raise RuntimeError("Duplex is not initialized; call init_unified() first")
        return self.duplex.streaming_prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            text_list=text_list,
            max_slice_nums=max_slice_nums,
        )

    def duplex_generate(
        self,
        decode_mode: str = "greedy",
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        listen_prob_scale: Optional[float] = None,
        listen_top_k: int = 5,
        text_repetition_penalty: Optional[float] = None,
        text_repetition_window_size: Optional[int] = None,
        length_penalty: float = 1.1,
        force_listen_override: bool = False,
    ):
        if self.duplex is None:
            raise RuntimeError("Duplex is not initialized; call init_unified() first")

        return self.duplex.streaming_generate(
            decode_mode=decode_mode,
            temperature=self.duplex.temperature if temperature is None else temperature,
            top_k=self.duplex.top_k if top_k is None else top_k,
            top_p=self.duplex.top_p if top_p is None else top_p,
            listen_prob_scale=self.duplex.listen_prob_scale if listen_prob_scale is None else listen_prob_scale,
            listen_top_k=listen_top_k,
            text_repetition_penalty=(
                self.duplex.text_repetition_penalty
                if text_repetition_penalty is None
                else text_repetition_penalty
            ),
            text_repetition_window_size=(
                self.duplex.text_repetition_window_size
                if text_repetition_window_size is None
                else text_repetition_window_size
            ),
            length_penalty=length_penalty,
            force_listen_override=force_listen_override,
        )

    def duplex_finalize(self):
        if self.duplex is None:
            raise RuntimeError("Duplex is not initialized; call init_unified() first")
        finalize = getattr(self.duplex, "finalize_unit", None)
        if finalize is not None:
            return finalize()
        return None

    def duplex_set_break(self):
        if self.duplex is None:
            raise RuntimeError("Duplex is not initialized; call init_unified() first")
        self.duplex.set_break_event()

    def duplex_clear_break(self):
        if self.duplex is None:
            raise RuntimeError("Duplex is not initialized; call init_unified() first")
        self.duplex.clear_break_event()

    def duplex_stop(self):
        if self.duplex is None:
            raise RuntimeError("Duplex is not initialized; call init_unified() first")
        self.duplex.set_session_stop()

    def duplex_is_break_set(self) -> bool:
        return bool(self.duplex is not None and self.duplex.is_break_set())

    def duplex_is_stopped(self) -> bool:
        return bool(self.duplex is not None and self.duplex.is_session_stop_set())


DuplexCapability = MiniCPMODuplex
