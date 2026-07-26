#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Thin unified facade for the vendored MiniCPM-o model.

This module intentionally keeps model behavior in ``modeling_minicpmo.py``.
It adds only the small runtime surface that the demo/backend imports:
``ProcessorMode``, ``MiniCPMO.init_unified()``, mode switching, and duplex
passthrough methods.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import time
import types
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from .modeling_minicpmo import gen_logits
from .modeling_minicpmo import MiniCPMO as BaseMiniCPMO
from .modeling_minicpmo import MiniCPMODuplex as DuplexCapability
from .processing_minicpmo import MiniCPMOProcessor
from .utils import StreamDecoder
from .utils import torch_clone_recursive
from .utils import TTSSamplingParams

logger = logging.getLogger(__name__)


class ProcessorMode(Enum):
    CHAT = "chat"
    STREAMING = "streaming"
    DUPLEX = "duplex"


def _patch_token2wav_bytesio_save() -> None:
    """Keep Token2wav BytesIO output compatible with torchcodec torchaudio."""

    try:
        import stepaudio2.token2wav as token2wav_mod
    except Exception:
        return
    torchaudio_mod = getattr(token2wav_mod, "torchaudio", None)
    if torchaudio_mod is None or getattr(torchaudio_mod.save, "_minicpmo_bytesio_patch", False):
        return

    original_save = torchaudio_mod.save

    def save_compat(uri, src, sample_rate, *args, **kwargs):
        if isinstance(uri, io.BytesIO):
            audio = src.detach().cpu().numpy()
            if audio.ndim == 2:
                audio = audio.T
            sf.write(uri, audio, sample_rate, format="WAV")
            uri.seek(0)
            return None
        return original_save(uri, src, sample_rate, *args, **kwargs)

    save_compat._minicpmo_bytesio_patch = True
    torchaudio_mod.save = save_compat


def _load_pt_state_dict(path: str) -> dict:
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model", "module"):
        if isinstance(state_dict, dict) and isinstance(state_dict.get(key), dict):
            return state_dict[key]
    return state_dict


def _plain_duplex_system_prompt(prefix: Optional[str]) -> str:
    text = prefix or "Streaming Omni Conversation."
    if text.startswith("<|im_start|>system\n"):
        text = text[len("<|im_start|>system\n") :]
    for marker in ("<|audio_start|>", "<|audio_end|>", "<|im_end|>"):
        text = text.replace(marker, "")
    return text.strip()


class MiniCPMO(BaseMiniCPMO):
    """Vendored MiniCPMO plus the demo runtime facade."""

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
        self.fc_duplex = FcDuplexCapability(
            model=self,
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

    def _require_fc_duplex(self) -> "FcDuplexCapability":
        if self.fc_duplex is None:
            raise RuntimeError("FC Duplex 未初始化，请先调用 init_unified()")
        return self.fc_duplex

    def fc_duplex_prepare(
        self,
        system_prompt: str,
        tools=None,
        ref_audio: Optional[np.ndarray] = None,
        prompt_wav_path: Optional[str] = None,
        generate_audio: Optional[bool] = None,
    ) -> dict:
        return self._require_fc_duplex().prepare(
            system_prompt=system_prompt,
            tools=tools,
            ref_audio=ref_audio,
            prompt_wav_path=prompt_wav_path,
            generate_audio=generate_audio,
        )

    def fc_duplex_streaming_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[List] = None,
        tool_responses=None,
        sample_rate: int = 16000,
        max_slice_nums: int = 1,
    ) -> dict:
        return self._require_fc_duplex().streaming_prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            tool_responses=tool_responses,
            sample_rate=sample_rate,
            max_slice_nums=max_slice_nums,
        )

    def fc_duplex_streaming_spoken_generate(
        self,
        max_tokens: int = 24,
        decode_mode: str = "greedy",
    ) -> dict:
        return self._require_fc_duplex().streaming_spoken_generate(
            max_tokens=max_tokens,
            decode_mode=decode_mode,
        )

    def fc_duplex_streaming_non_spoken_generate(
        self,
        decode_mode: str = "greedy",
        max_tokens: int = 1,
        close_reason: Optional[str] = None,
    ) -> dict:
        return self._require_fc_duplex().streaming_non_spoken_generate(
            decode_mode=decode_mode,
            max_tokens=max_tokens,
            close_reason=close_reason,
        )

    def fc_duplex_finalize_unit(self) -> dict:
        return self._require_fc_duplex().finalize_unit()

    def fc_duplex_decode_output_ids(self, output_ids=None, tools=None) -> dict:
        return self._require_fc_duplex().decode_output_ids(output_ids=output_ids, tools=tools)

    def fc_duplex_cleanup(self) -> None:
        self._require_fc_duplex().cleanup()

class FcDuplexCapability:
    """FC-slot duplex runtime, kept independent from the normal duplex path."""

    def __init__(
        self,
        model: "MiniCPMO",
        device: str = "cuda",
        temperature: float = 0.7,
        tool_format: str = "minicpm4_xml",
        forbidden_token_ids=None,
        **kwargs,
    ):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.tool_format = tool_format
        self.generate_audio = bool(kwargs.get("generate_audio", False))
        self.tts_temperature = torch.tensor(
            [kwargs.get("tts_temperature", 0.8)], dtype=torch.float, device=device
        )
        self.tts_repetition_penalty = kwargs.get("tts_repetition_penalty", 1.05)
        self.prompt_wav_path = None

        if not hasattr(model, "processor") or model.processor is None:
            model.processor = MiniCPMOProcessor.from_pretrained(
                model.config._name_or_path, trust_remote_code=True
            )
        self.processor = model.processor
        self.tokenizer = self.processor.tokenizer
        if forbidden_token_ids is None:
            forbidden_token_ids = [
                self.tokenizer.convert_tokens_to_ids("<|tts_pad|>"),
                *list(getattr(self.tokenizer, "bad_token_ids", [])),
            ]
        self.decoder = StreamDecoder(
            llm=model.llm,
            tokenizer=self.tokenizer,
            forbidden_token_ids=forbidden_token_ids,
        )

        self._sdk_tokenizer = None
        self._registry = None
        self._serializer = None
        self._normalize_tool_response_content = None
        self.K = None
        self.ids = {}
        self.id2name = {}
        self.max_special_id = 0
        self.tts_logits_processors = None
        self.tts_eos_token = None
        if getattr(model, "tts", None) is not None:
            self.tts_logits_processors = gen_logits(
                num_code=model.tts.config.num_audio_tokens,
                repetition_penalty=self.tts_repetition_penalty,
            )
            self.tts_eos_token = torch.tensor(
                [model.tts.config.num_audio_tokens - 1],
                dtype=torch.long,
                device=device,
            )
        self._reset_streaming_state()
        logger.info("[FcDuplexCapability] initialized")

    @property
    def protocol(self):
        self._ensure_protocol()
        return self

    def _ensure_protocol(self):
        if self._registry is not None:
            return
        from minicpm_o5_sdk import (
            get_o5_tool_serializer,
            load_builtin_o5_tokenizer,
            normalize_tool_response_content,
        )
        from minicpm_o5_sdk.protocols.duplex.special_tokens import (
            O5SpecialTokenKey,
            O5SpecialTokenRegistry,
        )

        self.K = O5SpecialTokenKey
        self._normalize_tool_response_content = normalize_tool_response_content
        self._sdk_tokenizer = load_builtin_o5_tokenizer()
        self._registry = O5SpecialTokenRegistry.from_tokenizer(self._sdk_tokenizer)
        self._serializer = get_o5_tool_serializer(self.tool_format)
        for key in self.K:
            try:
                resolved = self._registry.get(key)
            except Exception:
                continue
            self.ids[key.value] = resolved.token_id
            self.id2name[resolved.token_id] = resolved.display_name
        self.max_special_id = max(self.id2name) if self.id2name else 0
        logger.info(
            "[FcDuplexCapability] O5 protocol ready: %d special tokens, max_id=%d",
            len(self.id2name),
            self.max_special_id,
        )

    def sid(self, key) -> int:
        self._ensure_protocol()
        return self._registry.get(key).token_id

    def is_special(self, tid: int) -> bool:
        self._ensure_protocol()
        return tid in self.id2name

    def encode_text(self, text: str) -> list:
        self._ensure_protocol()
        return [t.token_id for t in self._sdk_tokenizer.encode_ordinary_with_offsets(text)] if text else []

    def decode_text(self, ids: list) -> str:
        self._ensure_protocol()
        return self._sdk_tokenizer.decode_ordinary(ids) if ids else ""

    def _flush(self, ids: list) -> str:
        try:
            return self.decode_text(ids)
        except Exception:
            parts = []
            for tid in ids:
                try:
                    parts.append(self.decode_text([tid]))
                except Exception:
                    parts.append(f"<id:{tid}>")
            return "".join(parts)

    def render_token_stream(self, ids: list) -> str:
        out, ordinary = [], []
        for tid in ids:
            if self.is_special(tid):
                if ordinary:
                    out.append(self._flush(ordinary))
                    ordinary = []
                out.append(self.id2name[tid])
            else:
                ordinary.append(tid)
        if ordinary:
            out.append(self._flush(ordinary))
        return "".join(out)

    def _normalize_tools(self, tools):
        if not tools:
            return None
        tools = list(tools) if isinstance(tools, (list, tuple)) else [tools]
        if all(not isinstance(tool, dict) for tool in tools):
            return tools
        try:
            from minicpm_o5_sdk import OpenAIToolDefinition
            return [
                OpenAIToolDefinition.model_validate(tool) if isinstance(tool, dict) else tool
                for tool in tools
            ]
        except Exception:
            return tools

    def _system_parts(self, system_prompt: str, tools=None, has_ref_audio=False):
        prefix = [self.sid(self.K.IM_START), *self.encode_text(system_prompt or "")]
        suffix = []
        if has_ref_audio:
            prefix.append(self.sid(self.K.AUDIO_START))
            suffix.append(self.sid(self.K.AUDIO_END))
        tools = self._normalize_tools(tools)
        if tools:
            block = self._serializer.render_tool_system_block(tools)
            suffix.extend(self.encode_text(block.preamble))
            suffix.extend(self.encode_text(block.definitions))
            suffix.extend(self.encode_text(block.guidelines))
        suffix.append(self.sid(self.K.IM_END))
        return prefix, suffix

    def _event_ids(self, events) -> list:
        if not events:
            return []
        ids = [self.sid(self.K.INPUT_EVENT_SLOT_START)]
        for item in events:
            if isinstance(item, dict):
                call_id = item.get("call_id") or item.get("id") or item.get("tool_call_id")
                event_type = item.get("type") or item.get("event") or "tool_response"
                raw = item.get("content") if "content" in item else item.get("response")
            else:
                call_id, raw = item
                event_type = "tool_response"
            if not call_id:
                raise ValueError("tool event missing call_id/tool_call_id")
            ids.extend([self.sid(self.K.TOOL_RESPONSE_EVENT_START), self.sid(self.K.TOOL_CALL_ID_START)])
            ids.extend(self.encode_text(str(call_id)))
            ids.append(self.sid(self.K.TOOL_CALL_ID_END))
            if event_type in ("tool_started", "started"):
                ids.extend([self.sid(self.K.TOOL_STARTED), self.sid(self.K.TOOL_RESPONSE_EVENT_END)])
                continue
            ids.append(self.sid(self.K.TOOL_RESPONSE_START))
            ids.extend(self.encode_text(self._normalize_tool_response_content(raw)))
            ids.extend([self.sid(self.K.TOOL_RESPONSE_END), self.sid(self.K.TOOL_RESPONSE_EVENT_END)])
        ids.append(self.sid(self.K.INPUT_EVENT_SLOT_END))
        return ids

    def _resize_embeddings(self) -> dict:
        self._ensure_protocol()
        required = self.max_special_id + 1
        current = int(self.model.llm.get_input_embeddings().weight.shape[0])
        info = {"old_vocab": current, "new_vocab": current, "resized": False, "need": required}
        if current < required:
            self.model.llm.resize_token_embeddings(required)
            new_size = int(self.model.llm.get_input_embeddings().weight.shape[0])
            info.update(new_vocab=new_size, resized=True)
            logger.warning(
                "[FcDuplexCapability] resized token embeddings %d -> %d to satisfy SDK "
                "O5 required size (max special id + 1). Newly added random embedding/lm_head "
                "rows are smoke-test-only and MUST NOT be used for FC quality evaluation.",
                current,
                new_size,
            )
        return info

    def _feed_ids(self, ids: list, want_logits=False):
        if not ids:
            return None
        self.output_ids.extend(ids)
        out = self.decoder.feed(self.decoder.embed_tokens(ids), return_logits=want_logits)
        return out[0] if want_logits else None

    def _feed_audio(self, waveform) -> int:
        if waveform is None or len(waveform) == 0:
            return 0
        data = self.processor.process_audio([np.asarray(waveform, dtype=np.float32)])
        nested = self.model.get_audio_embedding(
            data, chunk_length=self.model.config.audio_chunk_length
        )
        if not nested:
            return 0
        embeds = torch.cat([item for group in nested for item in group], dim=0)
        self.output_ids.extend([self.sid(self.K.AUDIO_PLACEHOLDER)] * int(embeds.shape[0]))
        self.decoder.feed(embeds)
        return int(embeds.shape[0])

    def _sample(self, logits, mode: str) -> int:
        if mode in ("greedy", "argmax"):
            return int(torch.argmax(logits[0]).item())
        probs = torch.softmax(logits[0] / max(float(self.temperature), 1e-5), dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def _safe_deserialize_tool_call(self, wire: str, tool_definitions=None) -> dict:
        definitions = self._normalize_tools(tool_definitions or self._tools)
        result = {"wire": wire, "name": None, "arguments": None, "error": None}
        if not definitions:
            result["error"] = "no tool definition available to deserialize"
            return result
        match = re.search(r'name="([^"]+)"', wire) or re.search(
            r'"name"\s*:\s*"([^"]+)"', wire
        )
        definition = definitions[0]
        if match:
            target = match.group(1)
            definition = next(
                (
                    item
                    for item in definitions
                    if getattr(getattr(item, "function", None), "name", None) == target
                ),
                definition,
            )
        try:
            call = self._serializer.deserialize_tool_call(wire, definition=definition)
            result["name"] = call.function.name
            result["arguments"] = call.function.arguments
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    def _reset_streaming_state(self):
        self.decoder.reset()
        self.output_ids = []
        self.units_info = []
        self._tools = None
        self._current_unit_idx = 0
        self._current_unit_open = False
        self._current_unit_info = None
        self._spoken_slot_open = False
        self._non_spoken_slot_open = False
        self._spoken_logits = None
        self._non_spoken_logits = None
        self._non_spoken_mode = None
        self._think_buf, self._tool_call_buf = [], []
        self.tts_text_start_pos = 0
        self.tts_past_key_values = None
        self.token2wav_initialized = False
        self.token2wav_buffer = []
        self.flow_cache_base = self.hift_cache_base = None
        self.pre_lookahead = 0

    def _init_token2wav_cache(self):
        if getattr(self.model, "tts", None) is None:
            raise RuntimeError("TTS model is not initialized")
        if getattr(self.model.tts, "audio_tokenizer", None) is None:
            self.model.init_token2wav(
                streaming=True,
                n_timesteps=getattr(self.model.config.tts_config, "s3_stream_n_timesteps", 10),
            )
        tokenizer = self.model.tts.audio_tokenizer
        tokenizer.cache = None
        flow_cache, hift_cache = tokenizer.set_stream_cache(self.prompt_wav_path)
        self.flow_cache_base = torch_clone_recursive(flow_cache)
        self.hift_cache_base = torch_clone_recursive(hift_cache)
        self.pre_lookahead = int(tokenizer.flow.pre_lookahead_len)
        self.token2wav_initialized = True

    def _reset_token2wav(self):
        if self.token2wav_initialized:
            tokenizer = self.model.tts.audio_tokenizer
            tokenizer.stream_cache = torch_clone_recursive(self.flow_cache_base)
            tokenizer.hift_cache_dict = torch_clone_recursive(self.hift_cache_base)
            self.token2wav_buffer = [4218] * 3

    def _tts_condition(self, results):
        tts = self.model.tts
        if not results:
            bos = tts.emb_text(torch.tensor(
                [tts.audio_bos_token_id], device=tts.emb_text.weight.device, dtype=torch.long
            ))
            return bos.unsqueeze(0)
        tokens = torch.tensor([x[0] for x in results], device=tts.emb_text.weight.device)
        hidden = torch.cat([x[1].squeeze(0) for x in results], dim=0)
        hidden = tts.projector_semantic(hidden)
        if getattr(tts.config, "normalize_projected_hidden", False):
            hidden = self.model._normalize_projected(hidden, tts.projector_semantic_norm)
        else:
            hidden = F.normalize(hidden, p=2, dim=-1)
        merged = tts.emb_text(tokens) + hidden
        bos = tts.emb_text(torch.tensor(
            [tts.audio_bos_token_id], device=tts.emb_text.weight.device, dtype=torch.long
        ))
        return torch.cat([merged, bos], dim=0).unsqueeze(0)

    def _spoken_audio(self, results, end_of_turn: bool) -> dict:
        empty = {
            "audio_waveform": None, "audio_sample_rate": None, "n_tts_tokens": 0,
            "cost_tts_prep": 0.0, "cost_tts": 0.0, "cost_token2wav": 0.0,
        }
        if not self.generate_audio or (not results and not end_of_turn):
            return empty
        if not self.prompt_wav_path:
            raise ValueError("prompt_wav_path is required when generate_audio=True")
        if not self.token2wav_initialized:
            self._init_token2wav_cache()
            self._reset_token2wav()
        prep = time.time()
        condition = self._tts_condition(results)
        prep_cost = time.time() - prep
        tts_start = time.time()
        new_tokens, old_kv = self.model.tts.generate_chunk(
            inputs_embeds=condition,
            temperature=self.tts_temperature,
            repetition_penalty=self.tts_repetition_penalty,
            eos_token=self.tts_eos_token,
            force_no_stop=False,
            max_new_token=26,
            min_new_tokens=0 if end_of_turn or self.tts_text_start_pos == 0 else 26,
            past_key_values=self.tts_past_key_values,
            logits_processors=self.tts_logits_processors,
            text_start_pos=self.tts_text_start_pos,
        )
        tts_cost = time.time() - tts_start
        if end_of_turn:
            self.tts_text_start_pos, self.tts_past_key_values = 0, None
        else:
            self.tts_past_key_values = old_kv
            self.tts_text_start_pos += condition.shape[1] + new_tokens.shape[1]
        wav_start = time.time()
        token_ids = new_tokens.reshape(-1).tolist()
        self.token2wav_buffer.extend(token_ids)
        pcm = []
        chunk_size = 25
        while len(self.token2wav_buffer) >= chunk_size + self.pre_lookahead:
            pcm.append(self.model.tts.audio_tokenizer.stream(
                self.token2wav_buffer[:chunk_size + self.pre_lookahead],
                prompt_wav=self.prompt_wav_path,
            ))
            self.token2wav_buffer = self.token2wav_buffer[chunk_size:]
        if end_of_turn and self.token2wav_buffer:
            pcm.append(self.model.tts.audio_tokenizer.stream(
                self.token2wav_buffer, prompt_wav=self.prompt_wav_path, last_chunk=True
            ))
            self.token2wav_buffer = []
            self._reset_token2wav()
        raw = b"".join(pcm)
        waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0 if raw else None
        return {
            "audio_waveform": waveform,
            "audio_sample_rate": 24000 if waveform is not None else None,
            "n_tts_tokens": int(new_tokens.numel()),
            "cost_tts_prep": prep_cost,
            "cost_tts": tts_cost,
            "cost_token2wav": time.time() - wav_start,
        }

    def prepare(self, system_prompt, tools=None, ref_audio=None, prompt_wav_path=None, generate_audio=None):
        resize_info = self._resize_embeddings()
        self._reset_streaming_state()
        self._tools = self._normalize_tools(tools)
        if generate_audio is not None:
            self.generate_audio = bool(generate_audio)
        self.prompt_wav_path = prompt_wav_path
        self.model.init_streaming_processor()
        if self.generate_audio:
            if not prompt_wav_path:
                raise ValueError("prompt_wav_path is required when generate_audio=True")
            self._init_token2wav_cache()
            self._reset_token2wav()
        prefix, suffix = self._system_parts(system_prompt, self._tools, ref_audio is not None)
        self._feed_ids(prefix)
        if ref_audio is not None:
            data = self.processor.process_audio([np.asarray(ref_audio, dtype=np.float32)])
            nested = self.model.get_audio_embedding(
                data, chunk_length=self.model.config.audio_chunk_length
            )
            if nested:
                self.decoder.feed(torch.cat([x for group in nested for x in group], dim=0))
        self._feed_ids(suffix)
        return {
            "prefill_ids": prefix + suffix,
            "resize_info": resize_info,
            "output_render": self.render_token_stream(prefix + suffix),
            "generate_audio": self.generate_audio,
            "prompt_wav_path": prompt_wav_path,
            "has_ref_audio": ref_audio is not None,
        }

    def streaming_prefill(
        self, audio_waveform=None, frame_list=None, tool_responses=None,
        sample_rate=16000, max_slice_nums=1
    ):
        del frame_list, sample_rate, max_slice_nums
        if self._current_unit_open:
            self.finalize_unit()
        info = {
            "unit": self._current_unit_idx, "n_audio": 0,
            "has_event": bool(tool_responses), "is_listen": None,
            "is_speaking": False, "spoken_ids": [], "non_spoken_ids": [],
            "non_spoken_terminator": None, "closed_spans": [],
        }
        self._current_unit_info = info
        self._feed_ids([self.sid(self.K.UNIT_START)])
        self._current_unit_open = True
        if audio_waveform is not None and len(audio_waveform):
            self._feed_ids([self.sid(self.K.USER_AUDIO_SLOT_START)])
            info["n_audio"] = self._feed_audio(audio_waveform)
            self._feed_ids([self.sid(self.K.USER_AUDIO_SLOT_END)])
        if tool_responses:
            self._feed_ids(self._event_ids(tool_responses))
        self._spoken_logits = self._feed_ids([self.sid(self.K.AI_SPOKEN_SLOT_START)], True)
        self._spoken_slot_open = True
        return dict(info)

    def streaming_spoken_generate(self, max_tokens=24, decode_mode="greedy"):
        if not self._spoken_slot_open:
            raise RuntimeError("spoken slot is not open; call streaming_prefill() first")
        start, logits = time.time(), self._spoken_logits
        ids, text_ids, tts_results = [], [], []
        is_listen = is_speaking = turn_eos = terminated = False
        reason = term_id = None
        terms = {
            self.sid(self.K.SPOKEN_SLOT_EOS): "spoken_slot_eos",
            self.sid(self.K.LISTEN): "listen",
            self.sid(self.K.TTS_PAD): "tts_pad",
        }
        for _ in range(max_tokens):
            tid = self._sample(logits, decode_mode)
            if tid == self.sid(self.K.AI_SPOKEN_SLOT_END):
                terminated, reason, term_id = True, "ai_spoken_slot_end", tid
                break
            ids.append(tid)
            self.output_ids.append(tid)
            is_listen |= tid == self.sid(self.K.LISTEN)
            is_speaking |= tid == self.sid(self.K.SPEAK)
            turn_eos |= tid == self.sid(self.K.SPOKEN_TURN_EOS)
            if not self.is_special(tid):
                text_ids.append(tid)
            logits, hidden = self.decoder.feed(self.decoder.embed_token(tid), return_logits=True)
            if is_speaking and not self.is_special(tid):
                tts_results.append((tid, hidden, turn_eos))
            if tid in terms:
                terminated, reason, term_id = True, terms[tid], tid
                break
        self._feed_ids([self.sid(self.K.AI_SPOKEN_SLOT_END)])
        self._spoken_slot_open = False
        audio = self._spoken_audio(tts_results, turn_eos)
        self._current_unit_info.update(
            is_listen=bool(is_listen), is_speaking=bool(is_speaking), spoken_ids=ids,
            spoken_slot_terminated=terminated, spoken_termination_reason=reason,
            spoken_termination_token_id=term_id,
            spoken_slot_unterminated=not terminated,
            spoken_generation_reached_max_tokens=len(ids) >= max_tokens and not terminated,
        )
        text = self._flush(text_ids) if text_ids else ""
        return {
            "is_listen": bool(is_listen), "is_speaking": bool(is_speaking),
            "spoken_ids": ids, "spoken_text": text, "text": text,
            "spoken_turn_eos": bool(turn_eos), "end_of_turn": bool(turn_eos),
            "spoken_slot_terminated": terminated,
            "spoken_slot_unterminated": not terminated,
            "spoken_generation_reached_max_tokens": len(ids) >= max_tokens and not terminated,
            "spoken_termination_reason": reason, "spoken_termination_token_id": term_id,
            "cost_llm": time.time() - start, **audio,
        }

    def _open_non_spoken(self):
        if not self._non_spoken_slot_open:
            self._non_spoken_logits = self._feed_ids(
                [self.sid(self.K.AI_NON_SPOKEN_SLOT_START)], True
            )
            self._non_spoken_slot_open = True

    def streaming_non_spoken_generate(self, decode_mode="greedy", max_tokens=1, close_reason=None):
        self._open_non_spoken()
        close_map = {
            "eos": self.K.NON_SPOKEN_EOS, "no_action": self.K.NO_ACTION,
            "budget_reached": self.K.NON_SPOKEN_BUDGET_REACHED,
            "hold": self.K.NON_SPOKEN_HOLD, "abort": self.K.NON_SPOKEN_ABORT,
        }
        if close_reason is not None:
            if close_reason not in close_map:
                raise ValueError(f"unsupported non-spoken close reason: {close_reason}")
            tids = [self.sid(close_map[close_reason])]
            self.output_ids.extend(tids)
            self._current_unit_info["non_spoken_ids"].extend(tids)
            self.decoder.feed(self.decoder.embed_token(tids[0]))
            terminated, reason, spans = True, close_reason, []
        else:
            tids, spans, terminated, reason = [], [], False, None
            natural = {
                self.sid(self.K.NON_SPOKEN_EOS): "eos",
                self.sid(self.K.NO_ACTION): "no_action",
                self.sid(self.K.NON_SPOKEN_ABORT): "abort",
            }
            for _ in range(max_tokens):
                tid = self._sample(self._non_spoken_logits, decode_mode)
                if tid == self.sid(self.K.AI_NON_SPOKEN_SLOT_END):
                    terminated, reason = True, "eos"
                    break
                tids.append(tid)
                self.output_ids.append(tid)
                self._current_unit_info["non_spoken_ids"].append(tid)
                if self._non_spoken_mode is None and tid in (
                    self.sid(self.K.THINK_START), self.sid(self.K.TOOL_CALL_START)
                ):
                    self._non_spoken_mode = "think" if tid == self.sid(self.K.THINK_START) else "tool_call"
                    self._think_buf, self._tool_call_buf = [], []
                elif self._non_spoken_mode == "think":
                    if tid == self.sid(self.K.THINK_END):
                        spans.append({"type": "think", "text": self._flush(self._think_buf)})
                        self._non_spoken_mode = None
                    elif not self.is_special(tid):
                        self._think_buf.append(tid)
                elif self._non_spoken_mode == "tool_call":
                    if tid == self.sid(self.K.TOOL_CALL_END):
                        wire = self._flush(self._tool_call_buf)
                        spans.append({
                            "type": "tool_call",
                            "wire": wire,
                            "tool_call": self._safe_deserialize_tool_call(wire),
                        })
                        self._non_spoken_mode = None
                    elif not self.is_special(tid):
                        self._tool_call_buf.append(tid)
                self._non_spoken_logits = self.decoder.feed(
                    self.decoder.embed_token(tid), return_logits=True
                )[0]
                if tid in natural:
                    terminated, reason = True, natural[tid]
                    break
        if terminated:
            self._feed_ids([self.sid(self.K.AI_NON_SPOKEN_SLOT_END)])
            self._non_spoken_slot_open = False
            self._current_unit_info["non_spoken_terminator"] = reason
        self._current_unit_info["closed_spans"].extend(spans)
        ordinary = [tid for tid in tids if not self.is_special(tid)]
        return {
            "token_ids": tids, "terminated": terminated, "close_reason": reason,
            "closed_spans": spans, "text": self._flush(ordinary) if ordinary else "",
        }

    def finalize_unit(self):
        if not self._current_unit_open:
            return {}
        if self._non_spoken_slot_open:
            self.streaming_non_spoken_generate(close_reason="budget_reached")
        if self._spoken_slot_open:
            self._feed_ids([self.sid(self.K.AI_SPOKEN_SLOT_END)])
            self._spoken_slot_open = False
        self._feed_ids([self.sid(self.K.UNIT_END)])
        info = dict(self._current_unit_info)
        self.units_info.append(info)
        self._current_unit_idx += 1
        self._current_unit_info = None
        self._current_unit_open = False
        return info

    def decode_output_ids(self, output_ids=None, tools=None):
        self._ensure_protocol()
        ids = list(self.output_ids if output_ids is None else output_ids)
        K = self.K
        unit_start = self.sid(K.UNIT_START)
        unit_end = self.sid(K.UNIT_END)
        spoken_start = self.sid(K.AI_SPOKEN_SLOT_START)
        spoken_end = self.sid(K.AI_SPOKEN_SLOT_END)
        non_spoken_start = self.sid(K.AI_NON_SPOKEN_SLOT_START)
        non_spoken_end = self.sid(K.AI_NON_SPOKEN_SLOT_END)
        listen = self.sid(K.LISTEN)
        speak = self.sid(K.SPEAK)
        spoken_terminators = {
            self.sid(K.SPOKEN_SLOT_EOS),
            self.sid(K.SPOKEN_TURN_EOS),
            self.sid(K.TTS_PAD),
        }
        think_start = self.sid(K.THINK_START)
        think_end = self.sid(K.THINK_END)
        tool_call_start = self.sid(K.TOOL_CALL_START)
        tool_call_end = self.sid(K.TOOL_CALL_END)
        no_action = self.sid(K.NO_ACTION)
        non_spoken_terminators = {
            self.sid(K.NON_SPOKEN_EOS): "eos",
            self.sid(K.NON_SPOKEN_BUDGET_REACHED): "budget_reached",
            self.sid(K.NON_SPOKEN_HOLD): "hold",
            self.sid(K.NON_SPOKEN_ABORT): "abort",
        }

        units = []
        current = None
        slot = None
        spoken_buffer = []
        non_spoken_mode = None
        think_buffer = []
        tool_call_buffer = []
        completed_thoughts = []
        tool_calls = []

        def new_unit():
            return {
                "is_listen": None,
                "spoken_text": "",
                "non_spoken_terminator": None,
                "raw_non_spoken": "",
            }

        for tid in ids:
            if tid == unit_start:
                current = new_unit()
                slot = None
                continue
            if tid == unit_end:
                if current is not None:
                    if spoken_buffer:
                        current["spoken_text"] += self._flush(spoken_buffer)
                    units.append(current)
                current = None
                slot = None
                spoken_buffer = []
                continue
            if current is None:
                continue
            if tid == spoken_start:
                slot = "spoken"
                spoken_buffer = []
                if current["is_listen"] is None:
                    current["is_listen"] = False
                continue
            if tid == spoken_end:
                current["spoken_text"] += self._flush(spoken_buffer) if spoken_buffer else ""
                spoken_buffer = []
                slot = None
                continue
            if tid == non_spoken_start:
                slot = "non_spoken"
                continue
            if tid == non_spoken_end:
                slot = None
                continue
            if slot == "spoken":
                if tid == listen:
                    current["is_listen"] = True
                elif tid == speak:
                    current["is_listen"] = False
                elif tid in spoken_terminators:
                    pass
                elif not self.is_special(tid):
                    spoken_buffer.append(tid)
                continue
            if slot == "non_spoken":
                current["raw_non_spoken"] += (
                    self.id2name[tid] if self.is_special(tid) else self._flush([tid])
                )
                if non_spoken_mode is None:
                    if tid == think_start:
                        non_spoken_mode = "think"
                        think_buffer = []
                    elif tid == tool_call_start:
                        non_spoken_mode = "tool_call"
                        tool_call_buffer = []
                    elif tid == no_action:
                        current["non_spoken_terminator"] = "no_action"
                    elif tid in non_spoken_terminators:
                        current["non_spoken_terminator"] = non_spoken_terminators[tid]
                elif non_spoken_mode == "think":
                    if tid == think_end:
                        completed_thoughts.append(self._flush(think_buffer) if think_buffer else "")
                        non_spoken_mode = None
                    elif tid in non_spoken_terminators:
                        current["non_spoken_terminator"] = non_spoken_terminators[tid]
                    elif not self.is_special(tid):
                        think_buffer.append(tid)
                elif non_spoken_mode == "tool_call":
                    if tid == tool_call_end:
                        wire = self._flush(tool_call_buffer)
                        tool_calls.append(
                            self._safe_deserialize_tool_call(wire, tools or self._tools)
                        )
                        non_spoken_mode = None
                    elif tid in non_spoken_terminators:
                        current["non_spoken_terminator"] = non_spoken_terminators[tid]
                    elif not self.is_special(tid):
                        tool_call_buffer.append(tid)

        if non_spoken_mode == "think" and think_buffer:
            completed_thoughts.append(self._flush(think_buffer))
        elif non_spoken_mode == "tool_call" and tool_call_buffer:
            tool_calls.append(
                self._safe_deserialize_tool_call(
                    self._flush(tool_call_buffer),
                    tools or self._tools,
                )
            )

        return {
            "units": units,
            "spoken_text": "".join(unit["spoken_text"] for unit in units),
            "think_text": "".join(completed_thoughts),
            "tool_calls": tool_calls,
            "output_ids": ids,
            "output_render": self.render_token_stream(ids),
        }

    def cleanup(self):
        self._reset_streaming_state()
