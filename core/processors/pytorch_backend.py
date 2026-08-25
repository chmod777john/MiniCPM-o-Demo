"""PyTorch MiniCPM-o backend implementation for session runtimes."""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional

import numpy as np
import soundfile as sf
import torch

from core.schemas.chat import ChatRequest
from core.schemas.common import GenerationConfig, ImageConfig, TTSConfig, TTSMode
from core.schemas.metrics import BackendMetrics
from core.schemas.common import Message
from core.schemas.duplex import DuplexConfig, DuplexGenerateResult
from core.schemas.fc_duplex import (
    FcDuplexPrepareRequest,
    FcDuplexPrepareResult,
    FcDuplexPrefillRequest,
    FcFinalizeUnitRequest,
    FcNonSpokenGenerateRequest,
    FcSpokenGenerateRequest,
    FcToolResponse,
)
from core.fc_duplex.system_input import (
    FcAudioPathInput,
    FcSystemContentInput,
)
from core.schemas.streaming import StreamingChunk, StreamingRequest, StreamingResponse
from core.sampling import tts_argmax_scope

logger = logging.getLogger("pytorch_backend")


class PyTorchBackend:
    """modeling.o5 PyTorch inference backend.

    持有一个 UnifiedProcessor 实例，提供三种推理模式。
    """

    def __init__(
        self,
        model_path: str,
        gpu_id: int,
        pt_path: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        duplex_pause_timeout: float = 60.0,
        compile: bool = False,
        chat_vocoder: str = "token2wav",
        attn_implementation: str = "auto",
        fc_model_family: Literal["o45", "o5"] = "o5",
    ):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.pt_path = pt_path
        self.ref_audio_path = ref_audio_path
        self.duplex_pause_timeout = duplex_pause_timeout
        self.compile = compile
        self.chat_vocoder = chat_vocoder
        self.attn_implementation = attn_implementation
        self.fc_model_family = fc_model_family

        self.status = "loading"
        self.processor = None
        self.spmd_is_driver = False
        self.spmd_is_worker = False
        self._trace_controller: Optional[Any] = None
        self._trace_writer: Optional[Any] = None
        self._trace_capture_mode = "tokens"
        self._token_trace_path: Optional[Path] = None
        self._token_trace_dir: Optional[Path] = None
        self._trace_unit_index = -1

        # Duplex 暂停超时监控 task
        self._duplex_timeout_task: Optional[asyncio.Task] = None

    def load_model(self) -> None:
        """加载模型（同步，在启动时调用）"""
        self.status = "loading"
        logger.info(f"[GPU {self.gpu_id}] Loading model from {self.model_path}...")
        startup_seed = os.environ.get("O5_STARTUP_SEED")
        if startup_seed is not None:
            self._seed_process(int(startup_seed))
            logger.info("[GPU %s] Startup seed set to %s", self.gpu_id, startup_seed)
        if os.environ.get("O5_CAPTURE_LAYERS", "0").lower() in {"1", "true", "yes", "on"}:
            os.environ["O5_LAYER_TRACE"] = "1"

        from core.processors.unified import UnifiedProcessor

        self.processor = UnifiedProcessor(
            model_path=self.model_path,
            pt_path=self.pt_path,
            ref_audio_path=self.ref_audio_path,
            compile=self.compile,
            chat_vocoder=self.chat_vocoder,
            attn_implementation=self.attn_implementation,
            fc_model_family=self.fc_model_family,
        )

        gc.collect()
        torch.cuda.empty_cache()

        self.status = "ready"
        logger.info(f"[GPU {self.gpu_id}] Model loaded successfully")

        self._install_spmd_method_wrappers()
        self._install_token_trace_if_requested()

        # 检查模型各组件的 device 分布
        self._log_device_map()

    @staticmethod
    def _seed_process(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def set_trace_unit_id(self, unit_id: Optional[str]) -> None:
        if self._trace_controller is None:
            return
        if unit_id is not None:
            self._trace_unit_index += 1
        self._trace_controller.set_unit(unit_id, self._trace_unit_index if unit_id is not None else None)

    @staticmethod
    def _safe_trace_session_id(session_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(session_id))
        return safe[:128] or "session"

    def set_trace_session_id(self, session_id: Optional[str]) -> None:
        if self._trace_controller is None:
            return
        if self._trace_writer is not None:
            remaining = self._trace_controller.drain()
            if remaining:
                self._trace_writer.append(remaining)
            self._trace_writer.close()
            self._trace_writer = None
        self._trace_controller.set_session(session_id)
        self._trace_unit_index = -1
        if session_id is None or self._trace_capture_mode != "replay":
            return

        from core.tracing import SessionBundleWriter

        if self._token_trace_dir is not None:
            root = self._token_trace_dir / self._safe_trace_session_id(session_id)
        elif self._token_trace_path is not None:
            root = self._token_trace_path.parent
        else:
            return
        self._trace_writer = SessionBundleWriter(
            root,
            source_implementation="demo-api",
            manifest_extra={
                "session_id": session_id,
                "capture_mode": self._trace_capture_mode,
                "deployment_mode": os.environ.get("O5_DEPLOY_MODE", "single_eager"),
                "model_path": self.model_path,
                "checkpoint": self.pt_path,
            },
        )
        session_events = self._trace_controller.drain()
        if session_events:
            self._trace_writer.append(session_events)

    def drain_trace_events(self, unit_id: Optional[str]) -> Optional[list[Dict[str, Any]]]:
        if self._trace_controller is None:
            return None
        from core.tracing import debug_trace_events

        events = self._trace_controller.drain(unit_id)
        if not events:
            return None
        if self._trace_writer is not None:
            self._trace_writer.append(events)
        return debug_trace_events(events) or None

    def _install_token_trace_if_requested(self) -> None:
        trace_path = os.environ.get("O5_TOKEN_TRACE_PATH")
        trace_dir = os.environ.get("O5_TOKEN_TRACE_DIR")
        if not (trace_path or trace_dir) or self.processor is None:
            return
        if self.spmd_is_worker:
            logger.info("[GPU %s] Session trace disabled on TP2 worker rank", self.gpu_id)
            return

        model = getattr(self.processor, "model", None)
        duplex = getattr(model, "duplex", None)
        if model is None or duplex is None:
            logger.warning("O5_TOKEN_TRACE_PATH set but duplex model is unavailable")
            return
        fc_duplex = getattr(model, "fc_duplex", None)

        from core.tracing import (
            DuplexTraceController,
            ForcingPolicy,
            MemoryTraceSink,
            ReplayReference,
        )

        self._token_trace_dir = Path(trace_dir) if trace_dir else None
        self._token_trace_path = Path(trace_path) if trace_path else None
        self._trace_capture_mode = os.environ.get("O5_SESSION_TRACE_MODE", "tokens").strip().lower()
        reference_path = os.environ.get("O5_REPLAY_REFERENCE")
        forcing = ForcingPolicy.parse(os.environ.get("O5_REPLAY_FORCING", "none"))
        if forcing.names() and not reference_path:
            raise RuntimeError("O5_REPLAY_FORCING requires O5_REPLAY_REFERENCE")
        reference = ReplayReference.load(Path(reference_path)) if reference_path else None
        self._trace_controller = DuplexTraceController(
            sink=MemoryTraceSink(),
            capture_mode=self._trace_capture_mode,
            reference=reference,
            forcing=forcing,
            tp_driver=self.spmd_is_driver,
            capture_layers=os.environ.get("O5_CAPTURE_LAYERS", "0").lower() in {"1", "true", "yes", "on"},
        ).install(duplex, fc_duplex=fc_duplex)
        target = self._token_trace_path if self._token_trace_path is not None else f"{self._token_trace_dir}/<session>"
        logger.info(
            "[GPU %s] Session trace enabled: target=%s mode=%s forcing=%s reference=%s",
            self.gpu_id,
            target,
            self._trace_capture_mode,
            forcing.names(),
            reference_path,
        )

    def _get_spmd_mirror(self) -> Any:
        model = getattr(self.processor, "model", None)
        return getattr(model, "_spmd_mirror", None)

    def _get_spmd_worker_loop(self) -> Any:
        model = getattr(self.processor, "model", None)
        return getattr(model, "_spmd_worker_loop", None)

    def _get_spmd_shutdown(self) -> Any:
        model = getattr(self.processor, "model", None)
        return getattr(model, "_spmd_shutdown", None)

    def _get_spmd_noop(self) -> Any:
        model = getattr(self.processor, "model", None)
        return getattr(model, "_spmd_noop", None)

    def _install_spmd_method_wrappers(self) -> None:
        """Mark driver backends that need idle SPMD heartbeat no-op calls.

        TP2 synchronization is intentionally confined to the LLM/graph runner
        boundary.  Backend chat/duplex/FC APIs should remain single-rank business
        logic and must not be mirrored wholesale.
        """
        deployment = getattr(self.processor, "_deploy", None)
        if deployment is not None and int(getattr(deployment, "world_size", 1)) > 1:
            self.spmd_is_driver = bool(getattr(deployment, "is_driver", False))
            self.spmd_is_worker = not self.spmd_is_driver
            return
        if self._get_spmd_noop() is not None:
            self.spmd_is_driver = True
            self.spmd_is_worker = False

    def call_spmd_noop(self) -> None:
        noop = self._get_spmd_noop()
        if noop is not None:
            noop()

    def _log_device_map(self) -> None:
        """打印模型各关键组件的 device，用于确认是否全部在 GPU 上"""
        if self.processor is None:
            return
        model = self.processor.model
        checks: list[tuple[str, str]] = []

        # LLM
        try:
            p = next(model.llm.parameters())
            checks.append(("LLM", str(p.device)))
        except Exception:
            checks.append(("LLM", "N/A"))

        # Vision encoder
        try:
            p = next(model.vpm.parameters())
            checks.append(("Vision (vpm)", str(p.device)))
        except Exception:
            checks.append(("Vision (vpm)", "N/A"))

        # Whisper / audio encoder
        for name in ("apm", "audio_encoder", "whisper"):
            if hasattr(model, name):
                try:
                    p = next(getattr(model, name).parameters())
                    checks.append((f"Audio ({name})", str(p.device)))
                except Exception:
                    checks.append((f"Audio ({name})", "no params"))
                break

        # TTS 模块
        if hasattr(model, "tts"):
            tts = model.tts
            # TTS 主体
            try:
                p = next(tts.parameters())
                checks.append(("TTS (main)", str(p.device)))
            except Exception:
                checks.append(("TTS (main)", "N/A"))

            # audio_tokenizer (Token2Wav 关键组件)
            if hasattr(tts, "audio_tokenizer"):
                tok = tts.audio_tokenizer
                try:
                    p = next(tok.parameters())
                    checks.append(("TTS audio_tokenizer", str(p.device)))
                except Exception:
                    checks.append(("TTS audio_tokenizer", "no params"))

                # hift (vocoder in Token2Wav)
                if hasattr(tok, "hift"):
                    try:
                        p = next(tok.hift.parameters())
                        checks.append(("TTS hift (vocoder)", str(p.device)))
                    except Exception:
                        checks.append(("TTS hift (vocoder)", "no params"))

            # CosyVoice2 / flow model
            for attr_name in ("cosyvoice", "cosyvoice2", "flow"):
                if hasattr(tts, attr_name):
                    try:
                        p = next(getattr(tts, attr_name).parameters())
                        checks.append((f"TTS {attr_name}", str(p.device)))
                    except Exception:
                        checks.append((f"TTS {attr_name}", "no params"))

        # Duplex decoder
        if hasattr(model, "duplex") and model.duplex is not None:
            try:
                p = next(model.duplex.decoder.parameters())
                checks.append(("Duplex decoder", str(p.device)))
            except Exception:
                checks.append(("Duplex decoder", "N/A"))

        logger.info(f"[GPU {self.gpu_id}] === Device Map ===")
        for name, device in checks:
            on_gpu = "cuda" in device
            marker = "✓" if on_gpu else "⚠ CPU!"
            logger.info(f"[GPU {self.gpu_id}]   {marker} {name}: {device}")

    # ========== Runtime backend surface ==========

    def metrics(self) -> Dict[str, Any]:
        """Return a sampled PyTorch backend metric snapshot."""
        if self.processor is None:
            return BackendMetrics(backend="pytorch").to_dict()
        return BackendMetrics(
            backend="pytorch",
            kv_cache_length=int(getattr(self.processor, "kv_cache_length", 0) or 0),
        ).to_dict()

    def chat_prefill(
        self,
        session_id: str,
        msgs: list,
        omni_mode: bool = False,
        max_slice_nums: Optional[int] = None,
        use_tts_template: bool = False,
        enable_thinking: bool = False,
    ) -> str:
        chat_view = self.processor.set_chat_mode()
        return chat_view.prefill(
            session_id=session_id,
            msgs=msgs,
            omni_mode=omni_mode,
            max_slice_nums=max_slice_nums,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
        )

    def chat_init_tts(self, ref_audio: Optional[np.ndarray]) -> None:
        if ref_audio is not None:
            self.processor.model.init_token2wav_cache(prompt_speech_16k=ref_audio)
            return

        if self.ref_audio_path:
            import librosa

            loaded_ref, _ = librosa.load(self.ref_audio_path, sr=16000, mono=True)
            self.processor.model.init_token2wav_cache(prompt_speech_16k=loaded_ref)

    def chat_streaming_generate(
        self,
        session_id: str,
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        length_penalty: float = 1.0,
    ) -> Iterator[StreamingChunk]:
        chat_view = self.processor.set_chat_mode()
        yield from chat_view.streaming_generate(
            session_id=session_id,
            generate_audio=generate_audio,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )

    def chat_non_streaming_generate(
        self,
        session_id: str,
        max_new_tokens: int = 256,
        generate_audio: bool = False,
        use_tts_template: bool = True,
        enable_thinking: bool = False,
        tts_ref_audio: Optional[np.ndarray] = None,
        length_penalty: float = 1.0,
    ) -> Any:
        chat_view = self.processor.set_chat_mode()
        return chat_view.generate(
            session_id=session_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            generate_audio=generate_audio,
            use_tts_template=use_tts_template,
            enable_thinking=enable_thinking,
            tts_ref_audio=tts_ref_audio,
            tts_sampling_params=None,
            length_penalty=length_penalty,
        )

    def chat_complete(
        self,
        messages: List[Message],
        max_new_tokens: int = 256,
        generate_audio: bool = False,
        use_tts_template: bool = True,
        omni_mode: bool = False,
        max_slice_nums: Optional[int] = None,
        enable_thinking: bool = False,
        tts_ref_audio: Optional[np.ndarray] = None,
        length_penalty: float = 1.0,
    ) -> Any:
        chat_view = self.processor.set_chat_mode()
        tts_config = TTSConfig(enabled=generate_audio, mode=TTSMode.AUDIO_ASSISTANT)
        if generate_audio:
            if tts_ref_audio is not None:
                ref_audio_f32 = np.asarray(tts_ref_audio, dtype=np.float32)
                tts_config = tts_config.model_copy(
                    update={"ref_audio_data": base64.b64encode(ref_audio_f32.tobytes()).decode("utf-8")}
                )
            elif self.ref_audio_path:
                tts_config = tts_config.model_copy(update={"ref_audio_path": self.ref_audio_path})

        response = chat_view.chat(
            ChatRequest(
                messages=messages,
                generation=GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    length_penalty=length_penalty,
                ),
                tts=tts_config,
                image=ImageConfig(max_slice_nums=max_slice_nums),
                use_tts_template=use_tts_template,
                omni_mode=omni_mode,
                enable_thinking=enable_thinking,
            ),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            generate_audio=generate_audio,
        )

        waveform = None
        if response.audio_data:
            wav_bytes = base64.b64decode(response.audio_data)
            waveform, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        return response.text, waveform

    def set_duplex_config(self, config: Optional[Dict[str, Any]]) -> None:
        if self.processor is None or not config:
            return
        duplex_view = self.processor.set_duplex_mode()
        duplex_view.config = DuplexConfig(**config)
        duplex_view.apply_config_to_model()

    def seed_runtime(self, seed: int) -> None:
        self._seed_process(seed)

    def _run_duplex_generate(self, force_listen: bool) -> DuplexGenerateResult:
        duplex_view = self.processor.set_duplex_mode()
        return duplex_view.generate(force_listen=force_listen)

    def duplex_prepare(
        self,
        system_prompt_text: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
        length_penalty: float = 1.0,
        sampling: Optional[Dict[str, Any]] = None,
        llm_seed: Optional[int] = None,
    ) -> str:
        if sampling:
            self.set_duplex_config(sampling)
        duplex_view = self.processor.set_duplex_mode()
        return duplex_view.prepare(
            system_prompt_text=system_prompt_text,
            ref_audio_path=ref_audio_path or self.ref_audio_path,
            prompt_wav_path=prompt_wav_path,
            llm_seed=llm_seed,
        )

    def duplex_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[list] = None,
        max_slice_nums: int = 1,
    ) -> Dict[str, Any]:
        duplex_view = self.processor.set_duplex_mode()
        return duplex_view.prefill(
            audio_waveform=audio_waveform,
            frame_list=frame_list,
            max_slice_nums=max_slice_nums,
        )

    def duplex_generate(self, force_listen: bool = False) -> DuplexGenerateResult:
        if os.environ.get("O5_TTS_ARGMAX", "0").lower() not in {"1", "true", "yes", "on"}:
            return self._run_duplex_generate(force_listen)

        with tts_argmax_scope(True):
            return self._run_duplex_generate(force_listen)

    def duplex_finalize(self) -> None:
        duplex_view = self.processor.set_duplex_mode()
        duplex_view.finalize()

    def duplex_stop(self) -> None:
        duplex_view = self.processor.set_duplex_mode()
        duplex_view.stop()

    def duplex_cleanup(self) -> None:
        if self.processor is None:
            return
        duplex_view = self.processor.set_duplex_mode()
        duplex_view.cleanup()
        self._cleanup_llm_runtime()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[GPU {self.gpu_id}] Duplex cleanup done, GPU memory released")

    # ========== FC Duplex ==========

    def fc_duplex_prepare(
        self,
        *,
        system: FcSystemContentInput,
        tts_prompt_audio: FcAudioPathInput | None,
        generate_audio: bool,
        fixed_tool_call_ids: list[str] | None,
    ) -> FcDuplexPrepareResult:
        """按 v3 canonical contract 初始化 FC Duplex View。

        参数:
            system: 有序多模态 system 与嵌套工具定义。
            tts_prompt_audio: 独立的 TTS 提示音频路径输入。
            generate_audio: 是否生成 spoken waveform。
            fixed_tool_call_ids: 评测专用的确定性内部工具调用 ID；None 使用默认生成器。

        返回:
            FC Duplex View 的初始化结果。
        """

        from core.processors.unified import FixedToolCallIdGenerator

        fc_view = self.processor.set_fc_duplex_mode()
        tool_call_id_generator = (
            FixedToolCallIdGenerator(fixed_tool_call_ids)
            if fixed_tool_call_ids is not None
            else None
        )
        return fc_view.prepare(
            FcDuplexPrepareRequest(
                system=system,
                tts_prompt_audio=tts_prompt_audio,
                generate_audio=generate_audio,
            ),
            tool_call_id_generator=tool_call_id_generator,
        )

    def fc_duplex_prefill(
        self,
        *,
        audio_data: Optional[str] = None,
        frame_list: Optional[list] = None,
        tool_responses: Optional[List[FcToolResponse]] = None,
        sample_rate: int = 16000,
    ) -> Any:
        fc_view = self.processor.set_fc_duplex_mode()
        return fc_view.streaming_prefill(
            FcDuplexPrefillRequest(
                audio_data=audio_data,
                frame_list=frame_list,
                tool_responses=tool_responses,
                sample_rate=sample_rate,
            )
        )

    def fc_duplex_spoken_generate(
        self,
        *,
        max_tokens: int = 24,
        decode_mode: str = "greedy",
    ) -> Any:
        fc_view = self.processor.set_fc_duplex_mode()
        return fc_view.streaming_spoken_generate(
            FcSpokenGenerateRequest(max_tokens=max_tokens, decode_mode=decode_mode)
        )

    def fc_duplex_non_spoken_generate(
        self,
        *,
        max_tokens: int = 1,
        decode_mode: str = "greedy",
        close_reason: Optional[str] = None,
    ) -> Any:
        fc_view = self.processor.set_fc_duplex_mode()
        return fc_view.streaming_non_spoken_generate(
            FcNonSpokenGenerateRequest(
                max_tokens=max_tokens,
                decode_mode=decode_mode,
                close_reason=close_reason,
            )
        )

    def fc_duplex_finalize(self) -> Any:
        fc_view = self.processor.set_fc_duplex_mode()
        return fc_view.finalize_unit(FcFinalizeUnitRequest())

    def fc_duplex_resume_boundary_status(self) -> Dict[str, Any]:
        """Return the View's public-history resume eligibility at the Unit boundary."""

        fc_view = self.processor.set_fc_duplex_mode()
        return fc_view.resume_boundary_status()

    def fc_duplex_terminate_non_spoken_text_stream(self, *, reason: str) -> Any:
        """Terminate one View text stream without advancing model/KV state."""

        fc_view = self.processor.set_fc_duplex_mode()
        return fc_view.terminate_non_spoken_text_stream(reason)

    def fc_duplex_resume_identity(self) -> Dict[str, Any]:
        """Return current model/tokenizer identity for public resume metadata."""

        fc_view = self.processor.set_fc_duplex_mode()
        return fc_view.resume_identity()

    def fc_duplex_replay_completed_unit(
        self,
        *,
        audio_data: Optional[str],
        frame_list: Optional[List[Any]],
        tool_responses: Optional[List[Any]],
        sample_rate: int,
        spoken_token_ids: List[int],
        non_spoken_token_ids: List[int],
        deferred_non_spoken_close: bool,
    ) -> Any:
        """Deterministically feed one historical FC Duplex Unit."""

        fc_view = self.processor.set_fc_duplex_mode()
        return fc_view.replay_completed_unit(
            audio_data=audio_data,
            frame_list=frame_list,
            tool_responses=tool_responses,
            sample_rate=sample_rate,
            spoken_token_ids=spoken_token_ids,
            non_spoken_token_ids=non_spoken_token_ids,
            deferred_non_spoken_close=deferred_non_spoken_close,
        )

    def fc_duplex_restore_generation_stream_sequence(
        self,
        *,
        next_stream_sequence: int,
    ) -> None:
        """Advance View stream IDs after stateless replay."""

        fc_view = self.processor.set_fc_duplex_mode()
        fc_view.restore_generation_stream_sequence(next_stream_sequence)

    def fc_duplex_restore_tool_call_sequence(
        self,
        *,
        tool_call_count: int,
    ) -> None:
        """Advance View/internal tool-call IDs after stateless replay."""

        fc_view = self.processor.set_fc_duplex_mode()
        fc_view.restore_tool_call_sequence(tool_call_count)

    def fc_duplex_dump_trace(self, *, path: str, session_id: Optional[str] = None, reason: Optional[str] = None) -> Any:
        if self.processor is None:
            return None
        return self.processor.set_fc_duplex_mode().dump_trace(path, session_id=session_id, reason=reason)

    def fc_duplex_cleanup(self) -> None:
        if self.processor is None:
            return
        self.processor.set_fc_duplex_mode().cleanup()
        self._cleanup_llm_runtime()
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"[GPU {self.gpu_id}] FC duplex cleanup done, GPU memory released")

    def _cleanup_llm_runtime(self) -> None:
        """Synchronize allocator cleanup for TP2's rank-local LLM state."""
        model = getattr(self.processor, "model", None)
        llm = getattr(model, "llm", None)
        cleanup = getattr(llm, "cleanup", None)
        if callable(cleanup):
            cleanup()

    def shutdown(self) -> None:
        """PyTorch backend currently has no external process to shut down."""
        return

    # ========== Half-Duplex ==========

    def half_duplex_prefill(self, request: StreamingRequest) -> str:
        """Half-Duplex 预填充"""
        half_duplex_view = self.processor.set_half_duplex_mode()
        prompt = half_duplex_view.prefill(request)
        return prompt

    def half_duplex_init_tts(self, ref_audio_data: Optional[np.ndarray] = None) -> None:
        """初始化 Half-Duplex TTS（在 generate 前调用，如需生成音频）
        
        Args:
            ref_audio_data: 前端上传的 ref audio ndarray (16kHz mono float32)。
                若提供则使用此数据，否则使用 worker 默认的 ref_audio_path。
        """
        half_duplex_view = self.processor.set_half_duplex_mode()
        if ref_audio_data is not None:
            half_duplex_view.init_ref_audio_from_data(ref_audio_data)
        else:
            half_duplex_view.init_ref_audio(self.ref_audio_path)

    def half_duplex_generate(
        self,
        session_id: str,
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        length_penalty: float = 1.0,
    ) -> Iterator[StreamingChunk]:
        """Half-Duplex 生成（yield StreamingChunk）"""
        half_duplex_view = self.processor.set_half_duplex_mode()
        yield from half_duplex_view.generate(
            session_id=session_id,
            generate_audio=generate_audio,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )

    def half_duplex_complete_turn(
        self,
        session_id: str,
        messages: List[Message],
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        output_audio_path: Optional[str] = None,
        length_penalty: float = 1.0,
    ) -> StreamingResponse:
        """Half-Duplex 完成一轮（便捷方法）"""
        half_duplex_view = self.processor.set_half_duplex_mode()
        return half_duplex_view.complete_turn(
            session_id=session_id,
            messages=messages,
            generate_audio=generate_audio,
            max_new_tokens=max_new_tokens,
            output_audio_path=output_audio_path,
            length_penalty=length_penalty,
        )

    def reset_half_duplex_session(self) -> None:
        """重置 Half-Duplex 模型 session（清除 KV cache）"""
        half_duplex_view = self.processor.set_half_duplex_mode()
        half_duplex_view._model.reset_session(reset_token2wav_cache=False)
        logger.info(f"[GPU {self.gpu_id}] Half-Duplex model session reset (KV cache cleared)")
