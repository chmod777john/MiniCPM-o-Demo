"""O5 FC 内层推理生命周期与 SDK token 边界回归测试。

本文件覆盖三项真实故障：prompt-specific Token2Wav cache 不得随 Session reset 丢失；
spoken turn 必须按 SDK 顺序关闭；TrainingData helper 必须显式选择 O5/O45_FC target。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from core.processors.unified import FcDuplexView, UnifiedProcessor
from minicpm_o5_sdk import O5TokenizerID
from modeling.o5.modeling_minicpmo_unified import FcDuplexCapability


class _FakeAudioTokenizer:
    """记录 prompt cache 构造次数的最小 Token2Wav tokenizer。"""

    def __init__(self) -> None:
        self.cache: object | None = object()
        self.flow = SimpleNamespace(pre_lookahead_len=3)
        self.calls: list[str] = []
        self.stream_cache: object | None = None
        self.hift_cache_dict: object | None = None

    def set_stream_cache(
        self,
        prompt_wav_path: str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """记录 prompt，并返回可深拷贝的缓存。"""

        self.calls.append(prompt_wav_path)
        return {"flow": torch.tensor([1.0])}, {"hift": torch.tensor([2.0])}


class _FakeDecoder:
    """为 spoken token 状态机提供确定性 hidden/logits。"""

    def embed_token(self, token_id: int) -> torch.Tensor:
        """返回一个与 token 绑定的占位 embedding。"""

        return torch.tensor([token_id], dtype=torch.long)

    def feed(
        self,
        _: torch.Tensor,
        *,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回下一步占位 logits 与 hidden。"""

        assert return_logits is True
        return torch.zeros(1, 8), torch.zeros(1, 1, 4)


def _new_cache_capability(tokenizer: _FakeAudioTokenizer) -> FcDuplexCapability:
    """构造只用于 Token2Wav cache 测试的 capability。"""

    capability = FcDuplexCapability.__new__(FcDuplexCapability)
    capability.model = SimpleNamespace(
        tts=SimpleNamespace(audio_tokenizer=tokenizer),
        config=SimpleNamespace(tts_config=SimpleNamespace(s3_stream_n_timesteps=10)),
    )
    capability.decoder = SimpleNamespace(reset=lambda: None)
    capability.token2wav_initialized = False
    capability.token2wav_buffer = []
    capability.flow_cache_base = None
    capability.hift_cache_base = None
    capability.pre_lookahead = 0
    capability._token2wav_prompt_wav_path = None
    return capability


def test_token2wav_prompt_cache_survives_session_reset_and_reuses_same_prompt() -> None:
    """同一 reference WAV 只能构建一次昂贵的 Token2Wav prompt cache。"""

    tokenizer = _FakeAudioTokenizer()
    capability = _new_cache_capability(tokenizer)

    capability.warm_token2wav(prompt_wav_path="/voices/ref.wav")
    capability._reset_streaming_state()
    capability.warm_token2wav(prompt_wav_path="/voices/ref.wav")

    assert tokenizer.calls == ["/voices/ref.wav"]
    assert capability.token2wav_initialized is True
    assert capability._token2wav_prompt_wav_path == "/voices/ref.wav"

    capability.warm_token2wav(prompt_wav_path="/voices/other.wav")
    assert tokenizer.calls == ["/voices/ref.wav", "/voices/other.wav"]


def test_token2wav_prompt_cache_reuse_can_be_disabled_for_equivalence_probe(
    monkeypatch: Any,
) -> None:
    """Cold 对拍可强制每个 prepare 重建 prompt cache。"""

    tokenizer = _FakeAudioTokenizer()
    capability = _new_cache_capability(tokenizer)
    monkeypatch.setenv("FC_DUPLEX_PROMPT_CACHE_REUSE", "0")

    capability.warm_token2wav(prompt_wav_path="/voices/ref.wav")
    capability.warm_token2wav(prompt_wav_path="/voices/ref.wav")

    assert tokenizer.calls == ["/voices/ref.wav", "/voices/ref.wav"]


def test_o5_spoken_turn_eos_stops_and_appends_slot_eos_before_slot_end() -> None:
    """O5 spoken turn 必须与 SDK/O45 一样补齐 turn_eos→slot_eos→slot_end。"""

    key = SimpleNamespace(
        SPEAK="speak",
        LISTEN="listen",
        TTS_PAD="tts_pad",
        SPOKEN_SLOT_EOS="spoken_slot_eos",
        SPOKEN_TURN_EOS="spoken_turn_eos",
        AI_SPOKEN_SLOT_END="ai_spoken_slot_end",
    )
    ids = {
        "speak": 1,
        "listen": 2,
        "tts_pad": 3,
        "spoken_slot_eos": 4,
        "spoken_turn_eos": 5,
        "ai_spoken_slot_end": 6,
    }
    ordinary_id = 7
    unexpected_after_turn_id = 8
    generated = iter(
        [ids["speak"], ordinary_id, ids["spoken_turn_eos"], unexpected_after_turn_id]
    )
    fed_close_ids: list[list[int]] = []

    capability = FcDuplexCapability.__new__(FcDuplexCapability)
    capability.K = key
    capability._spoken_slot_open = True
    capability._spoken_logits = torch.zeros(1, 9)
    capability._current_unit_info = {}
    capability.output_ids = []
    capability.decoder = _FakeDecoder()
    capability.sid = lambda semantic_key: ids[semantic_key]  # type: ignore[method-assign]
    capability.is_special = lambda token_id: token_id != ordinary_id  # type: ignore[method-assign]
    capability._sample = lambda _logits, _mode: next(generated)  # type: ignore[method-assign]
    capability._feed_ids = lambda token_ids, want_logits=False: fed_close_ids.append(  # type: ignore[method-assign]
        list(token_ids)
    )
    capability._flush = lambda token_ids: "正文" if token_ids else ""  # type: ignore[method-assign]
    capability._spoken_audio = lambda _results, _turn_eos: {  # type: ignore[method-assign]
        "audio_waveform": None,
        "audio_sample_rate": None,
        "n_tts_tokens": 0,
        "cost_tts_prep": 0.0,
        "cost_tts": 0.0,
        "cost_token2wav": 0.0,
    }

    result = capability.streaming_spoken_generate(max_tokens=8, decode_mode="greedy")

    assert result["spoken_ids"] == [ids["speak"], ordinary_id, ids["spoken_turn_eos"]]
    assert result["spoken_turn_eos"] is True
    assert result["spoken_text"] == "正文"
    assert fed_close_ids == [
        [ids["spoken_slot_eos"], ids["ai_spoken_slot_end"]]
    ]
    assert unexpected_after_turn_id not in capability.output_ids


def test_training_data_loader_requires_explicit_o5_target(monkeypatch: Any) -> None:
    """O5 offline helper 不得再静默使用 O45_FC tokenizer。"""

    observed: list[O5TokenizerID] = []
    training_data = SimpleNamespace(
        tokenize=lambda *, tokenizer_id: observed.append(tokenizer_id) or "tokenized"
    )
    fake_training_data_type = SimpleNamespace(
        load_structure=lambda _structure, data_root: training_data
    )
    monkeypatch.setattr(
        "minicpm_o5_sdk.O5DuplexTrainingData",
        fake_training_data_type,
    )

    loaded, tokenized = FcDuplexView._load_sdk_train_data(
        {},
        Path("/dataset"),
        tokenizer_target="o5",
    )

    assert loaded is training_data
    assert tokenized == "tokenized"
    assert observed == [O5TokenizerID.O5]


def test_processor_warms_full_o5_fc_prepare_path_before_returning_ready(
    tmp_path: Path,
) -> None:
    """配置 reference audio 时，Processor ready 前必须完成 APM/LLM/TTS 全路径预热。"""

    import soundfile as sf

    ref_path = tmp_path / "ref.wav"
    sf.write(ref_path, np.zeros(16_000, dtype=np.float32), 16_000)
    calls: list[tuple[object, str | None, bool | None]] = []
    processor = UnifiedProcessor.__new__(UnifiedProcessor)
    processor._is_initialized = False
    processor.fc_model_family = "o5"
    processor.preload_both_tts = True
    processor.ref_audio_path = str(ref_path)
    processor.model = SimpleNamespace(
        fc_duplex=SimpleNamespace(
            warm_prepare=lambda *, system_content, tts_prompt_audio_path, generate_audio: calls.append(
                (system_content, tts_prompt_audio_path, generate_audio)
            )
        )
    )

    processor._warm_fc_token2wav_if_configured()

    assert len(calls) == 1
    system_content, tts_prompt_audio_path, generate_audio = calls[0]
    assert len(system_content.segments) == 1
    assert system_content.segments[0].audio.get_tensor().shape == (16_000,)
    assert tts_prompt_audio_path == str(ref_path)
    assert generate_audio is True


def test_processor_startup_warm_can_be_disabled_for_cold_equivalence_probe(
    monkeypatch: Any,
) -> None:
    """Cold 对拍关闭 startup warm 时不得触碰 capability。"""

    calls: list[str] = []
    processor = UnifiedProcessor.__new__(UnifiedProcessor)
    processor._is_initialized = False
    processor.fc_model_family = "o5"
    processor.preload_both_tts = True
    processor.ref_audio_path = "/voices/ref.wav"
    processor.model = SimpleNamespace(
        fc_duplex=SimpleNamespace(
            warm_prepare=lambda **_: calls.append("unexpected")
        )
    )
    monkeypatch.setenv("FC_DUPLEX_STARTUP_WARM", "0")

    processor._warm_fc_token2wav_if_configured()

    assert calls == []


def test_processor_startup_warm_resolves_project_relative_reference_audio(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """FC schema receives an absolute path even when service config is relative."""

    import soundfile as sf

    ref_path = tmp_path / "assets" / "ref.wav"
    ref_path.parent.mkdir()
    sf.write(ref_path, np.zeros(16_000, dtype=np.float32), 16_000)
    calls: list[tuple[object, str | None, bool | None]] = []
    processor = UnifiedProcessor.__new__(UnifiedProcessor)
    processor._is_initialized = False
    processor.fc_model_family = "o5"
    processor.preload_both_tts = True
    processor.ref_audio_path = "assets/ref.wav"
    processor.model = SimpleNamespace(
        fc_duplex=SimpleNamespace(
            warm_prepare=lambda *, system_content, tts_prompt_audio_path, generate_audio: calls.append(
                (system_content, tts_prompt_audio_path, generate_audio)
            )
        )
    )
    monkeypatch.chdir(tmp_path)

    processor._warm_fc_token2wav_if_configured()

    assert calls[0][1] == str(ref_path)


def test_processor_startup_warm_rejects_unreadable_explicit_reference_audio(
    tmp_path: Path,
) -> None:
    """Audio-enabled FC startup must fail clearly instead of constructing a bad schema input."""

    processor = UnifiedProcessor.__new__(UnifiedProcessor)
    processor._is_initialized = False
    processor.fc_model_family = "o5"
    processor.preload_both_tts = True
    processor.ref_audio_path = str(tmp_path / "missing.wav")
    processor.model = SimpleNamespace(
        fc_duplex=SimpleNamespace(warm_prepare=lambda **_: None)
    )

    with pytest.raises(RuntimeError, match="readable reference audio file"):
        processor._warm_fc_token2wav_if_configured()


def test_o5_trace_dump_preserves_raw_tokens_without_session_close_error(
    tmp_path: Path,
) -> None:
    """O5 Session close 必须落盘 raw token，而不是抛 NotImplementedError。"""

    capability = FcDuplexCapability.__new__(FcDuplexCapability)
    capability._ensure_protocol = lambda: None  # type: ignore[method-assign]
    capability.output_ids = [248159, 248103, 42, 248143, 248114, 248160]
    capability.render_token_stream = lambda _ids: "<ai_spoken_slot>..."  # type: ignore[method-assign]
    capability.units_info = [{"unit": 0, "spoken_ids": [248103, 42, 248143]}]
    capability._current_unit_idx = 1
    capability._current_unit_open = False
    capability._spoken_slot_open = False
    capability._non_spoken_slot_open = False
    capability._non_spoken_mode = None
    capability._think_buf = []
    capability._tool_call_buf = []
    capability.token2wav_initialized = True
    capability._token2wav_prompt_wav_path = "/voices/ref.wav"
    capability._prepare_sequence = 2
    capability.decoder = SimpleNamespace(get_cache_length=lambda: 123)
    trace_path = tmp_path / "trace.json"

    result = capability.dump_trace(
        path=trace_path,
        session_id="sess_test",
        reason="session_close",
    )
    structure = json.loads(trace_path.read_text(encoding="utf-8"))

    assert result["trace_supported"] is True
    assert result["output_token_count"] == 6
    assert result["unit_count"] == 1
    assert structure["output_ids"] == capability.output_ids
    assert structure["session_id"] == "sess_test"
    assert structure["reason"] == "session_close"
    assert structure["prepare_sequence"] == 2
    assert structure["kv_cache_length"] == 123
