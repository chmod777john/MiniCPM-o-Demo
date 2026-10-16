"""O5 有界显存 prefill 的 CPU 契约测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from modeling.o5.bounded_prefill import (
    iter_prefill_chunks,
    prefill_chunk_tokens,
    prefill_experts_implementation,
)


class _Experts(torch.nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config


def test_prefill_chunks_cover_range_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """分块必须按顺序、无重叠地覆盖整个 prefill。"""

    monkeypatch.setenv("O5_PREFILL_CHUNK_TOKENS", "4")
    assert list(iter_prefill_chunks(10)) == [(0, 4), (4, 8), (8, 10)]
    assert list(iter_prefill_chunks(1)) == [(0, 1)]


def test_prefill_chunk_default_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """未设置时默认 512；非正数直接报错。"""

    monkeypatch.delenv("O5_PREFILL_CHUNK_TOKENS", raising=False)
    assert prefill_chunk_tokens() == 512
    monkeypatch.setenv("O5_PREFILL_CHUNK_TOKENS", "0")
    with pytest.raises(ValueError, match="正整数"):
        prefill_chunk_tokens()


def test_prefill_switches_only_batched_mm_and_restores() -> None:
    """batched_mm 在 prefill 内切到 eager，退出后恢复；其他实现不动。"""

    batched = SimpleNamespace(_experts_implementation="batched_mm")
    grouped = SimpleNamespace(_experts_implementation="grouped_mm")
    model = torch.nn.Sequential(_Experts(batched), _Experts(grouped), _Experts(batched))

    with prefill_experts_implementation(model):
        assert batched._experts_implementation == "eager"
        assert grouped._experts_implementation == "grouped_mm"
    assert batched._experts_implementation == "batched_mm"
    assert grouped._experts_implementation == "grouped_mm"


def test_prefill_restores_on_exception() -> None:
    """prefill 中途报错也必须恢复 decode 使用的实现。"""

    config = SimpleNamespace(_experts_implementation="batched_mm")
    model = _Experts(config)
    with pytest.raises(RuntimeError):
        with prefill_experts_implementation(model):
            raise RuntimeError("boom")
    assert config._experts_implementation == "batched_mm"


def test_prefill_attention_switches_only_eager_and_restores() -> None:
    """graph 固定的 eager attention 在 prefill 内切到 sdpa，退出后恢复；其他实现不动。"""

    from modeling.o5.bounded_prefill import prefill_attention_implementation

    eager = SimpleNamespace(_attn_implementation="eager")
    flash = SimpleNamespace(_attn_implementation="flash_attention_2")
    model = torch.nn.Sequential(_Experts(eager), _Experts(flash))

    with prefill_attention_implementation(model):
        assert eager._attn_implementation == "sdpa"
        assert flash._attn_implementation == "flash_attention_2"
    assert eager._attn_implementation == "eager"


def test_streaming_prefill_and_non_spoken_report_cost() -> None:
    """prefill / non_spoken / finalize 都要带耗时，否则无法回答「每个 Unit 慢在哪」。

    2026-10-01 之前只有 spoken 有 cost_llm，prefill 与 non_spoken 完全没计时，
    被问到「每 Unit prefill 多少 ms」时拿不出数。
    """

    import inspect

    from modeling.o5 import modeling_minicpmo_unified as mod

    prefill_src = inspect.getsource(mod.FcDuplexCapability.streaming_prefill)
    assert 'info["cost_prefill"] = time.time() - prefill_start' in prefill_src

    non_spoken_src = inspect.getsource(
        mod.FcDuplexCapability.streaming_non_spoken_generate
    )
    assert '"cost_llm": time.time() - non_spoken_start' in non_spoken_src

    finalize_src = inspect.getsource(mod.FcDuplexCapability.finalize_unit)
    assert 'info["cost_finalize"] = time.time() - finalize_start' in finalize_src


def test_prefill_eager_threshold_keeps_small_prefills_on_original_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """阈值以下的小 prefill 不切 eager；默认阈值 2 与引入时行为一致（任何 L>1 都切）。"""

    from modeling.o5.bounded_prefill import prefill_eager_min_tokens

    config = SimpleNamespace(_experts_implementation="batched_mm")
    model = _Experts(config)
    monkeypatch.delenv("O5_PREFILL_EAGER_MIN_TOKENS", raising=False)
    assert prefill_eager_min_tokens() == 2
    with prefill_experts_implementation(model, 2):
        assert config._experts_implementation == "eager"

    monkeypatch.setenv("O5_PREFILL_EAGER_MIN_TOKENS", "256")
    with prefill_experts_implementation(model, 40):
        assert config._experts_implementation == "batched_mm"
    with prefill_experts_implementation(model, 256):
        assert config._experts_implementation == "eager"
    with prefill_experts_implementation(model):
        assert config._experts_implementation == "eager"
    assert config._experts_implementation == "batched_mm"

    monkeypatch.setenv("O5_PREFILL_EAGER_MIN_TOKENS", "0")
    with pytest.raises(ValueError, match="正整数"):
        prefill_eager_min_tokens()


def test_prefill_experts_target_selects_long_prefill_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    """长 prefill 切到的实现可配置：默认 eager；batched_mm 等于不切换（TP2 实时部署用）。"""

    from modeling.o5.bounded_prefill import prefill_experts_target

    config = SimpleNamespace(_experts_implementation="batched_mm")
    model = _Experts(config)
    monkeypatch.setenv("O5_PREFILL_EAGER_MIN_TOKENS", "256")
    monkeypatch.delenv("O5_PREFILL_EXPERTS_IMPLEMENTATION", raising=False)
    assert prefill_experts_target() == "eager"
    monkeypatch.setenv("O5_PREFILL_EXPERTS_IMPLEMENTATION", "batched_mm")
    with prefill_experts_implementation(model, 5000):
        assert config._experts_implementation == "batched_mm"
    monkeypatch.setenv("O5_PREFILL_EXPERTS_IMPLEMENTATION", "grouped_mm")
    with prefill_experts_implementation(model, 5000):
        assert config._experts_implementation == "grouped_mm"
    with prefill_experts_implementation(model, 40):
        assert config._experts_implementation == "batched_mm"
    assert config._experts_implementation == "batched_mm"
    monkeypatch.setenv("O5_PREFILL_EXPERTS_IMPLEMENTATION", "fast")
    with pytest.raises(ValueError, match="只能是"):
        prefill_experts_target()
