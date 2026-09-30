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
