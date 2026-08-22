"""CPU-only tests for the runtime MoE implementation selector."""

import torch

from MiniCPMO45.moe_runtime import (
    prefill_experts_impl,
    temporary_experts_impl,
)


class _FakeMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"_experts_implementation": "batched_mm"})()


def test_hybrid_switches_only_long_prefill(monkeypatch):
    monkeypatch.setenv("O5_EXPERTS_IMPLEMENTATION", "hybrid")
    monkeypatch.setenv("O5_GROUPED_PREFILL_MIN_TOKENS", "100")

    assert prefill_experts_impl(99) == "batched_mm"
    assert prefill_experts_impl(100) == "grouped_mm"


def test_temporary_selector_restores_decode_implementation():
    module = _FakeMoE()
    with temporary_experts_impl(module, "grouped_mm"):
        assert module.config._experts_implementation == "grouped_mm"
    assert module.config._experts_implementation == "batched_mm"
