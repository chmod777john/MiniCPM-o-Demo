"""Optional runtime controls for Flash Linear Attention Triton kernels.

All controls default to ``auto``. A fixed value is intentionally opt-in because
these helpers reach FLA 0.5.0's internal Triton autotuners; review them when
changing the installed FLA version.
"""

from __future__ import annotations

import os
from typing import Any


FUSED_NORM_CONFIG_ENV = "O5_FLA_FUSED_NORM_CONFIG"
L2NORM_CONFIG_ENV = "O5_FLA_L2NORM_CONFIG"
CHUNK_OUTPUT_CONFIG_ENV = "O5_FLA_CHUNK_OUTPUT_CONFIG"
FUSED_NORM_CONFIGS = (
    "auto",
    "16x4",
    "16x8",
    "16x16",
    "32x4",
    "32x8",
    "32x16",
    "64x4",
    "64x8",
    "64x16",
)
L2NORM_CONFIGS = tuple(
    ["auto"]
    + [
        f"{block_tokens}x{num_warps}"
        for block_tokens in (8, 16, 32, 64, 128)
        for num_warps in (1, 2, 4, 8, 16)
    ]
)
CHUNK_OUTPUT_CONFIGS = (
    "auto",
    "32x32x2",
    "64x64x4",
    "128x128x8",
)


def _find_autotuner(kernel: Any) -> Any:
    candidate = kernel
    while not hasattr(candidate, "cache") and hasattr(candidate, "fn"):
        candidate = candidate.fn
    if not hasattr(candidate, "cache"):
        raise RuntimeError(f"could not locate Triton autotuner for {kernel!r}")
    return candidate


def configure_fused_norm(config_name: str | None = None) -> dict[str, Any]:
    """Optionally pin FLA's gated RMSNorm reduction to one Triton config.

    FLA 0.5.0 may select different candidates on different physical GPUs.
    The candidates are all valid, but their reduction orders need not produce
    bitwise-identical BF16 outputs. Pinning preserves the fused FLA path.
    """

    selected = config_name or os.environ.get(FUSED_NORM_CONFIG_ENV, "auto")
    if selected not in FUSED_NORM_CONFIGS:
        supported = ", ".join(FUSED_NORM_CONFIGS)
        raise ValueError(f"unsupported FLA fused norm config {selected!r}; expected one of: {supported}")
    if selected == "auto":
        return {"mode": "auto", "config": selected, "patched": False}

    import triton
    from fla.modules import fused_norm_gate

    block_tokens, num_warps = (int(value) for value in selected.split("x", maxsplit=1))
    autotuner = fused_norm_gate.layer_norm_gated_fwd_kernel.fn
    autotuner.configs = [triton.Config({"BT": block_tokens}, num_warps=num_warps)]
    autotuner.cache.clear()
    return {
        "mode": "fixed",
        "config": selected,
        "block_tokens": block_tokens,
        "num_warps": num_warps,
        "patched": True,
    }


def configure_l2norm(config_name: str | None = None) -> dict[str, Any]:
    """Optionally pin FLA's Q/K L2Norm reduction to one Triton config."""

    selected = config_name or os.environ.get(L2NORM_CONFIG_ENV, "auto")
    if selected not in L2NORM_CONFIGS:
        supported = ", ".join(L2NORM_CONFIGS)
        raise ValueError(f"unsupported FLA L2Norm config {selected!r}; expected one of: {supported}")
    if selected == "auto":
        return {"mode": "auto", "config": selected, "patched": False}

    import triton
    from fla.modules.l2norm import l2norm_fwd_kernel

    block_tokens, num_warps = (int(value) for value in selected.split("x", maxsplit=1))
    autotuner = _find_autotuner(l2norm_fwd_kernel)
    autotuner.configs = [triton.Config({"BT": block_tokens}, num_warps=num_warps)]
    autotuner.cache.clear()
    return {
        "mode": "fixed",
        "config": selected,
        "block_tokens": block_tokens,
        "num_warps": num_warps,
        "patched": True,
    }


def configure_chunk_output(config_name: str | None = None) -> dict[str, Any]:
    """Optionally pin FLA's delta-rule chunk output matmul configuration.

    Candidate K/V tilings have different FP32 accumulation orders, so an
    independently selected autotune winner is not a deterministic contract.
    """

    selected = config_name or os.environ.get(CHUNK_OUTPUT_CONFIG_ENV, "auto")
    if selected not in CHUNK_OUTPUT_CONFIGS:
        supported = ", ".join(CHUNK_OUTPUT_CONFIGS)
        raise ValueError(
            f"unsupported FLA chunk output config {selected!r}; expected one of: {supported}"
        )
    if selected == "auto":
        return {"mode": "auto", "config": selected, "patched": False}

    import triton
    from fla.ops.common.chunk_o import chunk_fwd_kernel_o

    block_k, block_v, num_warps = (int(value) for value in selected.split("x"))
    autotuner = _find_autotuner(chunk_fwd_kernel_o)
    autotuner.configs = [
        triton.Config(
            {"BK": block_k, "BV": block_v},
            num_warps=num_warps,
            num_stages=3,
        )
    ]
    autotuner.cache.clear()
    return {
        "mode": "fixed",
        "config": selected,
        "block_k": block_k,
        "block_v": block_v,
        "num_warps": num_warps,
        "num_stages": 3,
        "patched": True,
    }
