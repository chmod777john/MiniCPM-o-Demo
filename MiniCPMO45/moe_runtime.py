"""Runtime selection for Qwen3.5-MoE expert kernels.

``hybrid`` keeps graph capture and decode on ``batched_mm`` while using
``grouped_mm`` for sufficiently long variable-length prefill calls.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


VALID_IMPLS = {"eager", "batched_mm", "grouped_mm"}
HYBRID_IMPLS = {"hybrid", "prefill_grouped_mm", "grouped_prefill"}


def requested_experts_impl(default: str = "batched_mm") -> str:
    return os.environ.get("O5_EXPERTS_IMPLEMENTATION", default).strip().lower() or default


def is_hybrid_experts_impl(impl: str | None = None) -> bool:
    return (impl or requested_experts_impl()).lower() in HYBRID_IMPLS


def initial_experts_impl(impl: str | None = None) -> str:
    requested = (impl or requested_experts_impl()).lower()
    if is_hybrid_experts_impl(requested):
        return "batched_mm"
    if requested not in VALID_IMPLS:
        raise ValueError(f"Unsupported O5_EXPERTS_IMPLEMENTATION={requested!r}")
    return requested


def decode_experts_impl() -> str:
    return initial_experts_impl()


def hybrid_prefill_threshold() -> int:
    return max(1, int(os.environ.get("O5_GROUPED_PREFILL_MIN_TOKENS", "100")))


def prefill_experts_impl(seq_len: int) -> str:
    if is_hybrid_experts_impl() and seq_len >= hybrid_prefill_threshold():
        return "grouped_mm"
    return initial_experts_impl()


def _expert_configs(module: Any) -> list[Any]:
    configs = []
    seen: set[int] = set()
    modules = module.modules() if hasattr(module, "modules") else [module]
    for child in modules:
        config = getattr(child, "config", None)
        if config is None or id(config) in seen or not hasattr(config, "_experts_implementation"):
            continue
        configs.append(config)
        seen.add(id(config))
    return configs


def set_experts_impl(module: Any, impl: str) -> int:
    if impl not in VALID_IMPLS:
        raise ValueError(f"Unsupported experts implementation {impl!r}")
    configs = _expert_configs(module)
    for config in configs:
        config._experts_implementation = impl
    return len(configs)


@contextmanager
def temporary_experts_impl(module: Any, impl: str) -> Iterator[None]:
    if impl not in VALID_IMPLS:
        raise ValueError(f"Unsupported experts implementation {impl!r}")
    configs = _expert_configs(module)
    previous = [config._experts_implementation for config in configs]
    for config in configs:
        config._experts_implementation = impl
    try:
        yield
    finally:
        for config, value in zip(configs, previous):
            config._experts_implementation = value
