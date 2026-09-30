"""O5 MoE backbone 的有界显存 prefill。

根因（2026-09-30 定位）：transformers 的 ``batched_mm`` MoE 实现会按
``(token, top_k)`` 把被选中专家的权重 gather 成一份副本::

    gate_up_proj[expert_ids]   # (L * top_k, 2 * moe_intermediate, hidden)

O5 的 hidden=2048、moe_intermediate=512、top_k=8，所以**每个 prefill token 约 32 MiB**。
decode（L=1）只有 32 MiB，速度最快；但一次 ~550 token 的 system prefill 就要
~17 GiB，在单卡常驻 ~72 GiB 的完整 BF16 PT 上必然 OOM。TP2 把专家切半才躲过去。

因此本模块只做两件事，且对 decode 零影响：

1. prefill（L>1）期间把 ``batched_mm`` 临时切到 ``eager``（逐专家循环，显存 O(L)）；
2. 把长 prefill 切成不超过 ``O5_PREFILL_CHUNK_TOKENS`` 的块依次喂入同一个 cache，
   同时约束 eager full attention 的 ``[heads, L, max_cache_len]`` 分数矩阵。

分块 prefill 与一次性 prefill 在数学上等价（因果 attention + 递推的线性 attention 状态），
数值上只有 kernel 切分带来的 BF16 舍入差异。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch

DEFAULT_PREFILL_CHUNK_TOKENS = 512
_MEMORY_HEAVY_EXPERTS_IMPLEMENTATIONS = frozenset({"batched_mm"})
_PREFILL_EXPERTS_IMPLEMENTATION = "eager"


def prefill_chunk_tokens() -> int:
    """返回 prefill 分块上限（token 数）。

    返回:
        ``O5_PREFILL_CHUNK_TOKENS`` 的正整数值，未设置时为 512。

    异常:
        ValueError: 环境变量不是正整数。
    """

    raw = os.environ.get("O5_PREFILL_CHUNK_TOKENS")
    if raw is None:
        return DEFAULT_PREFILL_CHUNK_TOKENS
    value = int(raw)
    if value <= 0:
        raise ValueError(f"O5_PREFILL_CHUNK_TOKENS 必须为正整数，实际为 {raw!r}")
    return value


def iter_prefill_chunks(length: int) -> Iterator[tuple[int, int]]:
    """把长度为 ``length`` 的 prefill 切成 ``[start, end)`` 块。

    参数:
        length: 本次 feed 的 token/embedding 数。

    返回:
        按顺序覆盖 ``[0, length)`` 的区间迭代器。
    """

    chunk = prefill_chunk_tokens()
    for start in range(0, length, chunk):
        yield start, min(start + chunk, length)


def _experts_configs(model: torch.nn.Module) -> list[object]:
    cached = getattr(model, "_o5_experts_configs", None)
    if cached is not None:
        return cached
    seen: set[int] = set()
    configs: list[object] = []
    for module in model.modules():
        config = getattr(module, "config", None)
        if (
            config is not None
            and hasattr(config, "_experts_implementation")
            and id(config) not in seen
        ):
            seen.add(id(config))
            configs.append(config)
    model._o5_experts_configs = configs
    return configs


@contextmanager
def prefill_experts_implementation(model: torch.nn.Module) -> Iterator[None]:
    """prefill 期间把显存随 token 数线性放大的 MoE 实现切到 eager。

    参数:
        model: LLM backbone（或包含它的模块）。

    返回:
        上下文管理器；退出时恢复原实现，decode 路径（含已捕获的 CUDA graph）不受影响。
    """

    switched: list[tuple[object, str]] = []
    for config in _experts_configs(model):
        current = getattr(config, "_experts_implementation", None)
        if current in _MEMORY_HEAVY_EXPERTS_IMPLEMENTATIONS:
            switched.append((config, current))
            config._experts_implementation = _PREFILL_EXPERTS_IMPLEMENTATION
    try:
        yield
    finally:
        for config, original in switched:
            config._experts_implementation = original
