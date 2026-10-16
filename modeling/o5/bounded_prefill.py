"""O5 MoE backbone 的有界显存 prefill。

根因（2026-09-30 定位）：transformers 的 ``batched_mm`` MoE 实现会按
``(token, top_k)`` 把被选中专家的权重 gather 成一份副本::

    gate_up_proj[expert_ids]   # (L * top_k, 2 * moe_intermediate, hidden)

O5 的 hidden=2048、moe_intermediate=512、top_k=8，所以**每个 prefill token 约 32 MiB**。
decode（L=1）只有 32 MiB，速度最快；但一次 ~550 token 的 system prefill 就要
~17 GiB，在单卡常驻 ~72 GiB 的完整 BF16 PT 上必然 OOM。TP2 把专家切半才躲过去。

因此本模块只做三件事，且对 decode 零影响：

1. prefill（L>1）期间把 ``batched_mm`` 临时切到 ``eager``（逐专家循环，显存 O(L)）；
2. 把长 prefill 切成不超过 ``O5_PREFILL_CHUNK_TOKENS`` 的块依次喂入同一个 cache；
3. LLM graph 路径为了捕获 decode 图把 full attention 固定成 eager，而 eager 会按 StaticCache
   全宽物化 fp32 分数矩阵 ``[heads, L, max_cache_len]``（512×32768×16 头 = 1 GiB/块）。
   prefill 期间临时切到 ``sdpa``（memory-efficient kernel，不物化分数矩阵）。
   单卡常驻 ~77 GiB 时这 1 GiB 就是 OOM 与否的分界（2026-09-30 冒烟实测）。

分块 prefill 与一次性 prefill 在数学上等价（因果 attention + 递推的线性 attention 状态），
数值上只有 kernel 切分带来的 BF16 舍入差异。

⚠️ 代价（2026-10-09 定位 Demo 语音卡顿）：``eager`` 是按命中的专家逐个循环（O5 256 专家 × 40 层），
每次 prefill 上万次小 kernel 启动 + 每层一次 ``nonzero`` 同步。实时 Demo 每个 Unit 都要做几十 token 的
小 prefill，全部切 eager 后单 Unit 处理从 1 秒内被拖到 1–2 秒，语音一卡一卡。小 prefill 的显存本来就小
（每 token ~32 MiB，TP2 再减半），所以加 ``O5_PREFILL_EAGER_MIN_TOKENS``：只有 L 达到阈值的 prefill 才切 eager。
默认 2（= 任何 L>1 都切，与引入时行为一致，离线单卡评测不受影响）；TP2 实时部署显式调大。

⚠️ 代价二（2026-10-16 定位建会话慢）：开场的长 system prefill（Demo 带工具定义常见 2000–5000 token）仍达到阈值、
走 eager，实测每秒只有 ~230 token，5000 token 建会话要 22 秒。切到哪种实现由 ``O5_PREFILL_EXPERTS_IMPLEMENTATION``
决定：TP2 实时部署设为 ``batched_mm``（= 不切，仍按 512 token 分块，瞬时显存每块 ~8 GiB），2×A100 实测
1889 / 5021 token 建会话 8.2→3.0 / 22.1→7.7 秒，峰值显存 44→56 GiB，15 个 Unit 输出与 eager 逐个一致。
``grouped_mm`` 只在 H100（sm90）可用，A100 上 ``torch._grouped_mm`` 直接报错。默认仍是 eager（离线单卡不变）。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch

DEFAULT_PREFILL_CHUNK_TOKENS = 512
DEFAULT_PREFILL_EAGER_MIN_TOKENS = 2
_MEMORY_HEAVY_EXPERTS_IMPLEMENTATIONS = frozenset({"batched_mm"})
_PREFILL_EXPERTS_IMPLEMENTATIONS = ("eager", "grouped_mm", "batched_mm")
DEFAULT_PREFILL_EXPERTS_IMPLEMENTATION = "eager"
_PREFILL_ATTENTION_IMPLEMENTATION = "sdpa"


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


def prefill_eager_min_tokens() -> int:
    """返回「prefill 切 eager MoE」的最小 token 数。

    返回:
        ``O5_PREFILL_EAGER_MIN_TOKENS`` 的正整数值，未设置时为 2（任何 L>1 的 prefill 都切 eager）。
        阈值以下的 prefill 保留原实现（``batched_mm``），瞬时显存约为 ``阈值 × 每 token 专家副本``。

    异常:
        ValueError: 环境变量不是正整数。
    """

    raw = os.environ.get("O5_PREFILL_EAGER_MIN_TOKENS")
    if raw is None:
        return DEFAULT_PREFILL_EAGER_MIN_TOKENS
    value = int(raw)
    if value <= 0:
        raise ValueError(f"O5_PREFILL_EAGER_MIN_TOKENS 必须为正整数，实际为 {raw!r}")
    return value


def prefill_experts_target() -> str:
    """返回长 prefill（达到 ``prefill_eager_min_tokens()``）期间要切到的 MoE 专家实现。

    返回:
        ``O5_PREFILL_EXPERTS_IMPLEMENTATION`` 的值，未设置时为 ``eager``（逐专家循环，显存最省、最慢）。
        ``grouped_mm`` 按专家排序后一次分组矩阵乘，不复制专家权重（只支持 H100 / sm90，A100 上报错）；
        ``batched_mm`` 等于不切换
        （瞬时显存 = 分块 token 数 × 每 token 专家副本）。

    异常:
        ValueError: 取值不在上述三种之内。
    """

    value = os.environ.get("O5_PREFILL_EXPERTS_IMPLEMENTATION") or DEFAULT_PREFILL_EXPERTS_IMPLEMENTATION
    if value not in _PREFILL_EXPERTS_IMPLEMENTATIONS:
        raise ValueError(
            f"O5_PREFILL_EXPERTS_IMPLEMENTATION 只能是 {'/'.join(_PREFILL_EXPERTS_IMPLEMENTATIONS)}，实际为 {value!r}"
        )
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
def prefill_experts_implementation(
    model: torch.nn.Module,
    length: int | None = None,
) -> Iterator[None]:
    """prefill 期间把显存随 token 数线性放大的 MoE 实现切到 eager。

    参数:
        model: LLM backbone（或包含它的模块）。
        length: 本次 prefill 的 token 数；小于 ``prefill_eager_min_tokens()`` 时不切换。
            为 None 时无条件切换（兼容旧调用）。

    返回:
        上下文管理器；退出时恢复原实现，decode 路径（含已捕获的 CUDA graph）不受影响。
    """

    if length is not None and length < prefill_eager_min_tokens():
        yield
        return
    target = prefill_experts_target()
    switched: list[tuple[object, str]] = []
    for config in _experts_configs(model):
        current = getattr(config, "_experts_implementation", None)
        if current in _MEMORY_HEAVY_EXPERTS_IMPLEMENTATIONS and current != target:
            switched.append((config, current))
            config._experts_implementation = target
    try:
        yield
    finally:
        for config, original in switched:
            config._experts_implementation = original


def _attention_configs(model: torch.nn.Module) -> list[object]:
    cached = getattr(model, "_o5_attention_configs", None)
    if cached is not None:
        return cached
    seen: set[int] = set()
    configs: list[object] = []
    for module in model.modules():
        config = getattr(module, "config", None)
        if (
            config is not None
            and hasattr(config, "_attn_implementation")
            and id(config) not in seen
        ):
            seen.add(id(config))
            configs.append(config)
    model._o5_attention_configs = configs
    return configs


@contextmanager
def prefill_attention_implementation(model: torch.nn.Module) -> Iterator[None]:
    """prefill 期间把被 graph 固定的 eager full attention 临时切到 sdpa。

    参数:
        model: LLM backbone（LLM graph runner 的 ``self.model``）。

    返回:
        上下文管理器；退出时恢复 eager，已捕获的 decode graph 不受影响。
    """

    switched: list[tuple[object, str]] = []
    for config in _attention_configs(model):
        current = getattr(config, "_attn_implementation", None)
        if current == "eager":
            switched.append((config, current))
            config._attn_implementation = _PREFILL_ATTENTION_IMPLEMENTATION
    try:
        yield
    finally:
        for config, original in switched:
            config._attn_implementation = original
