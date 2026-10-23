"""长 prefill 的 MoE 专家实现 ``padded_bmm``。

``batched_mm`` 会按 ``(token, top_k)`` 把选中专家的权重复制一份，长工具返回时
大部分时间花在搬权重上。本实现先按专家把 token 分组，再用一次 ``bmm`` 算完，
每个专家的权重只加载一次。数学上与 ``batched_mm`` 是同一件事；decode 仍走原来的
实现，不要拿这个函数去捕获 CUDA graph。

来源：cjp/o5-fc-dev ``dda2f17`` 的 ``modeling/o5/prefill_moe.py``，原样搬入。
只在 prefill、且长度达到 ``O5_PREFILL_EAGER_MIN_TOKENS`` 时由
``bounded_prefill.prefill_experts_implementation`` 切过来。
"""

from __future__ import annotations

from typing import Any

import torch


def padded_bmm_experts_forward(
    self: Any,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """按专家分组计算 MoE，避免为每个 (token, expert) 复制一份权重。

    参数:
        self: transformers 的专家模块（``num_experts``、投影权重、bias、激活）。
        hidden_states: 形状 ``[num_tokens, hidden]`` 的输入。
        top_k_index: 形状 ``[num_tokens, top_k]`` 的专家下标；大于等于
            ``num_experts`` 的下标是无效 EP 路由，不参与计算。
        top_k_weights: 与 ``top_k_index`` 同形状的路由权重。

    返回:
        形状 ``[num_tokens, hidden]``、dtype 与输入相同的专家输出。
        空输入或全部路由都无效时返回全零。
    """

    num_tokens, hidden_dim = hidden_states.shape
    top_k = top_k_index.shape[-1]
    if num_tokens == 0:
        return torch.zeros_like(hidden_states)
    expert_ids = top_k_index.reshape(-1).clamp_max(self.num_experts)
    counts = torch.bincount(expert_ids, minlength=self.num_experts + 1)[: self.num_experts]
    capacity, valid_count = torch.stack((counts.max(), counts.sum())).tolist()
    if valid_count == 0:
        return torch.zeros_like(hidden_states)
    order = torch.argsort(expert_ids, stable=True)[:valid_count]
    offsets = counts.cumsum(0) - counts
    slots = torch.arange(valid_count, device=hidden_states.device) - torch.repeat_interleave(
        offsets, counts, output_size=valid_count
    )
    grouped_index = expert_ids[order] * capacity + slots
    padded = hidden_states.new_zeros(self.num_experts * capacity, hidden_dim)
    padded.index_copy_(0, grouped_index, hidden_states[order // top_k])
    padded = padded.view(self.num_experts, capacity, hidden_dim)

    up_weight = self.gate_up_proj if self.has_gate else self.up_proj
    up = torch.bmm(padded, up_weight if self.is_transposed else up_weight.transpose(1, 2))
    if self.has_bias:
        bias = self.gate_up_proj_bias if self.has_gate else self.up_proj_bias
        up = up + bias[:, None, :]
    activated = self._apply_gate(up) if self.has_gate else self.act_fn(up)
    down_weight = self.down_proj if self.is_transposed else self.down_proj.transpose(1, 2)
    down = torch.bmm(activated, down_weight)
    if self.has_bias:
        down = down + self.down_proj_bias[:, None, :]
    routed = down.reshape(-1, hidden_dim)[grouped_index]
    weighted = routed * top_k_weights.reshape(-1)[order, None]
    pairs = weighted.new_zeros(num_tokens * top_k, hidden_dim)
    pairs.index_copy_(0, order, weighted)
    return pairs.view(num_tokens, top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def register_padded_bmm() -> None:
    """把 ``padded_bmm`` 注册进 transformers 的专家实现表。重复调用无害。"""

    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    ALL_EXPERTS_FUNCTIONS.register("padded_bmm", padded_bmm_experts_forward)
