"""``padded_bmm`` 与 transformers ``batched_mm`` 逐元素对照。

覆盖空输入、专家分布极不均匀、EP 无效路由。来源：cjp/o5-fc-dev
``tests/test_o5_prefill_moe.py``。
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.integrations.moe import batched_mm_experts_forward

from modeling.o5.prefill_moe import padded_bmm_experts_forward


@pytest.mark.parametrize("layout", ["uniform", "skewed", "ep", "no_local_routes", "empty"])
@pytest.mark.parametrize("transposed,bias,gate", [(False, False, True), (True, True, True), (False, True, False)])
def test_grouped_tokens_preserve_routing_and_ep_semantics(layout, transposed, bias, gate):
    torch.manual_seed(8)
    experts, hidden, intermediate, top_k = 4, 8, 6, 2
    tokens = 0 if layout == "empty" else 17
    up = torch.randn(experts, intermediate * (2 if gate else 1), hidden)
    down = torch.randn(experts, hidden, intermediate)
    module = SimpleNamespace(
        num_experts=experts, has_gate=gate, has_bias=bias, is_transposed=transposed,
        gate_up_proj=up.transpose(1, 2) if transposed else up,
        up_proj=up.transpose(1, 2) if transposed else up,
        down_proj=down.transpose(1, 2) if transposed else down,
        gate_up_proj_bias=torch.randn(experts, intermediate * (2 if gate else 1)),
        up_proj_bias=torch.randn(experts, intermediate * (2 if gate else 1)),
        down_proj_bias=torch.randn(experts, hidden), act_fn=F.silu,
        _apply_gate=lambda value: F.silu(value.chunk(2, -1)[0]) * value.chunk(2, -1)[1],
    )
    inputs = torch.randn(tokens, hidden)
    indices = torch.randint(experts, (tokens, top_k))
    if layout == "skewed":
        indices.fill_(0)
    elif layout == "ep":
        indices[::2] = experts
    elif layout == "no_local_routes":
        indices.fill_(experts + 1)
    weights = torch.softmax(torch.randn(tokens, top_k), dim=-1)
    actual = padded_bmm_experts_forward(module, inputs, indices, weights)
    expected = (
        torch.zeros_like(inputs) if tokens == 0
        else batched_mm_experts_forward(module, inputs, indices, weights)
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
