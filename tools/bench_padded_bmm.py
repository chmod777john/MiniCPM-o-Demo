"""在 GPU 上对比 ``batched_mm`` 与 ``padded_bmm`` 的单层专家耗时。

形状按 O5 TP2 单卡：hidden=2048、moe_intermediate=256（专家维被切半）、
top_k=8、256 个专家。不加载完整模型，只测专家这一层；40 层的估算是单层耗时乘 40。

用法（仓库根目录，需要一张 CUDA 卡）::

    PYTHONPATH=. python tools/bench_padded_bmm.py
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers.integrations.moe import batched_mm_experts_forward

from modeling.o5.prefill_moe import padded_bmm_experts_forward

# TP2 把 moe_intermediate 从 512 切到 256，与线上单卡看到的专家形状一致。
EXPERTS = 256
HIDDEN = 2048
INTERMEDIATE = 256
TOP_K = 8
LAYERS = 40
LENGTHS = (256, 512, 1536)
WARMUP = 2
REPEAT = 3


def _module(device: torch.device) -> SimpleNamespace:
    """造一层与线上同形状的 SwiGLU 专家权重。"""

    gate = torch.randn(EXPERTS, INTERMEDIATE * 2, HIDDEN, device=device, dtype=torch.bfloat16)
    down = torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, device=device, dtype=torch.bfloat16)
    return SimpleNamespace(
        num_experts=EXPERTS,
        has_gate=True,
        has_bias=False,
        is_transposed=False,
        gate_up_proj=gate,
        up_proj=gate,
        down_proj=down,
        act_fn=F.silu,
        _apply_gate=lambda value: F.silu(value.chunk(2, -1)[0]) * value.chunk(2, -1)[1],
    )


def _time_ms(fn, repeats: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")
    device = torch.device("cuda")
    module = _module(device)
    rows = []
    for length in LENGTHS:
        hidden = torch.randn(length, HIDDEN, device=device, dtype=torch.bfloat16)
        index = torch.randint(EXPERTS, (length, TOP_K), device=device)
        weight = torch.softmax(torch.randn(length, TOP_K, device=device, dtype=torch.float32), dim=-1)
        for _ in range(WARMUP):
            padded_bmm_experts_forward(module, hidden, index, weight)
            batched_mm_experts_forward(module, hidden, index, weight)
        padded_ms = _time_ms(lambda: padded_bmm_experts_forward(module, hidden, index, weight), REPEAT)
        batched_ms = _time_ms(lambda: batched_mm_experts_forward(module, hidden, index, weight), REPEAT)
        rows.append(
            {
                "tokens": length,
                "batched_mm_ms": round(batched_ms, 1),
                "padded_bmm_ms": round(padded_ms, 1),
                "speedup": round(batched_ms / padded_ms, 2) if padded_ms else None,
                "batched_mm_x40_ms": round(batched_ms * LAYERS, 1),
                "padded_bmm_x40_ms": round(padded_ms * LAYERS, 1),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    print("BENCH " + json.dumps(rows, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
