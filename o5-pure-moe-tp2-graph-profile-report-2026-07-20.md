# o5 pure MoE TP2 CUDA Graph profile report

## Context

This worktree checks whether pure Qwen3.5-MoE / Transformers tensor-parallel decode can be CUDA-graph captured on two A100 GPUs, and whether that changes the earlier TP2 eager result.

Base branch and commit:

```text
branch: wt/o5-pure-moe-profile-2026-07-14
commit: 9f0c772e4176c3738b97314c58e6c9f7c265ba10
```

## Script

```text
scripts/profile_qwen35_moe_tp2_graph.py
```

The script uses:

```python
Qwen3_5MoeForCausalLM.from_pretrained(..., tp_plan="auto")
model.config._experts_implementation = "batched_mm"
```

It runs normal prefill and eager warmup, then captures one decode forward with `torch.cuda.CUDAGraph()` and replays it.

## Result

Task:

```text
task_id=596436
out_dir=/user/weihongliang/o5_qwen35_tp2_graph_profile_20260720_1
world_size=2
torch=2.8.0+cu126
transformers=5.5.4
device=A100-SXM4-80GB
```

Recorded result:

```text
graph_capture: ok
eager warm decode median: 76.45 ms/token
graph replay mean: 13.19 ms/token
graph replay throughput: 75.80 token/s
```

Important limitation:

```text
generated_token_ids are all 97698
generated_text repeats "作为"
```

This means the current script is a valid fixed-shape one-token forward replay cost probe, but not a correct autoregressive generation implementation. The replayed graph uses the captured cache object and a static token buffer. Although the token buffer is copied between replays, the generated sequence shows that the effective autoregressive state is not advancing as a normal eager decode loop would.

The job wrote `result.json` but remained alive at final distributed cleanup, so task `596436` was stopped manually after the result was saved. The script was updated to avoid a final barrier in `finally`.

## Interpretation

TP2 + CUDA Graph can capture and replay the one-token forward path. The measured replay cost is about the same order as the best single-card graph result:

```text
single-card graph: 13.97-14.00 ms/token
TP2 graph probe:  13.19 ms/token
```

However, unlike the single-card graph loop, this TP2 graph probe has not yet demonstrated correct autoregressive token progression. Therefore it should not be treated as a production decode result yet.

The useful conclusion is narrower:

1. TP2 eager is slow for batch=1 decode: about 75.5 ms/token.
2. TP2 one-token forward can be graph-captured: about 13.2 ms/replay.
3. More engineering is needed to make TP2 graph replay update cache/token state correctly and exit cleanly.

Given that single-card graph is already about 14 ms/token and correct enough for the existing probe, TP2 graph is not currently a clearly better latency path. It may still matter for memory capacity or throughput/batching scenarios.
