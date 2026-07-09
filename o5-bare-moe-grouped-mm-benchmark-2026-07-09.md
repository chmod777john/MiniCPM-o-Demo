# o5 裸 Qwen3.5 MoE grouped-mm 对比报告

## 代码定位

- 代码目录：`/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30`
- 分支：`wt/o5-inference-refactor-2026-06-30`
- 代码 commit：`34e79fa4ea13d39d3b246b6da1fdcd35043e2c80`
- 主要脚本：`scripts/benchmark_qwen35_moe_official_prefill_decode.py`
- 辅助脚本：`scripts/probe_qwen35_moe_official_class.py`

## 测试目的

这次测试只看 o5 使用的 Qwen3.5 MoE LLM 本体，不混入 MiniCPMO 外壳、audio encoder、TTS、token2wav 或 duplex 状态机。

目标是回答三个问题：

1. 在较新的 torch / transformers 组合下，A100 是否可以不切 eager，直接使用 grouped-mm。
2. grouped-mm 相比 eager 对 prefill 和 decode 分别有什么影响。
3. 裸 MoE 路径的显存和单 token decode 延迟大致是多少。

## 环境

实际可在 agent-dev A100 上运行的环境是：

```text
.venv-high-cu128
torch==2.11.0+cu128
torchvision==0.26.0+cu128
torchaudio==2.11.0+cu128
torchcodec==0.11.1+cu128
transformers==5.13.0
triton==3.6.0
```

该环境中 `torch.nn.functional.grouped_mm` 存在：

```text
has_functional_grouped_mm = true
```

曾尝试 `.venv-high` 的 cu130 组合：

```text
torch==2.11.0+cu130
transformers==5.13.0
```

但 agent-dev 节点驱动版本为 `12040`，无法加载 cu130 torch，报错为 driver too old。因此本次有效测速使用 cu128。

## 模型与权重

```text
MODEL_PATH=/user/weihongliang/MiniCPM-o-4_6
PT_PATH=/user/weihongliang/o5_weights/omni_sft2_main_run_iter1200.pt
```

脚本显式构造 Transformers 官方类：

```python
from transformers import Qwen3_5MoeForCausalLM
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

config = Qwen3_5MoeTextConfig.from_pretrained(MODEL_PATH)
model = Qwen3_5MoeForCausalLM(config)
```

只从 checkpoint 加载 `llm.*` 权重，不加载视觉、音频、TTS 模块。

## 测试方法

脚本不调用 `generate()` 做整体计时，而是手写 forward：

1. `model(input_ids=..., use_cache=True)` 计一次 prefill。
2. 从 prefill 的 logits 取 argmax 作为 next token。
3. 带 `past_key_values` 循环 decode 64 步。
4. 记录 decode mean / median / p90 / p99 / tokens/s。

共同参数：

```text
ATTN_IMPLEMENTATION=sdpa
DECODE_STEPS=64
WARMUP_DECODE_STEPS=4
PROMPT_REPEAT=1
prompt_tokens=12
```

对比变量：

```text
EXPERTS_IMPLEMENTATION=eager
EXPERTS_IMPLEMENTATION=grouped_mm
```

## 任务与结果路径

eager：

```text
cctl task: 142332
result: /user/weihongliang/o5_bare_moe_highcu128_eager_20260709_132441/result.json
```

grouped-mm：

```text
cctl task: 142334
result: /user/weihongliang/o5_bare_moe_highcu128_grouped_mm_20260709_132730/result.json
```

## 结果

| 实现 | prefill | decode mean | decode median | decode p90 | decode p99 | decode tok/s | peak mem |
|---|---:|---:|---:|---:|---:|---:|---:|
| eager | 37.677s | 0.1397s | 0.1389s | 0.1431s | 0.1491s | 7.16 | 64.82 GiB |
| grouped_mm | 36.164s | 0.1208s | 0.1204s | 0.1222s | 0.1259s | 8.28 | 64.82 GiB |

相对变化：

```text
prefill: 37.677s -> 36.164s，约 4.0% 改善
decode mean: 0.1397s -> 0.1208s，约 13.5% 改善
decode tokens/s: 7.16 -> 8.28，约 15.6% 提升
显存：基本不变
```

## 结论

1. 在 `torch==2.11.0+cu128` + `transformers==5.13.0` 下，A100 可以跑通 `grouped_mm`，不再需要像 torch 2.8 环境那样强制切到 eager。
2. 对这次 12-token prompt / 64-token decode 的裸 MoE 测试，grouped-mm 对 decode 有稳定收益，约 13%-16%。
3. prefill 仍然非常慢，grouped-mm 只带来小幅改善。这说明当前短 prompt prefill 的 30 多秒里可能混有首次调用、kernel 初始化、cache 初始化或其他非纯 MoE GEMM 开销；需要用更长 prompt、预热 prefill 或更细的模型内部计时继续拆分。
4. 裸 LLM 单卡显存约 64.8 GiB，接近 A100 80G 上限但可运行。该数值不包含 MiniCPMO 的 audio/TTS/duplex 额外开销。

## 注意事项

- cu130 high venv 已安装，但当前 agent-dev 节点驱动不支持，实际在集群上不能直接用。
- cu128 high venv 未写入 `requirements-high.txt`，而是单独写入 `requirements-high-cu128.txt`，避免和 cu130 方案混淆。
- 本实验是裸 MoE LLM，不代表完整 o5 duplex demo 的端到端速度。完整 demo 还会叠加 audio prefill、duplex 调度、TTS decode、token2wav 等耗时。
