# o5 Pure Qwen3.5 MoE 性能探查报告

日期：2026-07-14

## 代码与路径

当前 worktree：

```text
/user/weihongliang/MiniCPM-o-Demo-wt-o5-pure-moe-profile-2026-07-14
```

当前分支：

```text
wt/o5-pure-moe-profile-2026-07-14
```

来源：

```text
source branch: wt/o5-inference-refactor-2026-06-30
source commit: 07fcb1a421f5525689eb600655192eec8adcb6fb
```

Profiling 脚本：

```text
scripts/profile_qwen35_moe_backbone.py
```

模型目录只读使用：

```text
/user/weihongliang/wangkaiqi/o5_backbone_hf
```

## 当前实验设置

目标是排除 MiniCPM-o 外层、音频、TTS、duplex、websocket，只看纯 Qwen3.5 MoE backbone 的 prefill/decode 性能。

主要环境：

```text
venv: /user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-high-cu128
torch: 2.11.0+cu128
transformers: 5.13.0
GPU: A100-SXM4-80GB
attn_implementation: sdpa
experts_implementation: grouped_mm
```

运行时日志显示：

```text
[transformers] The fast path is not available because one of the required library is not installed.
Falling back to torch implementation.
```

这说明当前环境没有启用 FLA / causal-conv 相关 fast path。当前结果应理解为 high-cu128 fallback 环境下的性能，不代表加速依赖齐全后的最好结果。

## 实验结果目录

模块级 phase profile：

```text
/user/weihongliang/o5_pure_moe_profile_grouped_mm_phase_20260714_145049
```

底层 top-op profile：

```text
/user/weihongliang/o5_pure_moe_profile_topops_noshape_grouped_mm_20260714_153805
```

## 总体性能

在 top-op profile 这组实验中：

```text
prompt_tokens: 51
decode_steps: 8
prefill: 0.427 s
decode mean: 318.3 ms/token
decode throughput: 3.14 token/s
peak GPU memory: 64.8 GiB
```

在更重的 phase profile 实验中：

```text
prefill: 0.577 s
decode mean: 464.6 ms/token
decode throughput: 2.15 token/s
```

Profiler 自身会引入开销，因此不同 profile 配置下 wall time 有差异。共同现象是 decode 很慢。

## 模块级结论

模块级 CUDA event timing 显示，显性耗时主要在 MoE MLP，尤其是 experts。

Top-op profile 中 decode phase：

```text
mlp total:         2028.4 ms
mlp.experts:       1818.4 ms
self_attn:           81.0 ms
```

注意：`mlp` 包含 `mlp.experts`，二者不能相加。可以理解为 MLP 里面的 experts 是主要耗时来源。

prefill phase：

```text
mlp total:          330.7 ms
mlp.experts:        301.9 ms
self_attn:           10.9 ms
```

当前可见结果说明：在该环境和 batch=1 decode 场景下，显性瓶颈不是普通 attention，而是 MoE experts 路径。

## 底层 PyTorch Profiler 结论

Top-op profile 文件：

```text
profiler_cuda_top.txt
profiler_self_cuda_top.txt
profiler_cpu_top.txt
profiler_self_cpu_top.txt
```

核心统计：

```text
Self CPU time total:  3.556 s
Self CUDA time total: 275.866 ms
```

这说明实际 CUDA kernel 执行时间远小于 wall time。慢的主要嫌疑不是单个 GPU kernel 算不动，而是 CPU 调度、kernel launch、小 op 碎片化和 GPU stream 空洞。

关键 op：

```text
aten::_grouped_mm
  calls: 880
  CPU total: 2.281 s
  CPU avg: 2.592 ms/call
  CUDA total: 98.426 ms
  CUDA avg: 111.848 us/call

aten::mm
  calls: 229581
  CPU total: 532.6 ms
  CUDA total: 149.0 ms
  CUDA avg: 0.649 us/call
```

还有大量小 op：

```text
aten::copy_       calls 9118
aten::to          calls 10242
aten::_to_copy    calls 6246
aten::mul         calls 4884
aten::add         calls 4103
aten::topk        calls 440
aten::sort        calls 440
```

这些共同指向：decode=1 token 时 MoE 路由和专家计算路径被拆成大量小 op 和小 kernel。`grouped_mm` 的 GPU kernel 自身并不占绝对大头，但其调用路径和周边路由/搬运/小 op 造成很大的 wall time。

## 当前判断

1. pure MoE 本体已经很慢，问题不主要来自 demo 外层、duplex、TTS 或服务框架。
2. 当前环境下 decode 主要卡在 MoE experts 路径。
3. `grouped_mm` 在 batch=1、decode=1 token、top8 experts 场景下没有充分解决调度碎片问题。
4. 实际 CUDA kernel 总耗时远小于 wall time，说明 CPU/ATen 调度和 GPU 空洞是核心嫌疑。
5. FLA / causal-conv fast path 没开仍然需要修，但当前数据更强地指向 MoE experts 调度路径。

## 后续优化方向

优先级从低风险到高风险：

1. 对齐加速依赖，确认 FLA / causal-conv fast path 是否开启。
2. 对比 `grouped_mm` / `eager` / `batched_mm`，确认当前 experts 实现的相对收益。
3. 用 `nsys` 抓短 timeline，实锤 GPU stream 空洞和 kernel launch 碎片。
4. 尝试 `torch.compile` / CUDA graph / 固定形状 warmup，减少 decode 路径 Python 调度开销。
5. 对 MoE experts 路径做更深改造，例如更适合 batch=1 decode 的 expert batching/fusion，或 EP/TP 的多卡方案。

## 第一批优化实验更新

### `.venv-accel + grouped_mm` 在 A100 不可用

`.venv-accel` 环境包含 `flash_attn 2.8.3` 和 `causal_conv1d 1.6.1`，但它使用 `torch 2.8.0+cu126 / transformers 5.5.4`。在 A100 上尝试 `EXPERTS_IMPLEMENTATION=grouped_mm` 失败：

```text
RuntimeError: torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0
```

因此 `.venv-accel` 不能作为 A100 grouped_mm 路线。

### eager 对照

`.venv-accel + eager`：

```text
prefill: 0.921 s
decode mean: 158.1 ms/token
decode throughput: 6.32 token/s
```

`.venv-high-cu128 + eager`：

```text
prefill: 0.948 s
decode mean: 164.5 ms/token
decode throughput: 6.08 token/s
```

两者 decode 接近，说明 `.venv-accel` 里的 flash-attn / causal-conv 对本次 pure MoE decode 的主要瓶颈帮助有限。当前主要差异仍然在 MoE expert 实现。

### high-cu128 专家实现对照

同一环境下对比：

```text
venv: .venv-high-cu128
torch: 2.11.0+cu128
transformers: 5.13.0
PROFILE=0
decode_steps=32
```

| experts implementation | prefill | decode mean | throughput | decode mlp.experts total |
| --- | ---: | ---: | ---: | ---: |
| grouped_mm | 0.230 s | 139.8 ms/token | 7.15 token/s | 2366.5 ms |
| auto | 0.221 s | 139.5 ms/token | 7.17 token/s | 2360.9 ms |
| batched_mm | 0.165 s | 74.5 ms/token | 13.42 token/s | 303.8 ms |

这里 `auto` 实际选择了 grouped_mm。`batched_mm` 在 A100 + batch=1 decode 场景下明显更快：

```text
decode latency: 139.8 -> 74.5 ms/token
throughput: 7.15 -> 13.42 token/s
```

模块级 timing 也显示 batched_mm 显著降低 experts elapsed time：

```text
grouped_mm mlp.experts total: 2366.5 ms
batched_mm mlp.experts total: 303.8 ms
```

### 当前更新后的判断

1. `grouped_mm` 在 torch 2.11 / transformers 5.13 下能在 A100 跑，但 batch=1 decode 性能不如 `batched_mm`。
2. `batched_mm` 是目前最低风险、收益最大的开关级优化。
3. `.venv-accel` 的 fast path 依赖对本次 pure MoE decode 没有表现出决定性收益；并且 torch 2.8 的 grouped_mm 对 A100 不可用。

## 第二批实验：clean timing 与 nsys

### 最可靠的当前 clean baseline

去掉 PyTorch profiler 和 module hook，只保留逐 token wall time：

```text
/user/weihongliang/o5_pure_moe_opt_batched_clean_timing_20260714_174342
```

配置：

```text
venv: .venv-high-cu128
torch: 2.11.0+cu128
transformers: 5.13.0
experts_implementation: batched_mm
PROFILE=0
MODULE_TIMING=0
DECODE_STEPS=64
DECODE_ATTENTION_MASK=1
```

结果：

```text
prefill: 0.163 s
decode mean: 55.37 ms/token
decode throughput: 18.06 token/s
```

去掉 decode 阶段 attention_mask：

```text
/user/weihongliang/o5_pure_moe_opt_batched_nomask_timing_20260714_174914
```

结果：

```text
decode mean: 53.56 ms/token
decode throughput: 18.67 token/s
```

这说明 decode attention mask 不是主要瓶颈，收益只有约 3%。

### torch.compile 结果

```text
/user/weihongliang/o5_pure_moe_opt_batched_compile_timing_20260714_174559
```

结果：

```text
decode mean: 64.94 ms/token
```

比不 compile 更慢。日志里出现：

```text
torch._dynamo hit config.recompile_limit
cache_params.layers[9].has_previous_state == False
attention_mask rank mismatch. expected 2, actual 4
```

当前判断：直接 `torch.compile(model.model, dynamic=True)` 会被动态 cache / mask 形状影响，不能稳定复用图，因此不是现成优化。

### nsys timeline 结论

nsys 结果目录：

```text
/user/weihongliang/o5_pure_moe_nsys_batched_20260714_175600
```

配置：

```text
experts_implementation=batched_mm
PROFILE=0
MODULE_TIMING=0
DECODE_ATTENTION_MASK=0
CUDA_PROFILER_RANGE=1
DECODE_STEPS=8
```

nsys 自身有明显开销，wall time 变成：

```text
decode mean: 87.74 ms/token
```

但底层统计仍有参考价值：

```text
CUDA kernels: 41651
kernel total: 345.65 ms
kernel span: 1941.82 ms
same-stream gaps total: 1596.17 ms
cudaLaunchKernel total: 294.45 ms / 39971 calls
correlated launch runtime total: 307.25 ms / 41651 calls
```

Top kernel 按总耗时：

```text
vectorized_gather_kernel: 103.53 ms / 1041 calls
ampere bf16 GEMM 64x64: 58.61 ms / 562 calls
cutlass bf16 GEMM: 26.73 ms / 520 calls
gemvx kernel: 19.87 ms / 2640 calls
ampere bf16 GEMM 128x64: 17.13 ms / 560 calls
```

这个结果说明：

1. GPU 真正在算的 kernel 时间远小于端到端 wall time。
2. 单 stream 上有大量 kernel 间空洞，launch/调度碎片非常明显。
3. `batched_mm` 虽然比 `grouped_mm` 快很多，但仍然把 batch=1 decode 拆成大量小 kernel。
4. top kernel 里 gather 很重，说明 MoE 路由后的 expert 权重/hidden state 选择和搬运也是主要成本。

### 当前离 20 ms/token 的距离

当前最可信 clean baseline 是：

```text
53.56 ms/token
```

目标：

```text
<20 ms/token
```

也就是还需要约 2.7 倍加速。简单开关已经做过：

```text
grouped_mm -> batched_mm: 大收益
去 attention_mask: 小收益
直接 torch.compile: 负收益
FLA / causal-conv 环境: 没解决 MoE 主瓶颈
```

下一步需要验证的不是普通 attention 优化，而是减少 decode 阶段 MoE 路由/专家路径的小 kernel 和 CPU launch。

## 下一步优化假设

优先尝试：

1. 固定 decode 输入形状和 cache 形态，再测试 `torch.compile` 或 CUDA graph，避免动态图反复 recompile。
2. 对 MoE experts 路径做 batch=1 decode 专用实现，减少 gather / sort / topk 后的小 kernel 数。
3. 检查是否可以把 top-k routing 后的 expert 计算整理成更少的 batched GEMM 或 fused kernel。
4. 如果业务允许，一次生成多个 token 或并发多个 session，比单 session batch=1 更容易摊薄 launch 开销。

风险判断：

1. 只靠安装 flash-attn / causal-conv / FLA，不太可能把 53 ms/token 拉到 20 ms/token，因为 nsys 显示主要空洞来自 MoE experts 路径。
2. 多卡 EP/TP 对 batch=1 decode 未必直接加速，可能引入通信开销；它更适合解决显存或大 batch 吞吐。
3. 真要稳定低于 20 ms/token，可能需要 vLLM/SGLang 一类 serving kernel，或者专门为 Qwen3.5 MoE decode 写 fused expert path。

## 第三批实验：batch=1 experts patch / StaticCache

任务：

```text
cctl task: 143554
result dir: /user/weihongliang/o5_pure_moe_b1_static_agent_20260714_181452
```

配置共同部分：

```text
venv: .venv-high-cu128
torch: 2.11.0+cu128
transformers: 5.13.0
experts_implementation: batched_mm
PROFILE=0
MODULE_TIMING=0
DECODE_ATTENTION_MASK=0
LOGITS_TO_KEEP=1
DECODE_STEPS=64
```

### `LOGITS_TO_KEEP=1` baseline

```text
prefill: 0.163 s
decode mean: 52.81 ms/token
decode throughput: 18.94 token/s
```

相比上一轮 53.56 ms/token 略好，但不是主要矛盾。`logits_to_keep=1` 更主要是减少 prefill 阶段多余的全序列 lm_head 计算；decode 本来只有 1 个 token，所以收益有限。

### batch=1 experts forward patch

脚本里临时加了 `PATCH_B1_EXPERTS=1`，只在 profiling 脚本里 monkey-patch experts forward，不改 site-packages。它保持 batched_mm 数学路径，但对 `hidden_states.shape == (1, hidden)` 的 decode 情况避免部分通用路径开销。

结果：

```text
prefill: 0.167 s
decode mean: 50.89 ms/token
decode throughput: 19.65 token/s
```

收益：

```text
52.81 -> 50.89 ms/token，约 3.6%
```

结论：Python 层微调 experts forward 有小收益，但远不足以接近 20 ms/token。说明主要开销仍在更多小 kernel / gather / bmm / launch，而不是 `repeat_interleave` 这一类单点 Python/Tensor 操作。

### StaticCache 尝试

脚本里加了 `USE_STATIC_CACHE=1`，希望把 cache 形态固定，为后续 `torch.compile` / CUDA Graph 创造条件。

结果：失败。栈落在 Qwen3.5 linear attention 的 recurrent/conv state 更新：

```text
torch.AcceleratorError: CUDA error: device-side assert triggered
...
modeling_qwen3_5_moe.py:472, in forward
  mixed_qkv = self.causal_conv1d_update(...)
modeling_qwen3_5_moe.py:232, in torch_causal_conv1d_update
  conv_state.copy_(hidden_states_new[:, :, -state_len:])
```

判断：Qwen3.5 不只有普通 attention KV cache，还包含 linear attention / conv recurrent state。直接用通用 `StaticCache` 不是低风险路线，容易破坏 recurrent state 的形状或生命周期。要走 CUDA Graph，需要针对 Qwen3.5 这套混合 cache 先做专门的静态状态管理，而不是简单换成 `StaticCache`。

### 第三批结论

1. 当前最好的无风险结果约 `50.9 ms/token`。
2. 离 `<20 ms/token` 仍差约 2.5 倍。
3. `batched_mm + batch=1 patch` 已经接近开关级优化上限。
4. 真正值得投入的是：
   - Qwen3.5 recurrent/cache 静态化后再试 CUDA Graph；
   - 或者写/接入 fused MoE expert kernel，减少 top8 expert decode 的 gather + bmm + elementwise kernel 数。

## 当前优化边界

截至本轮，已经验证过的低风险手段：

| 手段 | 结果 | 判断 |
| --- | ---: | --- |
| `grouped_mm -> batched_mm` | 约 139.8 -> 74.5 ms/token | 大收益，应该保留 |
| 关闭 decode attention mask | 约 55.37 -> 53.56 ms/token | 小收益 |
| `LOGITS_TO_KEEP=1` | 约 53.56 -> 52.81 ms/token | 小收益，主要利好 prefill |
| batch=1 experts patch | 约 52.81 -> 50.89 ms/token | 小收益，不足以接近目标 |
| 直接 `torch.compile(model.model)` | 约 64.94 ms/token | 负收益，动态图/cache 导致 recompile |
| 通用 `StaticCache` | 失败 | Qwen3.5 linear attention recurrent state 不兼容 |

因此，`<20 ms/token` 目标不能靠现有开关自然达到。接下来需要从更底层入手：

1. 单独 microbench 一层 MoE experts，确认 experts 子模块自身的可优化空间。
2. 若 experts microbench 仍明显慢，优先考虑 fused top8 expert kernel。
3. 若 experts microbench 很快但整模型慢，说明主要是 40 层模型的 launch/gap，需要 CUDA Graph 或服务引擎级 decode graph。
4. Transformers 内置 `deepgemm` / `sonicmoe` 路线当前不适合 A100：代码里要求 SM90+ 或 SM100，A100 是 SM80。

## 第四批实验：单层 experts microbench

任务：

```text
cctl task: 143561
result dir: /user/weihongliang/o5_moe_experts_microbench_20260714_182008
script: scripts/bench_qwen35_moe_experts.py
```

这个实验只抽一层 `layer.mlp.experts`，输入是 batch=1、top8 experts，排除整模型其它部分。

no compile：

| implementation | mean |
| --- | ---: |
| HF `batched_mm` experts | 0.188 ms |
| profiling script `simple_bmm` | 0.163 ms |
| 逐 expert `F.linear` loop | 0.776 ms |

局部 `torch.compile`：

| implementation | mean |
| --- | ---: |
| HF `batched_mm` experts | 0.213 ms |
| profiling script `simple_bmm` | 0.197 ms |
| 逐 expert `F.linear` loop | 0.877 ms |

结论：

1. 单层 experts 子模块并不慢。即使用 `0.188 ms/layer * 40 layers` 粗略估计，也只有约 `7.5 ms/token`。
2. 整模型当前约 `50.9 ms/token`，说明主要剩余开销不是单层 expert GEMM 本身，而是整层 forward 中 attention、linear attention/recurrent state、routing、norm、lm_head、cache 更新以及大量小 kernel launch 的累计。
3. 单独优化 experts Python 路径只能拿到小收益，和第三批 `batch=1 experts patch` 的 3.6% 收益一致。
4. 局部 `torch.compile` 在这个 microbench 上也是负收益，不应该作为当前默认优化。

因此，要继续接近 `<20 ms/token`，更合理的下一步是整模型 decode graph / CUDA Graph / 专用 serving engine，而不是继续手写一个小的 experts wrapper。

## 第五批实验：模块级 timing 复查

任务：

```text
cctl task: 143562
result dir: /user/weihongliang/o5_pure_moe_module_timing_20260714_182444
```

配置：

```text
experts_implementation=batched_mm
MODULE_TIMING=1
PROFILE=0
DECODE_ATTENTION_MASK=0
LOGITS_TO_KEEP=1
DECODE_STEPS=16
```

注意：module hook 会显著增加 wall time，所以这组只看相对分布，不看绝对 decode latency。

decode phase 累计：

| module kind | total | count | mean |
| --- | ---: | ---: | ---: |
| `mlp` | 464.4 ms | 640 | 0.726 ms |
| `mlp.experts` | 156.2 ms | 640 | 0.244 ms |
| `self_attn` | 110.4 ms | 160 | 0.690 ms |
| `post_attention_layernorm` | 89.5 ms | 640 | 0.140 ms |
| `mlp.gate` | 88.4 ms | 640 | 0.138 ms |
| `input_layernorm` | 85.7 ms | 640 | 0.134 ms |
| `mlp.shared_expert` | 83.6 ms | 640 | 0.131 ms |
| `mlp.shared_expert_gate` | 34.4 ms | 640 | 0.054 ms |

这里 `mlp` 包含 `mlp.experts`、`mlp.gate`、`mlp.shared_expert` 等子模块，不能相加。这个结果支持第四批 microbench 的判断：experts 是重要部分，但不是唯一瓶颈；shared expert、router gate、norm、attention/linear attention 以及它们带来的小 kernel launch 也在累计消耗。

prefill phase 仍然主要由 `mlp.experts` 主导：

```text
mlp total: 148.1 ms
mlp.experts: 142.3 ms
self_attn: 2.6 ms
```

所以 prefill 和 decode 的优化侧重点不同：prefill 更像 MoE expert 吞吐问题；batch=1 decode 更像大量层级小 kernel 和调度空洞问题。

## 第六批实验：nsys + NVTX 模块分段

任务：

```text
cctl task: 143798
result dir: /user/weihongliang/o5_pure_moe_nsys_nvtx_modules_20260715_043214
script: scripts/profile_qwen35_moe_backbone.py
analysis: scripts/analyze_nsys_sqlite.py
```

配置：

```text
experts_implementation=batched_mm
PROFILE=0
MODULE_TIMING=0
NVTX_MODULES=1
DECODE_ATTENTION_MASK=0
LOGITS_TO_KEEP=1
DECODE_STEPS=8
CUDA_PROFILER_RANGE=1
```

nsys 会引入额外开销，所以这组不代表真实 clean latency。实际 run 内记录：

```text
decode mean: 87.30 ms/token
decode_loop: 699.53 ms / 8 token
decode_loop kernels: 25328
decode_loop kernel total: 125.68 ms
decode_loop stream gap total: 573.71 ms
```

这再次说明：wall time 主要不是单个 kernel 算得慢，而是大量 kernel launch / kernel gap。

### decode_loop 内模块分布

只统计 `decode_loop` 内的 NVTX range：

| range type | total | count | mean |
| --- | ---: | ---: | ---: |
| layer total | 678.31 ms | 320 | 2.120 ms |
| `mlp` | 238.74 ms | 320 | 0.746 ms |
| `linear_attn` | 224.76 ms | 240 | 0.937 ms |
| `mlp.experts` | 92.70 ms | 320 | 0.290 ms |
| `self_attn` | 73.55 ms | 80 | 0.919 ms |
| `post_attention_layernorm` | 54.97 ms | 320 | 0.172 ms |
| `input_layernorm` | 54.18 ms | 320 | 0.169 ms |
| `mlp.gate` | 51.91 ms | 320 | 0.162 ms |
| `mlp.shared_expert` | 43.52 ms | 320 | 0.136 ms |
| `mlp.shared_expert_gate` | 13.05 ms | 320 | 0.041 ms |

这里 `layer total` 包含子模块；`mlp` 包含 `experts`、`gate`、`shared_expert` 等，不能横向相加。它的意义是显示大块分布。

### 层类型结构

decode 里 40 层大致是：

```text
30 层 linear_attn
10 层 self_attn
```

每个 token 每层平均约 `2.1 ms`（在 nsys 开销下），linear-attn 层里 `linear_attn` 本身约 `0.93 ms`，MLP 约 `0.75 ms`。self-attn 层里 `self_attn` 本身约 `0.92 ms`，MLP 约 `0.74 ms`。

### 关键判断更新

之前关注 MoE experts 是对的，但现在更明确：decode 的大头不是单独 `mlp.experts`，而是：

1. `linear_attn` / recurrent state 路径；
2. MLP 整体，包括 router gate、experts、shared expert；
3. norm 和大量 elementwise/copy/gather 小 kernel；
4. kernel launch / stream gap。

因此，下一步比“继续改 experts wrapper”更有价值的是：

1. 让 FLA / causal-conv fast path 真正启用，再重测 `linear_attn`；
2. 做整模型 decode CUDA Graph，优先减少 25k+ kernel 的 launch/gap；
3. 如果换 H100，再试 `deepgemm` / `sonicmoe` 这类 SM90+ backend。

## 第七批实验：已有 accel venv 的 linear-attn fast path

任务：

```text
cctl task: 143799
result dir: /user/weihongliang/o5_pure_moe_accel_nvtx_20260715_044159
```

环境：

```text
venv: .venv-accel
torch: 2.8.0+cu126
transformers: 5.5.4
causal_conv1d: 1.6.1
fla: 0.5.0
flash_attn: 2.8.3
```

这套环境没有出现 high-cu128 里的 fallback warning，说明 Qwen3.5 linear attention fast path 应该已启用。为了避开 torch 2.8 在 A100 上 `grouped_mm` 不可用的问题，本实验仍使用：

```text
experts_implementation=batched_mm
DECODE_ATTENTION_MASK=0
LOGITS_TO_KEEP=1
```

### clean timing 对比

high-cu128 fallback 环境当前最好 clean baseline：

```text
decode mean: 52.81 ms/token
```

`.venv-accel` fast path 环境：

```text
decode mean: 60.82 ms/token
decode throughput: 16.44 token/s
prefill: 0.163 s
```

结论：这套 accel 环境反而更慢。

### nsys + NVTX 对比

`.venv-accel`：

```text
decode_loop: 709.04 ms / 8 token
decode_loop kernels: 25568
decode_loop kernel total: 127.17 ms
decode_loop stream gap total: 581.71 ms
```

high-cu128 fallback：

```text
decode_loop: 699.53 ms / 8 token
decode_loop kernels: 25328
decode_loop kernel total: 125.68 ms
decode_loop stream gap total: 573.71 ms
```

模块分布对比：

| module | high-cu128 fallback | `.venv-accel` fast path |
| --- | ---: | ---: |
| `mlp` | 238.74 ms | 257.88 ms |
| `linear_attn` | 224.76 ms | 214.32 ms |
| `mlp.experts` | 92.70 ms | 118.63 ms |
| `self_attn` | 73.55 ms | 74.88 ms |

fast path 让 `linear_attn` 只下降约 10.4 ms / 8 token，但 MLP / experts 更慢，整体没有收益。

### 第七批结论

1. 已有 `.venv-accel` 不是更优运行环境。
2. `linear_attn` fast path 本身可能有小收益，但被 torch/transformers 版本和 MoE experts 实现差异抵消。
3. 不能直接把“装齐 FLA / causal-conv”当作优化结论；需要在 high-cu128 这套更快的环境中单独补齐 causal-conv/flash-attn 后再测，或者继续走 CUDA Graph / fused path。

## 第八批实验：naive CUDA Graph decode probe

任务：

```text
cctl task: 143800
result dir: /user/weihongliang/o5_decode_graph_probe_20260715_044619
script: scripts/try_qwen35_decode_graph.py
```

环境：

```text
venv: .venv-high-cu128
torch: 2.11.0+cu128
transformers: 5.13.0
experts_implementation: batched_mm
```

结果：

```text
graph_capture: ok
eager warm decode after first step: ~55 ms/token
graph replay mean: 14.00 ms/token
```

这是目前最重要的结果：CUDA Graph 能把单步 replay 降到 `<20 ms/token`，说明之前 nsys 看到的 launch/gap 确实是核心瓶颈。

限制：这个 probe 只是捕获“同一个 decode step 输入/同一个 cache state”的重放，没有实现真实连续生成循环：

1. 没有在 graph 外更新 `graph_token` 为上一步输出 token；
2. 没有验证 replay 后 recurrent/KV state 是否可以作为下一步生成的合法状态持续推进；
3. 没有处理 argmax/sampling、结束条件、不同 session、不同 prompt 长度；
4. 没有接入 MiniCPM-o 外层或 duplex。

因此它不是可直接使用的推理实现，但已经证明方向成立。下一步应该做“可推进 graph decode loop”：固定输入 buffer，capture 一步 decode，replay 后把输出 token copy 回输入 buffer，并确认连续多 token 输出和 eager 一致或至少合法。

## 第九批实验：可连续推进的 CUDA Graph decode loop

任务：

```text
cctl task: 143801
result dir: /user/weihongliang/o5_decode_graph_loop_probe_20260715_044935
script: scripts/try_qwen35_decode_graph_loop.py
```

这次在第八批基础上增加了固定 `graph_token` buffer：

1. capture 一步 decode；
2. 每次 replay 后在图外做 `argmax`；
3. 把输出 token `copy_` 回 `graph_token`；
4. 连续 replay 32 步。

结果：

```text
graph_loop: ok
graph_loop_mean: 13.97 ms/token
```

生成文本片段：

```text
询问西安的历史、地理、文化和旅游特色，看来是希望获得一份全面且结构清晰的介绍。虽然问题重复了四次，但核心需求很明确，
```

这说明 pure Qwen3.5 backbone 的连续 greedy decode graph loop 是可行的，并且已经达到 `<20 ms/token`。

当前仍然是 probe，不是生产实现，原因：

1. 只覆盖 pure HF backbone，不含 MiniCPM-o 外层、duplex、TTS；
2. 只做 greedy argmax，未做 sampling/top-p/temperature；
3. capture 绑定了当前 prompt 后的 cache/state 形态，需要设计 graph 生命周期；
4. 没有处理 batch/session 切换、结束条件、不同最大 cache 长度；
5. 还没有验证长序列后 cache/recurrent state 的边界行为。

但优化方向已经明确：如果要把 o5 decode 压到 20ms/token 内，应该优先把这个 graph loop 工程化，而不是继续做小的 experts wrapper。

## 第十批实验：graph replay nsys

任务：

```text
cctl task: 144050
result dir: /user/weihongliang/o5_decode_graph_loop_nsys_20260715_063348
script: scripts/try_qwen35_decode_graph_loop.py
```

配置：

```text
CUDA_PROFILER_RANGE=1
NVTX_RANGES=1
STEPS=32
```

结果：

```text
graph_loop_mean: 15.00 ms/token   # nsys 下略慢于非 nsys 的 13.97
graph_replay_loop: 486.35 ms / 32 token
```

nsys 的 CUDA kernel 表只看到 graph 外部的 argmax kernel：

```text
ArgMax reduce kernel: 0.367 ms / 32 calls
```

CUDA runtime：

```text
cudaGraphLaunch: 29.65 ms / 32 calls, avg 0.93 ms
cudaDeviceSynchronize: 449.70 ms / 65 calls
cudaMemcpyAsync: 0.54 ms / 64 calls
cudaLaunchKernel: 0.40 ms / 32 calls   # argmax
cudaMemsetAsync: 0.24 ms / 32 calls
```

解释：Nsight Systems 默认不会把 CUDA Graph 内部节点像 eager kernel 那样完整展开到 `CUPTI_ACTIVITY_KIND_KERNEL` 统计里，所以这里不能直接用 top kernel 判断 graph 内部算子瓶颈。但它能说明 graph 外部开销：

1. 每步 graph launch 本身约 `0.93 ms`；
2. 图外 argmax kernel 很小，约 `0.011 ms/token`；
3. 当前测时脚本每步 `sync_time()` 都有 `cudaDeviceSynchronize`，这在服务里可以避免逐 token 强同步；
4. 如果服务端异步 pipeline 做得好，真实可用延迟可能更接近 graph replay 本身，而不是被每步 Python synchronize 放大。

下一步可优化点：

1. 不在每个 token 后 `cudaDeviceSynchronize`，改为 CUDA event 或异步队列测时；
2. 把 argmax / token copy 尽量纳入 graph，或者至少避免 CPU 同步；
3. 用更合适的 Nsight Compute / CUDA graph node profiling 方式看 graph 内部真实 kernel；
4. 工程化 graph decoder，确认真实服务路径是否还能保持 14-15 ms/token。

## 第十一批实验：graph loop 逐步同步 vs 统一同步

任务：

```text
cctl task: 144051
result dir: /user/weihongliang/o5_decode_graph_loop_async_20260715_063926
```

对比：

```text
SYNC_EACH_STEP=1: 14.09 ms/token
SYNC_EACH_STEP=0: 13.97 ms/token
```

两者输出文本一致，说明之前每步 `cudaDeviceSynchronize` 并不是 14ms 中的主要剩余开销。当前 14ms/token 已经更接近 graph 内实际计算，而不是测量同步造成的假慢。

这也更新了后续判断：

1. 单纯把同步从逐 token 改成批量同步，收益很小；
2. 图外 argmax/copy 也不是主要瓶颈；
3. 继续降到 10ms/token 以内，需要看 graph 内部算子本身，例如 linear attention、MLP/shared expert、norm/elementwise，而不是只优化 Python 调度。
4. 下一步应围绕 `batched_mm` 做 top-op profile，并考虑在 demo 推理代码中为 A100/batch=1 decode 默认选择 `batched_mm`。

## batched_mm top-op profile

batched_mm top-op profile 目录：

```text
/user/weihongliang/o5_pure_moe_profile_topops_batched_mm_20260714_173844
```

带 PyTorch Profiler 时：

```text
prefill: 0.188 s
decode mean: 108.3 ms/token
throughput: 9.24 token/s
```

由于 profiler 有开销，该数值比 timing-only 的 74.5 ms/token 慢，但仍显著快于 grouped_mm 的带 profiler结果。

模块级 decode phase：

```text
mlp total:          335.1 ms
mlp.experts:        116.3 ms
self_attn:           84.8 ms
mlp.gate:            67.3 ms
shared_expert:       60.7 ms
```

相比 grouped_mm top-op profile 中的 decode phase：

```text
grouped_mm mlp.experts: 1818.4 ms
batched_mm mlp.experts:  116.3 ms
```

底层 op 结构也发生变化。batched_mm 的主要 CUDA op：

```text
aten::index / vectorized_gather: ~99.6 ms CUDA total
aten::bmm:                       ~74.7 ms CUDA total
aten::mm:                        ~51.5 ms CUDA total
```

这与 grouped_mm profile 中 `_grouped_mm` CPU total 很高、周边小 op 很多的现象不同。batched_mm 把 experts 路径改造成更适合 A100/batch=1 decode 的执行形态，显著降低了 experts elapsed time。

当前最明确的工程建议：在 A100 + batch=1 streaming/decode 场景下，优先使用 `EXPERTS_IMPLEMENTATION=batched_mm`，不要让 `auto` 默认选择 grouped_mm。

## clean timing 与 torch.compile 结果

此前带 module hook / profiler 的数字会明显放大 decode 时间。加入 `MODULE_TIMING=0` 后重新测 clean timing：

```text
EXPERTS_IMPLEMENTATION=batched_mm
PROFILE=0
MODULE_TIMING=0
DECODE_STEPS=64
```

结果目录：

```text
/user/weihongliang/o5_pure_moe_opt_batched_clean_timing_20260714_174342
```

结果：

```text
prefill: 0.163 s
decode mean: 55.37 ms/token
decode throughput: 18.06 token/s
```

这应作为当前更可信的单卡 A100 batched_mm decode 基线。

尝试 `torch.compile(model.model, dynamic=True, mode=default)`：

```text
/user/weihongliang/o5_pure_moe_opt_batched_compile_timing_20260714_174559
```

结果：

```text
prefill: 0.193 s
decode mean: 64.94 ms/token
decode throughput: 15.40 token/s
```

compile 不但没有收益，还触发了 recompile limit：

```text
torch._dynamo hit config.recompile_limit
cache_params.layers[9].has_previous_state == False
attention_mask rank mismatch. expected 2, actual 4
```

当前判断：直接 compile 整个 `model.model` 不适合作为主线。decode 路径里 cache state 和 attention mask 形状变化会导致重编译/graph break。下一步优先减少 decode 输入动态性，例如避免每 token 重建 full attention_mask，或改用静态 cache/mask 后再考虑 CUDA graph / compile。

## 第十二批实验：CUDA Graph node 级 nsys

任务：

```text
cctl task: 144053
result dir: /user/weihongliang/o5_decode_graph_loop_nsys_node_20260715_064913
script: scripts/try_qwen35_decode_graph_loop.py
nsys: --cuda-graph-trace=node
```

这次打开 `--cuda-graph-trace=node`，可以看到 graph replay 内部节点。8 token 的统计：

```text
CUPTI kernels: 25864
kernel total: 115.88 ms
kernel span: 141.63 ms
```

按 8 token 粗略折算，graph 内核执行约 `14.5 ms/token`，与脚本测得的 `13.97-15.00 ms/token` 基本一致。这说明 CUDA Graph 后剩下的时间主要是实际 GPU kernel 工作，不是 Python launch 或逐步同步造成的假慢。

Top kernel：

| kernel 类别 | calls | total |
| --- | ---: | ---: |
| `vectorized_gather_kernel` | 640 | 15.49 ms |
| BF16 GEMM 64x64 | 328 | 12.61 ms |
| cuBLAS GEMV | 1760 | 12.58 ms |
| BF16 GEMM 128x64 | 320 | 10.20 ms |
| `direct_copy` | 2336 | 7.91 ms |
| BF16 GEMM 64x64 另一组 | 240 | 3.90 ms |
| elementwise mul | 1280 | 3.88 ms |
| CUTLASS 16x16 GEMM | 320 | 3.86 ms |
| copy/cast elementwise | 1280 | 3.57 ms |
| topk gather | 320 | 2.64 ms |

当前解释：

1. graph 之后主要瓶颈已经落到 MoE 路由、专家权重 gather、小 GEMM/GEMV、copy/elementwise 上；
2. linear attention recurrent kernel 和 depthwise conv 不是当前最大头；
3. 单纯减少 Python 调度已经不够，后续要么换更融合的 MoE kernel，要么在 H100/更新后端上验证 SonicMoE/DeepGEMM 这类实现。

## 第十三批实验：graph 后端对比

任务：

```text
cctl task: 144054
result dir: /user/weihongliang/o5_graph_backend_compare_20260715_065505
script: scripts/try_qwen35_decode_graph_loop.py
```

结果：

| experts implementation | batch1 patch | graph 状态 | graph decode |
| --- | --- | --- | ---: |
| `batched_mm` | false | ok | 13.98 ms/token |
| `batched_mm` | true | ok | 13.99 ms/token |
| `eager` | false | capture failed | - |
| `eager` | true | ok | 14.43 ms/token |

结论：

1. A100 上 graph decode 目前仍以 `batched_mm` 最好；
2. 之前尝试的 batch=1 experts patch 在 graph 化之后没有收益；
3. eager 不是更好的主线，未 patch 时甚至无法完成 graph capture，patch 后也慢于 batched_mm。

## 第十四批实验：full-step graph capture

任务：

```text
cctl task: 144052
result dir: /user/weihongliang/o5_decode_graph_fullstep_20260715_064632
script: scripts/try_qwen35_decode_graph_fullstep.py
```

目标是把 decode forward、argmax、token 写回、position/cache index 维护尽量放进同一个 CUDA Graph。结果失败：

```text
graph_fullstep: failed
error_type: AcceleratorError
error: cudaErrorStreamCaptureInvalidated
```

失败发生前普通 eager warm decode 约 `57 ms/token`。当前判断：full-step capture 里仍包含不适合 capture 的动态索引或 buffer 写入路径。这个方向不是完全不可行，但收益上限不大，因为第十一批实验已经证明 graph 外 argmax/copy 不是主要瓶颈。短期不应把它作为最高优先级。

## 第十五批实验：DeepGEMM A100 legacy 路径

实验 worktree：

```text
/user/weihongliang/MiniCPM-o-Demo-wt-o5-deepgemm-a100-2026-07-20
branch: wt/o5-deepgemm-a100-2026-07-20
commit: 74be9f0947cd2f6735de51a1ac45cf67d46eb893
report: o5-deepgemm-a100-notes-2026-07-20.md
```

DeepGEMM README 的主线 Quick start 要求 `SM90 or SM100`，但源码里仍有 `deep_gemm.legacy`，顶层注释为 `Legacy Triton kernels for A100`。因此本次单独验证了 A100/SM80 legacy Triton kernel，而不是 DeepGEMM 当前 SM90/SM100 主路径。

### legacy m-grouped GEMM smoke

任务和结果：

```text
task_id=595718
out_dir=/user/weihongliang/deepgemm_legacy_a100_probe_20260720_115954
torch=2.3.0a0+ebedce2
cuda=12.3
triton=2.2.0
device=A100-SXM4-80GB
shape: groups=8, m_per_group=128, M=1024, N=512, K=1024
legacy mean: 0.0932 ms
```

结论：DeepGEMM legacy Triton BF16 m-grouped GEMM 在 A100 上可以实际运行。之前不尝试它不是因为文档明确禁止 A100，而是因为 DeepGEMM 主路径偏 SM90/SM100，并且当前 decode 场景更关心小 batch/top8 expert routing。

### Qwen3.5-MoE expert 形状 smoke

Qwen3.5-MoE backbone 配置：

```text
hidden_size=2048
moe_intermediate_size=512
num_experts=256
num_experts_per_tok=8
```

对应两个 expert GEMM 近似为：

```text
gate/up: K=2048, N=1024
down:    K=512,  N=2048
```

在 legacy kernel 的 128 行对齐假设下，理想 grouped GEMM 结果：

| task | shape | legacy mean |
| --- | --- | ---: |
| `595741` | groups=8, m_per_group=128, M=1024, N=1024, K=2048 | 0.1302 ms |
| `595742` | groups=8, m_per_group=128, M=1024, N=2048, K=512 | 0.0989 ms |

这些结果说明 legacy GEMM 本体很快，但它要求每个 group 至少按 128 行对齐。真实 batch=1 decode 是 1 个 token 走 top8 experts，每个 expert 实际只有约 1 行输入。如果直接使用 legacy m-grouped GEMM，需要把每个 expert padding 到 128 行，产生大量无效计算。

### 单 token decode microbench

任务：

```text
task_id=595799
out_dir=/user/weihongliang/deepgemm_qwen_decode_bench_20260720_venv3
script: scripts/bench_deepgemm_legacy_qwen_decode.py
venv=/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel
torch=2.8.0+cu126
transformers=5.5.4
device=A100-SXM4-80GB
```

测试对象是第 0 层 MoE experts，单 hidden token、top8 experts。`hf_experts` 强制设置：

```python
model.config._experts_implementation = "batched_mm"
```

否则 torch 2.8 + transformers 5.5.4 会默认尝试 `grouped_mm`，在 A100 上报 `torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0`。

结果：

| path | mean ms |
| --- | ---: |
| `simple_bmm` | 0.1627 |
| `hf_experts` | 0.2189 |
| `deepgemm_padded` | 0.2746 |

数值误差：

```text
deepgemm_vs_simple_max_abs_diff=1.22e-4
deepgemm_vs_simple_mean_abs_diff=5.13e-7
```

结论：对当前单请求、单 token decode、top8 expert 场景，DeepGEMM legacy A100 路径即使正确，也因为 128 行 padding 慢于当前 `simple_bmm` / `batched_mm` 路径。它不适合直接作为当前 decode 主线优化。

DeepGEMM legacy 更可能适合 token 数足够多的 grouped GEMM，例如 prefill、多请求 batching 后 expert token 聚合，或者后续自研/接入真正支持小 M routing 的 fused MoE kernel。

## 第十六批实验：Transformers TP2 双卡

新增脚本：

```text
scripts/profile_qwen35_moe_tp2.py
```

运行方式：

```bash
torchrun --standalone --nproc_per_node=2 scripts/profile_qwen35_moe_tp2.py
```

脚本使用 Transformers 自带 tensor parallel：

```python
Qwen3_5MoeForCausalLM.from_pretrained(..., tp_plan="auto")
model.config._experts_implementation = "batched_mm"
```

`tp_plan="auto"` 会使用 Qwen3.5-MoE config 中的 base TP plan，包括 attention projection、MoE experts、shared expert 和 lm_head 的分片。这里仍强制 `batched_mm`，避免 torch 2.8 在 A100 上默认 `grouped_mm` 报 compute capability 9.0 的错误。

### 首次 TP2 eager 结果

```text
task_id=596377
out_dir=/user/weihongliang/o5_qwen35_tp2_profile_20260720_1
world_size=2
torch=2.8.0+cu126
transformers=5.5.4
input_tokens=103
decode_steps=32
```

结果：

```text
prefill_ms: 36639.1 ms
decode mean: 73.93 ms/token
```

prefill 包含明显的一次性初始化/通信/TP 懒加载开销，不适合作为 warmed prefill 对比；decode 已经稳定在约 74 ms/token。

### warmed TP2 eager 结果

```text
task_id=596398
out_dir=/user/weihongliang/o5_qwen35_tp2_profile_20260720_2
world_size=2
input_tokens=103
warmup_prefill_steps=2
warmup_decode_steps=16
decode_steps=64
```

结果：

```text
warm prefill min: 168.0 ms
measured prefill: 161.6 ms
decode mean: 75.49 ms/token
decode throughput: 13.25 token/s
```

对比当前单卡记录：

| 路径 | prefill | decode |
| --- | ---: | ---: |
| 单卡 `batched_mm` eager | 约 165 ms | 约 55 ms/token |
| 单卡 `batched_mm` CUDA Graph | - | 13.97-14.00 ms/token |
| TP2 `tp_plan=auto` eager | 约 162 ms | 75.5 ms/token |

结论：Transformers TP2 在当前 A100、batch=1、单请求 decode 场景下没有收益。prefill warmed 后大致持平，decode 反而慢于单卡 eager，更远慢于单卡 CUDA Graph。原因符合预期：TP 会减少每卡局部矩阵计算，但 batch=1 decode 的 MoE/attention 路径本来已经被大量小 kernel、gather/copy 和 launch/通信开销主导，TP 引入的跨卡通信会抵消甚至超过计算减少。

TP2 仍有价值的方向可能是：

1. 显存容量不够时分片承载模型；
2. batch 或并发足够大时提高吞吐；
3. 与更高层 serving batching / CUDA Graph / fused MoE kernel 配合后重新评估。

但对当前目标“单请求 decode latency 低于 20 ms/token”，TP2 不是直接加速路径。

## 第十七批实验：TP2 + CUDA Graph

实验 worktree：

```text
/user/weihongliang/MiniCPM-o-Demo-wt-o5-pure-moe-tp2-graph-profile-2026-07-20
branch: wt/o5-pure-moe-tp2-graph-profile-2026-07-20
commit: ff53e29
report: o5-pure-moe-tp2-graph-profile-report-2026-07-20.md
```

新增脚本：

```text
scripts/profile_qwen35_moe_tp2_graph.py
```

任务：

```text
task_id=596436
out_dir=/user/weihongliang/o5_qwen35_tp2_graph_profile_20260720_1
world_size=2
torch=2.8.0+cu126
transformers=5.5.4
device=A100-SXM4-80GB
```

结果：

```text
graph_capture: ok
eager warm decode median: 76.45 ms/token
graph replay mean: 13.19 ms/token
graph replay throughput: 75.80 token/s
```

但这个结果有一个关键限制：生成 token 全部相同，输出重复 `作为`。因此当前 TP2 graph 脚本只能说明“TP2 one-token forward 固定形状路径可以被 CUDA Graph capture，并且 replay 成本约 13.2 ms”，还不能说明它已经是一个正确的自回归 decode loop。

这和单卡 graph probe 的状态不同：TP2 graph 当前还没有证明 KV/cache/token 状态以正常 eager decode 的方式推进。任务写出 `result.json` 后还卡在最终分布式清理，已手动停止并在实验脚本里去掉 final barrier。

当前判断：TP2 + CUDA Graph 是有技术可行性的，但还需要额外工程处理状态推进和进程退出。即使做到正确，它目前的 replay 成本也只是略低于单卡 graph 的 `13.97-14.00 ms/token`，优势不明显；考虑到 TP2 eager 慢、实现复杂度高，短期不应作为主优化路径。

## 当前优化判断

截至 commit `5d63fa4`、DeepGEMM A100 legacy 补充实验 `74be9f0`、TP2 eager 补充实验和 TP2 graph 补充实验，A100 pure Qwen3.5 MoE backbone 的最可信最好结果仍是单卡 CUDA Graph：

```text
experts_implementation: batched_mm
CUDA Graph decode loop: 13.97-14.00 ms/token
```

这已经达到最初 `decode < 20 ms/token` 的实验目标。继续下降的主要路径不是继续改 Python wrapper，而是：

1. 在 H100/SM90 上验证 `sonicmoe` / DeepGEMM 主路径 / 新 fused MoE backend；
2. 或者自研更融合的 MoE decode kernel，减少 expert weight gather、copy、topk/gather、scatter/add 等碎片；A100 legacy DeepGEMM 已验证不适合直接优化 batch=1 decode；Transformers TP2 eager 已验证不适合直接优化 batch=1 latency；TP2 graph 虽可 capture，但尚未证明正确自回归推进；
3. 工程化 CUDA Graph 生命周期，让服务端真实 decode 路径能复用当前实验里的 graph loop。

从 node trace 粗略估算，单独消除显性 `vectorized_gather_kernel` 的理论上限约 `15.49ms / 8 = 1.94 ms/token`。如果 fused MoE 同时减少 gather、copy、elementwise 和小 GEMM/GEMV 调度，A100 等价收益可能在 `3-5 ms/token` 量级；H100 上需要实测，可能还会受 Tensor Core、内存带宽、后端实现和 kernel 可用性影响。
