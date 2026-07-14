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
