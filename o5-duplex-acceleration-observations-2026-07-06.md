# o5 双工推理加速实验记录

本文记录 o5 audio duplex probe 在基础环境和可选加速环境下的现象。实验目的不是做严格 kernel benchmark，而是确认当前 demo/probe 这条端到端双工路径是否能从可选 CUDA 加速依赖中获益。

## 环境设置

实验使用同一份代码、同一份模型代码目录、同一份 checkpoint、同一条 audio duplex probe。区别只在虚拟环境和 attention backend：

| 环境 | 虚拟环境 | attention | 可选加速依赖 |
|---|---|---|---|
| base | `.venv-base` | `sdpa` | 不安装 `flash-attn` / `causal-conv1d` |
| accel | `.venv-accel` | `flash_attention_2` | 安装 `flash-attn` / `causal-conv1d` |

为了避免只依赖 cctl 日志，probe 已经改为在输出目录写 `runtime_info.json`，记录实际运行时的 fast-path 状态。

## 用 cctl 运行双工 probe

实验使用两个 probe 脚本：

```bash
scripts/minimal_o5_vendored_duplex.py
scripts/minimal_o5_unified_model_duplex.py
```

它们都会从 `MODEL_PATH` 加载模型代码和 tokenizer，从 `PT_PATH` 加载 checkpoint 权重，合成或读取一段用户输入音频，然后按 audio duplex chunk 跑若干轮推理，最后写出音频、`runtime_info.json` 和 `timings.jsonl`。

必须传入的环境变量：

```bash
export MODEL_PATH=/path/to/model-code-and-tokenizer
export PT_PATH=/path/to/checkpoint.pt
export OUT_DIR=/path/to/output-dir
```

可选环境变量：

```bash
export REF_WAV=assets/ref_audio/ref_minicpm_signature.wav
export USER_TEXT='请详细介绍西安的历史文化、旅游景点、美食和城市特色。'
export USER_WAV=/path/to/user_input_16k.wav
```

在 GPU 节点上运行基础路径。把 `PROJECT`、`CLUSTER`、`IMAGE`、`RESOURCE_POOL` 替换成当前 cctl 环境中的实际值，把 `PROBE_SCRIPT` 替换成上面的任一 probe 脚本：

```bash
cctl job create \
  --project PROJECT \
  --cluster CLUSTER \
  --resource-pool RESOURCE_POOL \
  --image IMAGE \
  --gpu 1 \
  --cpu 16 \
  --memory 160 \
  --entry '
    cd "$PWD" &&
    source .venv/bin/activate &&
    export MODEL_PATH="/path/to/model-code-and-tokenizer" &&
    export PT_PATH="/path/to/checkpoint.pt" &&
    export OUT_DIR="/path/to/probe-output-base" &&
    export ATTN_IMPLEMENTATION=sdpa &&
    python PROBE_SCRIPT
  '
```

安装可选加速依赖后，运行加速路径：

```bash
cctl job create \
  --project PROJECT \
  --cluster CLUSTER \
  --resource-pool RESOURCE_POOL \
  --image IMAGE \
  --gpu 1 \
  --cpu 16 \
  --memory 160 \
  --entry '
    cd "$PWD" &&
    source .venv/bin/activate &&
    export MODEL_PATH="/path/to/model-code-and-tokenizer" &&
    export PT_PATH="/path/to/checkpoint.pt" &&
    export OUT_DIR="/path/to/probe-output-accel" &&
    export ATTN_IMPLEMENTATION=flash_attention_2 &&
    python PROBE_SCRIPT
  '
```

probe 输出包括：

- `runtime_info.json`：attention backend 和 fast-path availability
- `timings.jsonl`：每轮的 `prefill_s`、`generate_s`、`total_s`、文本和 token 信息
- `user_input_16k.wav`：合成或传入的用户输入音频
- `duplex_chunk_*.wav`：每个 chunk 生成的语音
- `duplex_output_timeline_24k.wav`：保留静音间隔的完整时间线音频
- `duplex_output_speech_only_24k.wav`：只拼接模型说话部分的音频

## runtime_info 证据

base 运行时记录：

```json
{
  "attn_implementation": "sdpa",
  "causal_conv1d_available": false,
  "flash_attn_2_available": false,
  "flash_linear_attention_available": true,
  "qwen_moe_fast_path": false
}
```

accel 运行时记录：

```json
{
  "attn_implementation": "flash_attention_2",
  "causal_conv1d_available": true,
  "flash_attn_2_available": true,
  "flash_linear_attention_available": true,
  "qwen_moe_fast_path": true
}
```

因此可以确认：accel 组确实启用了 FlashAttention2、causal-conv1d 和 Qwen MoE fast path。

## vendored probe 结果

vendored probe 使用 vendored modeling 里的 `MiniCPMO` 和 `MiniCPMODuplex`，直接组织双工 prefill/generate。每轮跑 10 个 unit，其中 5 个 listen、5 个 speak。

| 轮次 | base total mean | accel total mean | 现象 |
|---|---:|---:|---|
| r1 | 1.5484s | 1.6690s | accel 更慢 |
| r2 | 1.5681s | 1.6098s | accel 更慢 |
| r3 | 1.6407s | 1.7004s | accel 更慢 |

r3 的细分数据：

| 环境 | prefill mean | generate mean | total mean | total p50 |
|---|---:|---:|---:|---:|
| vendored-base-r3 | 0.6550s | 0.9857s | 1.6407s | 1.7733s |
| vendored-accel-r3 | 0.6673s | 1.0331s | 1.7004s | 1.8395s |

vendored 三轮结果方向一致：加速依赖启用后，端到端并没有变快，反而略慢。

## unified probe 结果

unified probe 使用 demo runtime 的 unified 外壳，调用 `init_unified()` 后走 `duplex_prefill()` / `duplex_generate()`。这更接近当前 demo 服务的实际调用路径。

早先一轮 unified 结果显示 accel 略快：

| 环境 | total mean |
|---|---:|
| unified-base | 1.3928s |
| unified-accel | 1.3685s |

但这轮差距只有约 1.7%，没有把 fast-path 状态写入输出目录，且处于短 probe 的正常抖动范围内。

之后补充 `runtime_info.json` 后重新运行，确认 accel 组确实启用了 fast path，但结果变成 accel 更慢：

| 环境 | prefill mean | generate mean | total mean | total p50 |
|---|---:|---:|---:|---:|
| unified-base | 0.6455s | 0.7015s | 1.3470s | 1.4770s |
| unified-accel | 0.6596s | 0.8022s | 1.4617s | 1.5449s |

这轮 unified 的 accel 输出文本比 base 略长，因此 `generate_s` 不是严格同 token 路径对比。不过 `prefill_s` 也没有变快。

## 当前结论

当前结论需要以 runtime_info 版本的实验为准：

1. 加速依赖确实可以安装并被运行时识别。
2. accel 组确实启用了 `flash_attention_2`、`causal_conv1d` 和 Qwen MoE fast path。
3. 在当前 10 unit 的短 audio-duplex probe 上，无论 vendored 还是 unified，都没有稳定观察到端到端速度收益。
4. vendored 连续三轮都是 accel 慢于 base。
5. unified 早先一轮 accel 略快，但差距很小；补充 runtime_info 后的新一轮显示 accel 更慢。

因此不能得出“当前 demo 双工路径可以从这些加速依赖中获利”的结论。更准确的说法是：

> 这些加速依赖已经启用，但当前短双工交互 probe 的端到端耗时没有体现收益，甚至多数轮次略慢。

## 可能原因

当前 probe 的计时是端到端粒度，`generate_s` 里混合了多种成本：

- LLM decode
- TTS token generate
- token2wav / waveform 生成
- Python 调度和 session 状态维护
- cache 管理
- CUDA 同步
- 采样导致的 token 数差异

同时，当前 probe 是 batch size 1、短 chunk、短上下文的实时双工场景。FlashAttention2 和 MoE fast path 的收益可能被小序列、小 batch、频繁调度和 TTS 路径开销抵消。

如果后续要严肃评估 kernel 加速，需要拆分更细粒度的计时，例如：

- LLM prefill
- LLM decode
- TTS decode
- token2wav
- Python/session overhead

当前实验只能说明：对这条短 audio-duplex demo 路径，端到端没有看到稳定加速收益。
