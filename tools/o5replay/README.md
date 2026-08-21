# O5 Session Replay

本目录提供 O5 双工 session 的录制、离线重放、teacher-forcing 和结果对比工具。

这套工具需要区分两类目录：

- **输入 session**：保存原始音频、视频帧、unit 顺序和运行配置。
- **reference/replay bundle**：某个模型实际运行后的结果，包含 token trace；在完整 replay 模式下还包含 hidden、logits、TTS condition 等 tensor。

同一个输入 session 可以分别交给 Canonical、Demo 单卡和 Demo TP2 运行。各运行结果写入不同目录，因此可以并行比较。

## 工具职责

| 工具 | 作用 |
| --- | --- |
| `make_video_session.py` | 从本地视频和参考音频生成输入 session，不加载模型、不调用 API |
| `run_session.py` | 统一的离线运行入口，支持 Canonical、Demo 单卡和 Demo TP2 |
| `compare.py` | 对比两个 replay bundle 的 token、tensor、概率分布和音频元数据 |
| `launch_matrix.py` | 通过 `cctl` 提交 reference 和多个并行 replay 任务 |
| `replay_token2wav.py` | 只加载 Token2Wav，从 turn 起点重建流式状态并导出指定音频 chunk |
| `session_io.py` | 读取 session、恢复媒体 blob、写出 replay 音频 |
| `loaders.py` | 加载 Canonical、Demo 单卡或 Demo TP2 runtime |

旧的 Canonical 专用脚本位于：

```text
tools/o5trace/session_canonical_replay.py
```

它是早期的 Canonical 自由重放入口，不支持 token、TTS condition 或 vocoder 的 teacher-forcing。需要 forcing 或运行 Demo 时使用本目录的 `run_session.py`。

## 记录模式

### API 普通 token trace

启动 Demo API 前设置：

```bash
export O5_TOKEN_TRACE_DIR=/path/to/sessions
export O5_SESSION_TRACE_MODE=tokens
```

API session 中会保存：

- `meta.json`
- `stream.jsonl`
- `blob/` 下的原始音频和视频帧
- 轻量 debug frame：LLM chunk、TTS chunk、T2W chunk

该模式不保存 hidden、logits、概率分布或逐层 tensor，适合检查 token 序列、unit/turn 关系和 T2W 输入区间，也可以作为 `llm` 和 `tts-token` forcing 的参考。

### API 完整 replay trace

需要数值对齐、TTS condition forcing 或 vocoder state forcing 时使用：

```bash
export O5_TOKEN_TRACE_DIR=/path/to/sessions
export O5_SESSION_TRACE_MODE=replay
```

典型 session 目录还会包含：

```text
trace_manifest.json
model_trace.jsonl
trace_tensors/
```

完整 replay 模式会保存 LLM hidden/logits、TTS condition、TTS forward 信息、TTS 概率以及 T2W/vocoder 所需的重放状态。完整 tensor 不通过 WebSocket 发送，而是写入本地 sidecar。

如果需要 LLM 每一层的 hidden：

```bash
export O5_CAPTURE_LAYERS=1
```

逐层记录只允许在 `replay` 模式下使用，开销和存储量都明显更高。

## 离线输入 session

没有 API 录制时，可以直接从视频构造输入 session：

```bash
python tools/o5replay/make_video_session.py \
  --video /path/to/input.mp4 \
  --ref-audio-path /path/to/reference.wav \
  --out-dir /path/to/input_session \
  --max-units 8 \
  --overwrite
```

这一步只切分输入音频、提取视频帧并写入 `stream.jsonl` 和 `blob/`，不运行模型。因此它是“无 API 的输入准备”，不是 Demo 推理。

## 统一离线入口

`run_session.py` 支持三个目标：

```text
--target canonical
--target demo-single
--target demo-tp2
```

### Canonical 自然运行并生成 reference

```bash
python tools/o5replay/run_session.py \
  --session-dir /path/to/input_session \
  --out-dir /path/to/canonical_ref \
  --target canonical \
  --forcing none \
  --capture-mode replay \
  --max-units 8 \
  --overwrite
```

如果要保存每个 LLM decoder layer：

```bash
  --capture-layers
```

这一步是自然运行，不是 teacher-forcing。输出目录可以作为后续 Canonical 或 Demo replay 的 `--reference-session`。

### Canonical teacher-forcing replay

```bash
python tools/o5replay/run_session.py \
  --session-dir /path/to/input_session \
  --reference-session /path/to/canonical_ref \
  --out-dir /path/to/canonical_tf \
  --target canonical \
  --forcing all \
  --capture-mode replay \
  --max-units 8 \
  --overwrite
```

也可以只强制某一个阶段：

```text
--forcing llm
--forcing tts-condition
--forcing tts-token
--forcing vocoder
--forcing llm,tts-token
```

### Demo 单卡自由运行

```bash
python tools/o5replay/run_session.py \
  --session-dir /path/to/input_session \
  --out-dir /path/to/demo_single_free \
  --target demo-single \
  --forcing none \
  --capture-mode replay \
  --max-units 8 \
  --overwrite
```

这里直接在离线进程中加载 Demo runtime，不启动 API server，也不经过 Gateway。

### Demo TP2 自由运行

```bash
torchrun --standalone --nproc_per_node=2 \
  tools/o5replay/run_session.py \
  --session-dir /path/to/input_session \
  --out-dir /path/to/demo_tp2_free \
  --target demo-tp2 \
  --forcing none \
  --capture-mode replay \
  --max-units 8 \
  --overwrite
```

### Demo 对 Canonical reference 做 forcing

例如只隔离 Demo 的 TTS 和下游差异：

```bash
torchrun --standalone --nproc_per_node=2 \
  tools/o5replay/run_session.py \
  --session-dir /path/to/input_session \
  --reference-session /path/to/canonical_ref \
  --out-dir /path/to/demo_tp2_tf \
  --target demo-tp2 \
  --forcing all \
  --capture-mode replay \
  --max-units 8 \
  --overwrite
```

`--forcing none` 用于观察真实自由运行的分叉；`--forcing all` 用于在保持真实 forward 的同时固定离散决策和下游状态。

## Forcing 的含义

teacher-forcing 不是直接读取 reference 的最终输出，而是继续执行被测模型的 forward，只替换指定阶段的输入或选择结果：

| 阶段 | 被替换的内容 | 仍然执行的计算 |
| --- | --- | --- |
| `llm` | LLM 本步选出的 token | 当前 LLM hidden、logits 和 KV cache 更新 |
| `tts-condition` | 送给 TTS 的 condition tensor | 当前 LLM 到 TTS condition 的计算以及 TTS forward |
| `tts-token` | TTS 每一步选出的 acoustic token | 当前 TTS hidden、logits 和概率分布 |
| `vocoder` | vocoder 使用的静态随机噪声 | Token2Wav、flow、HiFT/vocoder forward |

所有 forcing 都要求 reference 中的 `input_id`、unit 顺序和对应事件能够匹配。Replay 从 session 起点重新执行，不能直接从任意中间 KV cache 快照启动。

## 单独重放 Token2Wav

Token2Wav 的 streaming 路径不是无状态函数。每次 `stream()` 都会更新 flow cache 和
HiFT cache，并且非末尾 chunk 包含 3 个 lookahead token。因此，要重建一个中间音频
chunk，必须使用同一参考音频，从该 turn 的第一个 T2W 调用依次重放到目标调用。

`replay_token2wav.py` 只加载约 1.2 GB 的 Token2Wav assets，不加载 LLM、视觉编码器或
TTS token generator。例如从 API session 重建 `0039.wav`：

```bash
python tools/o5replay/replay_token2wav.py \
  --session-dir /path/to/session \
  --target-audio 0039.wav \
  --output /path/to/replayed_0039.wav
```

也可以按文本定位：

```bash
python tools/o5replay/replay_token2wav.py \
  --session-dir /path/to/session \
  --target-text "Hello Kitty graphic" \
  --output /path/to/replayed_hello_kitty.wav
```

需要定位声学差异时，可以同时保存每个调用的 `chunk_mel`、HiFT 输入/输出、CUDA RNG
状态和 Flow 的 `rand_noise`：

```bash
python tools/o5replay/replay_token2wav.py \
  --session-dir /path/to/session \
  --target-audio 0039.wav \
  --output /tmp/replayed_0039.wav \
  --capture-dir /tmp/t2w-capture
```

之后可以用 `--flow-rand-noise-from` 固定 Flow 初始噪声，或用
`--stream-rng-from` 恢复每个 T2W 调用前的 CUDA RNG 状态，从而分别观察 Flow 和 HiFT
随机性的影响。捕获目录包含 `capture.json`、`arrays.npz`、`flow_rand_noise.npz` 和每次
调用的 RNG 状态文件。

要做 token 影响分析，可在固定两类随机状态后替换目标调用中的一个 token：

```bash
python tools/o5replay/replay_token2wav.py \
  --session-dir /path/to/session \
  --target-t2w-seq 67 \
  --output /tmp/replayed_token_07.wav \
  --capture-dir /tmp/t2w-token-07 \
  --flow-rand-noise-from /tmp/t2w-capture \
  --stream-rng-from /tmp/t2w-capture \
  --perturb-token-index 7 \
  --perturb-token-id 4218
```

`4218` 是静音 token。这个实验测量的是 token 的因果影响，不直接等价于判断 token
“正确”或“错误”；要判断错误，需要另一个参考 token 序列进行同样的 counterfactual 对比。

普通 `tokens` trace 没有保存 vocoder 的 `rand_noise` tensor，因此这种重放会恢复 token、
lookahead 和流式 cache 演化，但不保证与原始 WAV bitwise 一致。需要严格复现时，应使用
`capture_mode=replay` 记录并固定 vocoder state。

## Reference 的要求

不同 trace 模式能支持的 forcing 不同：

| Reference 来源 | 可用 forcing |
| --- | --- |
| `tokens` 模式 | `llm`、`tts-token` |
| `replay` 模式 | `llm`、`tts-condition`、`tts-token`、`vocoder` |

原因是 `tokens` 模式只保存离散 token，不保存 TTS condition 和 vocoder noise tensor。因此要做 condition 或 vocoder forcing，必须使用完整 `replay` reference。

## 并行实验矩阵

可以让工具先生成一份 reference，再并行提交多个目标：

```bash
python tools/o5replay/launch_matrix.py \
  --session-dir /path/to/input_session \
  --out-root /path/to/matrix_run \
  --record-target canonical \
  --max-units 8
```

默认目标是：

```text
canonical
demo-single
demo-tp2
```

各任务只读同一份 reference bundle，各自写入独立输出目录。资源池选择优先检查 `agent-dev`，不足时再尝试 `agent-train`；脚本不会停止或修改已有任务。

## 在线 API replay

如果目的是验证真实 API、Gateway 和服务端事件链路，可以使用：

```text
tools/o5trace/session_api_replay.py
```

它读取已有 session 的音频和视频，再发送给一个正在运行的 realtime API，重新收集文字和音频输出。它本身不直接加载模型，因此不适合单独做模块级 tensor 对齐。

API 服务端也可以通过环境变量启用 forcing：

```bash
export O5_REPLAY_REFERENCE=/path/to/reference_session
export O5_REPLAY_FORCING=llm,tts-token
```

不过需要保证在线请求的 unit/input 顺序与 reference 一致。模型模块级对齐通常优先使用离线 `run_session.py`，API replay 用于验证服务链路。

## 输出与对比

每次 `run_session.py` 会生成：

```text
replay_outputs.jsonl
replay_summary.json
audio/
output_audio.wav
model_trace.jsonl
trace_manifest.json
trace_tensors/
```

其中 tensor 和 trace 文件只在完整 replay 模式下出现。对比两个结果：

```bash
python tools/o5replay/compare.py \
  /path/to/reference \
  /path/to/candidate \
  --out /path/to/comparison.json
```

对比内容包括：

- LLM 和 TTS token 是否相等，以及首次分叉位置；
- hidden、logits、condition 的 bitwise、最大绝对误差、relative RMS、cosine；
- LLM logits 或 TTS 概率分布的 KL 和总变差距离；
- T2W 输入 token、提交区间、lookahead 区间；
- 事件数量、事件配对情况和最终文字/音频元数据。

## 推荐实验顺序

针对一个短视频，建议按以下顺序执行：

1. API 录制 `tokens` 或 `replay` session。
2. Canonical 自然运行，生成 canonical reference。
3. Canonical 自身 teacher-forcing replay，检查 Canonical 可复现性。
4. Demo 单卡和 Demo TP2 自由运行，观察真实分叉。
5. Demo 单卡和 Demo TP2 使用 Canonical reference 做 `llm`、`tts-condition`、`tts-token` 分阶段 forcing。
6. 使用 `compare.py` 对比 token、hidden、logits、TTS condition 和音频指标。
7. 只有需要验证 Gateway/API 行为时，才额外使用在线 `session_api_replay.py`。

这样可以先区分 Canonical 自身的重复运行差异，再判断 Demo 的 TP2、batched-MM、CUDA Graph 或 TTS 优化分别带来了什么影响。
