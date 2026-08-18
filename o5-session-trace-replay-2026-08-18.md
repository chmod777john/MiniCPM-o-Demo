# O5 Session Trace 与 Replay

实现 commit：`630c51947dc75ddc63688bd6eee4e7beea732e78`

API chunk debug 收敛 commit：`d750fdd`

Token2Wav replay 状态诊断 commit：`040deff`

## 目标

同一份端到端 session 可以由 Demo API 或 Canonical 录制，并从会话起点在以下运行时重放：

- Canonical 单卡
- Demo 单卡优化路径
- Demo TP2 全加速路径

Replay 可独立强制以下状态，同时仍执行被测模块的真实 forward：

- `llm`：保留实际 logits，但把本步选中的 LLM token 替换为 reference token。
- `tts-condition`：先计算实际 condition，再把送入 TTS 的 tensor 替换为 reference condition。
- `tts-token`：保留实际 TTS 概率分布，但把采样结果替换为 reference acoustic token。
- `vocoder`：恢复 reference 的静态 `rand_noise`；token2wav/flow/vocoder 仍真实执行。

Replay 从 session 起点重新执行 `prepare -> (prefill -> generate -> finalize)*`，由真实执行重建 LLM/TTS KV cache，不支持从任意中间 KV 快照起跑。

## 两种记录模式

### 普通 API debug

`O5_SESSION_TRACE_MODE=tokens` 面向在线 API 排查。它只发送三种独立 WebSocket 帧：

```json
{"type":"debug","kind":"llm.chunk","token_ids":[...],"is_listen":false,"end_of_turn":false}
{"type":"debug","kind":"tts.chunk","source_llm_token_ids":[...],"token_ids":[...]}
{"type":"debug","kind":"t2w.chunk","input_token_ids":[...],"committed_range":[...],"lookahead_range":[...],"output_sample_range":[...]}
```

事件另带 `session_id`、`input_id`、`unit_index`、`turn_id`，用于还原 unit 和 turn 关系。粒度规则是：

- 每次 `streaming_generate()` 产生一个 `llm.chunk`。
- 每次 TTS `generate_chunk()` 产生一个 `tts.chunk`。
- 每次实际调用 `audio_tokenizer.stream()` 产生一个 `t2w.chunk`；受 buffer 和 prelook 影响，一个 TTS chunk 可以对应零个、一个或多个 T2W chunk。

该模式不记录逐 token decode/sample，不读取 tensor 内容，不计算 shape、dtype、numel 或 hash，也不创建 `model_trace.jsonl`、`trace_manifest.json` 和 `trace_tensors/`。Gateway 忠实写入的 `stream.jsonl` 是事实来源。文字、音频、listen 等业务帧不混入 `trace` 字段。

### 完整 replay sidecar

`O5_SESSION_TRACE_MODE=replay` 面向数值对齐和 teacher forcing。Backend 会额外创建以下 sidecar：

```text
session_dir/
  meta.json
  stream.jsonl
  blob/
  trace_manifest.json
  model_trace.jsonl
  trace_tensors/
  replay_outputs.jsonl
  replay_summary.json
  audio/
```

- `stream.jsonl` 与 `blob/` 是不可变输入。API recorder 会保存原始 `.f32` 音频和 JPEG。
- `model_trace.jsonl` 保存事件索引和轻量字段。
- `trace_tensors/` 保存 condition、hidden、logits、概率分布和可选逐层 hidden。
- `capture-mode=replay` 才能重放 condition/vocoder、保存 tensor 并做数值比较。

每个事件都带 `session_id`、`input_id`、`unit_index`、`turn_id` 和单调 `event_id`。T2W 事件额外记录输入 token、已提交区间、lookahead 区间和输出采样区间。

## API 记录

正常 API debug：

```bash
export O5_TOKEN_TRACE_DIR=/path/to/data/sessions
export O5_SESSION_TRACE_MODE=tokens
```

此时 Gateway 的 session 目录只有常规的 `meta.json`、`stream.jsonl` 和媒体 `blob/`；debug 帧位于 `stream.jsonl`。

需要完整 replay sidecar 时改为：

启动 Demo 前设置：

```bash
export O5_TOKEN_TRACE_DIR=/path/to/data/sessions
export O5_SESSION_TRACE_MODE=replay
```

Backend 会把逐步事件和 tensor 直接写入同一 session 目录，同时把精简后的 chunk debug 作为独立帧发给 Gateway。完整 tensor 不进入 WebSocket。

API teacher-forcing replay 额外设置：

```bash
export O5_REPLAY_REFERENCE=/path/to/reference_session
export O5_REPLAY_FORCING=llm,tts-condition,tts-token,vocoder
```

如果目标是重现最终文字和音频，而不是隔离 TTS condition 模块，可以不强制 condition：

```bash
export O5_REPLAY_FORCING=llm,tts-token,vocoder
```

此时 condition 仍由本次 LLM hidden 正常计算；TTS acoustic token 由 reference 强制，因此下游 Token2Wav 不依赖保存的 condition tensor。`vocoder` 会恢复 reference 的静态 `rand_noise`，所以该模式仍要求 reference 使用完整 replay sidecar 录制。普通 `tokens` debug 不保存该 tensor，当前不能直接作为 forcing reference。

逐层记录是高开销选项，并且只允许用于 `replay` 模式：

```bash
export O5_CAPTURE_LAYERS=1
```

## 离线执行

从视频生成一个短输入 session：

```bash
python tools/o5replay/make_video_session.py \
  --video /path/to/input.mp4 \
  --ref-audio-path /path/to/ref.wav \
  --out-dir /path/to/input_session \
  --max-units 8
```

录制 Canonical reference：

```bash
python tools/o5replay/run_session.py \
  --session-dir /path/to/input_session \
  --out-dir /path/to/reference \
  --target canonical \
  --capture-mode replay \
  --max-units 8
```

Demo 单卡 replay：

```bash
python tools/o5replay/run_session.py \
  --session-dir /path/to/reference \
  --reference-session /path/to/reference \
  --out-dir /path/to/demo_single \
  --target demo-single \
  --forcing all \
  --capture-mode replay \
  --max-units 8
```

Demo TP2 replay：

```bash
torchrun --standalone --nproc_per_node=2 tools/o5replay/run_session.py \
  --session-dir /path/to/reference \
  --reference-session /path/to/reference \
  --out-dir /path/to/demo_tp2 \
  --target demo-tp2 \
  --forcing all \
  --capture-mode replay \
  --max-units 8
```

## 并行矩阵

`launch_matrix.py` 先等待 record 成功，然后立即提交 Canonical、Demo 单卡和 Demo TP2 三个独立 replay job。三个任务只读同一 reference bundle，各写各的输出目录，可以并行执行。

```bash
python tools/o5replay/launch_matrix.py \
  --session-dir /path/to/input_session \
  --out-root /path/to/matrix_run \
  --record-target canonical \
  --max-units 8
```

资源池默认先检查 `agent-dev`，其 A100 配额不足时检查 `agent-train`。脚本不停止或修改任何已有任务。

## 对比

```bash
python tools/o5replay/compare.py \
  /path/to/reference \
  /path/to/candidate \
  --out /path/to/comparison.json
```

报告包含：

- LLM/TTS token 相等数和首次翻转位置
- tensor bitwise 相等数
- 最大绝对误差、relative RMS、cosine
- LLM logits 与 TTS 概率分布的 `KL(left || right)` 和总变差距离
- 两侧事件数量及无法配对的事件数量

事件按 `(kind, input_id, 同类事件序号)` 对齐，因此同一 unit 内的多个 LLM feed、TTS chunk 和 T2W call 可以稳定匹配。

## 验证结果

### 离线 Record/Replay 矩阵

使用 8 个 unit 完成了 Canonical 录制，以及 Canonical、Demo 单卡和 Demo TP2 三种全状态 forcing replay：

| 任务 | cctl job | 状态 |
| --- | --- | --- |
| Canonical record | `741351` | Succeeded |
| Canonical replay | `741428` | Succeeded |
| Demo single replay | `741447` | Succeeded |
| Demo TP2 replay | `741465` | Succeeded |

Reference bundle：

```text
/user/weihongliang/o5_session_trace_replay_runs/smoke-record-canonical-20260818
```

Replay matrix：

```text
/user/weihongliang/o5_session_trace_replay_runs/smoke-replay-matrix-v2-20260818
```

三种 replay 均完成 `118/118` 个事件配对；LLM selected token `13/13` 相等，TTS sample `24/24` 相等，T2W 输入、range 和输出长度相等。Demo 单卡与 Demo TP2 的 24 次 TTS 概率全部 bitwise 相等。三种 replay 的最终 WAV SHA256 均为：

```text
de5470a18f007ca4d80f27692030a49bf03a77b0662945287ac106cfcaf90b04
```

LLM hidden 和 logits 不满足 bitwise 相等，但在 forcing 下 selected decision 保持一致。横向报告位于 matrix 根目录的 `comparison-*.json`。

### Demo API 录制

`741544` 在 `agent-dev` 使用单卡 `single_opt` 完成 8-unit 视频 API smoke。运行时启用了 batched-MM、LLM graph、TTS graph、TTS fast、fused vision/audio 和 batch vision feed，关闭 vocoder graph。LLM/TTS CUDA Graph 均成功捕获。

```text
run:     /user/weihongliang/o5_session_trace_replay_runs/api-smoke-8u-20260818-v1
session: /user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-session-trace-replay-2026-08-18/data/sessions/sess_1786f7745b77
```

API 完成 8 个 unit，输出文本为“好的，没问题。”。Session bundle 校验结果：

- `trace_manifest.json.completed=true`
- `stream.jsonl` 23 帧
- `model_trace.jsonl` 130 个事件
- `trace_tensors/` 254 个 tensor
- Gateway `meta.json` 已写入 `ended_at`，`close_reason=session_end`

该作业通过同一节点的 `https://127.0.0.1:8009` 调用，避免平台 HTTP exposure 与内部 HTTPS 协议不匹配。probe 会等待服务端 `session.closed`，确认模型 trace 和 Gateway recorder 都完成关闭后再结束服务。

上面的 `741544` 使用旧的 `replay` API 透传格式，仅证明完整 sidecar 录制链路可用，不作为普通 API debug 的性能或协议基准。普通 API debug 的新格式由 `scripts/run_o5_session_trace_api_smoke.sh` 验证：业务帧无 `trace` 字段，`stream.jsonl` 包含三种 chunk debug，且 session 目录不存在 replay sidecar。

### Chunk debug API 验证

最终验证任务：

```text
cctl job: 741787
run:      /user/weihongliang/o5_session_trace_replay_runs/api-debug-chunks-order-8u-20260818
session:  /user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-session-trace-replay-2026-08-18/data/sessions/sess_c7493a912818
```

任务在 `agent-train` 使用单卡 `single_opt`，完成 8-unit 视频 API smoke。验证结果：

- Gateway 共记录 35 帧，其中 12 帧为 `type=debug`。
- 8 个 `llm.chunk`，每个输入 unit 恰好一个。
- 2 个说话 unit 各有一个 `tts.chunk` 和一个 `t2w.chunk`。
- 同 unit 顺序为 `llm.chunk -> tts.chunk -> t2w.chunk`。
- 业务帧携带 `trace` 的数量为 0。
- debug 帧不存在 shape、dtype、numel、hash、tensor 或概率分布。
- session 根目录仅有 `meta.json`、`stream.jsonl` 和 `blob/`，不存在 replay sidecar。

CPU 回归测试使用 accel venv，结果为 `24 passed`。

### 不强制 TTS condition 的 API replay

Reference 使用上面的 API session `sess_1786f7745b77`。Replay 只强制：

```text
llm,tts-token,vocoder
```

首次验证任务 `741822` 完成 8 个 unit，输出文字与 API 都是“好的，没问题。”。130/130 个 trace 事件对齐，LLM token、TTS token、T2W 输入/range 和 vocoder `rand_noise` 全部一致。没有强制的两次 TTS condition 也自然 bitwise 一致，说明重现该 case 不需要预先录制并替换 condition。

最终 WAV 未达到 bitwise 一致。为排除 WAV 写盘量化影响，`040deff` 在详细 replay 模式记录了 `audio_tokenizer.stream()` 的直接 int16 PCM，以及调用前后的 prompt、flow 和 HiFT cache 指纹。普通 API `tokens` 模式不计算这些字段。

诊断任务：

| 任务 | cctl job | 节点 | 状态 |
| --- | --- | --- | --- |
| T2W state replay | `741871` | `10.156.16.208` | Succeeded |
| 同条件重复 replay | `741885` | `10.156.16.208` | Succeeded |

输出目录：

```text
/user/weihongliang/o5_session_trace_replay_runs/api-reference-no-condition-t2w-state-20260818
/user/weihongliang/o5_session_trace_replay_runs/api-reference-no-condition-t2w-state-repeat-20260818
```

API 原始音频与 `741871` 的 T2W 直接 PCM 对比：

| T2W call | 样本数 | max abs PCM | relative RMS | cosine/correlation |
| --- | ---: | ---: | ---: | ---: |
| unit 5 非静音输出 | 9600 | 44 | 2.922% | 0.999579 |
| unit 6 输出 | 24000 | 162 | 0.487% | 0.999988 |

两次同节点 replay 的边界更明确：

- T2W input token、调用前后 prompt/flow/HiFT cache 和 vocoder noise 全部 bitwise 一致。
- unit 5 的 9600 个 PCM 样本只有 3 个相差 `+/-1`。
- unit 6 的 24000 个 PCM 样本全部 bitwise 一致。
- 两次 replay 的第一段 PCM relative RMS 为 `0.0122%`，第二段为 0。

因此结论需要区分两种“一模一样”：

- 离散输出可精确重现：LLM token、TTS token、T2W token 序列和文字可以由 teacher forcing 保证一致，TTS condition 不必作为 forcing 输入。
- 音频语义和波形可以高度一致，但默认不保证文件 SHA/PCM bitwise 一致。即使 T2W 的可见输入、cache 和 noise 完全相同，同一节点的连续 GPU vocoder 计算仍观察到 3 个 PCM 量化位的 `+/-1` 差异。

若测试目标是端到端正确性，应比较 token 精确相等，再对 PCM 使用 relative RMS、相关系数和语音指标；若硬性要求音频字节完全相同，需要额外约束 Token2Wav/HiFT 的确定性 kernel，或直接复用录制的 API 音频，后者不再验证 vocoder 重算。
