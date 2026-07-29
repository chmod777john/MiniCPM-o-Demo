# O5 FC Semantic API 使用手册

本文面向需要调用已启动 O5 API 的使用者和 Agent，只解释请求、事件和推理产物。
不涉及模型部署、计算资源申请、模型来源、Ground Truth 对拍或能力评价。

协议事件的唯一详细定义见
[`semantic-realtime-api-v2.md`](semantic-realtime-api-v2.md)。本文是使用教程，不重复定义协议。

## 1. 最短调用路径

准备：

- 一条 `O5DuplexTrainingData` JSON，或 JSONL 中的一行；
- TrainingData 引用媒体的根目录；
- 已启动 API 的 HTTP(S) 根地址。

调用：

```bash
./infer_o5_api.sh <training-data.json> \
  --data-root <media-root> \
  --base-url http://<host>:8009 \
  --output-dir <output-dir>
```

JSONL 输入使用零起始行号：

```bash
./infer_o5_api.sh <training-data.jsonl> \
  --line-index 17 \
  --data-root <media-root> \
  --base-url http://<host>:8009 \
  --output-dir <output-dir>
```

只检查输入投影，不连接 API：

```bash
./infer_o5_api.sh <training-data.json> \
  --data-root <media-root> \
  --base-url http://<host>:8009 \
  --output-dir <output-dir> \
  --prepare-only
```

## 2. TrainingData 如何进入 API

样例把 TrainingData 当作一个输入场景。

发送到 API：

- `system` 中的 system prompt；
- `system.tools` 中的工具定义；
- `system` 中可选的请求级参考音频路径；
- 完整 `unit_policy`；
- 按 TrainingData timeline 合成的 16 kHz 单声道 float32 用户音频 Unit。

不会发送到 API：

- `tracks.ai_spoken`；
- `tracks.ai_non_spoken`；
- 原 TrainingData 中的预期文本、预期工具调用和 AI 音频；
- 任何对拍字段。

因此，输入中的 AI Ground Truth 不会作为模型输入。`request_preview.json` 会明确记录
`source_ai_tracks_sent_to_api=false`，可供 Agent 审计。

当前样例不重放源 TrainingData 的 `input_event`。工具定义会发送，模型实际产生的
tool-call 会记录到结果，但样例不会自动注入源数据里的工具结果。需要真实工具闭环时，
调用方应基于 `response.tool_call.done` 接入自己的工具执行器。

## 3. WebSocket 生命周期

样例连接：

```text
ws://<host>:<port>/v1/realtime?mode=audio
```

HTTPS 对应 `wss://`。

基本顺序：

```text
Client connects
  <- session.queue_done
  -> session.init
  <- session.created
  -> input.append          # Unit 0
  <- semantic events
  <- response.unit.committed
  -> input.append          # Unit 1
  <- ...
  -> session.close
  <- session.closed
```

`session.init` 由脚本根据 TrainingData 构造。每个 `input.append` 携带：

```json
{
  "type": "input.append",
  "input": {
    "input_id": "training_data_unit_000000",
    "audio_base64": "<16 kHz mono float32 PCM>",
    "sample_rate": 16000
  }
}
```

脚本逐 Unit 发送，并等待同一 Unit 的 `response.unit.committed` 后继续。

## 4. 主要返回事件

### Think

```text
response.think.begin
response.think.delta
response.think.end
```

`response.think.end.full_text` 是该段完整 think 文本。

### Tool call

```text
response.tool_call.begin
response.tool_call.delta
response.tool_call.done
```

`response.tool_call.done.call` 包含工具名和结构化 arguments。样例将完整事件原样保存，
不执行工具、不判断调用是否正确。

### Spoken

```text
response.spoken.delta
response.spoken.end
```

`response.spoken.delta.steps` 携带流式文本，`audio` 携带 base64 float32 PCM，
`sample_rate` 当前通常为 24 kHz。

样例把同一 spoken turn 的音频 chunk 聚合为 WAV，并同时生成：

- 原始采样率 WAV，用于播放和人工检查；
- 16 kHz WAV，用于 replay TrainingData 的 SDK 媒体绑定。

### Unit

```text
response.unit.started
response.non_spoken.end
response.unit.committed
```

`response.unit.committed` 表示该 Unit 的模型状态和输出事件已经提交。脚本不把 Unit
是否符合预期解释为评价结论。

### Warning / Error

```text
response.warning
error
session.closed
```

warning 和 error 会原样进入结构化结果。API 提前返回 `error` 或关闭 Session 时，脚本
退出非零，并保留已经收到的 `history.jsonl`。

## 5. 输出目录

```text
<output-dir>/
├── input_training_data.json
├── request_preview.json
├── history.jsonl
├── response_history.jsonl
├── inference_result.json
├── replay_training_data.json        # 构造成功时存在
└── media/
    ├── user_audio.wav
    ├── ai_spoken_turn_000.wav
    └── ai_spoken_turn_000_16k.wav
```

### `history.jsonl`

完整双向 WebSocket 历史，按 `sequence` 排序。每行包含：

```json
{
  "sequence": 0,
  "direction": "up",
  "event": {"type": "session.init"}
}
```

这是最接近 API 原始事实的审计产物。

### `inference_result.json`

面向程序和 Agent 的结构化摘要，包含：

- `session_created`；
- `committed_unit_indices`；
- `think_texts`；
- `tool_calls`；
- `spoken_turns` 及其音频路径；
- `warnings` / `errors`；
- replay TrainingData 构造状态。

该文件不包含 expected/actual、准确率、token exact 或能力评分。

## 6. replay TrainingData

脚本会尽力写出：

```text
replay_training_data.json
```

存在两种模式。

### `token_equivalent`

当 API history 可由 resume canonicalizer 完整恢复时，脚本使用
`O5DuplexParser.parse()` 构造 token-equivalent replay `O5DuplexTrainingData`，并执行
parser round-trip 完整性检查。

### `semantic_replay`

当当前模型未开放可恢复边界，或源 TrainingData 含未发送的 input event 时，脚本根据
实际 API semantic events、实际用户音频和实际 AI 音频构造逻辑 replay
`O5DuplexTrainingData`。

该结构：

- 保留实际 system、用户音频、think、tool-call、spoken 文本和 AI 音频；
- 使用实际 API `unit_index` 作为全局时间锚点；
- 不恢复原始作者的 timing constraint；
- 不恢复未发送的 source `input_event`；
- 不声称与任何 Ground Truth token exact；
- spoken alignment 只使用整段文本对应整段音频，不伪造词级对齐。

`inference_result.json.replay_training_data` 会明确给出 `mode`、`status` 和说明。原始
history 与音频始终是事实源，不能只保留 replay TrainingData。

## 7. 代码入口

调用样例：

```text
examples/fc_duplex/infer_from_training_data.py
```

可复用环境包装：

```text
infer_o5_api.sh
```

协议规范：

```text
docs/fc-duplex/semantic-realtime-api-v2.md
```

现有 Ground Truth evaluator 位于 `scripts/fc_api_eval/`，不属于本使用手册，也不是调用
本样例的前置依赖。
