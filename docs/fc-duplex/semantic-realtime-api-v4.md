# MiniCPM-o 5 Realtime API v4

> ⚠️ **已被 v5 取代**（见 `semantic-realtime-api-v5.md`）：v5 新增 omni 模式（每秒一帧画面）。
> 按本文发 `protocol_version: "4"` 的客户端仍可用，等同于 v5 的 audio 模式。

本文档是对外调用 MiniCPM-o 5 全双工语音 API 的唯一说明。部署方会给你一个**端点地址**，
形如 `https://<host>:<port>/e/<端点名>`；下文用 `<端点>` 代指它。

- 一条 WebSocket 承载一个会话（Session）。
- 会话按 **Unit** 推进：你每发一段固定时长（通常 1 秒）的用户音频，模型就处理一个 Unit，
  并在这个 Unit 里决定说话、思考、调用工具或继续听。
- 同一端点背后可能有多个 Worker；忙不过来时你会先排队，排到后自动开始。

## 1. 会话前：查询端点信息

```http
GET <端点>/v1/info
```

无需鉴权，允许跨域。返回端点当前的模型与排队情况：

```json
{
  "endpoint": "mergeall",
  "protocol_version": "4",
  "realtime_path": "/e/mergeall/v1/realtime?mode=audio",
  "available_workers": 1,
  "queue_length": 0,
  "workers": [
    {"worker_id": "mergeall-1", "status": "idle", "draining": false, "profile_id": "mergeall-0930"}
  ],
  "models": [
    {
      "model": {
        "profile_id": "mergeall-0930",
        "sdk_version": "0.0.6",
        "pt_path": "…/model.pt",
        "session_defaults": {
          "tts_pad_protocol": "revised",
          "non_spoken_scheduling": "quality",
          "unit_policy": {"unit_sec": 1.0, "non_spoken_budgets_while_listening": [45],
                          "non_spoken_budgets_while_speaking": [25]}
        }
      },
      "workers": 1,
      "available": 1
    }
  ]
}
```

`model` 的其余字段说明加载的是哪个 checkpoint；`model.session_defaults` 是 **Session 不传时用的默认值**
（见 3.2）。一个端点下的所有 Worker 通常加载同一个模型；若不同，实际用的是哪个以
`session.created.model` 为准。

## 2. 连接与排队

```text
wss://<host>:<port>/e/<端点名>/v1/realtime?mode=audio
```

连上后服务端可能先发排队事件，再发 `session.queue_done` 表示已分到 Worker：

```json
{"type": "session.queued", "position": 2, "estimated_wait_s": 95.0, "ticket_id": "q_…", "queue_length": 2}
{"type": "session.queue_update", "position": 1, "estimated_wait_s": 40.0, "queue_length": 1}
{"type": "session.queue_done", "worker_id": "f9t4-1"}
```

`worker_id` 是这次会话分到的 Worker（只用于排查问题，不含地址）。排队期间断开连接即放弃排队。端点不存在时 WebSocket 以关闭码 `4404` 关闭。

## 3. 建立会话：`session.init`

收到 `session.queue_done` 后发送（也可以提前发，服务端会在分到 Worker 后处理）：

```json
{
  "type": "session.init",
  "payload": {
    "mode": "full_duplex",
    "fc_duplex": true,
    "protocol_version": "4",
    "system": {
      "segments": [
        {"kind": "text", "text": "你是一个语音助手。"},
        {"kind": "audio", "audio": {"source": "base64", "format": "wav", "data": "<WAV 文件字节的 base64>"}},
        {"kind": "text", "text": "保持简洁。"}
      ],
      "tools": [
        {"type": "function", "function": {
          "name": "web_search", "description": "联网搜索",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }}
      ]
    },
    "tts_prompt_audio": {"source": "base64", "format": "wav", "data": "<WAV 文件字节的 base64>"},
    "generate_audio": true,
    "config": {"decoding_policy": "constrained_decoding"}
  }
}
```

### 3.1 音频格式（v4 的核心变化）

所有放在 `session.init` 里的音频（system 里的音频段、`tts_prompt_audio`）都必须是：

```json
{"source": "base64", "format": "wav", "data": "<整个 WAV 文件字节的 base64>"}
```

- 必须是 **16 kHz、单声道** 的 WAV；服务端**不做重采样**，不符合直接报错。
- 单段解码后不超过 32 MB。
- **不接受服务端文件路径**（v3 的 `{"source": "path", "file_path": …}` 已废弃）。

`system.segments` 按给定顺序进入模型，文本段与音频段可以交错；`system.tools` 是工具定义
（OpenAI function 格式）。`tts_prompt_audio` 是模型说话所用的音色参考，与 system 里的音频相互独立；
`generate_audio: true` 时必填。

### 3.2 模型口径参数：可以传，不传用端点默认值

以下参数每个 Session 都可以自己传；不传就用端点公布的默认值（`/v1/info` 里的
`model.session_defaults`）。传了就以 Session 为准，不会因为与默认值不同而报错。

| 字段（`payload` 或 `payload.config`） | 取值 | 说明 |
|---|---|---|
| `tts_pad_protocol` | `revised` / `legacy` | 与 checkpoint 训练时的 SDK 对应：≥0.0.6a4 训的用 `revised`，更早的用 `legacy` |
| `unit_policy` | SDK `O5UnitPolicy` 对象 | 完整 Unit 策略：`unit_sec` 与逐 Unit 的听 / 说预算 |
| `non_spoken_budget_while_listening` / `non_spoken_budget_while_speaking` | 正整数 | 简写：常量预算。只传一个时，另一个取默认值 |
| `unit_sec` | 正数，默认 1.0 | 简写：每个 Unit 的音频时长（秒） |
| `non_spoken_scheduling` | `quality` / `latency` | 非语音部分的调度方式 |

只有 `checkpoint_profile_id` 不同：它是「我期望连到哪个模型」的断言，传了且与端点上的模型不一致时
会话直接失败，避免连错模型而不自知。

### 3.3 可选的会话参数（`payload.config`）

| 字段 | 取值 | 说明 |
|---|---|---|
| `decoding_policy` | `constrained_decoding`（推荐）/ `raise_protocol_error`（默认） | 模型输出违反协议时：约束解码纠正，或直接结束会话 |
| `max_spoken_tokens` | 正整数，默认 24 | 每个 Unit 最多生成的语音 token 数 |

### 3.4 `session.created`

```json
{
  "type": "session.created",
  "session_id": "sess_8c1f…",
  "mode": "full_duplex",
  "model": {"profile_id": "mergeall-0930", "pt_path": "…/model.pt", "session_defaults": {…}, "…": "…"},
  "session_config": {"tts_pad_protocol": "revised", "unit_policy": {…}, "non_spoken_scheduling": "quality",
                     "decoding_policy": "constrained_decoding", "checkpoint_profile_id": "mergeall-0930"},
  "resume": {"protocol_version": "4", "checkpoint_profile_id": "…", "tts_prompt_audio_sha256": "…", "…": "…"},
  "system_prefill": {"…": "…"}
}
```

`model` 是这次会话实际使用的模型；`session_config` 是本次会话**实际生效**的参数
（Session 传的值，缺的项取端点默认值）。system 内容较长（例如带很多工具）时，`session.created`
可能要十几秒后才到，这是在预填 system，属于正常现象。

## 4. 每个 Unit：发送用户音频

```json
{
  "type": "input.append",
  "input": {
    "input_id": "unit_000000",
    "audio_base64": "<16 kHz 单声道 float32 PCM 的 base64，无文件头>",
    "sample_rate": 16000
  }
}
```

- 每条消息一个 Unit，时长 = `unit_sec`（通常 1 秒 = 16000 个采样点）。
- 注意：这里是**裸 float32 PCM**，与 `session.init` 里的 WAV 文件不同。
- 实时对话：按真实时间每 `unit_sec` 发一条。离线评测：发一条、等到该 Unit 的
  `response.unit.committed` 再发下一条（锁步）。
- 用户没说话时也要发（静音），模型靠持续的 Unit 推进来决定何时开口。

## 5. 模型输出事件

每个 Unit 的事件顺序：

```text
response.unit.started
  → think / tool_call / spoken 事件（零个或多个）
  → response.non_spoken.end
  → response.unit.committed
```

| 事件 | 关键字段 | 含义 |
|---|---|---|
| `response.unit.started` | `unit_index`, `input_id`, `tool_events`, `standalone_events` | 该 Unit 开始；后两项回执本 Unit 消费了哪些工具结果 / 外部事件 |
| `response.think.begin` / `.delta` / `.end` | `steps`, `full_text`（end） | 模型的思考；可跨多个 Unit，只有 `end` 才算结束 |
| `response.tool_call.begin` / `.delta` / `.done` | `tool_call_id`, `call: {name, arguments}`（done） | 模型发起工具调用；解析失败时 `done` 带 `error` |
| `response.spoken.delta` | `steps`, `audio`（24 kHz float32 PCM base64）, `sample_rate: 24000` | 模型说的文字与语音 |
| `response.spoken.end` | `reason`: `slot_eos` / `slot_end` / `turn_eos`（带 `full_text`）/ `listen` | 本 Unit 说话结束；`turn_eos` 表示一整轮话说完 |
| `response.non_spoken.end` | `reason`: `no_action` / `eos` / `budget_reached` | 本 Unit 的非语音部分结束 |
| `response.unit.committed` | `unit_index` | 该 Unit 完成，可以发下一个 Unit |
| `response.warning` | `code`, `message` | 无法无损表达时的提示 |

`steps` 是有序列表，元素为 `{"kind": "text", "text": "…"}` 或 `{"kind": "pending"}`
（尚未凑成完整字符的 token）。把同一段的所有 `text` 依次拼接，等于结束事件里的 `full_text`。

## 6. 工具与外部事件

模型发出 `response.tool_call.done` 后，由**调用方**执行工具，再把结果送回：

```json
{"type": "input.tool_result", "tool_call_id": "tc_000002", "content": "result ok True; …"}
```

不关联任何工具调用的外部事件（例如后台任务的进度汇报）：

```json
{"type": "input.standalone_event", "content": "已找到两篇相关报道，正在核对数字", "event_id": "progress_1"}
```

两者都进入**下一个被处理的 Unit**，由该 Unit 的 `response.unit.started` 回执
（`tool_events` / `standalone_events`，后者按发送顺序回显 `event_id`）。

## 7. 结束会话

```json
{"type": "session.close"}
```

服务端回 `session.closed` 后关闭连接。服务端也可能主动结束会话：

| `session.closed.reason` | 含义 |
|---|---|
| `client_closed` | 你发了 `session.close` |
| `timeout` | 会话超过端点的最长时长 |
| `worker_removed` | 部署方强制下线了该 Worker |
| `model_protocol_error` | 模型输出违反协议（`decoding_policy=raise_protocol_error` 时）；带 `diagnostic` |
| `backend_error` | 服务端故障；带 `message` |

连接阶段的错误以事件形式返回，然后关闭连接：

```json
{"type": "error", "error": {"code": "queue_full", "message": "Queue full", "type": "server_error"}}
```

`code`：`queue_full`（排队已满）、`worker_busy`（没有可用 Worker）、`session_failed`
（分到的 Worker 无法连接或会话中断）。

## 8. 最小 Python 示例

```python
import asyncio, base64, json, ssl
import numpy as np, soundfile as sf, websockets

ENDPOINT = "wss://<host>:<port>/e/<端点名>/v1/realtime?mode=audio"

def wav_b64(path):                                 # session.init 里的音频：整个 WAV 文件
    return base64.b64encode(open(path, "rb").read()).decode()

async def main():
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE  # 自签证书
    async with websockets.connect(ENDPOINT, ssl=ctx, max_size=128 << 20) as ws:
        await ws.send(json.dumps({"type": "session.init", "payload": {
            "mode": "full_duplex", "fc_duplex": True, "protocol_version": "4",
            "system": {"segments": [{"kind": "text", "text": "你是一个语音助手。"}], "tools": []},
            "tts_prompt_audio": {"source": "base64", "format": "wav", "data": wav_b64("ref_16k_mono.wav")},
            "generate_audio": True, "config": {"decoding_policy": "constrained_decoding"},
        }}))
        while json.loads(await ws.recv())["type"] != "session.created":
            pass
        user, sr = sf.read("user_16k_mono.wav", dtype="float32")
        unit = sr                                       # unit_sec = 1.0
        for i in range(0, len(user), unit):
            chunk = np.zeros(unit, np.float32); seg = user[i:i + unit]; chunk[:len(seg)] = seg
            await ws.send(json.dumps({"type": "input.append", "input": {
                "input_id": f"unit_{i // unit:06d}", "sample_rate": 16000,
                "audio_base64": base64.b64encode(chunk.tobytes()).decode()}}))
            while True:                                 # 锁步：等本 Unit 提交
                event = json.loads(await ws.recv())
                if event["type"] == "response.spoken.end" and event.get("reason") == "turn_eos":
                    print("模型：", event["full_text"])
                if event["type"] == "response.unit.committed":
                    break
        await ws.send(json.dumps({"type": "session.close"}))

asyncio.run(main())
```

## 9. 与 v3 的差别

- `protocol_version` 改为 `"4"`。
- `session.init` 里的音频只收 base64 WAV，不再收服务端路径。
- `tts_pad_protocol`、budget、调度方式都是 Session 参数，不传时用端点默认值（`/v1/info` 的
  `model.session_defaults`）。
- `session.created` 新增 `model` 与 `session_config`；`resume.protocol_version` 反映实际协议版本。
- 新增端点地址 `/e/<端点名>/v1/realtime` 与 `/e/<端点名>/v1/info`。
- 事件语义（Unit、think、tool_call、spoken、tool_result、standalone_event）与 v2/v3 相同，
  细节见 `semantic-realtime-api-v2.md`。
