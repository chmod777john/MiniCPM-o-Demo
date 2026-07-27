# FC Duplex Semantic Realtime API v2

## 1. 目标

本协议以单 WebSocket 有序状态机为基础，只公开业务和 stateless resume 真正需要的信息。
协议不暴露 token ID，不重复传输同一语义，不为当前不存在的并发能力预埋 ID。

### 1.1 单 Job 多 GPU 临时验证入口

在单个 Cybertron `agent-dev` Job 内，可用
`scripts/start_o45_fc_api_multi_gpu.sh` 为每张可见 GPU 启动独立的
Backend、Worker 和 Gateway：

```bash
MODEL_PATH=/path/to/model \
PT_PATH=/path/to/model.pt \
REF_AUDIO_PATH=/path/to/ref.wav \
LOG_DIR=/tmp/o45-fc-multi-api \
bash scripts/start_o45_fc_api_multi_gpu.sh
```

`NUM_GPUS` 默认取 `CUDA_VISIBLE_DEVICES` 的设备数，未设置时为 `1`。Gateway、
Backend、Worker 端口分别从 `8009`、`22500`、`23510` 递增，也可通过
`BASE_GATEWAY_PORT`、`BASE_BACKEND_PORT`、`BASE_WORKER_PORT` 覆盖。启动器只安装并
验证一次正式 `minicpm-o5-sdk==0.0.5` wheel；所有实例禁用 FRP，并写入
`${LOG_DIR}/gpu_<id>/`。全部 `/health` 就绪后，Job 日志会输出可供批量调用方解析的
`[MULTI_API_READY]` 行及各 Gateway `base_urls`。结束父进程或任一实例异常退出时，
启动器只清理本次启动的子进程。

## 2. 不变量

- 一个 WebSocket 只承载一个 Session。
- 同时最多一个 active spoken turn。
- 同时最多一个 active think。
- Tool-call 可以异步等待结果，因此只有 tool-call 需要 `tool_call_id`。
- Unit 是模型调度边界；think、tool-call 和 spoken turn 可以跨 Unit。
- WebSocket 接收顺序就是事件顺序，不额外定义 event/step/batch/delta 序号。

### 2.1 UnitPolicy 与 Checkpoint Profile

Semantic Realtime API v2 以 `minicpm-o5-sdk==0.0.5` 的 `O5UnitPolicy` 为 Unit 资源
策略唯一 schema。TrainingData 调用方直接传递 load 后的完整策略：

```python
session_init_payload = {
    "checkpoint_profile_id": "profile_id",
    "unit_policy": training_data.unit_policy.model_dump(mode="json"),
    "config": {
        "non_spoken_scheduling": "quality",
    },
}
```

Runtime 使用 SDK `O5UnitPolicy.model_validate()` 解析，不复制字段、默认值或 sentinel
schema。每个 Unit 按 `unit_index` 调用：

```text
listen             → get_non_spoken_budget_while_listening(unit_index)
speak / tts_pad    → get_non_spoken_budget_while_speaking(unit_index)
```

列表越界行为由 SDK 定义为 tail-repeat。状态判定优先读取 spoken slot 的协议 token；
`tts_pad` 即使 legacy `is_speaking=false`，仍属于 speaking。

旧调用方可继续在 `config` 传递
`non_spoken_budget_while_listening`、`non_spoken_budget_while_speaking` 或单值
`non_spoken_budget_per_unit`。Runtime 只把这些 scalar 转成 `O5UnitPolicy` 后再执行，
内部不保留第二套策略实现。

通用 Runtime 不允许 Session 静默覆盖 checkpoint 事实。若进程环境通过
`CHECKPOINT_PROFILE_ID`、`FC_DUPLEX_UNIT_POLICY_JSON`、
`FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING`、
`FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING` 或
`FC_DUPLEX_NON_SPOKEN_SCHEDULING` 绑定了 Profile，Session 对应字段必须一致。现有部署
优先使用 `FC_DUPLEX_UNIT_POLICY_JSON` 保存完整 SDK policy；若只有 legacy scalar env，
Runtime 会校验完整 UnitPolicy 的每一项是否与已知 scalar 一致。可变列表或
`O5NoBudgetLimit` 不能绕过任一 Profile 检查。

`non_spoken_scheduling` 仍属于部署 Profile，而不是 `O5UnitPolicy`。当前 checkpoint
若只验证了 quality，Session 不能切换到 latency。

### 2.2 无样本级 budget 与基础设施上限

`O5NoBudgetLimit` 表示当前 Unit 不插入协议
`<|non_spoken_budget_reached|>`，Runtime 持续 decode 直到模型自然产生 `eos` /
`no_action`。它不表示进程可以无限执行。
若 latency 调度在该 Unit 自然结束前收到下一输入，Runtime 同样 fail-fast；不能为了
抢占当前 Unit 而伪造一个 `budget_reached`。

独立基础设施保护配置如下：

```json
{
  "infrastructure": {
    "max_non_spoken_steps": 4096
  }
}
```

也可由 `FC_DUPLEX_INFRASTRUCTURE_MAX_NON_SPOKEN_STEPS` 绑定。达到该上限时 Runtime
抛出 `RuntimeError` 并关闭失败 Session，不插入 `budget_reached`，避免把基础设施故障
伪装成模型协议输出。默认上限由 Demo 的 Pydantic 配置模型维护，不属于 SDK
UnitPolicy 默认值。

### 2.3 评测固定 Tool Call ID

评测可在 `session.init.payload` 提供：

```json
{
  "evaluation": {
    "fixed_tool_call_ids": ["ground_truth_call_0", "ground_truth_call_1"]
  }
}
```

该列表只注入 `FcDuplexView` 的内部 `FixedToolCallIdGenerator`，用于和 TrainingData
固定 ID 对拍。列表必须非空、每项为非空字符串且互不重复；生成器耗尽立即失败。
普通 Session 不提供 `evaluation` 时继续使用默认内部生成器。公共 API 始终独立分配
`tc_*`，不会暴露评测内部 ID。Stateless Resume canonicalizer 从原始
`session.init` 读取同一列表，确保 live 与 replay 的 input_event 内部 ID 完全一致。

## 3. 必要关联字段

协议只保留：

```text
unit_index     输出所属的模型调度 Unit
input_id       实际被该 Unit 消费的输入
tool_call_id   异步 tool result 关联
```

删除以下公开字段：

```text
block_id
span_id
response_id
每个事件重复的 session_id
event_index
step_index
batch_index
delta_index
source_step_indices
source_steps
```

## 4. Text step

每个 step 是 discriminated union：

```json
{"kind": "pending"}
```

```json
{
  "kind": "text",
  "text": "动物"
}
```

`pending` 表示一个 ordinary token 尚未产生安全 Unicode。每个 `text` step 自动覆盖同一
有序 semantic stream 中从上次 `text` 后累计的全部 `pending` 加当前 step；客户端可直接
从 steps 顺序推导，不重复发送计数。Transport 可以把多个 step 放进同一事件，但不能合并
或拆分单个 safe text。

## 5. Unit 生命周期

Backend 在实际完成 prefill 后发送：

```json
{
  "type": "response.unit.started",
  "unit_index": 12,
  "input_id": "input_000015",
  "tool_events": [
    {
      "type": "tool_result",
      "tool_call_id": "tc_000002"
    }
  ]
}
```

每个实际处理 Unit 都必须发送，包括空 events：

```json
{
  "type": "response.unit.started",
  "unit_index": 13,
  "input_id": "input_000016",
  "tool_events": []
}
```

`tool_events` 只表达实际 Unit 归属，不重复 tool result 内容；内容从此前
`input.tool_result` 读取。

Unit 完成：

```json
{
  "type": "response.unit.committed",
  "unit_index": 12,
  "resume": {
    "status": "unavailable",
    "reason": "deferred_close"
  }
}
```

在 finalize 前，每个 Unit 必须先发送且只发送一次：

```json
{
  "type": "response.non_spoken.end",
  "unit_index": 12,
  "reason": "budget_reached"
}
```

当前支持的 `reason`：

```text
no_action
eos
budget_reached
```

`eos` / `no_action` 是模型生成 token，各占一个有限 non-spoken budget；
`budget_reached` 只在有限 UnitPolicy budget 用尽或有限 budget 的 latency 调度截断时由
framework 插入，不占 budget。`O5NoBudgetLimit` 不插入该 token。Hold / Abort 尚未完整
实现，不进入当前 Public API。

事件顺序固定为：

```text
semantic begin/delta/end
→ response.non_spoken.end
→ finalize Unit / 计算 Resume
→ response.unit.committed
```

`budget_reached` 按协议固定表示 close token、slot end 和 Unit end 在下一 Unit prefill 前
进入 KV，因此不再额外发送 `deferred_model_feed`。`response.unit.committed` 不重复携带
reason，只承诺 Unit 状态和 Resume 结果已经提交。

## 6. Think

同一时刻最多一个 active think，因此不需要 ID。

```json
{
  "type": "response.think.begin",
  "unit_index": 5
}
```

```json
{
  "type": "response.think.delta",
  "unit_index": 5,
  "steps": [
    {"kind": "text", "text": "用户给我"},
    {"kind": "pending"}
  ]
}
```

跨 Unit 继续时仍属于同一个 think：

```json
{
  "type": "response.think.delta",
  "unit_index": 6,
  "steps": [
    {"kind": "text", "text": "设了个规则"}
  ]
}
```

只有模型产生 `</think>` 时发送：

```json
{
  "type": "response.think.end",
  "unit_index": 8,
  "full_text": "用户给我设了个规则……"
}
```

Budget 不发送 `think.end`，也不重置 ordinary DecodeStream。View 与 Capability 都保留
同一个 Think 的 decoder、pending token 和 semantic aggregate，直到 matching end。

## 7. Tool-call

Tool-call 需要 ID，因为外部结果异步返回。

```json
{
  "type": "response.tool_call.begin",
  "tool_call_id": "tc_000002",
  "unit_index": 16
}
```

```json
{
  "type": "response.tool_call.delta",
  "tool_call_id": "tc_000002",
  "unit_index": 16,
  "steps": [
    {
      "kind": "text",
      "text": "<function name=\"display_object_on_board\">"
    }
  ]
}
```

Budget 不结束 semantic tool-call，也不更换 `tool_call_id`。只有模型产生
`</tool_call>` 时发送一个最终事件：

```json
{
  "type": "response.tool_call.done",
  "tool_call_id": "tc_000002",
  "unit_index": 17,
  "full_text": "<function ...>...</function>",
  "call": {
    "name": "display_object_on_board",
    "arguments": {
      "name": "老鼠"
    }
  }
}
```

解析失败：

```json
{
  "type": "response.tool_call.done",
  "tool_call_id": "tc_000002",
  "unit_index": 17,
  "full_text": "...",
  "error": "invalid tool-call XML"
}
```

不再拆成 `args.end` 和 `args.raw` 两条事件。

## 8. Tool result

```json
{
  "type": "input.tool_result",
  "tool_call_id": "tc_000002",
  "content": {
    "status": "displayed",
    "name": "老鼠"
  }
}
```

等待结果或等待下一 Unit 注入时 checkpoint 暂时不可恢复：

```json
{
  "status": "unavailable",
  "reason": "pending_tool_result"
}
```

`response.unit.started.tool_events` 确认结果被某 Unit 消费后，可以重新 available。

## 9. Spoken

同时最多一个 active spoken turn，因此不需要 turn ID。

```json
{
  "type": "response.spoken.delta",
  "unit_index": 5,
  "steps": [
    {"kind": "text", "text": "好的"}
  ],
  "audio": "...base64...",
  "sample_rate": 24000
}
```

每个 Unit 的 spoken slot 结束：

```json
{
  "type": "response.spoken.end",
  "unit_index": 5,
  "reason": "slot_eos"
}
```

如果模型没有生成 `spoken_slot_eos`、仅由模板关闭 slot，则使用
`reason: "slot_end"`；该 reason 不对应额外模型 token。

模型产生 `SPEAK` 但没有 text/audio 时仍必须发送
`response.spoken.delta {steps: []}`，用于无歧义表达 SPEAK；纯模板空 slot 则只发送
`reason: "slot_end"`。

整个 turn 结束：

```json
{
  "type": "response.spoken.end",
  "unit_index": 8,
  "reason": "turn_eos",
  "full_text": "好的，你继续说。"
}
```

Listen：

```json
{
  "type": "response.spoken.end",
  "unit_index": 9,
  "reason": "listen"
}
```

Active turn 未 `turn_eos` 就出现 listen，Backend 必须 RuntimeError 并关闭 Session。

## 10. Warning

只在无法无损表达时发送：

```json
{
  "type": "response.warning",
  "unit_index": 8,
  "code": "incomplete_bpe_at_stream_end",
  "message": "文本边界包含未完成 BPE，Resume 不保证可复现"
}
```

## 11. Resume

客户端保存从 `session.init` 到 available `response.unit.committed` 的完整有序历史。
服务端：

1. 校验 Unit started/committed 连续。
2. 使用 semantic begin/end/done 恢复 protocol token。
3. 按有序 steps 统计每个 text 前累计的 pending ordinary token，并用 text re-encode
   结果逐项恢复。
4. 按 Unit started 显式归属恢复 tool events。
5. 按 `response.non_spoken.end.reason` 恢复 lane terminator 与 deferred feed。
6. 恢复到 checkpoint 后继续下一 Unit。

协议不依赖服务端旧 Session/KV，也不暴露 token ID。

## 12. 前端显示

- 一次 think begin→end 对应一张卡片。
- 一次 tool-call begin→done 对应一张卡片。
- Streaming 只显示 text steps。
- 同一卡片内部按 `unit_index` 显示 segment 样式。
- Full 只显示 end/done 的 `full_text`。
- 不自行添加 `<think>` / `<tool_call>`。
- 必须校验：

```text
join(all text steps) == full_text
```

Budget 只增加 Unit segment boundary，不关闭卡片。
