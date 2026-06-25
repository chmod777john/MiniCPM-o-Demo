# Duplex Tool Calling Protocol Draft

本文描述 backend 与 runtime 之间用于双工 tool calling 的最小扩展草案，不替代现有
session 生命周期、WebSocket 有序流、close 完成语义。

本文只定义最小必要协议面：

- 单条 WebSocket 连接内，事件接收顺序就是流式拼接顺序。
- `think` 不带 id，不带 seq。
- `tool_call` 带 `tool_call_id`，不带 seq。
- 工具结果由 runtime 通过同一个 `tool_call_id` 回填。

## 1. 传输与排序

本扩展沿用现有 WebSocket 数据通道。backend 下行事件与 runtime 上行输入都在同一个
session 内按接收顺序生效。

协议不要求 `idx`、`event_seq` 或 chunk-level `seq`。如果将来引入多连接转发、消息队列
重放或断线恢复，可以在不改变本文语义的前提下补充诊断序号。

## 2. 上行输入

### 2.1 input.standalone

`input.standalone` 表示双工模式下用户或上游系统给模型看的独立文本输入。

```json
{
  "type": "input.standalone",
  "text": "https://arxiv.org/abs/..."
}
```

`text` 是完整文本。本文不区分来源，不增加 `source` 字段。

### 2.2 input.tool_result

`input.tool_result` 表示外部工具执行完成后的结果。backend 将其作为 tool response 注入
模型上下文，并用 `tool_call_id` 与之前的模型 tool call 配对。

```json
{
  "type": "input.tool_result",
  "tool_call_id": "tc_xxx",
  "content": "工具返回内容"
}
```

约束：

- `tool_call_id` MUST 等于 backend 已经下发过的某个 `response.tool_call.args.begin`
  中的 id。
- `content` 是完整工具结果文本。当前草案不定义流式 tool result。
- 成功与失败不在协议字段中区分；错误也作为 `content` 返回。

## 3. 下行输出

### 3.1 response.think

`think` 是模型生成的非口语思考文本流。它需要 begin/end 边界，但不需要 id 或 seq。

```json
{ "type": "response.think.begin" }
```

```json
{ "type": "response.think.delta", "delta": "我需要先判断..." }
```

```json
{ "type": "response.think.end" }
```

约束：

- 同一 session 内同一时刻最多一个 active think。
- `response.think.delta` 的文本按 WebSocket 接收顺序拼接。
- `response.think.end` 表示当前 think span 正常闭合。

### 3.2 response.tool_call.args

`response.tool_call.args.*` 表示模型正在生成一个工具调用参数流。`tool_call_id` 由 backend
分配并贴到所有相关事件上。

```json
{
  "type": "response.tool_call.args.begin",
  "tool_call_id": "tc_xxx"
}
```

```json
{
  "type": "response.tool_call.args.delta",
  "tool_call_id": "tc_xxx",
  "delta": "<name>write_file</name>"
}
```

```json
{
  "type": "response.tool_call.args.end",
  "tool_call_id": "tc_xxx"
}
```

约束：

- `tool_call_id` MUST 由 backend 分配。
- runtime MUST 按接收顺序拼接同一个 `tool_call_id` 的 `delta`。
- `response.tool_call.args.end` 表示参数流闭合，可以解析并执行。
- 当前草案不定义 chunk-level `seq`。

`tool_call_id` 是 backend-worker wire 层对象 id。模型 token 流内部是否显式包含 id，
不由本文规定。

### 3.3 response.tool_call.abort

模型可能在参数流完成前放弃一个工具调用。backend 用 `response.tool_call.abort` 通知
runtime。

```json
{
  "type": "response.tool_call.abort",
  "tool_call_id": "tc_xxx"
}
```

约束：

- 如果 abort 发生在 `args.end` 之前，runtime MUST NOT 执行该 tool call。
- 如果 runtime 已经在 `args.end` 后开始执行，abort 表示取消请求；具体工具能否取消由
  runtime/tool 实现决定。
- 被 abort 的 tool call 不要求 runtime 回传 `input.tool_result`。

## 4. 最小生命周期

一次普通工具调用的下行与上行顺序如下：

```text
backend -> runtime: response.think.begin
backend -> runtime: response.think.delta*
backend -> runtime: response.think.end

backend -> runtime: response.tool_call.args.begin { tool_call_id }
backend -> runtime: response.tool_call.args.delta*
backend -> runtime: response.tool_call.args.end { tool_call_id }

runtime -> backend: input.tool_result { tool_call_id, content }
```

一次被放弃的工具调用：

```text
backend -> runtime: response.tool_call.args.begin { tool_call_id }
backend -> runtime: response.tool_call.args.delta*
backend -> runtime: response.tool_call.abort { tool_call_id }
```

独立文本输入与工具结果可以独立进入同一 session：

```text
runtime -> backend: input.standalone
runtime -> backend: input.tool_result
```

backend 负责把这些上行事件按模型内部 unit/text input slot 策略注入上下文。本文不规定
具体 unit 分配算法。

## 5. 工具参数解析

runtime 收到完整工具参数流后，按 WebSocket 接收顺序拼接同一个 `tool_call_id` 的
`response.tool_call.args.delta`：

```text
tool_call_content = concat(delta*)
```

如果参数格式是 SDK 默认的 `minicpm4_xml`，拼接结果可以交给 SDK tool serializer 反解析：

```python
from minicpm_o5_sdk.tool_serializer import (
    DEFAULT_TOOL_SERIALIZER_NAME,
    get_o5_tool_serializer,
    parse_and_validate_tools,
)

tools = parse_and_validate_tools(raw_openai_tools)
definition = tools[0]

serializer = get_o5_tool_serializer(DEFAULT_TOOL_SERIALIZER_NAME)
call = serializer.deserialize_tool_call(tool_call_content, definition)

name = call.function.name
arguments = call.function.arguments
```

解析结果是 OpenAI 风格 tool call 对象，而不是裸字符串：

```json
{
  "id": "",
  "function": {
    "name": "write_file",
    "arguments": {
      "path": "a.txt",
      "content": "hello"
    }
  }
}
```

约束：

- `deserialize_tool_call(content, definition)` MUST 使用对应 tool definition。
- 默认 XML 格式依赖 `definition.function.parameters` 做 schema-guided cast，例如把 `"3"`
  还原为 integer。
- serializer 返回的 `id` 是占位空串；工具执行与结果回填 MUST 使用 wire 层
  `tool_call_id`。
- runtime 可以只消费 `call.function.name` 与 `call.function.arguments`。

## 6. 与现有事件的关系

现有下行事件继续保留：

```text
response.output.delta kind=listen
response.output.delta kind=text
response.output.delta kind=audio
response.done
session.closed
```

本扩展只新增 think 与 tool-call 相关事件，以及上行的 `input.standalone` /
`input.tool_result`。字段结构保持
不变。
