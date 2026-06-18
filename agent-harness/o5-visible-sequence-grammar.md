# O5 可见序列文法草案

本文描述当前 SDK 主链路的 tokenized visible sequence。范围是去掉真实音频、视频、图像 embedding 数值之后，仍保留 runtime/builder 插入的结构 token、slot 边界、文本 payload、spoken decode、non-spoken decode 和 tool response 注入。

本文描述的是 serialized token sequence，不等同于“所有 token 都由模型自由采样”。slot 边界、输入 prefill、tool response 注入、budget 截断等 token 由 runtime/builder 插入；模型主要学习/采样输出 slot 内的决策和内容。

当前主链路已经不使用旧的 `<non_speak>` / `</non_speak>` 作为 non-spoken wrapper；对应位置改为五轨 slot 模板中的 `<ai_non_spoken_slot>` / `</ai_non_spoken_slot>`。

本文把结构合法性拆成两层：

```text
Layer 1: Visible Unit Template
  描述完整单序列里的 unit / input slot / spoken slot / non-spoken slot 骨架。

Layer 2: Non-Spoken Projection Language
  定义 π_non_spoken，把所有 unit 的 ai_non_spoken slot 内容抽出来拼接，
  再描述这个纯 non-spoken 流必须满足的生命周期规则。
```

完整合法性是这两层约束与语义约束的合取，而不是一个手写的巨大 CFG。

## Special Tokens

当前主链路 `TOKEN_SPECS` 中的 duplex special tokens：

```text
<unit>
</unit>

<user_video_slot>
</user_video_slot>
<image>
</image>
<slice>
</slice>

<user_audio_slot>
</user_audio_slot>

<text_input_slot>
</text_input_slot>
<duplex_tool_response>
</duplex_tool_response>
<tool_call_id>
</tool_call_id>
<tool_response>
</tool_response>
<|tool_started|>
<|tool_response_budget_reached|>
<|tool_response_streaming|>

<ai_spoken_slot>
</ai_spoken_slot>
<|listen|>
<|speak|>
<|spoken_slot_eos|>
<|spoken_turn_eos|>

<ai_non_spoken_slot>
</ai_non_spoken_slot>
<|no_action|>
<think>
</think>
<tool_call>
</tool_call>
<|non_spoken_eos|>
<|non_spoken_budget_reached|>
<|think_hold|>
<|think_abort|>
<|tool_call_hold|>
<|tool_call_abort|>
```

注：上表使用 O5 target 的 display name。O46 target 中 `SPOKEN_SLOT_EOS` 显示为 `<|chunk_eos|>`，`SPOKEN_TURN_EOS` 显示为 `<|turn_eos|>`。

`<|think_hold|>`、`<|think_abort|>`、`<|tool_call_hold|>`、`<|tool_call_abort|>` 已在 special token registry 中存在，但当前稳定 TrainingData content union 尚未闭环表达它们。本文把它们列为 pending-control 扩展。

`<|tool_started|>`、`<|tool_response_budget_reached|>`、`<|tool_response_streaming|>` 已在 registry 中存在，用于 duplex tool response 编排扩展；当前 tokenization builder 已实现 result frame 注入，created/streaming/budget frame 仍按 tool 协议后续闭环。

## Terminals

固定 special token 终结符：

```text
UNIT_START                       = <unit>
UNIT_END                         = </unit>

USER_VIDEO_SLOT_START            = <user_video_slot>
USER_VIDEO_SLOT_END              = </user_video_slot>
IMAGE_START                      = <image>
IMAGE_END                        = </image>
SLICE_START                      = <slice>
SLICE_END                        = </slice>

USER_AUDIO_SLOT_START            = <user_audio_slot>
USER_AUDIO_SLOT_END              = </user_audio_slot>

TEXT_INPUT_SLOT_START            = <text_input_slot>
TEXT_INPUT_SLOT_END              = </text_input_slot>
DUPLEX_TOOL_RESPONSE_START       = <duplex_tool_response>
DUPLEX_TOOL_RESPONSE_END         = </duplex_tool_response>
TOOL_CALL_ID_START               = <tool_call_id>
TOOL_CALL_ID_END                 = </tool_call_id>
TOOL_RESPONSE_START              = <tool_response>
TOOL_RESPONSE_END                = </tool_response>
TOOL_STARTED                     = <|tool_started|>
TOOL_RESPONSE_BUDGET_REACHED     = <|tool_response_budget_reached|>
TOOL_RESPONSE_STREAMING          = <|tool_response_streaming|>

AI_SPOKEN_SLOT_START             = <ai_spoken_slot>
AI_SPOKEN_SLOT_END               = </ai_spoken_slot>
LISTEN                           = <|listen|>
SPEAK                            = <|speak|>
SPOKEN_SLOT_EOS                  = <|spoken_slot_eos|>
SPOKEN_TURN_EOS                  = <|spoken_turn_eos|>

AI_NON_SPOKEN_SLOT_START         = <ai_non_spoken_slot>
AI_NON_SPOKEN_SLOT_END           = </ai_non_spoken_slot>
NO_ACTION                        = <|no_action|>
THINK_START                      = <think>
THINK_END                        = </think>
TOOL_CALL_START                  = <tool_call>
TOOL_CALL_END                    = </tool_call>
NON_SPOKEN_EOS                   = <|non_spoken_eos|>
NON_SPOKEN_BUDGET_REACHED        = <|non_spoken_budget_reached|>

THINK_HOLD                       = <|think_hold|>
THINK_ABORT                      = <|think_abort|>
TOOL_CALL_HOLD                   = <|tool_call_hold|>
TOOL_CALL_ABORT                  = <|tool_call_abort|>
```

抽象 payload 终结符类别：

```text
VISION_EMBED_TOKEN               = user video/image embedding placeholder
USER_AUDIO_EMBED_TOKEN           = user audio embedding placeholder
SPOKEN_TEXT_TOKEN                = ordinary token in spoken text payload
NON_SPOKEN_TEXT_TOKEN            = ordinary token in non-spoken text payload
THINK_TEXT_TOKEN                 = ordinary token in think payload
TOOL_CALL_PAYLOAD_TOKEN          = ordinary token in tool_call payload
TOOL_CALL_ID_TEXT_TOKEN          = ordinary token in tool_call_id text
TOOL_RESPONSE_TEXT_TOKEN         = ordinary token in tool_response content
```

payload token 必须 ordinary encode，不得被解释成 runtime special token。也就是说，payload 文本中出现形似 `<tool_call>`、`</think>`、`<unit>` 的字面量时，不能静默变成对应 special token id。

## Layer 1: Visible Unit Template

这一层只描述完整单序列的外层骨架，不在这里判断跨 unit 的 think/tool_call 生命周期。

### Nonterminals

```text
SESSION
PROMPT_PREFIX
UNIT
INPUT_SLOTS

USER_VIDEO_SLOT
USER_AUDIO_SLOT
TEXT_INPUT_SLOT
TOOL_RESPONSE_RESULT_FRAME

AI_SPOKEN_SLOT
SPOKEN_DECODE
SPOKEN_PAYLOAD
SPOKEN_TERMINATOR

AI_NON_SPOKEN_SLOT
AI_NON_SPOKEN_UNIT_BODY
AI_NON_SPOKEN_BODY_TOKEN
```

起始符号：

```text
SESSION
```

### Unit Rules

完整 session：

```text
SESSION :=
  PROMPT_PREFIX?
  UNIT+
```

`PROMPT_PREFIX` 是 system / tool definitions / reference audio 等全局前缀，当前文件不展开其内部结构。

unit：

```text
UNIT :=
  UNIT_START
    INPUT_SLOTS
    AI_SPOKEN_SLOT
    AI_NON_SPOKEN_SLOT
  UNIT_END

INPUT_SLOTS :=
    USER_VIDEO_SLOT
  | USER_AUDIO_SLOT
  | TEXT_INPUT_SLOT
  | USER_VIDEO_SLOT USER_AUDIO_SLOT
  | USER_VIDEO_SLOT TEXT_INPUT_SLOT
  | USER_AUDIO_SLOT TEXT_INPUT_SLOT
  | USER_VIDEO_SLOT USER_AUDIO_SLOT TEXT_INPUT_SLOT
```

`user_video`、`user_audio`、`text_input` 是输入 track。单个 input slot 在当前 unit 没有对应输入切片时可以省略，但每个合法 unit 至少必须包含一个输入 slot。

输入 slot：

```text
USER_VIDEO_SLOT :=
  USER_VIDEO_SLOT_START
    IMAGE_START VISION_EMBED_TOKEN+ IMAGE_END
    (SLICE_START VISION_EMBED_TOKEN+ SLICE_END)*
  USER_VIDEO_SLOT_END

USER_AUDIO_SLOT :=
  USER_AUDIO_SLOT_START
    USER_AUDIO_EMBED_TOKEN+
  USER_AUDIO_SLOT_END
```

text input slot：

```text
TEXT_INPUT_SLOT :=
  TEXT_INPUT_SLOT_START
    TOOL_RESPONSE_RESULT_FRAME+
  TEXT_INPUT_SLOT_END

TOOL_RESPONSE_RESULT_FRAME :=
  DUPLEX_TOOL_RESPONSE_START
    TOOL_CALL_ID_START TOOL_CALL_ID_TEXT_TOKEN+ TOOL_CALL_ID_END
    TOOL_RESPONSE_START TOOL_RESPONSE_TEXT_TOKEN* TOOL_RESPONSE_END
  DUPLEX_TOOL_RESPONSE_END
```

当前稳定 TrainingData 中，`tool_response` 属于 `text_input` 轨，是外部 observation / runtime 注入文本，`loss=0`。它不属于 `ai_non_spoken` 输出，也不是模型要学习生成的动作。

spoken slot：

```text
AI_SPOKEN_SLOT :=
  AI_SPOKEN_SLOT_START
    SPOKEN_DECODE
  AI_SPOKEN_SLOT_END

SPOKEN_DECODE :=
    LISTEN
  | SPEAK SPOKEN_PAYLOAD SPOKEN_TERMINATOR

SPOKEN_PAYLOAD :=
  SPOKEN_TEXT_TOKEN*

SPOKEN_TERMINATOR :=
    SPOKEN_SLOT_EOS
  | SPOKEN_TURN_EOS
```

`AI_SPOKEN_SLOT_START` / `AI_SPOKEN_SLOT_END` 是 runtime skeleton，`loss=0`。`LISTEN`、`SPEAK`、spoken payload、spoken terminator 是模型输出责任，默认 `loss=1`。

non-spoken slot：

```text
AI_NON_SPOKEN_SLOT :=
  AI_NON_SPOKEN_SLOT_START
    AI_NON_SPOKEN_UNIT_BODY
  AI_NON_SPOKEN_SLOT_END

AI_NON_SPOKEN_UNIT_BODY :=
  AI_NON_SPOKEN_BODY_TOKEN+

AI_NON_SPOKEN_BODY_TOKEN :=
    NO_ACTION
  | NON_SPOKEN_TEXT_TOKEN
  | THINK_START
  | THINK_TEXT_TOKEN
  | THINK_END
  | TOOL_CALL_START
  | TOOL_CALL_PAYLOAD_TOKEN
  | TOOL_CALL_END
  | NON_SPOKEN_EOS
  | NON_SPOKEN_BUDGET_REACHED
  | THINK_HOLD
  | THINK_ABORT
  | TOOL_CALL_HOLD
  | TOOL_CALL_ABORT
```

这一层把 `AI_NON_SPOKEN_UNIT_BODY` 视作 non-spoken token 的 unit-local 切片，只保证它位于 `<ai_non_spoken_slot>...</ai_non_spoken_slot>` 中。`<think>` 是否闭合、是否允许再开 `<think>`、budget 后是否跨 unit 继续，由第二层投影语言约束。

## Layer 2: Non-Spoken Projection Language

这一层描述把所有 unit 的 non-spoken 内容拼接后的纯 non-spoken 流。

### Projection

定义投影函数：

```text
π_non_spoken(SESSION):
  for each UNIT in order:
    take tokens inside AI_NON_SPOKEN_SLOT
    drop AI_NON_SPOKEN_SLOT_START / AI_NON_SPOKEN_SLOT_END
    append the remaining body tokens to output stream
  drop all other tokens
```

例如，两个 unit 中的 long think：

```text
unit 1 body:
  <think> THINK_TEXT_TOKEN+ <|non_spoken_budget_reached|>

unit 2 body:
  THINK_TEXT_TOKEN* </think> <|non_spoken_eos|>
```

投影后是：

```text
<think>
  THINK_TEXT_TOKEN+
  <|non_spoken_budget_reached|>
  THINK_TEXT_TOKEN*
</think>
<|non_spoken_eos|>
```

### Projection Nonterminals

```text
NON_SPOKEN_STREAM
NON_SPOKEN_EVENT

TEXT_EVENT
TEXT_BODY

THINK_EVENT
THINK_BODY
THINK_BODY_TOKEN

TOOL_CALL_EVENT
TOOL_CALL_BODY
TOOL_CALL_BODY_TOKEN

BUDGET_BOUNDARY
PENDING_CONTROL
```

### Projection Rules

```text
NON_SPOKEN_STREAM :=
  NON_SPOKEN_EVENT+

NON_SPOKEN_EVENT :=
    NO_ACTION
  | TEXT_EVENT
  | THINK_EVENT
  | TOOL_CALL_EVENT
  | PENDING_CONTROL
```

plain non-spoken text：

```text
TEXT_EVENT :=
  TEXT_BODY NON_SPOKEN_EOS

TEXT_BODY :=
  NON_SPOKEN_TEXT_TOKEN+ (BUDGET_BOUNDARY NON_SPOKEN_TEXT_TOKEN+)*
```

think：

```text
THINK_EVENT :=
  THINK_START THINK_BODY THINK_END NON_SPOKEN_EOS

THINK_BODY :=
  THINK_BODY_TOKEN*

THINK_BODY_TOKEN :=
    THINK_TEXT_TOKEN
  | BUDGET_BOUNDARY
```

tool call：

```text
TOOL_CALL_EVENT :=
  TOOL_CALL_START TOOL_CALL_BODY TOOL_CALL_END NON_SPOKEN_EOS

TOOL_CALL_BODY :=
  TOOL_CALL_BODY_TOKEN*

TOOL_CALL_BODY_TOKEN :=
    TOOL_CALL_PAYLOAD_TOKEN
  | BUDGET_BOUNDARY
```

budget boundary：

```text
BUDGET_BOUNDARY :=
  NON_SPOKEN_BUDGET_REACHED
```

pending-control 扩展：

```text
PENDING_CONTROL :=
    THINK_HOLD
  | THINK_ABORT
  | TOOL_CALL_HOLD
  | TOOL_CALL_ABORT
```

`PENDING_CONTROL` 当前是扩展分支。token registry 已有这些 token，但稳定 TrainingData schema 尚未闭环表达它们。

### Projection Consequences

第二层直接排除了以下情况：

```text
<think> THINK_TEXT_TOKEN* <think>
```

原因是 `THINK_BODY_TOKEN` 不包含 `THINK_START`，所以在前一个 `THINK_EVENT` 闭合前不能开启新的 think。

同理，在 `TOOL_CALL_EVENT` 闭合前也不能开启新的 tool call 或 think。

`NON_SPOKEN_BUDGET_REACHED` 不是逻辑事件结束；它只是允许当前 event 跨 unit 继续的 boundary token。自然完成必须由：

```text
TEXT_EVENT:      ... <|non_spoken_eos|>
THINK_EVENT:     ... </think> <|non_spoken_eos|>
TOOL_CALL_EVENT: ... </tool_call> <|non_spoken_eos|>
```

给出。

## Boundary Consistency

Layer 2 的投影语言描述拼接后的逻辑流；还需要和 Layer 1 的 unit 边界保持一致：

```text
NON_SPOKEN_BUDGET_REACHED:
  must be the final token before AI_NON_SPOKEN_SLOT_END in that unit
  the current logical event remains open in π_non_spoken

NO_ACTION:
  must occupy the whole AI_NON_SPOKEN_UNIT_BODY

NON_SPOKEN_EOS:
  must be the final token before AI_NON_SPOKEN_SLOT_END in that unit

PENDING_CONTROL:
  must be the final token before AI_NON_SPOKEN_SLOT_END in that unit
```

这组约束把“纯投影流合法”和“unit 内切片位置合法”连接起来。

## Complete Validity

完整合法性和前缀合法性分开：

```text
complete_valid(sequence):
  sequence satisfies Layer 1 Visible Unit Template
  π_non_spoken(sequence) satisfies Layer 2 Non-Spoken Projection Language
  boundary consistency constraints pass
  semantic constraints pass

prefix_valid(prefix):
  prefix has not entered error state in Layer 1 parser
  projected prefix has not entered error state in Layer 2 parser
  prefix can still be extended to a complete_valid sequence

allowed_next(prefix):
  token t is allowed iff prefix + t remains prefix_valid
  and semantic/runtime constraints do not ban t
```

实现上可以把 Layer 1 parser state 与 Layer 2 projection parser state 做笛卡尔积。当前 stable 骨架没有任意深度嵌套，主要结构可用有限状态机维护；payload 内部格式和 id 引用属于后面的语义层。

## Sampling Responsibility

默认采样责任可按当前 `default_loss` 粗分：

```text
runtime / builder / input / observation, default loss=0:
  unit boundary
  all slot boundaries
  user media wrappers and embedding placeholders
  text_input tool response frames
  NON_SPOKEN_BUDGET_REACHED

model output responsibility, default loss=1:
  LISTEN
  SPEAK
  spoken payload
  SPOKEN_SLOT_EOS
  SPOKEN_TURN_EOS
  NO_ACTION
  non-spoken text payload
  THINK_START / THINK_END and think payload
  TOOL_CALL_START / TOOL_CALL_END and tool-call payload
  NON_SPOKEN_EOS
  hold / abort extension tokens
```

`loss=0` 不等价于“模型物理上没有 logits”。推理时仍需要状态机 / logit mask 保证 runtime skeleton 不被模型自由采样，同时在正确位置由 runtime 插入。

## Semantic Constraints

以下约束不由上述结构文法单独完成，需要 parser 状态、payload schema 或 runtime policy 检查：

```text
tool_call_id must be unique among non-aborted tool calls
text_input tool_response.tool_call_id must reference a previous ai_non_spoken tool_call_id
tool_call payload must be valid according to the selected tool serializer
tool_call.name must exist in the declared tool table
PENDING_CONTROL must target the current pending kind
NON_SPOKEN_BUDGET_REACHED may only be inserted when budget is exhausted
side-effecting tool calls require runtime execution policy approval
```

## Examples

下面的例子只展示 visible token 结构。`[0]` 表示默认 `loss=0`，`[1]` 表示默认 `loss=1`。embedding placeholder 和普通 payload 只写符号名，不展开真实 token id。

最小合法输入、空 spoken、空 non-spoken：

```text
[0] <unit>
[0]   <user_audio_slot>
[0]     USER_AUDIO_EMBED_TOKEN+
[0]   </user_audio_slot>
[0]   <ai_spoken_slot>
[1]     <|listen|>
[0]   </ai_spoken_slot>
[0]   <ai_non_spoken_slot>
[1]     <|no_action|>
[0]   </ai_non_spoken_slot>
[0] </unit>
```

有用户音频输入，但模型继续听、不做 non-spoken 动作：

```text
[0] <unit>
[0]   <user_audio_slot>
[0]     USER_AUDIO_EMBED_TOKEN+
[0]   </user_audio_slot>
[0]   <ai_spoken_slot>
[1]     <|listen|>
[0]   </ai_spoken_slot>
[0]   <ai_non_spoken_slot>
[1]     <|no_action|>
[0]   </ai_non_spoken_slot>
[0] </unit>
```

模型只说话，不做 non-spoken 动作：

```text
[0] <unit>
[0]   <user_audio_slot>
[0]     USER_AUDIO_EMBED_TOKEN+
[0]   </user_audio_slot>
[0]   <ai_spoken_slot>
[1]     <|speak|>
[1]     SPOKEN_TEXT_TOKEN*
[1]     <|spoken_turn_eos|>
[0]   </ai_spoken_slot>
[0]   <ai_non_spoken_slot>
[1]     <|no_action|>
[0]   </ai_non_spoken_slot>
[0] </unit>
```

模型只 think，不说话：

```text
[0] <unit>
[0]   <user_audio_slot>
[0]     USER_AUDIO_EMBED_TOKEN+
[0]   </user_audio_slot>
[0]   <ai_spoken_slot>
[1]     <|listen|>
[0]   </ai_spoken_slot>
[0]   <ai_non_spoken_slot>
[1]     <think>
[1]       THINK_TEXT_TOKEN*
[1]     </think>
[1]     <|non_spoken_eos|>
[0]   </ai_non_spoken_slot>
[0] </unit>
```

同一个 unit 里模型说话，同时生成 non-spoken text 草稿：

```text
[0] <unit>
[0]   <user_audio_slot>
[0]     USER_AUDIO_EMBED_TOKEN+
[0]   </user_audio_slot>
[0]   <ai_spoken_slot>
[1]     <|speak|>
[1]     SPOKEN_TEXT_TOKEN*
[1]     <|spoken_slot_eos|>
[0]   </ai_spoken_slot>
[0]   <ai_non_spoken_slot>
[1]     NON_SPOKEN_TEXT_TOKEN+
[1]     <|non_spoken_eos|>
[0]   </ai_non_spoken_slot>
[0] </unit>
```

模型发起 tool call：

```text
[0] <unit>
[0]   <user_audio_slot>
[0]     USER_AUDIO_EMBED_TOKEN+
[0]   </user_audio_slot>
[0]   <ai_spoken_slot>
[1]     <|listen|>
[0]   </ai_spoken_slot>
[0]   <ai_non_spoken_slot>
[1]     <tool_call>
[1]       TOOL_CALL_PAYLOAD_TOKEN+
[1]     </tool_call>
[1]     <|non_spoken_eos|>
[0]   </ai_non_spoken_slot>
[0] </unit>
```

runtime 注入 tool response，然后模型继续生成后续 tool call：

```text
[0] <unit>
[0]   <text_input_slot>
[0]     <duplex_tool_response>
[0]       <tool_call_id>
[0]         TOOL_CALL_ID_TEXT_TOKEN+
[0]       </tool_call_id>
[0]       <tool_response>
[0]         TOOL_RESPONSE_TEXT_TOKEN*
[0]       </tool_response>
[0]     </duplex_tool_response>
[0]   </text_input_slot>
[0]   <ai_spoken_slot>
[1]     <|listen|>
[0]   </ai_spoken_slot>
[0]   <ai_non_spoken_slot>
[1]     <tool_call>
[1]       TOOL_CALL_PAYLOAD_TOKEN+
[1]     </tool_call>
[1]     <|non_spoken_eos|>
[0]   </ai_non_spoken_slot>
[0] </unit>
```

长 think 被 unit budget 截断，并在下一个 unit 继续。第二个 unit 不重新打开 `<think>`：

```text
[0] <unit>
[0]   <user_audio_slot>
[0]     USER_AUDIO_EMBED_TOKEN+
[0]   </user_audio_slot>
[0]   <ai_spoken_slot>
[1]     <|listen|>
[0]   </ai_spoken_slot>
[0]   <ai_non_spoken_slot>
[1]     <think>
[1]       THINK_TEXT_TOKEN+
[0]     <|non_spoken_budget_reached|>
[0]   </ai_non_spoken_slot>
[0] </unit>

[0] <unit>
[0]   <user_audio_slot>
[0]     USER_AUDIO_EMBED_TOKEN+
[0]   </user_audio_slot>
[0]   <ai_spoken_slot>
[1]     <|listen|>
[0]   </ai_spoken_slot>
[0]   <ai_non_spoken_slot>
[1]       THINK_TEXT_TOKEN*
[1]     </think>
[1]     <|non_spoken_eos|>
[0]   </ai_non_spoken_slot>
[0] </unit>
```

该例子的 non-spoken 投影是：

```text
<think>
  THINK_TEXT_TOKEN+
  <|non_spoken_budget_reached|>
  THINK_TEXT_TOKEN*
</think>
<|non_spoken_eos|>
```

pending-control 扩展示例。当前 token registry 已有这些 token，但 TrainingData schema 尚未闭环表达：

```text
[0] <unit>
[0]   <user_audio_slot>
[0]     USER_AUDIO_EMBED_TOKEN+
[0]   </user_audio_slot>
[0]   <ai_spoken_slot>
[1]     <|listen|>
[0]   </ai_spoken_slot>
[0]   <ai_non_spoken_slot>
[1]     <|think_abort|>
[0]   </ai_non_spoken_slot>
[0] </unit>
```

不可扩展前缀示例：在未闭合 think 内再次打开 think。

```text
<ai_non_spoken_slot> <think> THINK_TEXT_TOKEN* <think>
```

这个前缀已经进入错误状态，不能通过追加后续 token 变成合法完整序列。原因是在投影语言里，`THINK_BODY_TOKEN` 不包含 `THINK_START`。
