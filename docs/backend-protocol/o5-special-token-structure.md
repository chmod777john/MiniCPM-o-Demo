# O5 Special Token Nesting Structure

本文记录当前 `/user/sunweiyue/lib/swy-dev/omni_agent_research/minicpm_o5_sdk`
中 active duplex token registry 的 special token 嵌套结构，作为 backend/runtime
协议对齐时的参考。

当前 active registry 规模：

- `O5SpecialTokenKey`: 47 个。
- active bundle `o5_duplex.special_token_ids`: 47 个。
- active bundle `o45_fc_duplex_v0_1_0.special_token_ids`: 47 个。
- 其中 HF `special=True` control token 41 个，`special=False` named content token 6 个。

## 1. 顶层结构

```text
SAMPLE :=
  PROMPT_PREFIX?
  UNIT*
```

`PROMPT_PREFIX` 是 system/context 前缀。`UNIT` 是 duplex 时分复用的一帧。

## 2. System Prefix

```text
PROMPT_PREFIX :=
  <|im_start|>
    system_text
    REFERENCE_AUDIO*
    TOOL_DEFINITIONS?
  <|im_end|>

REFERENCE_AUDIO :=
  <|audio_start|>
  <|audio_end|>
```

说明：

- `system_text` 是普通 payload，不是 special token。
- 当前 SDK 实现中，reference audio 只发 `<|audio_start|><|audio_end|>` 骨架；
  真实 audio embedding 接入留给 collate/后续阶段。
- 当前 SDK 实现中，tool definitions 由 `tool_serializer.render_tool_system_block()`
  渲染为普通 payload 文本；`<tools>...</tools>` 不属于这 47 个 registry special token。
- system prefix 全部作为 context，默认 loss 为 0。

## 3. Unit Frame

每个 unit 内 slot 顺序固定：

```text
UNIT :=
  <unit>
    USER_VIDEO_SLOT?
    USER_AUDIO_SLOT?
    INPUT_EVENT_SLOT?
    AI_SPOKEN_SLOT
    AI_NON_SPOKEN_SLOT
  </unit>
```

出现条件：

- `USER_VIDEO_SLOT`: sample-level optional。`user_video is None` 时整条样本都不出现；否则每个 unit 都出现。
- `USER_AUDIO_SLOT`: sample-level optional。`user_audio is None` 时整条样本都不出现；否则每个 unit 都出现。
- `INPUT_EVENT_SLOT`: unit-level optional。当前 unit 有 input event segment 落点时才出现。
- `AI_SPOKEN_SLOT`: 每个 unit 必出现。
- `AI_NON_SPOKEN_SLOT`: 每个 unit 必出现。

## 4. User Video Slot

```text
USER_VIDEO_SLOT :=
  <user_video_slot>
    IMAGE_BLOCK+
  </user_video_slot>

IMAGE_BLOCK :=
  <image>
    <|image_placeholder|> x 64
  </image>

HD_SLICE_BLOCK :=
  <slice>
    <|image_placeholder|> x N
  </slice>
```

说明：

- 当前 slot 实现每个 image block 固定 64 个 `<|image_placeholder|>`。
- `<slice>` / `</slice>` 在 registry 中存在，用于 HD slice 结构；是否生产取决于具体视频路径。
- placeholder token 是 embedding placeholder，不是文本 payload。

## 5. User Audio Slot

```text
USER_AUDIO_SLOT :=
  <user_audio_slot>
    <|audio_placeholder|> x round(unit_sec * 10)
  </user_audio_slot>
```

说明：

- user audio 内部没有 `<audio>...</audio>` 包裹。
- `<|audio_placeholder|>` 是等时 audio embedding placeholder。

## 6. Input Event Slot

```text
INPUT_EVENT_SLOT :=
  <input_event_slot>
    INPUT_EVENT_FRAME+
  </input_event_slot>

INPUT_EVENT_FRAME :=
    TOOL_CREATED_FRAME
  | TOOL_RESPONSE_FRAME
  | STANDALONE_EVENT_FRAME
```

### 6.1 Tool Created

```text
TOOL_CREATED_FRAME :=
  <tool_response_event>
    <tool_call_id>
      id_text
    </tool_call_id>
    <|tool_started|>
  </tool_response_event>
```

说明：

- 这是框架自动注入的握手帧。
- `id_text` 是普通 payload，默认 loss 为 0。

### 6.2 Tool Response

```text
TOOL_RESPONSE_FRAME :=
  <tool_response_event>
    <tool_call_id>
      id_text
    </tool_call_id>
    <tool_response>
      response_text
    </tool_response>
  </tool_response_event>
```

说明：

- `response_text` 是普通 payload，不经 tool serializer。
- 成功和失败不通过独立 special token 区分，错误也写进 `response_text`。

### 6.3 Standalone Event

```text
STANDALONE_EVENT_FRAME :=
  <standalone_event>
    standalone_text
  </standalone_event>
```

说明：

- standalone event 不带 `tool_call_id`。
- `standalone_text` 是普通 payload。

### 6.4 Reserved Input Event Tokens

```text
<|event_budget_reached|>
<|tool_response_streaming|>
```

说明：

- 当前 registry 中存在。
- 当前 v0.0.1 主路径不生产 streaming tool response，也不生产 input event budget 截断帧。

## 7. AI Spoken Slot

```text
AI_SPOKEN_SLOT :=
  <ai_spoken_slot>
    SPOKEN_DECODE
  </ai_spoken_slot>

SPOKEN_DECODE :=
    <|listen|>
  | <|tts_pad|>
  | <|speak|>
      spoken_text
    SPOKEN_TERMINATOR

SPOKEN_TERMINATOR :=
    <|spoken_slot_eos|>
  | <|spoken_turn_eos|>
```

说明：

- `<|listen|>` 表示本 unit 不发声。
- `<|tts_pad|>` 表示该 unit 在 spoken segment timeline 内，但没有新 spoken text token。
- `<|speak|>` 后接普通 spoken text payload，再用 spoken terminator 收束。
- `spoken_slot_eos` 是 unit 内 spoken slice 结束；`spoken_turn_eos` 是整个 spoken turn 结束。

## 8. AI Non-Spoken Slot

```text
AI_NON_SPOKEN_SLOT :=
  <ai_non_spoken_slot>
    NON_SPOKEN_DECODE
  </ai_non_spoken_slot>

NON_SPOKEN_DECODE :=
    <|no_action|>
  | THINK_DECODE
  | TOOL_CALL_DECODE
  | <|non_spoken_hold|>
  | <|non_spoken_abort|>

THINK_DECODE :=
  <think>
    think_text
  </think>
  NON_SPOKEN_TERMINATOR

TOOL_CALL_DECODE :=
  <tool_call>
    tool_call_payload
  </tool_call>
  NON_SPOKEN_TERMINATOR

NON_SPOKEN_TERMINATOR :=
    <|non_spoken_eos|>
  | <|non_spoken_budget_reached|>
```

说明：

- `<|no_action|>` 表示本 unit 没有 non-spoken action。
- `<think>...</think>` 内是普通 think payload。
- `<tool_call>...</tool_call>` 内是 tool serializer 产出的普通 payload。
- `non_spoken_eos` 表示本 unit 没有更多 non-spoken token。
- `non_spoken_budget_reached` 表示本 unit decode budget 用尽，后续 unit 继续。
- `non_spoken_hold` / `non_spoken_abort` 是模型侧 non-spoken 控制动作。

## 9. 47 个 Token 按层级归类

```text
system:
  <|im_start|>
  <|im_end|>
  <|audio_start|>
  <|audio_end|>

unit:
  <unit>
  </unit>

user_video:
  <user_video_slot>
  </user_video_slot>
  <image>
  <|image_placeholder|>
  </image>
  <slice>
  </slice>

user_audio:
  <user_audio_slot>
  <|audio_placeholder|>
  </user_audio_slot>

input_event:
  <input_event_slot>
  </input_event_slot>
  <tool_response>
  </tool_response>
  <tool_response_event>
  </tool_response_event>
  <standalone_event>
  </standalone_event>
  <tool_call_id>
  </tool_call_id>
  <|tool_started|>
  <|event_budget_reached|>
  <|tool_response_streaming|>

ai_spoken:
  <ai_spoken_slot>
  </ai_spoken_slot>
  <|listen|>
  <|speak|>
  <|spoken_slot_eos|>
  <|spoken_turn_eos|>
  <|tts_pad|>

ai_non_spoken:
  <ai_non_spoken_slot>
  </ai_non_spoken_slot>
  <|no_action|>
  <think>
  </think>
  <tool_call>
  </tool_call>
  <|non_spoken_eos|>
  <|non_spoken_budget_reached|>
  <|non_spoken_hold|>
  <|non_spoken_abort|>
```

## 10. 核心层级模型

```text
sample
  prompt_prefix
  unit
    slot
      frame / modality wrapper / decode span
        payload text or embedding placeholder
```

所有结构 token 由 SDK builder 显式注入 token id；payload text 走 ordinary encode，
不应让用户文本里的 `<tool_call>`、`<image>` 等字面量命中 control token。
