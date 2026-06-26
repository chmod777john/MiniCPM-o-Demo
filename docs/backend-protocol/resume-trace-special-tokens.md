# Resume Trace and Special Tokens

本文讨论 API 是否应该返回 O5 special token，以及 agent resume / KV-cache 重建时需要保存什么。

结论先行：

- 普通应用层 API 不应该把 special token 当作 assistant 内容返回。
- 需要 resume 时，API 应该提供一个可选、版本化的 `resume_trace`；这个 trace 可以包含 special token
  的 `token_id` 和调试用 `token_text`，但它不是展示层内容，也不应该要求客户端理解 O5 SDK 模板。
- 如果只返回 special token 字符串，仍然不足以稳定重建 KV cache；真正需要的是模型可见序列的规范记录：
  text/control token id、multimodal embedding span、tool/input event、模板和 tokenizer 版本。

## 1. 为什么普通 API 不直接返回 special token

普通 API 面向应用和 agent runtime，应该返回稳定的语义事件：

```text
text delta
audio delta
listen / speak state
think span
tool_call begin/delta/end/raw
tool_result accepted
done
```

special token 是模型内部 schema，不适合作为普通输出契约：

- token 名称和嵌套结构随 tokenizer / o5-sdk bundle / prompt template 变化。
- 客户端拿到 `<unit>`、`<|speak|>`、`<|non_spoken_eos|>` 后仍然无法安全判断业务语义，
  还要复刻 SDK parser。
- 对用户可见文本来说，special token 是泄漏的控制面噪声。
- 对 tool calling 来说，runtime 已经应该消费 backend 解析后的 `response.tool_call.args.raw`，
  不应该再自己解析 `<tool_call>...</tool_call>`。

所以普通 response payload 里应该过滤掉 special token，只保留 typed event。

## 2. resume 需要的不是 special token 字符串

resume 的目标是：agent 把过往会话过程发给 backend，backend 可以继续以前的会话。这里至少有两种语义：

```text
semantic resume:
  从历史消息和工具事件重新 canonicalize，再重新 prefill。
  适合普通 agent 续聊，不承诺 bit-exact。

exact resume:
  从之前导出的模型可见 trace 重放，尽量复原同一条 KV-cache 序列。
  适合调试、审计、断线恢复、跨 backend 迁移。
```

仅靠 special token 字符串不够做 exact resume，原因是：

- LLM prefill 路径实际喂的是 `inputs_embeds`，不是纯文本 token。图像、音频会把 placeholder token
  替换成 embedding。
- full-duplex 路径通过 decoder 增量 `feed()` token embedding、vision embedding、audio embedding；
  schema 里会出现 `("img", dim)`、`("audio", dim)` 这样的 embedding span，而不是普通文本。
- TTS 有独立的 `tts_past_key_values`、`tts_text_start_pos`、token2wav buffer/cache；
  如果要在“正在说话的中间”恢复，光有 LLM 文本 token 不能恢复音频生成状态。
- 流式音频输入还有 mel processor state、audio cache、chunk index、padding / extra-context 策略。
- 采样续跑如果要可复现，还要记录 generation 参数和 RNG 状态，或者只允许在稳定边界恢复。

因此，resume trace 应该记录“模型可见输入”，special token 只是其中一类 token record。

## 3. omni-dev 里的实现依据

在 `weihongliang/omni-dev` 中可以看到这些状态边界：

- half-duplex LLM cache 存在 `llm_past_key_values`。`_register_chunk()` 保存 chunk 的
  `input_ids` 和 decoded 文本；`_rebuild_cache_from_history()` 会拼接保存的 `input_ids`，
  再调用 LLM `use_cache=True` 重建 KV cache。
- 但 `non_streaming_prefill()` / `streaming_prefill()` 先通过 processor 构造
  `input_ids`、图像/音频特征，再调用 `get_vllm_embedding()`、`get_omni_embedding()`，
  最终把 `inputs_embeds` 喂给 LLM。只保存 token 字符串会丢掉 embedding 来源。
- full-duplex 的 cache 不直接读 `model.llm_past_key_values`，而是读
  `model.duplex.decoder.cache`。duplex prefill 会显式 feed `<unit>`、image/audio embedding
  和 generated token，并维护 `prefill_schema_tokens` / `total_ids`。
- full-duplex TTS 另有 `tts_past_key_values`、`tts_text_start_pos`、token2wav cache/buffer。
- speculative rollback 保存的是 cache 长度/checksum、audio cache clone、mel processor snapshot、
  RNG 状态、turn state，而不只是文本。

这些实现说明：KV-cache 重建可以从 token/embedding trace 重放，但不能只从展示文本重建。

## 4. 建议 API 分层

### 4.1 默认 response

默认 API 不返回 special token：

```json
{
  "type": "response.output.delta",
  "kind": "text",
  "delta": "你好"
}
```

tool call 继续用 typed event 和 parsed raw：

```json
{
  "type": "response.tool_call.args.raw",
  "tool_call_id": "tc_123",
  "raw": {
    "type": "function_call",
    "name": "read_file",
    "arguments": "{\"path\":\"README.md\"}"
  }
}
```

这里的 `raw` 内不重复 `tool_call_id`。

### 4.2 可选 resume trace

当客户端请求可恢复会话时，backend 可以额外返回 trace：

```json
{
  "type": "response.resume_trace.delta",
  "trace_version": "o5-resume-trace.v1",
  "tokenizer_bundle": "o5_duplex",
  "items": [
    {
      "kind": "token",
      "track": "llm",
      "token_id": 248094,
      "token_text": "<unit>",
      "special": true,
      "source": "backend_schema"
    },
    {
      "kind": "embedding_span",
      "track": "llm",
      "modality": "audio",
      "placeholder_token_id": 248096,
      "placeholder_count": 10,
      "media_ref": "input_audio_0007",
      "feature_spec": {
        "sample_rate": 16000,
        "chunk_idx": 3,
        "online_streaming": true
      }
    },
    {
      "kind": "tool_call",
      "tool_call_id": "tc_123",
      "raw": {
        "type": "function_call",
        "name": "read_file",
        "arguments": "{\"path\":\"README.md\"}"
      }
    }
  ]
}
```

约束：

- `token_id` 是重放主键；`token_text` 只用于日志和人读，不作为解析主键。
- 上例 token id 来自 `o5_duplex` bundle；不同 bundle 的 id 不同，恢复时必须以 trace metadata
  里的 bundle / tokenizer fingerprint 为准。
- `trace_version`、`model_id`、`tokenizer_fingerprint`、`template_version` 必须可校验。
- trace item 顺序就是模型可见顺序。
- multimodal span 必须能定位原始 media，或定位 backend 已保存的 feature/embedding artifact。
- tool result 应记录 canonical text payload 和对应 `tool_call_id`；不需要 tool-result raw。
- 如果发生 sliding window，需要记录保留下来的上下文窗口，而不是假装完整历史都还在 KV cache 里。

### 4.3 resume input

resume 时建议支持两种入口：

```json
{
  "type": "session.resume",
  "mode": "semantic",
  "messages": [],
  "tool_events": []
}
```

```json
{
  "type": "session.resume",
  "mode": "exact",
  "resume_trace": {
    "trace_version": "o5-resume-trace.v1",
    "items": []
  }
}
```

`semantic` 模式由 backend 按当前模板重新构造上下文。`exact` 模式要求 backend 校验 trace 与当前
模型/tokenizer/template 兼容，然后重放 token 和 embedding span 重建 cache。

## 5. 恢复边界建议

第一版建议只承诺在稳定边界恢复：

- turn 完成后。
- full-duplex `finalize_unit()` 完成后。
- 没有 pending tool-call args delta。
- 没有 pending TTS chunk / token2wav buffer，或 backend 明确保存了 TTS trace/cache snapshot。

如果要支持任意中间点恢复，需要额外导出：

- LLM / duplex decoder cache snapshot，或完整可重放 token+embedding trace。
- audio streaming processor snapshot。
- TTS `past_key_values` / `tts_text_start_pos` / token2wav buffer/cache。
- generation 参数和 RNG 状态。

这类中间点恢复更像 backend checkpoint，不应该伪装成普通聊天 transcript resume。

## 6. 最终建议

API 不应该把 special token 暴露为默认输出内容。为了 resume，应该暴露一个显式 opt-in 的
`resume_trace` 调试/恢复通道，其中 special token 以 `token_id` 为主、`token_text` 为辅出现。

换句话说：

```text
normal API: typed semantic events, no special-token leakage
resume API: canonical model-visible trace, may include special token ids/text
debug API: may expose decoded schema for inspection, not for client parsing
```

这样客户端不需要依赖 o5-sdk 解析内部 token，同时 backend 仍然保留精确重建 KV cache 的信息。
