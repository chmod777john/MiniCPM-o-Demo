# O5 Vendored / Unified / TP2 Notes

Date: 2026-07-22

This note records the current reasoning around the O5 inference refactor, with
O45 trusted code as the reference baseline.

## Code Layers

The comparison uses three layers:

- O45 trusted code: `/user/weihongliang/autoshow_omni/models/MiniCPM-o-4_5`
- O5 vendored code: `MiniCPMO45/modeling_minicpmo.py`
- O5 unified adapter: `MiniCPMO45/modeling_minicpmo_unified.py`

The intended refactor shape is:

```text
O5 unified adapter
  inherits O5 vendored code
  keeps only serving/demo/runtime deltas
```

Therefore, for every method that unified overrides, the first question is not
"why does unified override this?". The first question is:

```text
Did the underlying O5 vendored method change relative to O45 trusted code?
```

If O5 vendored did not change, then a large unified override is probably a
serving/runtime/performance adaptation rather than an O5 model adaptation.

## Unified Overrides With Vendored Counterparts

For methods that exist both in O5 unified and O5 vendored, the current function
level comparison against O45 trusted code is:

| Unified override | O5 vendored vs O45 trusted | Notes |
| --- | --- | --- |
| `MiniCPMO.__init__` | changed | O5 model init, Qwen3.5/M-RoPE, wrapper-sensitive class handling |
| `MiniCPMO.chat` | changed | O5 chat TTS boundaries, speaker/ref-audio handling, image marker changes |
| `MiniCPMO.streaming_prefill` | changed | O5 canvas M-RoPE `position_ids` are required |
| `MiniCPMO.streaming_generate` | unchanged | O45 trusted and O5 vendored bodies are effectively identical |
| `MiniCPMO.get_sys_prompt` | unchanged | No material O5-vs-O45 change |
| `DuplexCapability.from_existing_model` | unchanged | O45 trusted and O5 vendored bodies are effectively identical |
| `DuplexCapability._reset_streaming_state` | unchanged | O45 trusted and O5 vendored bodies are effectively identical |
| `DuplexCapability.prepare` | changed | O5 vendored uses batched token feed for prompt pieces |
| `DuplexCapability.streaming_prefill` | unchanged | O45 trusted and O5 vendored bodies are effectively identical |
| `DuplexCapability.streaming_generate` | unchanged | O45 trusted and O5 vendored bodies are effectively identical |

Summary:

```text
10 unified overrides have vendored counterparts.
4 have O5-vs-O45 changes in vendored.
6 do not have material O5-vs-O45 changes in vendored.
```

For duplex specifically:

```text
DuplexCapability.from_existing_model      unchanged
DuplexCapability._reset_streaming_state   unchanged
DuplexCapability.prepare                  changed, mostly performance
DuplexCapability.streaming_prefill        unchanged
DuplexCapability.streaming_generate       unchanged
```

This means the large unified duplex streaming overrides are not explained by
O5 vendored model changes. They mostly come from serving/runtime additions such
as profiling, result shaping, delayed finalization, and performance tweaks.

## Whether O5 Vendored Changes Are Absorbed By Unified

For the four methods where O5 vendored differs from O45 trusted code:

### `MiniCPMO.__init__`

Unified calls:

```python
super().__init__(config)
```

So O5 vendored initialization is preserved. This is important because O5 init
sets up the Qwen3.5/M-RoPE model stack and wrapper-sensitive logic.

### `MiniCPMO.chat`

Unified delegates to:

```python
BaseMiniCPMO.chat(...)
```

and wraps it for serving needs such as temporary waveform output, token stats,
and optional TTS reference audio override. This preserves O5 vendored chat logic,
including TTS region boundaries, speaker/reference handling, and image marker
changes.

### `MiniCPMO.streaming_prefill`

O5 vendored changed streaming prefill from `position_ids=None` to explicit canvas
M-RoPE position ids. Unified absorbs this through:

```python
_compute_unified_prefill_position_ids(...)
```

which mirrors vendored O5 logic:

```text
if image content exists:
  use _compute_canvas_position_ids(...)
else:
  combine text position ids with expanded 3D M-RoPE ids
then add cache_length when continuing from an existing cache
```

This is the most important O5-specific unified adaptation. Without it, O5
streaming and duplex prefill use wrong position ids.

### `DuplexCapability.prepare`

O5 vendored changed some system-prompt feeding from token-by-token to batched
embedding feed:

```python
self.decoder.feed(self.decoder.embed_tokens(tokens))
```

Unified also uses batched `embed_tokens(...); decoder.feed(...)` for the same
prompt pieces, so the performance-oriented change is absorbed. This is less
semantically critical than M-RoPE, but relevant for realtime latency.

## LLMGraphRunner Boundary

`LLMGraphRunner` is not a replacement for the whole HuggingFace LLM. It exposes
only a narrow execution interface:

```python
runner.reset()
runner.prefill(inputs_embeds, start_pos)
runner.decode(inputs_embeds, pos)
```

It is currently used only inside `StreamDecoder.feed()`.

Without `llm_graph`, `StreamDecoder.feed()` calls the LLM directly:

```text
self.m.model(...) + self.m.lm_head(...)
or
self.m(...) + self.m.lm_head(...)
```

With `llm_graph`, the same high-level operation is routed through:

```text
LLMGraphRunner.prefill/decode(...)
self.m.lm_head(...)
```

The intended semantic operation is the same:

```text
feed embeddings into the LLM backbone
update cache
return hidden state / logits for the last token when requested
```

The implementation mechanics are different:

- dynamic cache becomes `StaticCache`
- cache position and attention mask become explicit
- single-token decode uses CUDA graph replay
- long sessions are bounded by `O5_LLM_CACHE`; overflow currently resets context

Because `LLMGraphRunner` does not cover the complete LLM surface, it cannot be
used as a transparent `self.llm` replacement. The raw LLM still exposes:

```text
forward / __call__
generate
model
lm_head
embedding helpers
config / generation_config
HF generation helper methods
```

## StreamDecoder Design Choices

There are two possible directions:

1. Align `StreamDecoder` as closely as possible with O5 no-FC, so it directly
   calls the raw LLM in the same shape as vendored code. In this design, graph
   cannot remain as ad-hoc logic inside `StreamDecoder.feed()`.
2. Let `StreamDecoder` depend on a small feed-runtime abstraction, with normal
   and graph implementations. For example:

   ```python
   runtime.feed(embeds, return_logits=False)
   runtime.reset()
   runtime.cache
   ```

   The runtime would own cache position, StaticCache/DynamicCache, graph replay,
   overflow behavior, and last-token lm-head calculation.

The current code is closer to option 2 in spirit but not fully abstracted: graph
details still live directly inside `StreamDecoder.feed()`.

## TP2 And Whole-Backend SPMD Mirror

There are two TP2 synchronization shapes in the current code:

```text
O5_LLM_GRAPH=0
  Use DistributedTPLLM worker loop.
  Synchronization is closer to the LLM boundary.

O5_LLM_GRAPH=1
  Use whole-backend SpmdMirror.
  Rank 1 mirrors selected backend public methods.
```

The whole-backend mirror still exists and is active in the default graph path.
The mirrored backend methods include:

```text
chat_complete
chat_prefill
chat_init_tts
chat_prepare
chat_streaming_generate
chat_non_streaming_generate
duplex_prepare
duplex_prefill
duplex_generate
duplex_finalize
duplex_stop
duplex_cleanup
```

This is coarse but currently needed when `O5_LLM_GRAPH=1`, because graph/cache
state is owned outside the raw HF LLM module, primarily in `StreamDecoder.feed()`:

```text
LLMGraphRunner
StaticCache
_static_pos
attention-mask buffer
cache_position
graph replay state
```

If only the LLM boundary is synchronized, rank 0 advances this outer graph/cache
state while rank 1 does not. The whole-backend mirror makes both ranks execute
the same high-level Python path so these states advance in lockstep.

Longer term, if graph state is moved behind a proper LLM feed runtime, TP2 could
sync at that runtime boundary instead of mirroring whole backend calls.

## Non-Streaming Chat Beam Setting

O5 uses 3D M-RoPE position ids shaped like:

```text
[3, batch, seq]
```

Transformers beam search assumes ordinary position ids shaped like:

```text
[batch, seq]
```

and expands along dimension 0. That corrupts the M-RoPE axis and fails in rotary
embedding. The current non-streaming chat path therefore forces:

```python
num_beams=1
```

until beam expansion is made M-RoPE aware. This keeps non-streaming chat on the
direct `model.chat(...)` path while avoiding the incompatible beam-search path.

