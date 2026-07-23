# Worktree: o5-no-fc-speedup-tp2-thin-unified-clean-llmgraph-narrow-no-tts-fix-2026-07-26

- Created: 2026-07-26
- Base branch: `wt/o5-no-fc-speedup-tp2-thin-unified-2026-07-24`
- Base commit: `c0bb591cc9a37b465548b2ae13ba2f63e579873d`
- Reason: build a clean replacement candidate for the thin unified O5 no-FC TP2 demo. Keep the thin unified facade, remove the later houyueran TTS behavior changes that are not trusted for this path, and then port the narrower LLMGraph/TP2 synchronization boundary from `o5-no-fc-speedup-tp2-llmgraph-narrow-2026-07-23`.
- Guardrail: do not modify or depend on uncommitted local changes in `/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24-our-base-2026-07-25`.
- LLMGraph source: cherry-pick from `4627e3f Narrow TP2 sync to LLM graph runner` on `o5-no-fc-speedup-tp2-llmgraph-narrow-2026-07-23`.

# Worktree: o5-no-fc-speedup-tp2-thin-unified-2026-07-24

- Created: 2026-07-24
- Base branch: `o5-no-fc-speedup-tp2`
- Base commit: `38410c757619d5a5f53279a37cc37a1e7cf5b802`
- Reason: shrink O5 `modeling_minicpmo_unified.py` toward the vendored implementation, keep only the minimum API-facing compatibility layer, and verify raw vendored offline vs thin unified offline before any API or acceleration experiments.

## Validation

- Offline raw vs thin-unified video duplex probe passed on `omni_demo_duplex_01.mp4`, max 8 units.
- Canonical output hash: `d68ab9fb5695b62e6815493e214f9654c783d307612327e6d39e370423920a36`.
- Raw output: `/user/weihongliang/thin_unified_runs/raw_video01_max8_20260724_01/canonical_units.json`.
- Unified output: `/user/weihongliang/thin_unified_runs/unified_video01_max8_20260724_01/canonical_units.json`.
- API video probe passed against local SSH service on `https://127.0.0.1:8051`, session `sess_fc907dbda1f4`.
- API audio probe passed against the same service.
- Service startup needed `O5_ATTN_IMPLEMENTATION=sdpa` for this single-card validation; `auto` selected FlashAttention2 and hit a single-card startup OOM.
- Full 36-unit offline raw/thin-unified canonical hash:
  `97a673b3acfd05f6fefa75be92eacf94092c32c153875cdc387c34db11f289a6`.
- Acceleration strict-alignment report:
  `o5-duplex-acceleration-alignment-2026-07-24.md`.
- Strict-aligned acceleration subset on the 36-unit probe:
  `tts_fast + lmhead + fuse_vision_audio`, with `tts_graph/vocoder_graph/batch_vision_feed/batched_mm/llm_graph` disabled.
