# Worktree: o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26

- Created: 2026-07-26
- Base branch: `wt/o5-no-fc-speedup-tp2-thin-unified-2026-07-24`
- Base commit: `c0bb591cc9a37b465548b2ae13ba2f63e579873d`
- Reason: align the realtime/API end-to-end O5 duplex path against the already aligned canonical offline path, assuming the unified import probe is already matched to raw/vendored inference.

## Scope

- Add a realtime `/v1/realtime?mode=video` API probe that emits the same `canonical_units.json` shape as the offline duplex probe.
- Add canonical-unit comparison tooling for API-vs-offline checks.
- Keep the model logic close to the thin-unified baseline; API-only representation differences such as listen/no-audio are normalized in the probe output, not in the server.

## API vs Canonical Validation

- Report: `o5-api-canonical-alignment-2026-07-26.md`.
- Offline canonical: `/user/weihongliang/thin_unified_runs/unified_trace_current_video01_max8_ttsargmax_20260726_02`.
- API run: `/user/weihongliang/o5_alignment_runs/api_vs_current_noaccel_video01_max8_ttsargmax_20260726_11`.
- Result: first 8 units match exactly under deterministic TTS argmax eval (`diff_count=0`).
- The generated TTS token hashes and the token slices entering `token2wav.stream()` also match.

## Inherited Baseline

- Parent worktree: `o5-no-fc-speedup-tp2-thin-unified-2026-07-24`.
- Parent base branch: `o5-no-fc-speedup-tp2`.
- Parent base commit: `38410c757619d5a5f53279a37cc37a1e7cf5b802`.
- Parent reason: shrink O5 `modeling_minicpmo_unified.py` toward the vendored implementation, keep only the minimum API-facing compatibility layer, and verify raw vendored offline vs thin unified offline before any API or acceleration experiments.

## Parent Validation

- Offline raw vs thin-unified video duplex probe passed on `omni_demo_duplex_01.mp4`, max 8 units.
- Canonical output hash: `d68ab9fb5695b62e6815493e214f9654c783d307612327e6d39e370423920a36`.
- Raw output: `/user/weihongliang/thin_unified_runs/raw_video01_max8_20260724_01/canonical_units.json`.
- Unified output: `/user/weihongliang/thin_unified_runs/unified_video01_max8_20260724_01/canonical_units.json`.
- API video probe passed against local SSH service on `https://127.0.0.1:8051`, session `sess_fc907dbda1f4`.
- API audio probe passed against the same service.
- Service startup needed `O5_ATTN_IMPLEMENTATION=sdpa` for this single-card validation; `auto` selected FlashAttention2 and hit a single-card startup OOM.
- Full 36-unit offline raw/thin-unified canonical hash: `97a673b3acfd05f6fefa75be92eacf94092c32c153875cdc387c34db11f289a6`.
- Acceleration strict-alignment report: `o5-duplex-acceleration-alignment-2026-07-24.md`.
- Strict-aligned acceleration subset on the 36-unit probe: `tts_fast + lmhead + fuse_vision_audio`, with `tts_graph/vocoder_graph/batch_vision_feed/batched_mm/llm_graph` disabled.
