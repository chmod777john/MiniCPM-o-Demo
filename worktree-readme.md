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
