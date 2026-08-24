# Worktree

- Created: 2026-08-24
- Branch: `align-enhance-fc`
- Base branch: `o5-fc-dev`
- Base commit: `28a54d23c1373620871ca122122623ed6965cc77`
- Source to integrate: `o5-no-fc-speedup-tp2`
- Source commit at setup: `ad070af99a726a9ac41da8fb2326bb2503deef74`

## Purpose

Integrate the validated TP2 and acceleration history into the FC implementation
while preserving the source commits where Git can do so. The resulting branch
must retain FC capability, preserve the no-FC duplex replay/alignment tooling,
and support controlled canonical, unaccelerated, and fully accelerated
teacher-forcing comparisons.

The original FC and speedup worktrees are not modified by this worktree.

## Current continuation

- `cc0d07c fix(deploy): preserve explicit llm cache length`: keeps an explicit
  `O5_LLM_CACHE`, with TP2 defaulting to 32768 and single-card optimization to
  8192 when neither env nor config provides a value.
- `54acb58 fix(tp2): synchronize llm allocator cleanup`: propagates the
  rank-0 session cleanup to the rank-1 LLM/graph worker so both ranks run
  allocator cleanup. This stays below the FC/duplex API boundary.
- Focused FC tests: 52 passed in the existing accel venv. GPU alignment runs
  are still pending; use `agent-dev` first and `agent-train` only when needed.

## Inherited worktree history

# Worktree: o5-fc-dev

- Created: 2026-07-26
- Branch: `o5-fc-dev`
- Base branch: `wt/o5-no-fc-speedup-tp2-thin-unified-clean-llmgraph-narrow-no-tts-fix-fc-api-fc-board-job616069-2026-07-26`
- Base commit: `c732ae24d0d0dfd35afc151f3c968d212652c217`
- Source worktree: `/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24-clean-llmgraph-narrow-no-tts-fix-2026-07-26-fc-api-2026-07-26-fc-board-job616069-2026-07-26`
- Reason: establish the formal O5 FC development branch from the clean thin-unified TP2 implementation after FC API integration, FC Board MVP integration, TauVoice validation, and Board API same-schedule alignment.
- Baseline validation: TauVoice tasks `622868` and `622892`; FC Board task `622959`.

# Worktree: o5-no-fc-speedup-tp2-thin-unified-clean-llmgraph-narrow-no-tts-fix-fc-api-fc-board-job616069-2026-07-26

- Created: 2026-07-26
- Base branch: `wt/o5-no-fc-speedup-tp2-thin-unified-clean-llmgraph-narrow-no-tts-fix-fc-api-2026-07-26`
- Base commit: `6065a28eb10900ec54b24e133201979e41013fcf`
- Reason: merge the existing FC Board MVP into the clean thin-unified FC branch while preserving the original Board commits and implementation.
- FC Board source branch: `o5-no-fc-speedup-tp2-llmgraph-narrow-2026-07-23-fc-api-2026-07-23-fc-board-job616069-2026-07-24`
- FC Board source commit: `da8918efcf3829d5c116af6a0fcaa1bf0ec8112a`
- Integration policy: use a real `--no-ff` Git merge; retain the four source commit hashes without reimplementation or squashing.

# Worktree: o5-no-fc-speedup-tp2-thin-unified-clean-llmgraph-narrow-no-tts-fix-fc-api-2026-07-26

- Created: 2026-07-26
- Base branch: `wt/o5-no-fc-speedup-tp2-thin-unified-clean-llmgraph-narrow-no-tts-fix-2026-07-26`
- Base commit: `efb44621bd0d59c91596afca347f7186b38d799a`
- Reason: integrate the existing FC API implementation into the clean thin-unified, no-debug-TTS, narrow-LLMGraph TP2 branch while preserving the original FC implementation and Git history.
- FC source branch: `o5-no-fc-speedup-tp2-llmgraph-narrow-2026-07-23-fc-api-2026-07-23`
- FC source commit: `a6da42ff9b0625eddfbab59fbc0266bba6eb1a0a`
- Integration policy: use a real Git merge. Do not replace source FC behavior with a separately reimplemented equivalent.

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

## Merged FC Source: o5-no-fc-speedup-tp2-llmgraph-narrow-2026-07-23-fc-api-2026-07-23

- Created: 2026-07-23
- Base worktree: `/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-llmgraph-narrow-2026-07-23`
- Base branch: `o5-no-fc-speedup-tp2-llmgraph-narrow-2026-07-23`
- Base commit: `537387b84c302a4af653a4d01aeda1c84e5639fe`
- New branch: `o5-no-fc-speedup-tp2-llmgraph-narrow-2026-07-23-fc-api-2026-07-23`

Purpose: integrate O5 FC duplex API on top of the cleaned TP2/LLMGraphRunner base. The target is TP2 with all optimization flags enabled, with FC tool-call events exposed through the realtime API and tool results fed back into the duplex stream.

## FC Merge And Validation

- Merge commit: `82ecedbfde799aa75a70b2c7f42f3fce50eab437`, with first parent `b761f685ad37568a576ae93f3888ca47336b93b0a` and FC source parent `a6da42ff9b0625eddfbab59fbc0266bba6eb1a0a`.
- The FC schema, runtime, probes, report, and demo assets are byte-identical to the committed FC source. `FcDuplexCapability` is copied verbatim into the thin facade; only imports, configuration, initialization, and forwarding hooks are adapted to the thin structure.
- Model path: `/user/weihongliang/MiniCPM-o-4_6`.
- Checkpoint: `/user/weihongliang/o5_weights/iter_0001500_o5_with_tts.pt`.
- TP2 backbone safetensors: `/user/weihongliang/o5_weights/o5_backbone_hf_iter1500_with_tts`.
- Strict TauVoice FC alignment passed in cctl deploy task `622868`: the API emitted `convert_decimal_to_binary(255)` followed by `convert_decimal_to_binary(383)`, and accepted the tool results fed back by the probe.
- Full FC audio path passed in cctl deploy task `622892`: the API emitted a valid `convert_decimal_to_binary(383)` call and returned three non-empty 1-second float32 audio events.
- Strict result: `run-logs/cctl_tp2_smoke_20260726_124248/fc_tauvoice_probe.json`.
- Audio result: `run-logs/cctl_tp2_smoke_20260726_125704/fc_tauvoice_probe.json`.
- Live audio planning is intentionally not required to reproduce the strict two-call sequence. The source FC report documents drift between direct spoken answers, a single `383` call, and multiple calls; live audio validation requires non-empty audio and validates any emitted TauVoice calls.

## Merged FC Board Source

- Created: 2026-07-24
- Base branch: `o5-no-fc-speedup-tp2-llmgraph-narrow-2026-07-23-fc-api-2026-07-23`
- Base commit: `a6da42ff9b0625eddfbab59fbc0266bba6eb1a0a`
- Source branch: `o5-no-fc-speedup-tp2-llmgraph-narrow-2026-07-23-fc-api-2026-07-23-fc-board-job616069-2026-07-24`
- Source commit: `da8918efcf3829d5c116af6a0fcaa1bf0ec8112a`
- Reason: add the FC Board MVP demo surface on top of the O5 FC + TP2 runtime and run it with the `job616069` O5 FC MVP checkpoint.
- Checkpoint: `/user/heweiquan/models/MiniCPM-o5/trained_model/20260724/job616069/iter_0000500_o5.pt`.

## FC Board Validation

- Current branch validation commit: `900cfd5cf0621278bad2457fa966405aba096580`.
- cctl deploy task `622959` passed the API same-schedule probe for Board case `04702` with TP2 and the `job616069 iter500` checkpoint.
- The GT and API tool calls were semantically identical: `display_object_on_board({"name":"红外感应相机"})`.
- The API emitted the call at `unit_037`; the GT tool result was replayed successfully at `unit_039`, with no unsent or remaining responses.
- Result: `/user/weihongliang/fc_board_api_runs/board_api_same_schedule_04702_20260726_133856/board_same_schedule.json`.
- Full experiment record: `o5-fc-api-offline-alignment-report-2026-07-23.md`, section `FC Board API 同调度对齐（2026-07-26）`.
- This run did not enable audio and did not rerun the offline batch evaluator on the current HEAD.
