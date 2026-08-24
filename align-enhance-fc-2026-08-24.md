# align-enhance-fc

## Scope

This report records the integration and alignment work for the FC branch with
the validated TP2 speedup implementation. Results are appended as experiments
complete.

## Baselines

- FC base: `o5-fc-dev` at `28a54d23c1373620871ca122122623ed6965cc77`.
- TP2 source: `o5-no-fc-speedup-tp2` at `ad070af99a726a9ac41da8fb2326bb2503deef74`.
- Common ancestor: `c0bb591cc9a37b465548b2ae13ba2f63e579873d`.
- Latest fetch of `codeup-modelbest/o5-fc-dev`: no new commit; remote remains
  `28a54d2`.
- This worktree: branch `align-enhance-fc`.

## Integration policy

Use real Git integration so the TP2 source commits remain inspectable. Keep FC
runtime and API behavior from the FC branch, and resolve only the actual
interfaces between FC, unified duplex processing, TP2, graph state, and replay.
Do not rewrite a source optimization as an unrelated semantic equivalent.

## Environment

Grouped-MM experiments must record the high-cu128 acceleration environment in
the run manifest and logs. The code path currently points at the shared accel
venv used by the validated TP2 launcher; the exact executable and torch/CUDA
versions will be recorded before GPU execution.

## Required experiments

1. Run Canonical twice with fixed FLA settings, seed, inputs, and checkpoint.
   Compare hidden, full logits, raw argmax, shapes, dtypes, and hashes. The
   main question is whether Canonical itself is bitwise stable under high-cu128.
   Then run `align-enhance-fc` with TP2 and all accelerations disabled against
   the same canonical input and settings.
2. Run non-FC duplex on `align-enhance-fc` with TP2 and all accelerations on,
   except vocoder graph. Teacher-force LLM and TTS tokens and compare the
   replay against Canonical, including the first reversal and TTS inputs.
3. Use one FC benchmark case. First run FC with every acceleration disabled.
   Then run FC with TP2 and all accelerations on, except vocoder graph, using
   teacher forcing. Compare FC token planning, opener/tool events, TTS tokens,
   and the first reversal.

## Current status

- Worktree created from the clean FC tip.
- FC remote fetched and confirmed unchanged.
- TP2 integration is present on the current branch. The integration keeps the
  FC runtime and the LLM/graph-boundary synchronization model; it does not
  restore the removed whole-backend SPMD mirror.
- `cc0d07c fix(deploy): preserve explicit llm cache length` fixes an actual
  configuration precedence bug: an explicit `O5_LLM_CACHE` now wins over the
  old hard-coded `cfg.get(..., 8192)` fallback. With no explicit value, TP2
  defaults to 32768 and single-card optimization defaults to 8192.
- `54acb58 fix(tp2): synchronize llm allocator cleanup` adds a small LLM
  boundary cleanup command. FC/duplex session cleanup on rank 0 now asks the
  rank-1 LLM/graph worker to run `gc.collect()` and `torch.cuda.empty_cache()`
  as well. This addresses the observed rank-1 reserved-memory plateau without
  mirroring FC or duplex methods.
- Static checks passed: Python compilation, shell syntax checks, and
  `git diff --check`.
- Using the existing project accel venv
  `/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel`
  with `PYTHONPATH=.`, the focused FC tests passed: **52 passed**.
- The replay CLI fixed-FLA wiring was added in `0fd0905
  test(replay): expose fixed FLA configs`. It exposes and records:
  `32x8` fused gated RMSNorm, `16x8` L2Norm, and `128x128x8` chunk output.

## Experiment 1: Canonical fixed-FLA self-replay

- Input: `/user/weihongliang/o5_replay_strategy_hd_input_8u_20260822_canonical_hd`
  (8 units from `omni_demo_duplex_01.mp4`).
- Checkpoint: `chenmoye ... iter_100.pt` under
  `/user/weihongliang/o5_weights/`.
- Runtime: commit `0fd0905`, `.venv-accel`, seed `0`, greedy LLM decode,
  `O5_TTS_ARGMAX=1`, and all three fixed FLA configs above.
- Jobs: `779159` and `779158`, both 1 GPU in `agent-train`.
- Comparison:
  `/user/weihongliang/o5_align_enhance_fc_exp_20260824/canonical-fixedfla-comparison.json`

Result: Canonical is bitwise stable for this 8-unit run. All 320 trace events
paired; LLM feed hidden/logits (93/93), decode logits (24/24), TTS hidden/logits
and probabilities (89/89), TTS condition (4/4), TTS tokens (4/4), token2wav
inputs and PCM chunks (4/4), and merged PCM16 audio are bitwise equal. The
merged audio SHA256 is
`f195a4e1a5979399f5fec642aca0379bd11e9dc1829b850b5d2a6d3154fc25dd`.

This establishes a stable Canonical reference for the next comparisons; it
does not yet establish alignment of `align-enhance-fc` with Canonical.

## Replay interface adaptation

The first `single_eager` no-acceleration run (`779192`) reached model loading
but stopped before the first unit because FC's `PyTorchBackend.duplex_prepare()`
did not expose the existing `llm_seed` argument. The underlying O5 model already
accepted it, and the speedup replay path had used it to make the session seed
explicit. The fix is a narrow pass-through in `DuplexView.prepare()` and
`PyTorchBackend.duplex_prepare()`; it does not add backend mirroring or change
FC behavior. The failed task was not retried after the fix yet.
