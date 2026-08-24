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

## Deterministic TTS sampling correction

The first fixed-FLA comparison exposed an experiment-control bug before it
exposed an FC or TP2 numerical difference. The replay command exported
`O5_TTS_ARGMAX=1`, but the two targets did not consume that setting at the same
layer:

- Demo `PyTorchBackend.duplex_generate()` temporarily replaced the global
  `torch.multinomial` with argmax.
- Canonical had no backend wrapper and therefore continued to use seeded
  `torch.multinomial`.
- When `DuplexTraceController` was installed, the Demo backend replacement
  also bypassed the trace wrapper, so its TTS sample events were incomplete.

This explained the observed first TTS divergence: LLM selected tokens matched,
but Canonical selected acoustic content while Demo could select EOS under
argmax. It was not evidence that the FC wrapper had already changed the LLM
path.

The correction is in the shared `core/sampling.py` utility. Backend generation
enters `tts_argmax_scope()`, while the common trace controller consults the same
scope/environment and records the argmax decision. The scope is aware of an
installed trace wrapper, so backend generation cannot replace it and silently
lose trace events. The old behavior remains for untraced API generation.

Validation:

- `python -m py_compile` passed for the changed modules.
- With the project `.venv-accel`, `tests/test_session_trace_replay.py` passed:
  **17 passed**.
- The new regression test verifies that a traced TTS run under the backend
  argmax scope records all three sample calls, including the EOS call.

The fixed-FLA Canonical/Demo replay pair must be rerun with this correction
before any TP2 or acceleration conclusion is considered valid.

## Single-eager buffer mismatch

The corrected argmax replay exposed a second, independent baseline issue. The
FC demo's legacy `single_eager` loader called `model.bfloat16()`, which casts
floating-point buffers as well as parameters. Canonical's controlled loader
casts parameters only and deliberately keeps the LLM RoPE `inv_freq` buffer in
float32. The TP2 deployment builder already followed the latter policy, but
the single-eager path had not been updated.

The first `llm.feed` embeddings were equal while hidden states diverged from
the first call, which is the expected signature of this buffer mismatch. The
single-eager path now uses the same parameter-only placement helper as the
canonical and TP2 controlled paths. A focused unit test checks parameter BF16
conversion and float buffer preservation.

## Single-eager attention configuration mismatch

The next investigation found a separate configuration propagation bug. The
replay command passed `--attn-implementation sdpa`, and the outer
`MiniCPMOConfig` received it, but `MiniCPMO.__init__()` creates the Qwen text
config independently with `AutoConfig.for_model()`. The subsequent public
field copy uses `config.to_dict()`, while Transformers intentionally omits the
private `_attn_implementation` field from that dictionary. In the shared
accel environment this produced:

```text
outer config: sdpa
new Qwen text config: None
```

Therefore the previous demo comparison did not prove an SDPA-vs-Canonical
comparison; the text backbone's attention implementation was not explicitly
resolved at construction. The fix explicitly copies
`config._attn_implementation` into the newly created text config. Replay
manifests now record both the requested and the actual text-backbone attention
implementation so this class of experiment-control error is visible.

The corrected single-eager replay is pending. No TP2 or acceleration result
will be treated as a baseline until this run is compared with the stable
Canonical reference again.
