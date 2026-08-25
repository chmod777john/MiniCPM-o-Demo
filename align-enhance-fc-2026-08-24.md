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

## Experiment-control audit (2026-08-25)

The existing replay artifacts were compared locally with the repository's
`tools/o5replay/compare.py` without starting another model process.

- The two current `demo-single` layer-trace runs are self-consistent:
  `demo-single-fixedfla-layer-trace-1u` and its repeat have 13/13 paired
  events, all recorded LLM tensors bitwise equal, identical selected tokens,
  and identical 24 kHz audio SHA256. This rules out ordinary repeat
  nondeterminism caused solely by the layer hooks.
- Comparing the current layer-trace demo against the current Canonical
  reference shows equal feed embeddings but different hidden/logits from the
  first feed. This is a path difference, not an input-session mismatch.
- The current no-acceleration candidate
  `demo-single-noaccel-fixedfla-sdpa-attn-fix` has 299/320 common events:
  all 8 accepted LLM chunks and 24/24 LLM decode decisions match, while its
  TTS sample decisions do not match the Canonical reference (79 compared
  events, 0 equal). Its first four completed audio units were identical only
  because the TTS path had not diverged yet; the candidate then produced a
  different unit count/audio length.
- The older precision matrix contains a genuinely aligned single-card
  all-off run (`demo-single-current-local-alloff-free-8u`): 93/93 LLM feed
  hidden tensors, 24/24 decode decisions, and 75/75 TTS samples matched the
  Canonical reference. That artifact predates this FC worktree's latest
  loader/config changes and has no complete deployment manifest, so it is a
  reference for the target behavior, not proof that the current command is
  equivalent.

The controlled next run must therefore explicitly set and record:
`single_eager`, `experts_implementation=eager`, `attn_implementation=auto`
(which resolves to SDPA in the shared environment), all graph/fast/batched
flags off, float-buffer preservation on, and the fixed FLA settings. The replay
manifest now records these deployment settings and the CUDA runtime facts so
the result cannot be misclassified from a directory name alone.

At this point the cctl API is still returning HTTP 503 for pool queries, and
the reserved 8-GPU SSH host is unreachable by SSH timeout. No GPU task or
other user's process was changed. Once either approved resource path is
available, the controlled single-eager run is the first required experiment;
TP2 and acceleration runs remain downstream of that baseline.

## Same-GPU demo repeat (2026-08-25)

The corrected single-eager path was run twice in one process on the same GPU
(`deploy` task `780344`). Both runs used the same input bundle, checkpoint,
backbone, eager attention, eager experts, fixed FLA settings, and deterministic
CUDA controls. Both completed all 8 units and returned the same text:
`好的，没问题。`

The replay comparison is:

- 210/210 trace events paired;
- all accepted LLM chunks and 16/16 LLM decode decisions equal;
- all TTS sample decisions equal (44/44);
- all TTS conditions, TTS chunks, and Token2Wav ranges equal;
- all 8 per-unit PCM files and the combined audio file bitwise equal.

This rules out ordinary run-to-run nondeterminism in the current demo path for
this controlled case. The earlier canonical comparison was invalid because its
reference bundle had no FLA overrides while the demo bundle used
`32x8/16x8/128x128x8`; a matching canonical run was submitted as task `780365`.

## Same-GPU canonical vs demo (2026-08-25)

The corrected comparison was rerun sequentially on one A100 in deploy task
`780462`: canonical first, then `demo-single`, with the same checkpoint,
backbone, input bundle, fixed FLA settings, eager attention/experts, and all
graph/fast/lmhead/batching accelerations disabled. The command used the shared
`.venv-accel` Python executable and explicitly invoked `run_session.py` through
Python.

Artifacts:

- Canonical:
  `/user/weihongliang/o5_align_enhance_fc_exp_20260824/canonical-fixedfla-samegpu-c`
- Demo:
  `/user/weihongliang/o5_align_enhance_fc_exp_20260824/demo-single-fixedfla-samegpu-c`
- Comparison:
  `/user/weihongliang/o5_align_enhance_fc_exp_20260824/canonical-vs-demo-samegpu-c.json`

The model-side result is fully aligned:

- 213/213 trace events paired.
- LLM accepted tokens: 8/8 equal.
- LLM decode selected and local-argmax tokens: 17/17 equal.
- LLM feed embeddings, hidden states, and logits: 86/86 entries bitwise equal.
- LLM decode logits: 17/17 bitwise equal, with zero argmax reversals.
- TTS chunks, conditions, hidden/logits, probabilities, and selected samples:
  all entries equal (45/45 TTS samples).
- Token2Wav input token IDs and committed/lookahead/output ranges: all equal.

The only remaining difference is inside Token2Wav/vocoder execution:

- `vocoder.state.rand_noise`: 1/1 tensor differs.
- Token2Wav output PCM: 0/2 bitwise equal.
- Per-unit audio: the first four listen units and final listen unit are
  bitwise equal; the two spoken units differ, while their TTS tokens and
  Token2Wav input ranges remain equal.
- Combined audio has the same shape and duration but different PCM hashes.

Therefore this experiment establishes that the current FC merge, in the
single-card all-off path, is aligned with canonical through TTS token sampling
and the complete Token2Wav input contract. The audio mismatch is a downstream
vocoder random-noise initialization/consumption issue, not an LLM, TTS, TP2,
or FC token-planning divergence. It must be isolated separately before using
PCM equality as the acceptance criterion for later TP2/acceleration runs.

## TP2 all-off replay (2026-08-25)

The first valid TP2 replay was task `780602`. An earlier task used the
`torchrun` shim from an old environment and was excluded; `780602` invoked
`python -m torch.distributed.run` through the same `.venv-accel` environment
used by the Canonical control. Its manifest records:

- Demo commit: `329f44e1` (`align-enhance-fc`)
- mode: `tp2`, two A100 ranks, `experts_implementation=eager`
- attention: `eager`
- LLM/TTS/vocoder graph, TTS fast, lmhead, vision fusion and vision batching:
  all disabled
- fixed FLA: `32x8`, `16x8`, `128x128x8`
- deterministic CUDA controls enabled, greedy decode, seed `0`

The comparison against
`canonical-fixedfla-samegpu-c` is
`/user/weihongliang/o5_align_enhance_fc_exp_20260824/canonical-vs-demo-tp2-alloff-fixedfla-accel.json`.
Both bundles use the same input session, checkpoint, backbone and fixed-FLA
settings. The result is not bitwise aligned:

- LLM accepted chunks: `6/8` token-id lists equal.
- LLM decode decisions: `10/15` equal; `5` local-argmax reversals.
- The first accepted-token divergence is `unit_000004`.
- LLM feed hidden/logits differ before that discrete divergence; feed embeddings
  are identical for the common early calls and then differ when the previous
  TP2 trajectory has diverged.
- TTS samples: `1/42` equal among the common trace positions.
- The TP2 run produced 8 units while Canonical produced 8 source units but the
  event alignment reached only 7 common audio units after the trajectory split;
  combined audio shapes were different (`167040` vs `185280` samples).

This is the same class of TP2 eager numerical drift documented in the original
speedup-tp2 work: the TP2 LLM hidden state differs from single-card eager even
with all graph/fast switches off, and TTS autoregression amplifies that small
continuous difference into token and unit-boundary changes. It is therefore not
evidence of an FC merge regression. The result also confirms that the current
FC branch has not yet reached strict Canonical/TP2 token alignment; the
teacher-forced experiments below are needed to separate forward-distribution
drift from downstream TTS effects.

## TP2 full-acceleration replay deadlock root cause (2026-08-24)

The first full-acceleration TP2 teacher-forced run was stopped after rank 1
reported an NCCL `BROADCAST` timeout. The failure was in the replay harness,
not in the checkpoint, scheduler, or graph capture:

- `tools/o5replay/run_session.py` installs `DuplexTraceController` only on
  rank 0; rank 1 enters the model-provided worker loop and does not execute
  the high-level decoder sampling hooks.
- `core/tracing/session_trace.py` nevertheless called
  `torch.distributed.broadcast(selected)` from the rank-0-only forced LLM
  decode hook.
- Rank 1 therefore waited for the next graph/LLM command while rank 0 entered
  an extra NCCL collective. This shifted the collective sequence and eventually
  produced the observed `BROADCAST` timeout.

The selected token does not need a separate collective. It is converted into
the next decoder embedding on the driver, and that embedding is already
broadcast by `DistributedTPLLM` in eager TP2 or by `LLMGraphRunner` in graph
TP2. The extra broadcast was removed while retaining the `tp_driver` argument
for call-site compatibility.

Regression coverage in `tests/test_session_trace_replay.py` forces
`tp_driver=True` and fails if the replay hook invokes any distributed
`broadcast`; the focused test set passes in the shared `.venv-accel`:
`2 passed, 18 deselected`.

The next run is a narrow TP2 + LLM-graph replay with TTS graph disabled. A
successful run will establish that graph command synchronization is restored
before enabling the remaining TTS optimizations.

## TP2 LLM graph after replay fix (task 780833, 2026-08-24)

The narrow run used the new commit `07016e9` on two A100s in `deploy`:

- `llm_graph=1`, `tts_graph=0`, `tts_fast=1`, `lmhead=1`;
- `experts_implementation=batched_mm`;
- `vocoder_graph=0`, `fuse_vision_audio=1`, `batch_vision_feed=1`;
- eager attention, fixed FLA (`32x8`, `16x8`, `128x128x8`), deterministic replay;
- `--forcing all`, 8 units, same canonical reference and input session.

The task captured the LLM graph on both ranks, entered rank 1's graph worker
loop, and completed all 8 units without a collective timeout. The generated
text was `好的，没问题。`.

Comparison artifact:
`/user/weihongliang/o5_align_enhance_fc_exp_20260824/demo-tp2-llmgraph-only-fix/comparison.json`

The comparison had 213/213 paired events:

- accepted LLM tokens: `8/8` equal;
- LLM decode selections and local argmax: `17/17` equal;
- TTS chunks: `3/3` equal;
- TTS sampled tokens: `45/45` equal;
- Token2Wav input IDs and ranges: all equal;
- LLM hidden/logits: numerically different, but all 17 decode argmax decisions
  remained equal (`cosine_mean=0.9975` for decode logits);
- TTS hidden/logits/probabilities: bitwise equal because their conditions and
  sampled tokens were forced from the same reference;
- audio shape and duration: equal; PCM differed only in the known downstream
  vocoder execution path.

This isolates the previous failure to the removed driver-only replay collective.
