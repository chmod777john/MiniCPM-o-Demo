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

## TP2 full acceleration without vocoder graph (task 780864, 2026-08-24)

The next run enabled `tts_graph=1` on top of the successful LLM-graph
configuration, while keeping `vocoder_graph=0`. It again used two A100s in
`deploy`, commit `169e36d`, fixed FLA, eager attention, `batched_mm`, and
deterministic `--forcing all` replay.

The task captured both LLM and TTS CUDA graphs and completed all 8 units. The
comparison remained structurally complete at 213/213 events:

- accepted LLM tokens: `8/8` equal;
- LLM decode selections and local argmax: `17/17` equal;
- TTS sampled tokens: `45/45` equal;
- Token2Wav input IDs and ranges: all equal;
- TTS graph hidden/logits: `20/45` bitwise equal, but all 45 logits argmax
  decisions equal; one probability-vector argmax differed numerically;
- generated audio shape/duration: equal, with the same known PCM-level
  vocoder-state differences.

The forcing result means this run proves graph execution and replay lifecycle,
not free-running TTS equivalence. The next experiment enables `vocoder_graph`
as the only remaining acceleration switch before a no-forcing comparison.

## TP2 all-acceleration free run (task 780921, 2026-08-24)

The complete configuration (`llm_graph=1`, `tts_graph=1`, `tts_fast=1`,
`lmhead=1`, `batched_mm`, vision fusion/batching, and `vocoder_graph=1`) was
then run with `--forcing none` for 8 units on two A100s. It completed normally
and produced the same text `好的，没问题。`.

Against the Canonical reference:

- all 17 LLM decode decisions and all 8 accepted LLM token lists matched;
- LLM hidden/logits still showed the expected TP2 continuous drift;
- TTS conditions were numerically different, as expected from the TP2 LLM
  hidden state;
- TTS sample decisions matched `44/45`; TTS logits had 2 local-argmax
  reversals out of 45;
- the first discrete divergence was therefore in the TTS autoregressive loop,
  followed by a different final TTS chunk and audio length.

Artifact:
`/user/weihongliang/o5_align_enhance_fc_exp_20260824/demo-tp2-fullaccel-free/comparison.json`

This does not establish that `tts_graph` caused both reversals: TP2 hidden
drift alone can move a close TTS probability boundary. The paired free run
with `tts_graph=0` is required to separate those causes.

## TP2 LLM graph free run with TTS graph disabled (task 780971, 2026-08-24)

The paired run kept `llm_graph=1`, `batched_mm`, `tts_fast=1`, and
`vocoder_graph=1`, but changed only `tts_graph` to `0`. It also completed all
8 units without synchronization errors.

Compared with Canonical, the same LLM path still had 17/17 decode decisions
equal. TTS behavior was materially worse than the graph-enabled free run:

- TTS sample decisions: `27/45` equal;
- TTS logits local argmax: `26/45` equal, with `19` reversals;
- only `1/3` TTS chunks were bitwise equal;
- the generated TTS/audio trajectory diverged before the final spoken unit.

This is a useful ablation result, not a new code failure: on this workload,
the TTS graph is the more reproducible execution path. The graph-enabled run
had `44/45` TTS sample decisions and 2 logits reversals. The remaining work is
therefore to retain TTS graph in the production acceleration profile and
validate the FC-specific replay path separately.

## FC Board API v3 probe hardening (2026-08-25)

The first FC Board trace attempt exposed a sequence of test-harness contract
issues before model execution. They were fixed as separate commits:

- `bffb5b8 fix(fc): pass case reference audio to backend`: derive the case's
  absolute `HTRef06.wav` path and pass it to both single-card and TP2 backend
  launchers, so FC Token2Wav warmup does not consume the relative config
  fallback.
- `cb33082 fix(fc-probe): send semantic v3 board init`: project the case's
  ordered system text/audio segments and tools into the required Semantic
  Realtime v3 payload, including `protocol_version="3"` and
  `tts_prompt_audio`.
- `ff2e8a8 fix(fc-probe): make unit policy authoritative`: remove the legacy
  scalar budget from the v3 request when the case provides a complete
  `unit_policy`.
- `a8c1d93 fix(fc-probe): wait for committed unit events`: advance the probe
  using the mandatory `response.unit.committed` event rather than optional
  `response.debug` events.

With these fixes, task `781170` loaded the model, warmed the FC path, accepted
the v3 session, and entered actual generation. It then failed at the model's
first non-spoken generation step:

```text
FC non_spoken ordinary token arrived before stream opener: 220
```

The checkpoint used there was `/user/weihongliang/o5_weights/iter_0001500_o5_with_tts.pt`.
The same failure class is documented by inherited commits `84e9a31` and
`ef7c33e`: an FC checkpoint can emit an ordinary token before `<think>`,
`<tool_call>`, or `<no_action>`, which is a model protocol failure rather than
an API transport/parser failure. The O5 adapter intentionally keeps
`drops_unclassified_non_spoken_tokens=False`; relaxing that would hide a
checkpoint defect and invalidate token/replay alignment. The next FC run uses
the available Chenjinpeng FC-CE checkpoint to separate checkpoint behavior
from the integrated runtime.

## FC token-only startup warmup fix (2026-08-25)

The first Chenjinpeng `000747` token-only tasks (`781300`, `781304`) were
started from commit `951179d`. They reached the service launcher but failed
before the probe because the generic service config supplied the relative
default path `assets/ref_audio/ref_minicpm_signature.wav` to
`FcAudioPathInput`, whose schema requires an absolute server-readable path.
The token-only request itself did not need Token2Wav at all.

Commit `04db2cb fix(fc): make token2wav startup warmup path-safe` addresses
both sides of this boundary:

- startup warmup resolves project-relative reference audio to an absolute path
  and checks that it is a readable file before constructing the FC schema;
- an audio-enabled run still fails early with the configured and resolved
  paths when the reference file is invalid;
- `run_fc_board_trace_replay_cctl_entry.sh` exports
  `FC_DUPLEX_STARTUP_WARM=0` for `GENERATE_AUDIO=0`, so token-only replay does
  not pay for or validate an unnecessary Token2Wav warmup;
- focused FC runtime tests cover disabled warmup, relative-path resolution,
  invalid audio failure, cache reuse, and protocol boundaries: `9 passed`.

The two old tasks are intentionally not treated as valid results because they
run commit `951179d`; they must be replaced by tasks from `04db2cb` before
comparing FC output.

## FC spoken slot boundary fix (2026-08-25)

The official Chenjinpeng `000747` trajectory showed a protocol pattern that
the integrated FC View rejected:

```text
Unit N:   <speak> ordinary spoken tokens <spoken_slot_eos>
Unit N+1: <listen>
```

The existing `FcDuplexView._decode_generation_steps()` kept the spoken text
decoder alive after every `spoken_slot_eos` and treated any later `listen` as
`listen before spoken_turn_eos`. That interpretation is too strict for the
actual O5 FC checkpoint, where a slot boundary can be the end of the current
spoken response if the next Unit listens, while repeated `speak` opens a
continuation on the same stream. `tts_pad` has the same lazy-boundary behavior.

Commit `975a36e fix(fc): accept listen after spoken slot boundary` fixes this
without dropping cross-Unit BPE state:

- `spoken_slot_eos` and `tts_pad` mark a pending slot boundary but do not
  destroy the decoder immediately;
- a following `speak` clears that marker and reuses the same stream;
- a following `listen` closes the pending spoken stream before accepting the
  listen token;
- `listen` without either a pending slot boundary or `spoken_turn_eos` still
  raises the protocol error;
- both legacy and semantic-v2 stateless resume validators use the same state
  transition.

Verification on the shared accel venv:

```text
py_compile: passed
focused FC tests: 54 passed
```

The regression tests cover both `spoken_slot_eos -> listen` acceptance and
direct `listen` rejection, while retaining the existing cross-Unit
`spoken_slot_eos -> speak` BPE test. The next cctl run must use commit
`975a36e` and the Chenjinpeng `000747` token-only case; the previous tasks
`781300` and `781304` remain invalid because they ran the pre-warmup-fix
commit.

## FC checkpoint all-off replay (task 781690, 2026-08-25)

The corrected FC Board replay was run with the official FC model-path assets,
the Chenjinpeng iter-500 checkpoint, and the materialized `000747` case:

- model path: `/user/weihongliang/fc_align_enhance_fc_modelpaths/official-o5-fc-with-local-assets`
- checkpoint: `/user/chenjinpeng/training/o5-duplex-fc-ce/inference/iter_0000500_seed0/iter_0000500_o5.pt`
- mode: `single_eager`, one A100, `sdpa`, greedy, seed `0`, TTS argmax
- all deployment acceleration flags disabled, token-only FC replay
- code: `81a9aafa9ce6dd927065dbae077a6e87fc20ae69`
- artifact: `/user/weihongliang/fc_align_enhance_fc_runs/fc-000747-official-modelpath-alloff-20260825-r10`

The service loaded successfully and entered real FC generation. It reached
Unit 3, then the checkpoint emitted ordinary token `98258` before a valid FC
stream opener. The FC runtime correctly rejected it with:

```text
FC non_spoken ordinary token arrived before stream opener: 98258
```

This is the same checkpoint/protocol failure class previously seen with the
other FC checkpoint, and it occurs with one-card all-off inference. It is
therefore not evidence that TP2, LLM Graph, batched MM, or TTS acceleration
caused this failure. The all-off run did not produce a completed FC trace, so
it cannot be used as a completed token-alignment baseline for this case.

## FC full-acceleration TP2 replay (task 781708, 2026-08-25)

The same case, model path, checkpoint, deterministic settings, and token-only
probe were then started with two A100s and every deployment optimization
enabled:

```text
mode=tp2, O5_LLM_CACHE=32768
experts=batched_mm
llm_graph=1, tts_graph=1, vocoder_graph=1, tts_fast=1
lmhead=1, fuse_vision_audio=1, vision_batch=1
```

Both ranks loaded and captured the LLM and vocoder graphs successfully. The
first FC request then failed before producing a unit because `batched_mm`
prefill attempted to allocate `15.26 GiB` while each A100 had only about
`6.0 GiB` free (`~73.1 GiB` already in use). The task was `781708`; artifact:

`/user/weihongliang/fc_align_enhance_fc_runs/fc-000747-tp2-fullaccel-20260825-r1`

This is a capacity failure of the requested 32K StaticCache plus all graph
and batched-MoE resources for this FC load shape. It is independent of the
`98258` protocol failure observed by the one-card all-off run. A reduced-cache
full-acceleration run is needed to test the FC runtime and replay behavior
without changing the model or acceleration flags; that run is diagnostic only
and does not establish 32K serving capacity.

## FC replay SDK compatibility failure (tasks 781977, 2026-08-25)

The first attempt to rerun the FC replay with the checkpoint's documented
inference environment was task `781977`. It failed before model loading because
the launcher still used the stale default model path
`/user/weihongliang/MiniCP-o-4_6`, which does not exist in the deployment image.
That was a launcher configuration error, not a model or runtime result.

The corrected attempts used:

- code: `a2fdbb9` on branch `align-enhance-fc`
- SDK/runtime: `/user/chenjinpeng/.venvs/o5-fc-infer` (`minicpm-o5-sdk==0.0.5a1`)
- model path: `/user/weihongliang/fc_align_enhance_fc_modelpaths/official-o5-fc-with-local-assets`
- checkpoint: `/user/chenjinpeng/training/o5-duplex-fc-ce/inference/iter_0000500_seed0/iter_0000500_o5.pt`
- O5 TP2 backbone: `/user/weihongliang/o5_backbones/job616069_iter0000500_hf`

Both corrected tasks loaded the model and reached the first FC websocket
request, but both then closed with:

```text
'O5UnitPolicy' object has no attribute 'validate_execution_capacity'
```

This is an FC runtime/SDK API mismatch. The semantic-v2 runtime introduced in
`c08aeea` called a method from the newer SDK contract, while the documented
Chenjinpeng venv's `0.0.5a1` exposes `validate_unit_policy()` instead. It is
independent of TP2 and acceleration: it reproduced in both all-off task
`782011` (single card) and all-acceleration task `782016` (two cards).

The runtime compatibility fix is commit `a89708f`:

- prefer `validate_execution_capacity()` when the installed SDK provides it;
- fall back to the existing `validate_unit_policy()` validator on SDK `0.0.5a1`;
- do not modify the shared SDK venv or the policy schema.

## Reference-runtime SDK alignment (task 782213, 2026-08-25)

The reference throughput task `715753` uses the shared project runtime:

- Python: `/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel/bin/python`
- `minicpm-o5-sdk==0.0.5`
- `torch==2.8.0+cu126`
- `transformers==5.5.4`

The installed environment was verified directly. It exposes both
`O5UnitPolicy.validate_execution_capacity()` and
`MiniCPMO5Tokenizer.is_ordinary_token_id()`. The previous FC replay attempts
used `/user/chenjinpeng/.venvs/o5-fc-infer` with SDK `0.0.5a1`, so they were not
runtime-comparable to task `715753`.

Task `782213` reran the same FC all-off replay at commit `a89708f` with the
reference `.venv-accel`, `single_eager`, SDPA, all acceleration flags disabled,
and the same checkpoint/model-path/case/reference as the preceding attempts.
The model loaded successfully and neither of the old SDK compatibility errors
occurred. The run then stopped during replay forcing with:

```text
RuntimeError: reference exhausted: kind=llm.decode input_id=unit_000 index=2
```

The reference contains two `llm.decode` events for `unit_000`, while this
runtime requested three. Its manifest reports 62 units and 327 LLM decode
events. This is a reference/case scheduling mismatch and is not an SDK-load or
model-capacity result; task `782213` must not be used as an all-off alignment
verdict until the canonical reference is regenerated from the same execution
contract.

Validation before rerun: `34 passed` across FC runtime and checkpoint tests,
Python compilation and `git diff --check` passed. The two failed tasks produced
no alignment trace and are not evidence about FC or acceleration numerical
alignment. They must be rerun at commit `a89708f`.

## No-FC full acceleration with LLM teacher forcing (task 782343, 2026-08-25)

This is the first valid no-FC acceleration gate after the single-card control
and the TP2 all-off drift run. It used the reference task's model assets and
runtime:

- code: `a89708f` on branch `align-enhance-fc`;
- pool: `deploy`, two A100 ranks, target `demo-tp2`;
- Python: `/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel/bin/python`;
- SDK: `minicpm-o5-sdk==0.0.5`, PyTorch `2.8.0+cu126`;
- checkpoint: `/user/weihongliang/o5_weights/chenmoye_minicpm_5o_moe_omni_long_context_sft_stage2_sft2_8k_audio_online_process_on_online_audio_process_v2_iter_100.pt`;
- backbone: `/user/weihongliang/o5_weights/o5_backbone_hf_chenmoye_minicpm_5o_moe_omni_long_context_sft_stage2_sft2_8k_audio_online_process_on_online_audio_process_v2_iter_100`;
- input: `/user/weihongliang/o5_replay_strategy_hd_input_8u_20260822_canonical_hd`;
- canonical reference: `/user/weihongliang/o5_align_enhance_fc_exp_20260824/canonical-fixedfla-samegpu-c`.

Acceleration settings were enabled for TP2, LLM Graph, TTS Graph, TTS fast,
LM head, vision fusion, vision batching, and `batched_mm`. `vocoder_graph` was
explicitly disabled. Attention was eager; FLA remained fixed at `32x8`,
`16x8`, and `128x128x8`; cache length was 32768; seed was `0`; decoding was
greedy and deterministic. The replay used `--forcing llm`: LLM decisions came
from the canonical reference, while TTS autoregression remained free. This
isolates TTS sensitivity from the already-known TP2 LLM continuous drift.

Artifact:

`/user/weihongliang/o5_align_enhance_fc_exp_20260824/demo-tp2-fullaccel-no-vocoder-llmforce-a89708f-ref715753`

Comparison:

`/user/weihongliang/o5_align_enhance_fc_exp_20260824/canonical-vs-demo-tp2-fullaccel-no-vocoder-llmforce.json`

The run completed all 8 source units and produced 221 trace events. The
canonical and demo traces share 213 events because the accelerated run has
eight extra events after the TTS trajectory diverges. Results:

- LLM accepted token lists: `8/8` equal.
- LLM selected tokens and local argmax decisions: `17/17` equal.
- LLM logits are numerically different, as expected for TP2 plus accelerated
  execution, but have zero decode argmax reversals.
- TTS conditions are close but not bitwise equal (`cosine_mean=0.999987`).
- TTS forward-logit argmax: `43/45` equal, `2` reversals.
- TTS sampled token IDs: `44/45` equal.
- The first sampled-token divergence is `unit_000006`, TTS step `14`:
  canonical selected `6561`, while TP2 selected `5838`.
- The divergence adds four TTS tokens to that chunk, changing its Token2Wav
  input range from `[25,45]` to `[25,49]` and its output from `23040` to
  `26880` samples.
- Six of seven common per-unit WAV files are shape-compatible; five are
  bitwise equal. The combined duration changes from `167040` to `170880`
  samples because of the extra TTS tokens.

This passes the intended practical no-FC gate for the current workload: all
LLM decisions remain aligned and the full acceleration profile without
`vocoder_graph` has only two TTS argmax reversals out of 45 positions. It does
not claim bitwise TTS or PCM equality. The remaining TTS difference is a
near-boundary numerical sensitivity of the TP2/graph path, and is the correct
baseline to carry into the FC comparison. `vocoder_graph` remains excluded
from this conclusion as requested.

## FC short-board all-off record (task 783188, 2026-08-25)

To establish an FC-side reference with the same short case used by reference
job `715753`, task `783188` ran this branch at commit `07ab0b7` with:

- case: `/user/weihongliang/o5_fc_short_case_715753/case.json`;
- model path: `/user/weihongliang/MiniCPM-o-4_6`;
- checkpoint: `/user/sunweiyue/checkpoints/model_tunnel_moe_deploy/family_001_730-sft2-agent-duplex/treatment_004_vision12-agent6-tau3-ui2-mvp1-steps4000-decay2000/runs/job_672317/checkpoints/iter_0004000/model.pt`;
- TP2 backbone: `/user/sunweiyue/checkpoints/model_tunnel_moe_deploy/family_001_730-sft2-agent-duplex/treatment_004_vision12-agent6-tau3-ui2-mvp1-steps4000-decay2000/runs/job_672317/checkpoints/iter_0004000/backbone`;
- runtime: `/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel/bin/python`;
- one A100, `single_eager`, SDPA, greedy/seed 0/TTS argmax, all deployment
  acceleration flags disabled, 32K cache;
- artifact: `/user/weihongliang/fc_align_enhance_fc_runs/fc-715753-alloff-record-07ab0b7-20260825`;
- trace: `/user/weihongliang/fc_align_enhance_fc_runs/fc-715753-alloff-record-07ab0b7-20260825/trace_sessions/sess_5cb7855e559a`.

The service loaded successfully and completed all `46/46` units with `235`
API trace events. It did not emit the case's expected
`display_object_on_board` tool call, so the ground-truth tool response was not
sent. This is not a service failure, but this artifact is not a complete
tool-call behavior reference. It remains a valid deterministic all-off FC
trace for the generated path before any tool-call branch; the difference
between this result and the existing TP2 free run, which did emit a tool-call
span, is itself recorded as a discrete behavior difference rather than being
attributed to runtime failure.

## FC TP2 full acceleration replay (task 783244, 2026-08-25)

Task `783244` was submitted in `deploy` with the same short case, checkpoint,
backbone, model path, runtime, seed, and 32K cache. It enables TP2,
`batched_mm`, LLM Graph, TTS Graph, TTS fast, LM head, vision/audio fusion,
and vision batching; `vocoder_graph` is explicitly disabled. The replay uses
the all-off trace above with `O5_REPLAY_FORCING=all`, so the comparison is
teacher-forced at every trace stage while still executing the accelerated
model's forward path.

## FC TP2 full acceleration replay, complete trace (task 783453, 2026-08-25)

Task `783453` is the valid replacement for the earlier incomplete FC replay
task `783244`. It ran commit `6755d5d` (`fix(fc): flush model traces at
protocol unit boundaries`) in the `deploy` pool with the same 46-unit short
board case and FC checkpoint as the all-off record. The runtime and assets
were:

- case: `/user/weihongliang/o5_fc_short_case_715753/case.json`;
- model path: `/user/weihongliang/MiniCPM-o-4_6`;
- checkpoint: `/user/sunweiyue/checkpoints/model_tunnel_moe_deploy/family_001_730-sft2-agent-duplex/treatment_004_vision12-agent6-tau3-ui2-mvp1-steps4000-decay2000/runs/job_672317/checkpoints/iter_0004000/model.pt`;
- TP2 backbone: `/user/sunweiyue/checkpoints/model_tunnel_moe_deploy/family_001_730-sft2-agent-duplex/treatment_004_vision12-agent6-tau3-ui2-mvp1-steps4000-decay2000/runs/job_672317/checkpoints/iter_0004000/backbone`;
- Python: `/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel/bin/python`;
- deployment: `tp2`, 32K LLM cache, seed `0`, greedy/TTS argmax;
- acceleration: TP2, `batched_mm`, LLM Graph, TTS Graph, TTS fast, LM head,
  vision/audio fusion, and vision batching enabled; `vocoder_graph` disabled;
- forcing: `O5_REPLAY_FORCING=all`.

Both sides completed the full trace with identical event structure:

- all-off trace: `/user/weihongliang/fc_align_enhance_fc_runs/fc-715753-alloff-record-6755d5d-20260825/trace_sessions/sess_ec255bc2e4c6`;
- TP2 trace: `/user/weihongliang/fc_align_enhance_fc_runs/fc-715753-tp2-fullaccel-forceall-6755d5d-20260825/trace_sessions/sess_986dbc8d0b03`;
- comparison: `/user/weihongliang/fc_align_enhance_fc_runs/fc-715753-fc-alloff-vs-tp2-fullaccel-forceall-compare-6755d5d.json`.

Each trace contains `1405` events and `2664` tensor sidecars; both manifests
have `completed=true`. The alignment results are:

- LLM accepted tokens: `184/184` equal;
- selected LLM tokens under forcing: `207/207` equal;
- TTS condition input IDs: `7/7` equal;
- TTS sampled token IDs under forcing: `182/182` equal;
- Token2Wav input IDs and all input/output ranges: `8/8` equal;
- free local LLM argmax: `204/207` equal, `3` reversals;
- free TTS forward-logit argmax: `181/182` equal, `1` reversal;
- TTS conditions are numerically close but not bitwise equal
  (`cosine_mean=0.9999988`);
- TTS forward logits are also close (`cosine_mean=0.9999815`,
  `tv_mean=0.00867`) but not bitwise equal;
- LLM decode logits show the larger TP2/accelerated numerical drift
  (`cosine_mean=0.99646`, `tv_mean=0.01110`), while the forced token path
  remains identical.

This confirms that FC can be teacher-forced and compared stage by stage on
the current short board case. It does **not** claim free-running FC output is
bitwise identical: the accelerated path changes continuous hidden/logit
values and has a small number of local argmax reversals. Because forcing is
enabled at every stage, those reversals do not alter this run's TTS tokens or
Token2Wav inputs. The next FC experiment should use the same case and setting
with either no forcing or only LLM forcing to measure actual trajectory
reversals; `vocoder_graph` remains outside this conclusion.

## FC TP2 full acceleration with LLM forcing (task 783590, 2026-08-25)

Task `783590` repeats the same short board case and acceleration profile, but
changes the forcing policy to `O5_REPLAY_FORCING=llm`. The canonical/all-off
LLM token trajectory is fixed; TTS autoregression, Token2Wav, and vocoder are
free to run. This is the useful FC comparison for actual TTS trajectory
reversals. The task used commit `c3d4983` (the report commit after the
trace-flush fix), the `deploy` pool, two A100s, the reference `.venv-accel`,
the same FC checkpoint/backbone/model path, and the same seed/greedy/32K
cache configuration as task `783453`.

Artifacts:

- output: `/user/weihongliang/fc_align_enhance_fc_runs/fc-715753-tp2-fullaccel-llmforce-6755d5d-corrected-20260825`;
- trace: `/user/weihongliang/fc_align_enhance_fc_runs/fc-715753-tp2-fullaccel-llmforce-6755d5d-corrected-20260825/trace_sessions/sess_940485583c84`;
- comparison: `/user/weihongliang/fc_align_enhance_fc_runs/fc-715753-fc-alloff-vs-tp2-fullaccel-llmforce-6755d5d.json`.

The task completed all `46/46` units. The candidate and all-off reference
again have identical trace structure: `1405` events and `2664` tensor
sidecars. Results:

- LLM accepted tokens: `184/184` equal;
- selected LLM tokens: `207/207` equal;
- LLM local argmax: `206/207` equal, one unused local-argmax reversal at
  `unit_013`, decode ordinal `11`;
- TTS condition source token IDs and end-of-turn flags: `7/7` equal;
- TTS forward-logit argmax: `178/182` equal, `4` reversals;
- TTS sampled token IDs: `179/182` equal, `3` actual reversals;
- TTS chunks: `5/7` token lists equal; the differing chunks are `unit_012`
  and `unit_014`;
- Token2Wav input token lists: `5/8` equal, with differences only caused by
  those three TTS token substitutions; all eight input/committed/lookahead
  and output ranges remain equal;
- TTS condition cosine mean: `0.9999987`;
- TTS forward-logit cosine mean: `0.999507`, total variation mean `0.02682`;
- LLM decode-logit cosine mean: `0.996331`, total variation mean `0.01277`;
- vocoder graph remained disabled, as required.

The three actual TTS substitutions are:

```text
unit_012, TTS step 13: 1645 -> 1564
unit_012, TTS step 14: 5650 -> 5651
unit_014, TTS step 24: 4431 -> 4432
```

This is a small but real free-TTS divergence, not an artifact of the
all-forced replay. It confirms that FC TP2 plus the selected acceleration
profile preserves the LLM trajectory and produces only `3/182` TTS token
reversals on this short reference case. The next scope is to test whether
the same rate holds on another FC case and to separate the individual TTS
graph/fast/batched-MM contributions only if this combined result is not
sufficient for acceptance.
