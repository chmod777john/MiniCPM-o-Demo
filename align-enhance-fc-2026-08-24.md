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
- TP2 merge not yet completed.
- No GPU experiment has been started from this worktree.
