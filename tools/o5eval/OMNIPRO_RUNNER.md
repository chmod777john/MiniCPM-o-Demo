# OmniPro Demo Runner

This worktree adds a Demo-side inference adapter for HumanEvalKit's
`omnipro_online` benchmark. The benchmark data adapter and scorer remain in
HumanEvalKit; the Demo runner calls `core.deploy` directly under TP2.

The default contract is:

- one input video chunk per second;
- full-duration 1fps processing, including videos longer than 64 seconds;
- `max_new_speak_tokens_per_chunk=1024`;
- sampling enabled with seed `0` unless overridden;
- TTS/audio generation disabled, because OmniPro's primary score uses text and
  response timing;
- four independent TP2 workers when using the 8-GPU launcher.

Inference and judging are separate. Each sample writes a `result.json` with the
canonical sample, resolved request, timestamped `DuplexOutput`, engine flags,
sampling values, and seed. After inference, use HumanEvalKit's
`scripts/rejudge_from_results.py` with `--benchmark omnipro_online` and
`gemini-3-flash-preview` to perform the remote content judging.

Example:

```bash
CHECKPOINT_PATH=/path/to/iter_4000.pt \
BACKBONE_DIR=/path/to/backbone \
OUT_ROOT=/user/weihongliang/omnipro_demo_tp2_eval_iter4000 \
bash scripts/run_omnipro_tp2_demo_eval_8gpu.sh
```

The launcher uses one 2-GPU torchrun per shard. It does not alter the base
`o5-no-fc-speedup-tp2` worktree.
