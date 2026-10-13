# Demo external HF acceptance (2026-10-13)

Base: codehub/o5-fc-dev, 732b075841f837d67d9f512835cbf1aec174911e.
Environment: /user/weihongliang/environments/o5-3drope-fc-20261013/.venv.
HF: /user/weihongliang/worktrees/minicpm-o-5-fix-3d-rope-o5fc-20261013.

## Source analysis

Default single_eager originally bypassed core.deploy and imported vendored code;
mode switching imported the vendored enum separately. Both now select external
HF through the retained facade when O5_CANONICAL_PATH is set. Config and
processor retain the existing model_path asset mapping. Full PT state is read
directly; missing weights fail explicitly, with only llm.mtp training heads
excluded. FP32 model buffers retain their precision.

The ordinary facade methods derive from the fixed Demo base. FC implementation
is attached from the external HF package. Flatten wrappers accept arbitrary
positional/keyword arguments and preserve API v4 options and audio fields.
HF exceptions convert explicitly to Demo exceptions with diagnostic fields.
No processor registration shim remains; HF owns its registration.

External tp2/tp2_llm use one process and accel TP/EP. They never construct
DistributedTPLLM or enter the legacy graph worker loop. Startup scripts select
python -m py_backend.server and need no extracted backbone. Resize memory staging
comes from accel. Ordinary and FC continue to share the existing session runtime.

## CPU checks

- External facade MRO resolves _demo_external_o5.modeling_minicpmo.
- After importing facade and constructing config, modeling.o5 imports: [].
- Synthetic FC flatten forwarding preserves max_tokens, tts_pad_protocol and
  decoding_policy unchanged.
- HF ConstrainedDecodingError conversion preserves context and is caught by
  the existing Demo exception class.
- Checkpoint wrapped state_dict unwrap: PASS.
- External tp2 registry world_size=1, requires_spmd=False; WORLD_SIZE=2 fails
  before CUDA work with explicit launch-without-torchrun message.
- bash -n both modified scripts: PASS.
- launch_tp2.sh with O5_CANONICAL_PATH=/external and PYTHON=/bin/echo outputs
  '-m py_backend.server --port 1234' without requiring O5_BACKBONE_DIR.
- py_compile four changed Python modules and git diff --check: PASS.

## Remaining validation

No GPU or service was started by this agent. Root owns checkpoint loading,
ordinary/FC generation, API v4 audio, accelerator precision and TP/EP GPU checks.
Vendored modeling stays available for deployments without O5_CANONICAL_PATH.
