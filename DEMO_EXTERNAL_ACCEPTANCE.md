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

## Follow-up startup audit

- External compile=True now fails before loading; facade compile entry also
  fails explicitly. All external acceleration belongs to accel.
- Parent launcher exports PYTHON as the selected VENV_DIR/bin/python.
- External FC profiles retain SDK 0.0.6 enforcement and allow tp2 without
  backbone_dir; profile schema/path validation and start_fc_service agree.
- CPU profile-without-backbone validation and installed SDK 0.0.6: PASS.
- SDK registry required rows: 248168. HF ordinary modeling/utils contain no
  spk_full or resize_token_embeddings call. FC prepare checks trained coverage;
  HF attach startup check was removed by the HF owner, so both modes attach.
- T2W facade reuses matching assets/float16/timestep signature. Ordinary duplex
  construction calls init_tts with matching defaults (10 timesteps); if config
  deliberately chooses another timestep, it selects a different initializer.
  FC prepare builds session prompt caches; reset restores its cloned base,
  and cleanup releases them. Ordinary reset_session clears ordinary prompt
  cache on mode transitions, preserving the existing lifecycle.
- Board default training case exists. Tauvoice default training case is absent
  on this host. EXTERNAL_O5.md includes startup environment and probe entrypoints.
- py_compile updated modules, bash -n launchers, git diff --check: PASS.
