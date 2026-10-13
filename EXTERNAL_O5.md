# External O5 runtime

The existing Demo, API v4, workers and session mode selection are retained.
Set `O5_CANONICAL_PATH` to the HF source checkout and `O5_CHECKPOINT_PATH`
to the original full PT checkpoint. The existing `model_path` continues to
locate config/tokenizer/processor assets. A wrapped checkpoint under
`state_dict`, `model` or `module` is accepted; vocabulary rows follow the
checkpoint embedding and LM head. No weights are exported.

Optional `O5_ASSETS_DIR` selects the Token2wav assets directory (or its parent).
Optional `O5_ACCELERATION_PATH` selects an accel checkout with
`src/minicpm_o_accel`. Optimized deployment modes attach that package after
HF and FC initialization. `single_eager` remains eager. External TP/EP uses one
serving process; accel owns the two device layout (`O5_MOE_PARALLEL=tp` or `ep`).
Launch external modes without torchrun.

`core.external_o5` loads HF in one isolated package namespace. The retained
ordinary facade is derived from the fixed Demo baseline; model primitives,
processor classes and FC capabilities come from HF. FC options pass unchanged
through the flatten methods, and HF protocol errors convert explicitly to the
existing Demo exception types. Prompt caches are initialized by HF prepare.

CPU checks cover imports, facade option forwarding and exception conversion.
GPU generation, audio and TP lifecycle require the root integration checks.

## Startup parameters

Run `scripts/start_o5_tp2_cctl_service.sh` with these environment variables:

```bash
export PROJECT_DIR=/user/weihongliang/worktrees/o5-fc-dev-external-hf-20261013
export VENV_DIR=/user/weihongliang/environments/o5-3drope-fc-20261013/.venv
export O5_CANONICAL_PATH=/user/weihongliang/worktrees/minicpm-o-5-fix-3d-rope-o5fc-20261013
export O5_CHECKPOINT_PATH=/absolute/path/to/original-checkpoint.pt
export O5_ACCELERATION_PATH=/absolute/path/to/accel-checkout
export O5_ASSETS_DIR=/absolute/path/to/token2wav_causal_vocoder
export O5_MOE_PARALLEL=tp
export BACKEND_PORT=22510 WORKER_PORT=22410
export GATEWAY_PORT=8009 GATEWAY_INTERNAL_PORT=8010
export GATEWAY_HTTPS=0
bash "$PROJECT_DIR/scripts/start_o5_tp2_cctl_service.sh"
```

Use free ports chosen by the resource owner. Two GPUs must be visible. There is
one backend process; the service mode name remains tp2. MODEL_PATH defaults to
O5_CANONICAL_PATH, but can separately select config/processor assets. PT_PATH
defaults to O5_CHECKPOINT_PATH. No BACKBONE_DIR is required. PYTHON is exported
as VENV_DIR/bin/python, so the backend, worker and gateway use the same venv.
Set `O5_ACCELERATION_HARDWARE` only to an accepted accel hardware preset when
auto detection is unsuitable. Keep the application config `compile=false`;
external acceleration belongs to the accel package.

A deployment profile is optional for startup. For explicit SDK/protocol binding,
set FC_DEPLOYMENT_PROFILE to an O5 profile JSON with sdk_version 0.0.6,
deployment_mode tp2, tp_size 2, tokenizer_target o5, required_model_rows 248168,
model_path, pt_path and unit_policy (null means session owned). backbone_dir is
optional in the external mode. Reference audio is needed when requesting audio;
it can be supplied by the profile or session. FC capability attaches at startup;
trained row coverage is checked when FC prepare begins. Ordinary sessions retain
their existing mode selection.

## Existing probes

The board case exists at
`/user/weihongliang/o5_fc_assets/board_mvp_20260724/delivery_train_data/dob_midtrain_v1_20260628_animal_seed_ct01_and_04702.json`.
The default tauvoice path in `scripts/probe_fc_tauvoice_backend.py` was absent
on this host at audit time. Supply another real case explicitly.

For the board backend protocol probe:

```bash
"$VENV_DIR/bin/python" scripts/probe_fc_board_backend.py \
  --backend http://127.0.0.1:22510 --path /backend \
  --case /user/weihongliang/o5_fc_assets/board_mvp_20260724/delivery_train_data/dob_midtrain_v1_20260628_animal_seed_ct01_and_04702.json \
  --out /absolute/evidence/path/board.json
```

The public API evaluation entry is `scripts/evaluate_fc_api_from_training_data.py`
(training_data positional, --base-url, --output-dir, optional --profile-json).
Its existing Semantic client targets `/v1/realtime?mode=audio`; inspect the
selected evaluation profile's API version and audio options before claiming
API v4 coverage.
Backend board/tauvoice probes exercise the backend WebSocket protocol; public
API acceptance must also use the public evaluation entry.

## Updated acceptance and deployment

See `docs/external-hf/DEPLOYMENT.md` for concrete three-material paths, environment,
reference jobs and validation boundaries. `DEMO_EXTERNAL_ACCEPTANCE.md` records
CPU delivery and subsequent root GPU/API follow-up. Ordinary session options and
the inactive retained length_penalty control are documented in
`docs/external-hf/ORDINARY_OPTIONS.md`.
