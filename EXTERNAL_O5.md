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
