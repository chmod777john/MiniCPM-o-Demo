# O5 API vs Canonical Alignment

Date: 2026-07-26

Code under test:

- Branch: `wt/o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26`
- Base commit: `c0bb591cc9a37b465548b2ae13ba2f63e579873d`
- Report commit: `TBD`
- API probe: `tools/o5trace/api_canonical_video_probe.py`
- Compare tool: `tools/o5trace/compare_canonical_units.py`
- cctl runner: `scripts/run_api_canonical_align_job.sh`

## Question

The offline thin-unified path had already been aligned with the canonical
vendored/raw path. This run checks whether the realtime API video duplex path
can reproduce the same unit-level output when using the same media, model path,
checkpoint, seed, and deterministic TTS sampling policy.

## Inputs

- Video: `/user/weihongliang/omni_demo_duplex_01.mp4`
- Model path: `/user/weihongliang/MiniCPM-o-4_6`
- Checkpoint: `/user/weihongliang/o5_weights/houyueran_o5_MB_omni-sft2-8k-hyr_a2_i2_0722_iter_0002800.pt`
- Prompt wav: `/user/weihongliang/MiniCPM-o-4_6/assets/audio_cases/paimon__system_ref_audio.wav`
- Units: first 8 one-second units

Media parity was checked directly:

- `input_16k.wav` sha256: `8aa235fa30ae173c56e7f03535f5b5e1f47f1c2e81596cbea6b56dbecd8726ad`
- The first 8 extracted JPEG frame hashes match between offline and API runs.

## Deterministic Eval Settings

The alignment run uses an explicit deterministic eval mode:

- `seed = 0`
- `O5_STARTUP_SEED = 0`
- `O5_TTS_ARGMAX = 1`
- `TTS_N_TIMESTEPS = 5`
- text decode mode: `greedy`
- `O5_ATTN_IMPLEMENTATION = sdpa`
- `O5_PRELOAD_BOTH_TTS = 0`

`O5_TTS_ARGMAX` is only enabled by the alignment runner. It is not a production
default; it exists to make the discrete TTS token path comparable with the
offline `--tts-argmax` canonical probe.

## Runs

Offline current canonical argmax run:

```text
/user/weihongliang/thin_unified_runs/unified_trace_current_video01_max8_ttsargmax_20260726_02
```

API argmax run:

```text
/user/weihongliang/o5_alignment_runs/api_vs_current_noaccel_video01_max8_ttsargmax_20260726_11
```

Compare result:

```json
{
  "equal": true,
  "num_a": 8,
  "num_b": 8,
  "first_diff": null,
  "diff_count": 0,
  "text_a": "好的，现在电梯已经到 20层了，还有 4层",
  "text_b": "好的，现在电梯已经到 20层了，还有 4层"
}
```

## Token Trace

The generated TTS token chunks match the offline argmax canonical trace:

| unit | tts token hash |
| --- | --- |
| 4 | `96be99f01f94bc8f881c78205db42b4fced15de1b70b5dd2ff22551a9d5b5f64` |
| 5 | `413bb9c9cb28676e0738cc638d61909ac8407a0334963fb562427308cf17c2f4` |
| 6 | `654207113501ae538cd3b488c73a17878a266ffef727279591b87f890e1da5c0` |
| 7 | `d53e5ae7688dc62210c249515cc28483f37303f539f58d2dd0ca13c832e8200f` |

The token slices entering `token2wav.stream()` also match:

| unit | token2wav input hash |
| --- | --- |
| 5 | `0821099ea18a66f86b11611aaa4a6ac041b46309b2fab3a5a3766fae36cfd9ed` |
| 6 | `09ebdc089a4f52a2d74c6cedcadf8aa2342086b64bdb53a75864b6e0f7877831` |
| 7 | `27616ee87ba02ac6dd9d03edd5f8b71958b6c6e6fa39b6dd235aee4e8e8241a4` |

## Conclusion

For the first 8 units of `omni_demo_duplex_01.mp4`, realtime API video duplex
and offline canonical thin-unified are exactly aligned under deterministic TTS
argmax evaluation:

- unit-level canonical fields match
- text matches
- audio hashes match
- generated TTS tokens match
- tokens entering `token2wav` match

The earlier mismatch was not caused by media slicing or API transport. The
difference came from TTS sampling randomness: the historical canonical run used
deterministic argmax-style TTS sampling, while the API path was initially using
normal stochastic `torch.multinomial` sampling.

Remaining limitation: this report proves the deterministic first-8-unit case.
A full 36-unit API run should be added if we need the same evidence over the
entire video.
