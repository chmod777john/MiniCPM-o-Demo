# Unified O45/O5 FC SDK 0.0.5 Validation

## Scope

This record validates branch `duplex-fc-unified-sdk005` with one shared
Semantic Realtime API v2, one `FcDuplexView`, and server-selected O45/O5
ModelAdapters.

## CPU gates

- Shared Semantic API, UnitPolicy, View, Resume and Profile tests: `72 passed`.
- MCore DCP converter, checkpoint row validation and text-only backbone config:
  `5 passed`.
- O45 and O5 modeling packages import independently.
- Formal SDK rows:
  - O45_FC: `151772`
  - O5: `248168`

## O45 real-model gate

Cybertron Job `628125` used:

- Profile: `o45_fc_board_overfit100_sdk005_step100`
- SDK: installed `minicpm-o5-sdk==0.0.5`
- Checkpoint SHA256:
  `6c5f9c733ec0ac5dcf0e674df65cd9b6f955a12381eb4d45fdeb6671b37a6e3e`
- Modeling package: `modeling.o45`

Result:

- Backend loaded from repository-owned O45 modeling.
- 42/42 Units completed.
- Public Semantic v2 history reconstructed the complete token stream.
- `full_exact=true`
- expected and actual token lengths: `1708`
- SDK parser validate: passed.
- Parser TrainingData replay round-trip remains the known Trigger limitation.

Raw result:

```text
run-logs/o45_semantic_v2_exact/
20260727T132854Z__dob_midtrain_v1_20260628_animal_seed_ct01_and_04842__
o45_fc_board_overfit100_sdk005_step100/result.json
```

O45 Full4850 TTS Profile validation also passed:

```text
profile: o45_fc_board_tts_full4850_sdk005_step400
training Job: 624224
embedding/lm_head rows: 151772
checkpoint SHA256:
d7e8c539c935eed8a64cef95b7015599ff8425795ca7a02875b978253aec750f
```

O45 Full4850 audio Job `628757` completed a 42-Unit Session:

- `errors=[]`
- two valid tool calls and two tool-result injections;
- six non-empty spoken audio events at `24000` Hz;
- generated spoken text was non-empty.

## O5 checkpoint conversion

The current Demo runtime accepts a dense PT plus a paired TP2 HF backbone.
Formal SDK 0.0.5 training checkpoints are Megatron/MCore DCP, so this branch
adds deterministic conversion tools.

MVP100 LLM+TTS:

```text
source Job: 625367
source step: 100
dense PT Job: 628076
PT rows: 248168
PT SHA256: 29eba81bd84a368997bb555c835169bcf6aa909410084c1c2ce72e13bd81365c
runtime key count: 1746
```

Full4850 LLM+TTS:

```text
source Job: 625368
source step: 400
dense PT Job: 628505
PT rows: 248168
PT SHA256: 6d2f07d0f3e5e8ee0b4203aa4116b27fe17300b82fbccf0c2eb194957c7ab884
runtime key count: 1746
TP2 backbone Job: 628529
```

The converted TTS PT has the same runtime key set as the previously validated
O5 PT except for two config-derived rotary buffers, and differs in tensor shape
only at the intentional SDK vocabulary boundary (`248168` vs historical
`248174`).

## O5 TP2/LLM Graph gate

Validated runtime facts:

- Repository-owned `modeling.o5` loaded successfully.
- Formal SDK O5 target loaded with max special token ID `248167`.
- TP2 loaded 693 LLM tensors across two A100 ranks.
- LLM CUDA Graph capture passed on both ranks.
- Rank 1 entered the SPMD graph worker loop.
- Vision, APM, TTS and vocoder were resident on GPU.
- Shared `FcDuplexView` emitted Semantic API v2 events.

Full4850 Job `628543` completed a 51-Unit Session:

- `errors=[]`
- four valid tool calls were emitted;
- four tool results were injected and consumed by later Units;
- Session closed normally;
- O5 Stateless Resume remained explicitly unavailable as planned for MVP.

The strict token-exact CLI returned nonzero because O5 Resume is intentionally
deferred, not because API execution failed.

Full4850 audio Job `628631` completed a second 42-Unit Session:

- `errors=[]`
- two valid tool calls and two tool-result injections;
- seven non-empty spoken audio events;
- six audio chunks contained `128000` base64 characters and the final chunk
  contained `158720`;
- sample rate: `24000`;
- generated spoken text was non-empty.

## User acceptance deployment

Temporary one-hour Cybertron Job `628825` is running with:

- Profile: `o5_fc_board_tts_full4850_sdk005_step400`
- runtime: O5 MoE TP2 + LLM Graph
- branch: `duplex-fc-unified-sdk005`
- public FC Board: `https://47.95.219.248:7040/fc_board`

Automated browser verification passed:

- page title and shared FC Board layout loaded;
- checkpoint Profile displayed correctly;
- listening/speaking budgets displayed as `30/15`;
- reference audio and training defaults loaded;
- public `/health`, `/fc_board` and `/workers` returned HTTP 200;
- worker registered as idle with the expected O5 Full4850 Profile.
