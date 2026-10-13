# Final retained runtime smoke, 2026-10-13

Job1210413, agent-dev A100x2 TP; service product code Demo1eeba42c
(subsequent633a8f11 docs only), HF335bfdd, accelf06e7d7 product code
(subsequent6e33931 harness/evidence only); project env SDK0.0.6/utils1.0.7.
Original allmerge checkpoint iter300.

Sequence on one process: reference-audio non-stream chat -> ordinary duplex8 ->
FC Board8 -> ordinary duplex4. Text/audio chat returned 你好。 and 48960 float32
samples at24kHz; actual canonical TTS params and reference audio override exercised.
Ordinary8: 8/8 output, no errors, retained page length_penalty=1.05 accepted
with explicit inactive warning, wall p50=324.8ms/max1138.6ms. FC8:8/8 committed;
this bounded Board session has no tools/audio, only a mode-lifecycle check;
real tool result evidence lives in20261013-fc-ckpt. Final ordinary4:4/4,
no errors, wall p50=306.8ms/max308.4ms. Backend log has no ERROR/Traceback.
No browser rendering, full evaluation score, or long-context timing claim.
Raw artifacts: `/user/weihongliang/analysis-runs/20261013-o5-three-materials/api-final/`.
