# O5 双工离线加速对齐记录

日期：2026-07-24

代码定位：

- 分支：`wt/o5-no-fc-speedup-tp2-thin-unified-2026-07-24`
- 工具提交：`2fa7076bc2be65e058cf30b07600e23b8693f50b`
- worktree：`/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24`
- probe：`tools/o5trace/thin_duplex_video_probe.py`

## 测试口径

- 输入：`omni_demo_duplex_01.mp4`，按 1 秒切 36 个 unit。
- 解码：`seed=0`、文本 `greedy`、TTS 采样通过 `--tts-argmax` 固定为 argmax。
- 精度口径：先比较 `canonical_units.json`，包括 listen/speak、文本、文本 token 数、TTS token 数、音频 hash；必要时再比较 `token_trace.json` 中的 TTS token 和真正进入 token2wav 的 token slice。
- baseline：不开 O5 加速，thin unified 已和 vendored/raw 离线路径全量对齐。

baseline full hash：

```text
97a673b3acfd05f6fefa75be92eacf94092c32c153875cdc387c34db11f289a6
```

baseline full 文本：

```text
好的，现在电梯已经到 20层了，还有 4层就到 24层了。到 24层了，可以出电梯了。你刚才开门了，但是没有完全打开。
```

## 结论

严格 bitwise 对齐的安全加速子集：

```text
tts_fast = on
lmhead = on
fuse_vision_audio = on
tts_graph = off
vocoder_graph = off
batch_vision_feed = off
experts_implementation = eager
llm_graph = off
```

这组在完整 36-unit 视频上和 baseline 完全一致：

- 输出：`/user/weihongliang/thin_unified_runs/unified_o5opt_no_tts_graph_video01_full_20260724_01`
- hash：`97a673b3acfd05f6fefa75be92eacf94092c32c153875cdc387c34db11f289a6`

## 加速项对齐情况

| 加速项 | 结果 | 现象 |
| --- | --- | --- |
| `tts_fast + lmhead + fuse_vision_audio` | 通过 | 36-unit 完全等于 baseline。 |
| `tts_graph` | 不通过 | 8-unit 文本一致，但 unit 5 开始音频 hash 不同，unit 6 的 TTS token 数也不同。 |
| `batch_vision_feed` | 不通过 | 8-unit 文本已变化：`20层` 变成 `2 0楼`。 |
| `vocoder_graph` | token 级通过，wav bitwise 不通过 | 8-unit 完全一致；36-unit 的 TTS token 和 token2wav 输入完全一致，但 unit 13 开始音频 hash 不同。 |
| `batched_mm` | 单卡不可用 | 进入 MoE `batched_mm_experts_forward` 时 OOM，需要额外约 4.66 GiB。 |
| `llm_graph + batched_mm` | 单卡不可用 | 同样在 MoE `batched_mm_experts_forward` OOM。 |

## Trace 结果

`vocoder_graph` full trace：

- baseline：`/user/weihongliang/thin_unified_runs/unified_trace_baseline_video01_full_20260724_01`
- vocoder graph：`/user/weihongliang/thin_unified_runs/unified_trace_vocoder_no_tts_graph_video01_full_20260724_01`

对比结果：

```text
generated_tts_chunks: equal, len=13
token2wav_stream_inputs: equal, len=13
canonical wav hash: not equal, first diff unit=13
```

这说明 `vocoder_graph` 当前没有改变 TTS token，也没有改变进入 token2wav 的 token slice；差异发生在 vocoder/token2wav 内部数值路径，属于 waveform bitwise 不一致。

## 建议

如果目标是和未加速离线路径严格一致，当前只采用安全子集：`tts_fast/lmhead/fuse_vision_audio`，不要启用 `tts_graph/vocoder_graph/batch_vision_feed/batched_mm/llm_graph`。

如果目标是性能优先，可以单独考虑 `vocoder_graph`，但它需要人工听感或更高层音频指标验收，因为 token 级一致不等于 waveform bitwise 一致。

`batched_mm` 和 `llm_graph` 暂时不适合单卡 strict-alignment probe；需要在 TP2 或更省显存的 MoE 实现下继续验证。
