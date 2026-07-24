# O5 双工离线加速对齐记录

日期：2026-07-24

代码定位：

- 分支：`wt/o5-no-fc-speedup-tp2-thin-unified-2026-07-24`
- 工具提交：`2fa7076bc2be65e058cf30b07600e23b8693f50b`
- TP2 probe：随本报告当前提交提交，具体以 `git log -1` 为准
- worktree：`/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24`
- probe：`tools/o5trace/thin_duplex_video_probe.py`
- TP2 probe：`tools/o5trace/tp2_duplex_video_probe.py`

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

## TP2 双卡安全子集

新增 TP2 backend mirror probe 后，按单卡 strict-alignment 的安全子集测试：

```text
deployment_mode = tp2
call_path = backend
o5_llm_cache = 32768
experts_implementation = eager
llm_graph = off
tts_graph = off
batch_vision_feed = off
tts_fast/lmhead/fuse_vision_audio = on
decode_mode = greedy
tts_argmax = on
```

8-unit 结果：

- 单卡安全子集仍和 baseline 完全一致：hash `d68ab9fb5695b62e6815493e214f9654c783d307612327e6d39e370423920a36`。
- TP2 backend mirror 可运行，32K StaticCache 下没有 OOM。
- TP2 backend mirror 两次自洽复现：
  - `tp2_backend_safe_video01_max8_20260724_01`
  - `tp2_backend_safe_video01_max8_20260724_02`
  - 两次 hash 都是 `4f6a86fbbff18f6bee25a9284fba209719432c0bbcf997830c5001707da182fd`。
- TP2 文本和单卡一致：`好的，现在电梯已经到 20层了，还有 4层`。
- TP2 没有和单卡 strict bitwise 对齐：first diff 在 unit 5 音频 hash。
- trace 显示差异已经发生在生成的 TTS token，不是只发生在 waveform/vocoder 末端。

unit 5 的第一段 TTS token diff：

```text
single-card: ..., 1517, 1735, 2031, 4299, 4299, ..., 5485, ..., 4382, ...
tp2       : ..., 4492, 3972, 4299, 6486, 6486, ..., 5486, ..., 2195, ...
```

补充观察：

- `tp2_llm` 和 `tp2 backend mirror` 的 generated TTS token / token2wav stream input 一致，说明 backend mirror 本身没有额外改变 TTS token 序列。
- 但两者 waveform hash 仍不同，说明在 token2wav/vocoder 状态或数值路径上还存在 bitwise 差异。

结论：TP2 路径已经证明 32K cache + 安全开关可以跑通且自身可复现，因此此前单卡 OOM 的配置有继续在双卡上实验的价值；但当前不能声称 TP2 和单卡 strict-alignment 已通过。下一步如果继续推进双卡，需要优先解释 TP LLM hidden state / TTS conditioning 的数值差异如何影响 TTS token。

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
