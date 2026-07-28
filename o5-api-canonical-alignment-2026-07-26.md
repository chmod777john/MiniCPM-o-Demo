# O5 API vs Canonical Alignment

Date: 2026-07-26

Code under test:

- Branch: `wt/o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26`
- Base commit: `c0bb591cc9a37b465548b2ae13ba2f63e579873d`
- Code commit: `04462c74cca97296b63e7c8a445fe699e0186678`
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
- Units: first 8 one-second units, then full 36 one-second units

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

Offline current canonical argmax run, first 8 units:

```text
/user/weihongliang/thin_unified_runs/unified_trace_current_video01_max8_ttsargmax_20260726_02
```

API argmax run:

```text
/user/weihongliang/o5_alignment_runs/api_vs_current_noaccel_video01_max8_ttsargmax_20260726_11
```

First-8 compare result:

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

Full-36 canonical baseline:

```text
/user/weihongliang/thin_unified_runs/unified_trace_baseline_video01_full_20260724_01
```

Full-36 API run:

```text
/user/weihongliang/o5_alignment_runs/api_vs_baseline_noaccel_video01_full36_ttsargmax_20260726_01
```

cctl task:

```text
622380
```

Full-36 compare result:

```json
{
  "equal": true,
  "num_a": 36,
  "num_b": 36,
  "first_diff": null,
  "diff_count": 0,
  "text_a": "好的，现在电梯已经到 20层了，还有 4层就到 24层了。到 24层了，可以出电梯了。你刚才开门了，但是没有完全打开。",
  "text_b": "好的，现在电梯已经到 20层了，还有 4层就到 24层了。到 24层了，可以出电梯了。你刚才开门了，但是没有完全打开。"
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

For the full-36 run, token trace comparison also passed:

- generated TTS chunks: `13 / 13`, all matching
- `token2wav.stream()` inputs: `13 / 13`, all matching

## Conclusion

For the full 36 units of `omni_demo_duplex_01.mp4`, realtime API video duplex
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

Remaining limitation: this proves deterministic eval parity. Normal stochastic
TTS sampling is still expected to vary unless the eval-only argmax path is
enabled.

## Browser Session Replay: `sess_69109c78ee45`

本节记录一次真实浏览器双卡 TP2 demo session 和 canonical 单卡离线 replay
之间的对齐情况。这个 session 不是 deterministic eval 配置下产生的，而是
8040 服务当时的默认交互配置。

### 环境和输入

- 浏览器服务：`https://82.157.64.212:8040/omni`
- cctl task：`623820`
- session id：`sess_69109c78ee45`
- session 目录：
  `/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26-session-replay-canonical-2026-07-26/data/sessions/sess_69109c78ee45`
- canonical replay 任务：`623860`
- canonical replay 输出目录：
  `/user/weihongliang/session_replay_runs/sess_69109c78ee45_canonical_20260726_1804`

session manifest 中记录的关键配置：

- `seed = 245465768`
- `force_listen_count = 3`
- `decode_mode = sampling`
- `temperature = 0.7`
- `top_k = 20`
- `top_p = 0.8`
- `tts_temperature = 0.8`
- `O5_DEPLOY_MODE = tp2_llm`
- `O5_LLM_GRAPH = 1`
- `O5_LLM_CACHE = 32768`
- `O5_ATTN_IMPLEMENTATION = auto`
- `O5_TTS_ARGMAX` 未设置

### 浏览器/API 实际输出

浏览器实际返回的 21 个 audio chunks 已按 `stream.jsonl` 中的顺序拼接：

```text
/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26-session-replay-canonical-2026-07-26/data/sessions/sess_69109c78ee45/response_audio_concat.wav
```

音频统计：

- chunks：`21`
- 时长：`22.48s`
- 采样率：`24000`
- 声道：单声道
- RMS：`0.0889`
- PCM sha256：`2310fd111da91db852515f799ae146dc009ccac090f1ddcb4d446d7348c1fe13`

浏览器/API 文本：

```text
好的，现在到 18层了，24减 18等于 6，还需要再上 6层。已经到 24层了，一共上了 6层。哎，电梯门怎么打不开呢？哦，原来是到了安全通道这里，安全通道的门是要手动打开的。你刚才开门的时候发出了吱呀的声音，应该是门轴缺润滑油了。
```

### Canonical Replay 输出

canonical replay 连续音频：

```text
/user/weihongliang/session_replay_runs/sess_69109c78ee45_canonical_20260726_1804/continuous.wav
```

音频统计：

- units：`36`
- listen units：`22`
- speak units：`14`
- 时长：`37.08s`
- 采样率：`24000`
- RMS：`0.0493`
- absmax：`0.6212`

canonical 文本：

```text
好的，现在到 18层了，24减 18等于 6，还需要再上 6层。已经到 24层了，可以下电梯了。你刚才开门了，但是没有完全打开，又关上了。
```

### 对齐结论

两边文字存在一段公共前缀，最长公共前缀为 41 个中文字符：

```text
好的，现在到 18层了，24减 18等于 6，还需要再上 6层。已经到 24层了，
```

之后开始分叉：

- 浏览器/API：`一共上了 6层。哎，电梯门怎么打不开呢？...`
- canonical：`可以下电梯了。你刚才开门了，但是没有完全打开，又关上了。`

这说明 session replay 的媒体输入、基础配置、seed 记录已经能把生成轨迹对齐到前半段，
但这次旧 session 仍然不是完全可复现。主要剩余随机源和不稳定源是：

1. 文本 decode 仍是 `sampling`，会受 `temperature/top_k/top_p` 和随机数状态影响。
2. TTS token 仍使用 stochastic `torch.multinomial`，因为服务当时没有设置 `O5_TTS_ARGMAX=1`。
3. session seed 是服务端自动分配并记录的，canonical 可以读取它，但不同运行之间不会固定成同一个 seed。
4. TP2 + graph + `auto` attention 和单卡 canonical 在数值上可能存在微小差异；在 sampling 下，这类微小差异可能被放大成不同 token 轨迹。

### 后续固定方案

为了让下一次浏览器 session 更适合做 canonical replay，对 8040 验证服务引入
deterministic replay 模式，而不改变项目的普通生产默认值：

- `O5_DETERMINISTIC_REPLAY=1`
- `O5_SESSION_SEED=0`
- `O5_TTS_ARGMAX=1`
- 当请求没有显式传 `decode_mode` 时，服务端补齐：
  - `decode_mode = greedy`
  - `temperature = 0.0`
  - `top_k = 0`
  - `top_p = 1.0`

这样下一次 `session.created.replay_manifest` 应该能看到：

- `resolved.seed = 0`
- `resolved.seed_source = deterministic_replay`
- `resolved.deterministic_replay = true`
- `resolved.duplex_config.decode_mode = greedy`
- `runtime.env.O5_TTS_ARGMAX = 1`

该配置用于 replay/eval 对齐验证，不代表要替代正常 stochastic 交互体验。

### 8040 重启验证

旧 8040 服务 task `623820` 已停止。新的 deterministic replay 服务为：

- cctl task：`623957`
- URL：`https://82.157.64.212:8040/omni`
- 启动日志目录：
  `/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26-session-replay-canonical-2026-07-26/run-logs/o5_session_replay_tp2_8040_det_20260726_183131`

启动日志确认：

```text
[tp2-start] deterministic_replay=1 session_seed=0 tts_argmax=1
```

最小 websocket 初始化检查已通过，检查 session 为 `sess_1a3ab60b6f73`。
该检查只发送 `session.init`，不发送音视频输入。`session.created.replay_manifest`
中的关键字段为：

```json
{
  "seed": 0,
  "seed_auto_assigned": false,
  "seed_source": "deterministic_replay",
  "deterministic_replay": true,
  "llm_seed": 0,
  "duplex_config": {
    "decode_mode": "greedy",
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0
  },
  "env": {
    "O5_TTS_ARGMAX": "1",
    "O5_DETERMINISTIC_REPLAY": "1",
    "O5_SESSION_SEED": "0"
  }
}
```

因此从 `623957` 开始，新浏览器 session 的主要采样随机性已经固定：

- 文本侧不再走 sampling，而是 greedy。
- TTS 离散 token 不再走 stochastic multinomial，而是 argmax。
- 未显式传 seed 的浏览器 session 使用固定 seed `0`。

仍需注意：TP2/graph/attention kernel 与单卡 canonical 之间可能还有浮点数值差异。
在 greedy/argmax 下这类差异通常不会像 sampling 那样被随机采样放大，但如果某一步
logit 极接近并列，仍理论上可能导致分支不同。

## Browser Session Replay: `sess_2972da2bf587`

本节记录 8040 重启到 deterministic replay 配置后的第一条浏览器 session replay。

### Session 配置

- session id：`sess_2972da2bf587`
- session 目录：
  `/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26-session-replay-canonical-2026-07-26/data/sessions/sess_2972da2bf587`
- 8040 服务 task：`623957`
- deploy canonical replay task：`624897`
- canonical replay 输出目录：
  `/user/weihongliang/session_replay_runs/sess_2972da2bf587_canonical_deploy_20260727_0243`

manifest 已确认 deterministic replay 配置生效：

- `seed = 0`
- `seed_source = deterministic_replay`
- `deterministic_replay = true`
- `decode_mode = greedy`
- `temperature = 0.0`
- `top_k = 0`
- `top_p = 1.0`
- `O5_TTS_ARGMAX = 1`

### 浏览器/API 实际输出

浏览器实际返回音频已按 `stream.jsonl` 顺序拼接：

```text
/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26-session-replay-canonical-2026-07-26/data/sessions/sess_2972da2bf587/response_audio_concat.wav
```

音频统计：

- chunks：`14`
- 时长：`12.68s`
- 采样率：`24000`
- RMS：`0.0753`
- absmax：`0.6673`
- PCM sha256：`14efa0a5733250814f7de9987d45f9661cc470df0e38b20d453d1e8e11c6543a`

浏览器/API 文本：

```text
好的，现在电梯在 1 7层，到 2 4层还有 7层。已经到 24层了，可以准备出电梯啦。你刚才开门了，但是门没有完全打开。
```

### Canonical Replay 输出

canonical replay 连续音频：

```text
/user/weihongliang/session_replay_runs/sess_2972da2bf587_canonical_deploy_20260727_0243/continuous.wav
```

音频统计：

- units：`36`
- listen units：`20`
- speak units：`16`
- 时长：`34.32s`
- 采样率：`24000`
- RMS：`0.0526`
- absmax：`0.7079`
- PCM sha256：`a43b424fc569373f255fc4b3faf51e3626a9c16fca7958d02c06c7fbf8148395`

canonical 文本：

```text
好的，现在电梯在 1 6层，到 2 4层还有 8层。到了到了，电梯已经到达 2 4层了。你刚才开门了，你拧动了门把手，然后推开了门。
```

### 对齐结论

两边仍未对齐，最长公共前缀只有 11 个字符：

```text
好的，现在电梯在 1
```

第一个明显差异发生在楼层判断：

- 浏览器/API：`7层，到 2 4层还有 7层`
- canonical：`6层，到 2 4层还有 8层`

这说明固定采样随机性后，浏览器 TP2/graph 路径和单卡 canonical 路径仍可能在很早的
语义判断上分叉。此时主要嫌疑不再是普通 sampling 随机性，而是：

1. TP2 + graph + `auto` attention 路径与单卡 `sdpa` canonical 的数值/执行路径不同。
2. 浏览器/API 侧当前 `token_trace.json` 没记录到 `generated_tts_chunks` 和
   `token2wav_stream_inputs`，无法直接比较进入 TTS/token2wav 的离散 token。
3. 仍需进一步确认 canonical replay 和浏览器路径是否在所有媒体输入、模型路径、
   attention backend、graph 开关、TP2 wrapper 层面完全等价。

因此，`sess_2972da2bf587` 证明：deterministic replay 配置已经生效，但还不足以让
双卡浏览器输出与单卡 canonical 完全复现。
