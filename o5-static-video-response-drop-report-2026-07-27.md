# O5 静态视频双工输出异常记录

日期：2026-07-27

## 代码状态

Demo / API worktree：

- 路径：`/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26-session-replay-canonical-2026-07-26`
- 分支：`wt/o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26-session-replay-canonical-2026-07-26`
- HEAD commit：`e54a1e6084352fe6be4da8a2320ccb7b25bba275`
- 备注：当前 worktree 还有未提交的 replay/probe/server 相关改动；本报告记录的是该工作区当前状态下的运行现象。

8040 服务使用的模型代码：

- `MODEL_PATH=/user/weihongliang/MiniCPM-o-4_6`
- 分支：`whl/transformers-513-processing-compat`
- HEAD commit：`de378d98cec8c220cf9bc02ab840ef8559d95815`
- 备注：该目录有 `processing_minicpmo.py` 未提交修改。

canonical 使用的裸推理模型代码：

- 路径：`/user/weihongliang/MiniCPM-o-4_6-modelbest-moe-35b-a3b_debug-session-replay-canonical-2026-07-26`
- 分支：`wt/moe-35b-a3b_debug-session-replay-canonical-2026-07-26`
- HEAD commit：`1acad597022003c0df2690331bb11d665dffb849`

## 权重和配置

所有本次实验都使用同一个 2800 checkpoint：

```text
/user/weihongliang/o5_weights/houyueran_o5_MB_omni-sft2-8k-hyr_a2_i2_0722_iter_0002800.pt
```

8040 服务额外使用从同一 checkpoint 抽出的 LLM safetensors backbone：

```text
/user/weihongliang/o5_weights/o5_backbone_hf_houyueran_o5_MB_omni-sft2-8k-hyr_a2_i2_0722_iter_0002800
```

公共推理设置：

- `decode_mode=greedy`
- `force_listen_count=0`
- `max_new_speak_tokens_per_chunk=20`
- `n_timesteps=5`
- canonical：`--tts-argmax`
- 8040 服务：`O5_DETERMINISTIC_REPLAY=1`，`O5_SESSION_SEED=0`，`O5_TTS_ARGMAX=1`
- 参考音频：`/user/weihongliang/MiniCPM-o-4_6/assets/audio_cases/paimon__system_ref_audio.wav`

## 运行任务

8040 服务：

- cctl task：`623957`
- 状态：`Running`
- URL：`https://82.157.64.212:8040/omni`

case003：

- 视频：`/backup/user/wenyuyang/omni_demo/omni_demo_collect_202607/omni_static_demo_collect_202607/omni_static_videos/omni_demo_static_long_003.mp4`
- 视频时长：约 `350.9s`
- 8040 API 任务：`626450`
- canonical 任务：`626447(seed0)`，`626448(seed1)`，`626449(seed2)`

case002：

- 视频：`/backup/user/wenyuyang/omni_demo/omni_demo_collect_202607/omni_static_demo_collect_202607/omni_static_videos/omni_demo_static_long_002.mp4`
- 视频时长：约 `334.4s`
- 本次只跑前 `200` 个 1s unit
- 8040 API 任务：`626675`
- canonical 任务：`626671(seed0)`，`626672(seed1)`，`626674(seed2)`

## 现象概述

这两个 case 的视频内容中后段仍持续有提问/对话，但模型输出在较早时间点后变成长期 `listen`，不再继续说话。

该现象同时出现在：

- 8040 TP2 API 路径
- houyueran/debug canonical 裸推理路径

因此它不是只由 8040 websocket/API 包装层导致的现象；canonical 路径也复现。

所有已检查输出里均未出现 `think` / `<think>` / `</think>`。

## case003 结果

case003 跑完整视频，共 `351` 个 unit。

| 路径 | 任务 | 文本输出 | 最后 text unit | 最后 audio unit | 后续现象 |
| --- | --- | --- | ---: | ---: | --- |
| 8040 API | `626450` | `好的，没问题。` | 32 | 33 | 之后基本全是 listen |
| canonical seed0 | `626447` | `好的，我会全程帮你记录的。` | 34 | 34 | 之后基本全是 listen |
| canonical seed1 | `626448` | `好的，我会帮你统计的。` | 33 | 34 | 之后基本全是 listen |
| canonical seed2 | `626449` | `好的，没问题。` | 32 | 33 | 之后基本全是 listen |

输出目录：

```text
/user/weihongliang/session_replay_runs/omni_demo_static_long_003_8040_api_20260727_075753
/user/weihongliang/session_replay_runs/omni_demo_static_long_003_canonical_seed0_20260727_075753
/user/weihongliang/session_replay_runs/omni_demo_static_long_003_canonical_seed1_20260727_075753
/user/weihongliang/session_replay_runs/omni_demo_static_long_003_canonical_seed2_20260727_075753
```

## case002 结果

case002 跑前 `200` 个 unit。8040 和 canonical 都在 `unit_088/089` 附近停止说话；从约 `90s` 到 `200s` 基本全是 `listen`。

| 路径 | 任务 | units | text events | audio events | listen events | 最后 text unit | 最后 audio unit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8040 API | `626675` | 200 | 60 | 62 | 137 | 88 | 89 |
| canonical seed0 | `626671` | 200 | 59 | 62 | 137 | 88 | 89 |
| canonical seed1 | `626672` | 200 | 59 | 62 | 137 | 88 | 89 |
| canonical seed2 | `626674` | 200 | 59 | 62 | 137 | 88 | 89 |

8040 文本：

```text
我知道，是《武林外传》。是第10集，佟湘玉被白展堂点了穴。好的，没问题。现在出现的是金钏，她是同福客栈的跑堂伙计，性格活泼爱八卦。金钏在帮佟湘玉梳头，还给她讲外面的趣事。好的，金钏正在给佟湘玉梳头，佟湘玉说她和白展堂要去怡红院。金钏继续帮她梳头，佟湘玉说等白展堂醒了就告诉他。金钏说让佟湘玉去东小院。现在出现的是郭芙蓉，她跟金钏说话。金钏让郭芙蓉把佟湘玉叫来。郭芙蓉转身离开。现在画面中出现的是佟湘玉。接着出现的是王剑，她是佟湘玉的妹妹。王剑走到佟湘玉面前。佟湘玉让王剑把郭芙蓉带出去。
```

canonical 三个 seed 的文本完全一致，SHA256：

```text
6678e42c75132eea9865be5d93809037007dcba6bc419bb715e14bee442ad9c6
```

canonical 文本：

```text
我知道，是《武林外传》。是第10集，名为“同福客栈遇贵人”。好的，没问题。现在出现的是金钏，她是同福客栈的丫鬟，性格活泼爱八卦，和佟湘玉关系很好。金钏正在给佟湘玉梳头。好的，没问题。现在金钏正在给佟湘玉梳头，同时跟她说自己明天要和太师一起去怡红院。然后她让佟湘玉等太师醒了之后就告诉太师。现在画面切换了，出现了一个女人，她是太师的母亲。她正在训斥金钏。金钏被太师母亲打了一巴掌。现在出现的是王金，她是同福客栈的掌柜，为人精明能干，是客栈的核心人物。她正在和太师母亲说话。
```

输出目录：

```text
/user/weihongliang/session_replay_runs/omni_demo_static_long_002_8040_api_max200_20260727_084254
/user/weihongliang/session_replay_runs/omni_demo_static_long_002_canonical_seed0_max200_20260727_084254
/user/weihongliang/session_replay_runs/omni_demo_static_long_002_canonical_seed1_max200_20260727_084254
/user/weihongliang/session_replay_runs/omni_demo_static_long_002_canonical_seed2_max200_20260727_084254
```

## 初步判断

1. 两个 case 的关键异常不是“输出音频生成失败”，而是模型在某个时间点后持续判定为 `listen`，没有继续产生 text/audio。
2. case002 中 8040 与 canonical 的停止位置高度一致：最后 text 都在 `unit_088`，最后 audio 都在 `unit_089`，随后 `unit_090-199` 基本全 listen。
3. case003 中停止更早：只在 `unit_031-034` 附近短暂输出，后续长期 listen。
4. canonical 路径也复现，说明问题更可能在模型/duplex 判定逻辑/输入构造或权重行为层面，而不是单纯的 8040 API 服务层。
5. case002 canonical 三个 seed 文本完全一致，主要原因是本次使用 `decode_mode=greedy` 和 `--tts-argmax`，文本 token 选择路径基本确定。
