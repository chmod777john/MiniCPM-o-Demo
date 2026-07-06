# MiniCPM-o 5.0 推理接入 Demo 问题记录

本文面向 o5 推理代码维护者，记录我们把 Codeup 上的 MiniCPM-o 5.0 35a3 推理代码接入实时交互 demo 时遇到的几个问题。这里不讨论 demo 内部工程结构，只关注推理代码、依赖、运行参数和可复现实验现象。

## 背景

当前接入目标是：把 o5 35a3 的推理代码和 `.pt` checkpoint 接到实时交互 demo，使其支持以下能力：

1. 文本/语音单工问答，包含流式和非流式。
2. 音频双工，按 1s chunk 输入，模型逐 chunk 决定 listen/speak。
3. 视频双工，按视频帧 + 音频 chunk 输入，模型逐 chunk 决定 listen/speak。

本次使用的模型代码来源是 Codeup：

```text
git@codeup.aliyun.com:thunlp/infra/MiniCPM-o-4_6.git
branch: moe-35b-a3b
```

主要 checkpoint 包括：

```text
omni_sft2_main_run_iter1200.pt
```

我们也参考了 humanevalkit 中 o5 35a3 的评测说明和运行方式。

## 需要推理侧关注的问题

### 1. 实时双工性能压力较大

o5 的 duplex 设计是按 `chunk_seconds=1.0` 逐 chunk 处理输入，每个 chunk 决定 listen 或 speak。对于实时 demo 来说，理想状态是一个 chunk 的 prefill + generate + TTS 合成尽量稳定小于 1s，否则输入会持续积压，交互延迟会越来越高。

我们做过 minimal duplex probe，调用方式等价于：

```python
prefill = duplex.streaming_prefill(audio_waveform=chunk)
result = duplex.streaming_generate()
```

probe 中记录了：

```text
prefill_s
generate_s
total_s = prefill_s + generate_s
is_listen
```

从已有 probe 看，listen unit 通常比 speak unit 快很多；speak unit 因为包含文本 token 生成、TTS token 生成和 token2wav，耗时明显更高。以下数字来自 10 个 1s chunk 的小样本 probe，只用于说明数量级，不作为正式 benchmark：

| 配置 | listen mean / p90 | speak mean / p90 | prefill 量级 | 备注 |
| --- | --- | --- | --- | --- |
| o45 vendored probe | 0.194s / 0.331s | 0.766s / 0.817s | 约 0.09s | o45 基线 |
| o5 iter1200 no compile | 0.766s / 0.801s | 1.968s / 2.138s | 约 0.74s | A100，小样本 |
| o5 iter1200 compile | 0.841s / 0.867s | 1.971s / 2.129s | 约 0.81s | 该小样本未体现收益 |
| o5 iter1200 rerun no compile | 0.711s / 0.721s | 1.807s / 1.999s | 约 0.69s | A100，小样本 |
| o5 iter1200 vendored probe | 1.093s / 1.123s | 2.123s / 2.327s | 约 0.75s | 调用路径略不同 |

可以看到 o5 的 listen unit 已经接近 1s chunk 边界，speak unit 常见在 1.8-2.3s；相比 o45，实时余量明显更小。humanevalkit 结果中的 `cost_all` 也显示，audio duplex 带 TTS 时 speak unit 往往在秒级，部分 case 接近或超过 2s。

这个问题会直接影响实时 demo：即使模型逻辑正确，只要 speak unit 平均耗时高于输入 chunk 速度，服务层就会积压。


### 2. 当前 torch / transformers 接入版本存在轻微接口适配

为了跑 o5 35a3，我们不能简单复用 o45 demo 的旧依赖。o5 路径会涉及新版 transformers cache、MoE grouped-mm、Qwen3.5 MoE 相关实现，以及新的 processor/config 文件。

当前 demo 接入环境已经按较新的 torch / transformers 组合安装，但不等于二者都是当前最新稳定版。`requirements.txt` 和本地 `.venv` 实际版本一致：

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
torchaudio==2.11.0+cu128
torchcodec==0.11.1+cu128
transformers==5.13.0
accelerate==1.12.0
CUDA runtime: 12.8
```

这个版本组合可以跑起 o5，但接入过程中暴露出若干接口层面的轻微不兼容；这些问题更像是“推理代码需要明确支持/声明的依赖边界”，而不是模型能力问题。

接入时遇到过几类轻微兼容问题：

#### DynamicCache / cache 结构

新版 transformers 的 generation cache 表示和旧版 tuple/list cache 不完全一致。外部如果需要读取 cache 或 debug generation state，需要兼容 `DynamicCache` 等结构。

希望推理侧确认：o5 推荐的 transformers 版本范围，以及 cache 相关 public/semipublic 读取方式。最好在 requirements 或部署文档里固定。

#### `chunk_generate()` 参数签名

o45 旧路径里存在一些额外 generation 参数，例如 `suppress_forbidden_tokens` 一类开关。o5 `chunk_generate()` 的签名和旧代码不完全一致，外部调用需要按 o5 签名调整。

希望推理侧确认：o5 对外稳定支持的 streaming generation 参数集合。这样 demo / serving 可以只透传稳定参数，避免旧参数误传。

#### TTS config 中的 `rope_theta`

streaming 路径曾遇到：

```text
'LlamaConfig' object has no attribute 'rope_theta'
```

从现象看，TTS 内部构造/使用 `LlamaConfig` 时会访问 `rope_theta`。如果某些构造路径没有显式带上该字段，就会报错。

希望推理侧确认：

1. `rope_theta` 是否应作为 o5 TTS config 的必备字段。
2. 是否应该在模型初始化时由推理代码统一补齐，而不是由外部 demo 补。
3. 对旧 checkpoint/config 是否需要兼容默认值。

#### grouped-mm / MoE kernel

o5 MoE 路径会用到 grouped expert 计算。实际底层可能依赖当前 torch / transformers / qwen3_5_moe 版本中的 grouped-mm 实现。

希望推理侧给出明确推荐：

1. torch 版本。
2. transformers 版本。
3. CUDA / GPU 架构要求。
4. 是否需要特定的 `qwen3_5_moe` 或其它本地包版本。

这样 demo 和 serving 才能避免“在某个环境能跑、换环境轻微报错或性能变化”的问题。

### 3. non-streaming 单工 chat 语音存在吞尾音

demo 中发现，单工非流式之下，生成的音频有结尾吞音现象，稳定复现。目前还没找到根因。或许 humanevalkit 评测可以提供参考。

需要的评测 setting 是：

```text
inference_mode = chat
response_mode = text_and_speech
generate_audio = true
stream = false
```

对应调用形态类似：

```python
model.init_tts()
res = model.chat(
    msgs=msgs,
    use_tts_template=True,
    generate_audio=True,
    output_audio_path="output.wav",
)
```
