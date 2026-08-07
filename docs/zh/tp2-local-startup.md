# O5 TP2 本地启动指南

本文适用于没有 `cctl`、直接在一台有 NVIDIA GPU 的 Linux 机器上启动一套 O5 TP2 服务的情况。
一套服务使用两张 GPU：一个 TP2 backend、一个 worker 和一个 HTTPS gateway。

## 1. 目录和代码

启动代码使用本 Demo 分支：

```text
o5-no-fc-speedup-tp2
HEAD: 129a65483a46ddaba13967a2a180ff29728606e3
```

`MODEL_PATH` 不要指向 Demo 根目录，而是指向模型代码和配置目录。按下面的方式获取指定分支：

```bash
git clone --branch moe-35b-a3b \
  git@codeup.aliyun.com:modelbest/multi-modal/inference/MiniCPM-o-4_6.git

export MODEL_PATH=$PWD/MiniCPM-o-4_6
```

如果本机使用 HTTPS Git 地址，仓库地址应写成：

```text
https://codeup.aliyun.com/modelbest/multi-modal/inference/MiniCPM-o-4_6.git
```

仍然需要通过 `--branch moe-35b-a3b` 选择分支；网页查看地址是
`https://codeup.aliyun.com/modelbest/multi-modal/inference/MiniCPM-o-4_6/tree/moe-35b-a3b`。

## 2. 创建项目 venv

不要全局安装 Python 包。所有依赖安装在 Demo 项目自己的 `.venv` 中：

```bash
cd /path/to/MiniCPM-o-Demo

# 需要 Python 3.10+
PYTHON=python3.10 bash install.sh
source .venv/bin/activate
```

`requirements.txt` 当前固定了 CUDA 12.6 对应的 PyTorch 2.8.0。需要 CUDA 扩展加速时，
在同一个 venv 中执行：

```bash
bash install.sh --with-accel
```

该命令会额外安装 `flash-attn` 和 `causal-conv1d`，不会创建或修改全局 Python 环境。

## 3. 权重

当前使用的 O5 checkpoint 是：

```text
/user/weihongliang/o5_weights/chenmoye_minicpm_5o_moe_omni_long_context_sft_stage2_sft2_8k_audio_online_process_on_online_audio_process_v2_iter_100.pt
```

用自己的机器时，将该文件放到本地可访问的位置，并设置：

```bash
export PT_PATH=/path/to/chenmoye_iter_100.pt
```

TP2 不直接从完整 `.pt` 的 `llm.*` 参数建立张量并行骨干，而是使用从同一个 `.pt` 抽取出的
Hugging Face backbone。每个 checkpoint 都要抽取对应的 backbone，不能混用不同 checkpoint
的抽取结果。

## 4. 抽取 TP2 backbone

抽取脚本是：

```text
tools/o5deploy/extract_backbone.py
```

在 Demo 根目录执行：

```bash
cd /path/to/MiniCPM-o-Demo

export WORKTREE=$PWD
export PYTHONPATH=$PWD
export MODEL_PATH=/path/to/MiniCPM-o-4_6
export PT_PATH=/path/to/chenmoye_iter_100.pt
export BACKBONE_DIR=/path/to/o5_backbone_hf_iter_100

.venv/bin/python tools/o5deploy/extract_backbone.py
```

脚本会：

1. 从 `MODEL_PATH` 读取 O5 配置，并在 meta device 上构建完整模型。
2. 加载 `PT_PATH` 的完整 checkpoint。
3. 提取 `model.llm`，转换为 bf16。
4. 将 backbone 保存为带 `config.json`、index 和分片 safetensors 的目录。

默认产物约 65 GB；完整 `.pt` 约 75 GB。`BACKBONE_DIR` 应使用新的空目录，并确保磁盘和内存
足够。启动 TP2 时必须使用这个 checkpoint 对应的 `BACKBONE_DIR`。

## 5. 启动一套双卡服务

单套本地编排脚本是：

```text
scripts/start_o5_tp2_cctl_service.sh
```

文件名中的 `cctl` 是历史命名；脚本本身不调用 `cctl`，会在本机启动 gateway、worker 和
`torchrun --nproc_per_node=2` 的 TP2 backend。

下面的命令使用两张卡 `0,1`，监听本机 HTTPS `8009`，并关闭 FRP：

```bash
cd /path/to/MiniCPM-o-Demo

export PROJECT_DIR=$PWD
export VENV_DIR=$PWD/.venv
export MODEL_PATH=/path/to/MiniCPM-o-4_6
export PT_PATH=/path/to/chenmoye_iter_100.pt
export BACKBONE_DIR=/path/to/o5_backbone_hf_iter_100

export O5_DEPLOY_MODE=tp2
export O5_ATTN_IMPLEMENTATION=auto
export O5_EXPERTS_IMPLEMENTATION=batched_mm
export O5_LLM_GRAPH=1
export O5_TTS_GRAPH=1
export O5_VOCODER_GRAPH=1
export O5_TTS_FAST=1
export O5_LMHEAD=1
export O5_FUSE_VISION_AUDIO=1
export O5_VISION_BATCH=1
export O5_LLM_CACHE=32768

# 保留正常采样；不启用确定性回放。
export O5_DETERMINISTIC_REPLAY=0
export O5_TTS_ARGMAX=0

export BACKEND_HOST=127.0.0.1
export BACKEND_PORT=22510
export WORKER_HOST=127.0.0.1
export WORKER_PORT=22410
export GATEWAY_HOST=0.0.0.0
export GATEWAY_PORT=8009
export GATEWAY_INTERNAL_PORT=8010
export ENABLE_FRP=0
export LOG_DIR=$PWD/run-logs/o5-tp2-local

CUDA_VISIBLE_DEVICES=0,1 bash scripts/start_o5_tp2_cctl_service.sh
```

服务启动后访问：

```text
https://127.0.0.1:8009/omni
```

仓库使用自签名证书，浏览器首次访问时需要接受证书例外。实时 API 的协议和路径保持 Demo
原有定义，HTTPS gateway 对外提供入口，worker 和 TP2 backend 仅监听本机内部端口。

## 6. 健康检查和日志

另开终端检查：

```bash
curl -k https://127.0.0.1:8009/health
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:22410/health
curl http://127.0.0.1:22510/health
```

日志目录由 `LOG_DIR` 指定，主要文件是：

```text
gateway.log
worker.log
backend_tp2.log
```

停止整套服务时，在启动脚本所在终端按 `Ctrl-C`；脚本会连同 gateway、worker 和 torchrun
进程一起清理。
