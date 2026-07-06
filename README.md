# MiniCPM-o Demo Runtime

这个仓库用于启动 MiniCPM-o 推理 demo 服务。所有 Python 依赖都应安装在项目目录内的虚拟环境中，不要全局安装。

## 安装

使用 Python 3.10+，在仓库根目录创建项目内虚拟环境：

```bash
bash install.sh
```

这个命令会创建 `.venv`，从 `requirements.txt` 安装基础运行环境，并检查主要依赖是否可用。基础环境不依赖可选 CUDA 扩展；如果没有 FlashAttention2，attention 会走 PyTorch 自带的 SDPA 路径。

如果需要把可选加速依赖安装到同一个 `.venv`：

```bash
bash install.sh --with-accel
```

可选加速依赖写在 `requirements-accel.txt`：

- `flash-attn`
- `causal-conv1d`

安装这些依赖后，可以配置模型使用 `flash_attention_2`，并启用 Qwen MoE fast path。没有这些依赖时，同一份代码仍应能通过基础 PyTorch 路径运行。

安装时可用的环境变量：

```bash
PYTHON=python3.10 bash install.sh
MAX_JOBS=16 bash install.sh --with-accel
```

## 用 cctl 启动 demo 服务

浏览器 demo 可以通过 service helper 启动：

```bash
source .venv/bin/activate
MODEL_PATH=/path/to/model-code-and-tokenizer \
PT_PATH=/path/to/checkpoint.pt \
PUBLIC_PORT=8445 \
bash scripts/start_o5_cctl_service.sh
```

这个 helper 会启动本地 backend，启动 cctl GPU worker，把 worker 注册到 backend，并按配置暴露浏览器访问入口。模型路径、checkpoint、日志和生成文件应放在项目目录或明确分配的数据目录里。

查看 cctl 任务：

```bash
cctl task list
```

查看某个任务日志：

```bash
cctl job logs tasks/<task-id>
```

停止自己的任务：

```bash
cctl job stop tasks/<task-id>
```
