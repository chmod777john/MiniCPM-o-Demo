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

## 启动 demo 服务

准备模型代码目录和 checkpoint：

```bash
source .venv/bin/activate
export MODEL_PATH=/path/to/model-code-and-tokenizer
export PT_PATH=/path/to/checkpoint.pt
```

启动 backend。backend 负责加载模型和执行推理：

```bash
PYTHONPATH=. python -m py_backend.server \
  --host 127.0.0.1 \
  --port 22510 \
  --model-path "$MODEL_PATH" \
  --pt-path "$PT_PATH" \
  --gpu-id 0
```

另开一个终端，启动 worker。worker 作为 gateway 和 backend 之间的运行时入口：

```bash
source .venv/bin/activate
PYTHONPATH=. python worker.py \
  --host 127.0.0.1 \
  --port 22410 \
  --gpu-id 0 \
  --backend-server-url http://127.0.0.1:22510
```

再开一个终端，启动 gateway。gateway 提供浏览器页面和 WebSocket 入口：

```bash
source .venv/bin/activate
PYTHONPATH=. python gateway.py \
  --host 0.0.0.0 \
  --port 8009 \
  --internal-port 8010 \
  --https \
  --ssl-certfile certs/cert.pem \
  --ssl-keyfile certs/key.pem
```

最后把 worker 注册到 gateway：

```bash
curl -X PUT \
  -H 'content-type: application/json' \
  --data '{"endpoint":"127.0.0.1:22410","gpu_group":"local-gpu-0","labels":{"model":"o5","runtime":"local"}}' \
  http://127.0.0.1:8010/internal/workers/local-o5-worker
```

然后访问 gateway 地址，例如 `https://127.0.0.1:8009/`。浏览器麦克风和摄像头 API 通常要求 HTTPS；如果只做后端调试，也可以把 gateway 改为 `--http`。
