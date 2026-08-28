# Docker 部署与 FC Case Probe

本文说明如何在一台已经安装 NVIDIA Container Toolkit 和 Docker Compose 的
GPU 主机上，直接使用预构建镜像启动 MiniCPM-o O5 FC + TP2，并回放一条 FC
TrainingData case。

本文使用的镜像是 Docker Hub 上的预构建镜像：

```text
device0/minicpm-o-worker-backend:align-enhance-fc-speedup-full-bundle-20260828
device0/minicpm-o-gateway:align-enhance-fc-speedup-full-bundle-20260828
```

Docker 运行时不需要 `MODEL_PATH` 或 `PT_PATH`。worker 需要的是完整的
safetensors bundle 和 processor/Token2Wav assets。

## 1. 准备目录

把本项目目录放到 GPU 主机上。例如：

```bash
export PROJECT_DIR=/opt/MiniCPM-o-Demo
cd "$PROJECT_DIR"
```

准备以下两个宿主机目录：

```bash
export WEIGHTS_HOST_PATH=/data/models/o5-full-bundle
export ASSETS_HOST_PATH=/data/models/MiniCPM-o-4_5-assets

test -f "$WEIGHTS_HOST_PATH/model.safetensors.index.json"
test -f "$WEIGHTS_HOST_PATH/llm/config.json"
test -f "$ASSETS_HOST_PATH/preprocessor_config.json"
test -d "$ASSETS_HOST_PATH/token2wav"
```

`WEIGHTS_HOST_PATH` 是完整 O5 safetensors bundle，不是只有 LLM backbone 的
目录。`ASSETS_HOST_PATH` 包含 processor 配置、Token2Wav 权重及其运行时文件。
两个目录会以只读方式挂载到 worker 容器：

```text
宿主机 WEIGHTS_HOST_PATH -> /models/o5-full-bundle
宿主机 ASSETS_HOST_PATH  -> /models/o5-assets
```

浏览器体验需要 HTTPS 证书。测试时可以使用项目中的 `certs/`，正式部署应
替换为自己的 `cert.pem` 和 `key.pem`：

```bash
export CERTS_HOST_PATH="$PROJECT_DIR/certs"
test -f "$CERTS_HOST_PATH/cert.pem"
test -f "$CERTS_HOST_PATH/key.pem"
```

## 2. 不重新构建，直接使用镜像

这套 `docker-compose.deploy.yml` 只使用 `image:`，没有 `build:`。因此不需要
安装 Python venv，也不需要执行 `docker compose build`。

先登录 Docker Hub，并拉取两个镜像：

```bash
docker login

export WORKER_IMAGE=device0/minicpm-o-worker-backend:align-enhance-fc-speedup-full-bundle-20260828
export GATEWAY_IMAGE=device0/minicpm-o-gateway:align-enhance-fc-speedup-full-bundle-20260828

docker compose -p o5-fc-docker \
  -f docker-compose.deploy.yml --profile tp2 pull
```

`docker compose pull` 只下载镜像，不启动服务，也不修改 GPU 进程。

## 3. 启动双卡 TP2

下面的配置启动一个 gateway 和一个同时占用两张 GPU 的 TP2 worker。一个
worker 容器内部启动两个 `torchrun` rank；它不是两个独立的单卡 worker。

```bash
export DEPLOY_PROJECT_NAME=o5-fc-docker
export GATEWAY_HOST_PORT=8006
export TP2_GPU0=0
export TP2_GPU1=1

docker compose -p "$DEPLOY_PROJECT_NAME" \
  -f docker-compose.deploy.yml --profile tp2 \
  up -d --no-build --pull never
```

如果宿主机的 `8006` 已被占用，改用已经放通的端口，例如：

```bash
export GATEWAY_HOST_PORT=18010
```

然后用相同的 `up` 命令启动。不要同时启用 `single` 和 `tp2` profile；它们
会创建不同的 worker topology。Compose project name 要保持唯一，避免误操作
其他项目的容器。

查看启动状态：

```bash
docker compose -p "$DEPLOY_PROJECT_NAME" \
  -f docker-compose.deploy.yml --profile tp2 ps

curl -k "https://127.0.0.1:${GATEWAY_HOST_PORT}/health"
curl -k "https://127.0.0.1:${GATEWAY_HOST_PORT}/status"
```

健康状态应满足：

```json
{"status":"healthy"}
```

`/status` 中 `total_workers` 应为 `1`，`idle_workers` 应为 `1`。如果 gateway
刚被重新创建过而显示 `total_workers: 0`，worker 进程可能仍然健康但它之前
的注册已被 gateway 的内存状态清掉；重启同一 Compose project 的 TP2 worker
即可重新注册：

```bash
docker compose -p "$DEPLOY_PROJECT_NAME" \
  -f docker-compose.deploy.yml --profile tp2 restart worker-tp2
```

## 4. FC case 的路径要求

FC v3 在 `generate_audio=true` 时要求 `tts_prompt_audio` 是服务端可读取的
绝对路径。system audio 和 TTS prompt 是两个独立字段；服务端不会自动从
system audio 推断 TTS prompt。

因此 case 目录应包含 JSON 和它引用的 `media/`，例如：

```text
case-dir/
  case.json
  media/
    system_reference/HTRef06.wav
    ...
```

直接把宿主机 case 目录复制到正在运行的 worker 容器中，可以保证 probe 生成
的路径同时对 probe 和 backend 可见：

```bash
export CASE_HOST_DIR=/data/cases/my_fc_case
export WORKER_CONTAINER="${DEPLOY_PROJECT_NAME}-worker-tp2"

test -f "$CASE_HOST_DIR/case.json"
docker exec "$WORKER_CONTAINER" rm -rf /tmp/fc-case
docker cp "$CASE_HOST_DIR" "$WORKER_CONTAINER:/tmp/fc-case"
```

复制后，容器内的 case 路径是：

```text
/tmp/fc-case/case.json
```

不要把只在宿主机存在的路径，例如 `/data/cases/my_fc_case/media/x.wav`，
直接作为 API 的 `file_path` 发送给 worker；该路径在容器内不可读，会导致：

```text
TTS prompt audio path is required when generate_audio is true
```

或者后续的 audio path not found 错误。

## 5. 运行 FC API probe

### 5.1 正式公网 API 回放

`examples/fc_duplex/infer_from_training_data.py` 走正式 gateway API：

```text
/v1/realtime?mode=audio
```

它发送请求侧 TrainingData 输入，不把 AI ground-truth 输出发送给模型，并保存
原始 API history、spoken audio 和可选 replay 产物。在 worker 容器中执行时，
通过 Compose 网络访问 gateway 的服务名 `gateway`：

```bash
docker exec "$WORKER_CONTAINER" \
  python examples/fc_duplex/infer_from_training_data.py \
  /tmp/fc-case/case.json \
  --data-root /tmp/fc-case \
  --base-url https://gateway:8006 \
  --insecure \
  --output-dir /app/data/fc-api-eval
```

结果位于 worker 容器的 `/app/data/fc-api-eval`。需要复制到宿主机时：

```bash
docker cp "$WORKER_CONTAINER:/app/data/fc-api-eval" ./fc-api-eval
```

要求 case 的 system segments 中有可读取的 audio segment，因为这个示例会
使用首个 system audio 作为独立的 TTS prompt。若 case 没有 system audio，使用
下面的 backend diagnostic probe，并显式传入容器内可读的 WAV 路径。

### 5.2 Backend diagnostic probe

`scripts/probe_fc_board_backend.py` 直接连 TP2 worker 容器中的 backend，适合
先确认 FC 状态机、unit 调度、tool call 和 TTS 音频是否能工作。它不经过
gateway 队列，因此不是公网 API 验证。

```bash
docker exec "$WORKER_CONTAINER" \
  python scripts/probe_fc_board_backend.py \
  --backend http://127.0.0.1:22500 \
  --path /backend \
  --case /tmp/fc-case/case.json \
  --out /app/data/fc-board-probe.json \
  --normalize-tools \
  --tool-response-schedule gt \
  --generate-audio
```

如果 case 没有 system audio，但 assets 中有参考音频，可以显式指定：

```bash
docker exec "$WORKER_CONTAINER" \
  python scripts/probe_fc_board_backend.py \
  --backend http://127.0.0.1:22500 \
  --path /backend \
  --case /tmp/fc-case/case.json \
  --ref-audio-path /models/o5-assets/system_ref_audio.wav \
  --out /app/data/fc-board-probe.json \
  --normalize-tools \
  --tool-response-schedule gt \
  --generate-audio
```

查看结果：

```bash
docker cp "$WORKER_CONTAINER:/app/data/fc-board-probe.json" ./fc-board-probe.json
jq '.summary | {
  units_sent,
  spoken_text,
  spoken_audio_events,
  spoken_audio_bytes,
  spoken_audio_sha256,
  spoken_audio_nonzero_bytes,
  tool_calls_semantic_exact
}' ./fc-board-probe.json
```

成功的生成音频 probe 应有非零的 `spoken_audio_bytes` 和
`spoken_audio_nonzero_bytes`；tool case 还应有
`tool_calls_semantic_exact: true`。

## 6. 单卡启动

如果只需要一张 GPU，把 profile 改成 `single`：

```bash
export GPU_ID=0
export SINGLE_DEPLOY_MODE=single_eager

docker compose -p o5-fc-docker-single \
  -f docker-compose.deploy.yml --profile single \
  up -d --no-build --pull never
```

单卡默认使用 `single_eager`。如需单卡优化模式，可以设置：

```bash
SINGLE_DEPLOY_MODE=single_opt
```

单卡和 TP2 仍使用相同的镜像、权重挂载和 gateway 协议；差别只在 worker
容器绑定的 GPU 数量和 backend launcher。

## 7. 加速选项

`docker-compose.deploy.yml` 的默认值是：

```text
O5_ATTN_IMPLEMENTATION=auto
O5_EXPERTS_IMPLEMENTATION=batched_mm
O5_LLM_GRAPH=1
O5_TTS_FAST=1
O5_LMHEAD=1
O5_TTS_GRAPH=1
O5_VOCODER_GRAPH=0
O5_FUSE_VISION_AUDIO=1
O5_VISION_BATCH=1
O5_UNIT_PREFILL_BATCH=1
```

启动时可以覆盖，例如关闭 LLM graph 和 batched MM：

```bash
O5_LLM_GRAPH=0 \
O5_EXPERTS_IMPLEMENTATION=eager \
docker compose -p o5-fc-docker \
  -f docker-compose.deploy.yml --profile tp2 \
  up -d --no-build --pull never
```

`O5_VOCODER_GRAPH` 默认关闭，因为它是独立的实验性加速项。修改环境变量
后需要重新创建 worker，模型会重新加载；不需要重新构建镜像。

## 8. 查看日志和停止服务

```bash
docker compose -p "$DEPLOY_PROJECT_NAME" \
  -f docker-compose.deploy.yml logs -f worker-tp2

docker compose -p "$DEPLOY_PROJECT_NAME" \
  -f docker-compose.deploy.yml logs -f gateway
```

只停止自己使用的 Compose project：

```bash
docker compose -p "$DEPLOY_PROJECT_NAME" \
  -f docker-compose.deploy.yml --profile tp2 down
```

不要使用不带 `-p` 的 `docker compose down`，也不要按端口或模糊名称删除
其他项目的容器。
