# MiniCPM-o Demo Adapter

本分支使用 **HF + accel + Demo adapter** 三份物料运行 O5，支持普通单工、音视频双工及 FC 双工。模型推理由外部 HF 提供，加速由外部 accel 以 patch 方式接入；本仓库负责 API、会话、页面和必要的接口桥接。

## 1. 准备三份物料

| 物料 | 分支 | 职责 | 启动时指定 |
|---|---|---|---|
| HF | `fc` | 3D-RoPE 模型与普通 / FC 推理逻辑 | `O5_CANONICAL_PATH` |
| accel | `hf-fc-accel` | 卡型适配、双卡并行、Graph、合并 prefill 等加速 | `O5_ACCELERATION_PATH` |
| Demo（本仓库） | `demo-adapter-fc` | backend、worker、gateway、页面与桥接 | `PROJECT_DIR` |

三份物料的配套 commit 记录在 [materials.lock.json](materials.lock.json)。部署时按这份组合准备代码，并记录实际使用的三个 commit；其中 `last_runtime_validation` 和 `post_cleanup_validation` 分别说明运行验收与清理后的检查范围。

目录可以分别放置，例如：

```text
/path/to/o5/
├── hf/                 # HF fc 分支，含 config/tokenizer/processor 资产
├── accel/              # hf-fc-accel 分支
├── demo/               # 本仓库
├── weights/model.pt    # 完整 O5 checkpoint
└── token2wav_causal_vocoder/
```

HF、accel 通过路径加载。HF 目录需要包含 `configuration_minicpmo.py`、`modeling_minicpmo.py`、`processing_minicpmo.py`、`fc/` 及配套资产。模型使用完整 `.pt` 权重；声码器资产另外准备。

## 2. Python 环境

所有 Python 依赖安装到项目内的 venv。环境需要与 GPU 和 accel 配套，下面是现有环境记录中的关键版本，**不是完整安装清单**：

| 组件 | A100 环境 | Pro5000 Docker 环境 |
|---|---|---|
| Python | 3.10 | 3.10 |
| torch | `2.11.0+cu128` | `2.14.0+cu130` |
| transformers | `5.13.0` | `5.13.0` |
| flash-attn | `2.8.3` | `2.8.3.post1` |
| minicpm-o5-sdk | `0.0.6` | `0.0.6` |
| minicpmo-utils | `1.0.7` causal 版本 | `1.0.7` causal 版本 |

`minicpmo-utils` 使用配套的 causal wheel。它的部分依赖元数据与运行栈不同，安装方式应遵循环境锁定清单。根目录 `install.sh` / `requirements.txt` 仍是上游的另一套依赖版本，不能直接用于复现上表环境。

Pro5000 的固定包、源码和安装方式见 [Docker pip 构建说明](docker/external-hf/pip/README.md) 与 [版本清单](docker/external-hf/requirements-pro5000.lock)。以下启动示例假定已准备好对应环境。

## 3. 双卡启动 API 服务

下面以 A100 双卡为例。将所有 `/path/to/...` 替换为实际绝对路径：

```bash
export PROJECT_DIR=/path/to/o5/demo
export VENV_DIR="$PROJECT_DIR/.venv"
export O5_CANONICAL_PATH=/path/to/o5/hf
export O5_ACCELERATION_PATH=/path/to/o5/accel
export O5_CHECKPOINT_PATH=/path/to/o5/weights/model.pt
export O5_ASSETS_DIR=/path/to/o5/token2wav_causal_vocoder

export CUDA_VISIBLE_DEVICES=0,1
export O5_DEPLOY_MODE=tp2
export O5_MOE_PARALLEL=tp
export O5_ACCELERATION_HARDWARE=a100
export O5_ATTN_IMPLEMENTATION=sdpa
export DUPLEX_SLOT_PROTOCOL=1

export O5_LLM_GRAPH=1
export O5_TTS_GRAPH=1
export O5_VOCODER_GRAPH=1
export CANONICAL_UNIT_PREFILL_BATCH=1
export O5_COMPACT_LOGITS=1
export O5_TTS_FA2_WHOLE_DECODE=1
export O5_CANVAS_WINDOW_EXACT_CHAIN=1
```

其他卡型应使用对应环境和 accel 配置；`O5_ACCELERATION_HARDWARE=auto` 可用于自动识别。普通双工的 `DUPLEX_SLOT_PROTOCOL` 需要与 checkpoint 匹配：slot 词表由 SDK 动态提供，权重的 embedding / lm_head 必须覆盖所需 token。

在项目目录保存运行配置，例如 `run-config.json`，其中路径也要替换：

```json
{
  "model": {
    "model_path": "/path/to/o5/hf",
    "pt_path": "/path/to/o5/weights/model.pt",
    "deployment_mode": "tp2",
    "attn_implementation": "sdpa"
  },
  "audio": {
    "ref_audio_path": "/path/to/reference-16k.wav",
    "chat_vocoder": "token2wav"
  },
  "service": {
    "compile": false,
    "data_dir": "/path/to/o5/demo/data"
  }
}
```

参考音频使用 16 kHz 单声道 WAV。`MODEL_PATH` 默认取 `O5_CANONICAL_PATH`，也可以单独指向模型 config / processor 资产目录。`service.compile` 保持 `false`，上述加速由 accel 接管。

启动服务：

```bash
cd "$PROJECT_DIR"
export O5_DEMO_CONFIG_PATH="$PROJECT_DIR/run-config.json"
export GATEWAY_PORT=8009
export GATEWAY_INTERNAL_PORT=8010
export BACKEND_PORT=22510
export WORKER_PORT=22410
export GATEWAY_HTTPS=1
export GATEWAY_API_ONLY=0
export ENABLE_FRP=0
bash scripts/start_o5_tp2_cctl_service.sh
```

脚本会启动 gateway、backend、worker，等待就绪并自动注册 worker，退出时清理它启动的子进程。日志默认在 `run-logs/o5_tp2_cctl/`。脚本名中的 `cctl` 不代表会申请集群资源；应在已分配 GPU 的机器或容器中执行。双卡模型由外部 accel 在一个 backend 进程中管理。

HTTPS 使用 `certs/cert.pem` 和 `certs/key.pem`，请准备浏览器信任的证书，或通过 TLS 反向代理访问。仅调试 HTTP API 时可设 `GATEWAY_HTTPS=0`；浏览器麦克风 / 摄像头需要安全上下文。

### FC 部署参数

FC 还需要绑定 checkpoint 对应的 `unit_policy`、`non_spoken_scheduling` 和 `tts_pad_protocol`。使用部署方提供的 Profile，在启动上述服务前设置：

```bash
export FC_DEPLOYMENT_PROFILE=/path/to/fc-profile.json
```

Profile 中的模型路径、权重和部署模式应与本次部署一致。字段定义见 [profiles.py](core/fc_duplex/profiles.py)。Gallery 页面从所选 API 的 `/v1/info` 读取模型默认参数，preset 主要描述 system 与参考音频。FC 协议见 [Realtime API v5](docs/fc-duplex/semantic-realtime-api-v5.md)。

## 4. API 与应用服务

Gateway 现在只提供模型目录、命名实时端点和 Worker 调度，不再提供 HTML 页面或文档站。模型输入输出契约见 [Gateway Worker transport](docs/gateway-thin-transport.md)。

`demos/` 下的页面与应用接口由 `demo_server.py` 独立提供。浏览器从应用服务取得 HTML / JS，然后直接连接 Gateway 的 `/e/<endpoint>/v1/realtime?mode=audio|omni`。在另一个终端执行（重新设置本终端的 `PROJECT_DIR` 和 `VENV_DIR`）：

```bash
export PROJECT_DIR=/path/to/o5/demo
export VENV_DIR="$PROJECT_DIR/.venv"
cd "$PROJECT_DIR"
export DEMO_DEFAULT_API=https://your-api-host:8009/e/default
export DEMO_API_ENDPOINTS='[{"name":"O5","url":"https://your-api-host:8009/e/default"}]'
export DEMO_AUDIO_ROOTS=/path/to/reference-audio-directory
"$VENV_DIR/bin/python" demo_server.py --host 0.0.0.0 --port 8012 --https \
  --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem
```

访问 `https://your-demo-host:8012/demos`。API 地址必须能从浏览器访问；`DEMO_AUDIO_ROOTS` 是 Demo 服务允许读取参考音频的目录，多个目录用冒号分隔。应用组织与各 Demo 配置见 [demos/README.md](demos/README.md)。

## 5. 其他部署方式

- **单卡**：使用 `scripts/start_o5_cctl_service.sh`，设置 `O5_DEPLOY_MODE=single_eager` 或 `single_opt`，并同步调整运行配置；本地启动设 `ENABLE_FRP=0`。`single_eager` 直接运行外部 HF，`single_opt` 可通过 `O5_ACCELERATION_PATH` 接入 accel。显存需能容纳完整模型和运行缓存。
- **Docker**：按 [固定 pip 安装说明](docker/external-hf/pip/README.md) 准备环境输入；[assemble_sources.py](docker/external-hf/assemble_sources.py) 从三个干净 checkout 打包源代码和页面资产，并记录实际 commit 与文件哈希到 `materials/source-manifest.json`。打包前自行选定配套版本，脚本不会自动切换到 `materials.lock.json` 指定的 commit。权重与声码器资产按部署配置挂载。
