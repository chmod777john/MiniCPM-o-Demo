# O5 API 一键启动

本文只说明如何用已经准备好的 O5 模型目录启动 API，不涉及 checkpoint 来源、转换、
评测方法或 Cybertron/CCTL 调度。

## 1. 克隆可复现版本

```bash
git clone --branch o5-fc-dev \
  git@codeup-modelbest:modelbest/multi-modal/MiniCPM-o-Demo.git

cd MiniCPM-o-Demo
git checkout --detach <交付方提供的一键启动 commit>
```

该环境已经提供可直接复用的 Python/CUDA/SDK 运行环境，不需要重新安装依赖。

## 2. 准备模型目录

推荐使用标准目录：

```text
<model-dir>/
├── model.pt
└── backbone/
    ├── config.json
    ├── model.safetensors.index.json
    └── model-*.safetensors
```

为兼容已有资产，启动器也接受：

- 模型目录顶层唯一的 `*.pt`；
- 模型目录一级子目录中，唯一同时包含 `config.json` 和
  `model.safetensors.index.json` 的 backbone。

如果找到多个候选，启动器会明确报错，不会猜测。

## 3. 启动 API

默认启动：

```bash
./run_o5_api.sh <model-dir>
```

指定运行产物目录和 Gateway 端口：

```bash
./run_o5_api.sh <model-dir> \
  --storage-dir <storage-dir> \
  --port 8009
```

仅检查模型资产和复用环境：

```bash
./run_o5_api.sh <model-dir> --check-only
```

启动器会自动：

1. 发现完整 PT 与 TP2 backbone；
2. 读取仓库维护的内部固定设置；
3. 生成临时 Deployment Profile 和 ServiceConfig；
4. 隔离日志、Session、cache 和运行配置；
5. 启动 Gateway、TP2 Backend 和 Worker；
6. 等待内部服务通过健康检查。

## 4. 外部接口

默认使用 HTTP，由部署基础设施负责 TLS 和公网映射：

```text
HTTP API:  http://<host>:8009
WebSocket: ws://<host>:8009/v1/realtime
Health:    http://<host>:8009/health
FC Board:  http://<host>:8009/fc_board
```

健康检查：

```bash
curl http://127.0.0.1:8009/health
```

以下日志全部出现后，服务才真正可用：

```text
backend-tp2 ready
worker ready
service ready
```

## 5. 存储目录

未传 `--storage-dir` 时，默认写入：

```text
run-logs/<model-id>-<UTC timestamp>/
```

目录结构：

```text
<storage-dir>/
├── logs/
│   ├── gateway.log
│   ├── backend_tp2.log
│   └── worker.log
├── data/
│   └── sessions/
├── cache/
└── runtime/
    ├── deployment_profile.json
    └── service_config.json
```

## 6. 两层设置边界

部署使用者只感知：

- 模型目录；
- 可选存储目录；
- 可选 Gateway 端口；
- 可选 `--check-only`。

Demo 维护者在
`configs/fc_deployment/o5_api_internal_settings.json` 中统一管理：

- 可复用 Python/CUDA/SDK 环境；
- 公共基础模型 config/tokenizer/processor 资产目录；
- O5 SDK 版本与 UnitPolicy；
- TP2、LLM Graph、cache、attention；
- Gateway、Backend、Worker 内部地址和端口；
- FRP/HTTPS 默认关闭。

参考音频、评测 case、system prompt 和评测指标不属于部署设置，由 API 请求或上层评测系统决定。

## 7. 运行前提

- 当前进程至少可见两张 CUDA GPU；
- `<model-dir>` 位于运行节点可访问的文件系统；
- 内部固定设置声明的复用环境和基础模型资产可访问。

计算资源如何申请、Job 如何创建和停止，不属于本项目启动入口的职责。
