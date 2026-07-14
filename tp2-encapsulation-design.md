# O5 TP2 封装理想形态

本文记录 TP2 双卡推理的理想封装边界。目标是：相对于 `single_opt`，开启双卡只是一种部署 / 执行后端切换，不要求 processor、chat、duplex、TTS 等业务逻辑维护另一套分支。

## 目标

理想状态下，业务层仍然只看到一个普通 backend/model：

```python
backend = create_backend(config)

backend.chat_complete(...)
backend.duplex_prepare(...)
backend.duplex_prefill(...)
backend.duplex_generate(...)
backend.duplex_finalize(...)
```

切换单卡优化和双卡 TP2 只改配置和启动方式：

```json
{
  "model": {
    "deployment_mode": "single_opt"
  }
}
```

```json
{
  "model": {
    "deployment_mode": "tp2",
    "backbone_dir": "/path/to/o5_backbone_hf",
    "llm_cache_len": 32768
  }
}
```

业务代码不应该感知这些概念：

```text
rank / world_size
torch.distributed
SPMD descriptor
dist.broadcast
token broadcast
tp_plan
worker_loop
```

## 进程模型

短期仍建议用 `torchrun` 作为 TP2 的启动方式：

```bash
torchrun --nproc_per_node=2 python -m py_backend.server
```

语义上只有 rank0 是 server：

```text
rank0: 初始化 backend，监听 HTTP/WebSocket，接真实请求。
rank1: 初始化 backend，不监听端口，进入 worker loop，镜像 rank0 的模型计算。
```

因此进程数是 2，但真正 bind 端口的 server 只能有 1。rank1 必须在 server bind 之前被 deployment runtime 接管。

理想 server 入口只保留很薄的分流：

```python
def main():
    config = load_config()
    runtime = DeploymentRuntime.from_config(config)

    if runtime.worker_only:
        runtime.run_worker_loop()
        return

    backend = runtime.build_backend()
    serve(backend)
```

`server.py` 可以知道 `worker_only`，但不应该知道 descriptor、broadcast、token sync 等细节。

## 构建边界

`backend_factory.py` 的理想职责只是分发到 deployment runtime：

```python
def create_backend(config):
    runtime = DeploymentRuntime.from_config(config)
    return runtime.build_backend()
```

不同模式由 registry / builder 处理：

```text
single_eager -> 原始单卡模型
single_opt   -> 单卡优化模型
tp2          -> 双卡 TP 模型
```

`backend_factory.py` 不应该包含模型加载细节、rank 通信细节或 duplex 特判。

## 模型执行边界

最理想的封装是：TP2 只是 `self.llm` 的一种实现。

```text
single_opt: self.llm = LocalOptimizedLLM(...)
tp2:        self.llm = DistributedTPLLM(...)
```

上层 modeling 仍然写普通调用：

```python
outputs = self.llm(
    inputs_embeds=inputs_embeds,
    position_ids=position_ids,
    attention_mask=attention_mask,
    past_key_values=cache,
    use_cache=True,
)
```

如果这一层暂时做不到，退一步可以封装在 backend/model public API 层：

```text
chat_complete
duplex_prepare
duplex_prefill
duplex_generate
duplex_finalize
```

但不希望 processor 里显式写 `_mcall("duplex_generate")`。processor 应该像调用普通 backend 一样调用这些方法。

## Distributed Runtime 职责

TP2 相关逻辑集中在 `core/deploy` 或后续的 `core/deploy/runtime.py`：

```text
rank/world 初始化
rank1 worker loop
rank0/rank1 调用同步
descriptor broadcast
token broadcast
barrier/synchronize
错误传播与退出
```

这些逻辑不应该出现在：

```text
core/processors/unified.py
chat view
duplex view
websocket handler
TTS 状态机
业务 API schema
```

## 当前分支的泄漏点

当前 TP2 分支相对于 `single_opt` 已经基本把大块新增放在 `core/deploy` 和 `tools/o5deploy`，但仍有几个泄漏点：

1. `core/processors/unified.py` 里有 `_mcall()`，并且 `duplex_prepare/prefill/generate/finalize` 显式通过 mirror 调用。
2. 单工 chat 路径也被调整为更靠近 `prepare + generate`，以便 serving mirror。
3. `py_backend/server.py` 有 backend-level `_spmd_call()`，用于 mirror `chat_complete`、`duplex_prefill` 等 backend 方法。
4. `MiniCPMO45/modeling_navit_siglip_fast.py` 和 `MiniCPMO45/utils.py` 有少量兼容修复。这些应尽量作为通用修复进入 `single_opt` base，而不是 TP2 专属差异。

这些改法能跑通，但说明 TP2 runtime 仍然泄漏到了 processor/server 层。

## 理想文件差异

相对于 `single_opt`，理想 TP2 增量应主要是：

```text
A core/deploy/base.py
A core/deploy/registry.py
A core/deploy/runtime.py
A core/deploy/modes.py
A core/deploy/spmd.py
A core/deploy/launch_tp2.sh
A core/deploy/config.example.tp2.json

M config.py
M py_backend/server.py
M core/processors/backend_factory.py

A tools/o5deploy/extract_backbone.py
A tools/o5deploy/deploybench.py
A tools/o5deploy/grid_2card.py
A tools/o5deploy/live_client.py
```

尽量避免或消除：

```text
M core/processors/unified.py
M MiniCPMO45/modeling_minicpmo.py
M MiniCPMO45/modeling_minicpmo_unified.py
M MiniCPMO45/utils.py
```

如果必须改这些文件，应优先判断是否是通用 bug fix 或 `single_opt` 应有能力，而不是 TP2 特判。

## 迁移方向

1. 把 `_mcall` 从 `DuplexView` 下沉到 backend/model wrapper，使 processor 调用普通方法即可。
2. 把 `_spmd_call` 从 `server.py` 下沉到 `DistributedBackend`，让 server 只处理 `worker_only` 分流。
3. 把 rank1 worker 入口集中到 `DeploymentRuntime.run_worker_loop()`。
4. 明确 `single_opt` base：所有 batched_mm、LLM graph、TTS graph、vocoder graph、fuse feed、lmhead 优化都属于 base，不应作为 TP2 diff 重新出现。
5. 将 TP2 特有能力限制为：TP backbone 加载、distributed runtime、token broadcast、32K StaticCache 配置。

## 本次迭代

本次先完成一层 backend public API 封装，不触碰更底层的 LLM forward 封装：

1. `PyTorchBackend.load_model()` 在发现模型带 `_spmd_mirror` 后，自动包装需要跨 rank 同步的 backend public methods。
2. `DuplexView` 不再显式 `_mcall()`，恢复为普通 `self._model.duplex_*()` 调用。
3. `py_backend/server.py` 不再使用 backend-level `_spmd_call()`，chat/duplex handler 只调用普通 backend 方法。
4. rank1 startup 仍由 `server.py` 做 worker-only 分流；这是当前保留的启动层泄漏，后续可继续下沉到 deployment runtime。

验证结果：

```text
non-streaming chat: 通过，返回“测试”
streaming chat:     通过，返回 delta + done
duplex smoke:       通过，1s 静音 force_listen 返回 listen
```

一个重要修正是 `chat_prefill` 也必须进入 backend wrapper。streaming chat 是 `chat_prefill -> chat_streaming_generate` 两段调用；如果只 mirror generate，rank0/rank1 会进入不同的 collective 序列，最终触发 NCCL timeout。

## 验收标准

1. `single_opt -> tp2` 只需改 config 和 launcher。
2. processor 中没有 rank / mirror / `_mcall`。
3. modeling 中没有 TP2-specific branch。
4. rank1 在 server bind 前进入 worker loop。
5. 所有 public backend/model API 在 single_opt 和 tp2 下语义一致。
6. rank0/rank1 出错时能一起退出，不会单边挂死。
7. 32K cache、token broadcast、SPMD descriptor 只存在于 deploy/runtime 层。
