# O5 TP2 封装理想形态

本文记录 TP2 双卡推理的理想封装边界。目标是：相对于 `single_opt`，开启双卡只是一种部署 / 执行后端切换，不要求 processor、chat、duplex、TTS 等业务逻辑维护另一套分支。

## 目标

理想状态下，业务层仍然只看到一个普通 backend/model，更底层的
`MiniCPMO` 也仍然只看到一个普通 HuggingFace 风格 LLM 对象：

```text
MiniCPMO / chat / streaming / duplex
        |
        v
self.llm  # LLM-compatible object
        |
        +-- single_opt: local Qwen3_5MoeForCausalLM
        +-- tp2:        DistributedTPLLM
```

业务层继续调用普通 backend API：

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

这里的目标不是只包住 `forward`。`self.llm` 必须作为一个完整的
LLM-compatible object 暴露当前 MiniCPMO 实际依赖的 contract：

```text
forward / __call__
generate
config / generation_config
model / model.embed_tokens
lm_head
get_input_embeddings / set_input_embeddings
get_output_embeddings / set_output_embeddings
parameters / eval / train / to / cuda / bfloat16 等 nn.Module lifecycle
past_key_values / cache_position / position_ids 的 forward 语义
必要的属性透传
```

也就是说，上层不应该写：

```python
if deployment_mode == "tp2":
    ...
else:
    ...
```

也不应该为了 TP2 在 `chat`、`streaming_generate`、`duplex_generate`
等流程里维护第二套调用路径。差异应集中在 `self.llm` 的构造和
`DistributedTPLLM` 内部。

本项目当前还必须尊重 Kaiqi 优化分支已经存在的 `LLMGraphRunner` 边界。
`LLMGraphRunner` 在 `MiniCPMO45.utils.StreamDecoder.feed()` 中创建和推进，
维护 `_static_pos`、StaticCache、graph input buffer 和 attention mask buffer。
因此：

```text
O5_LLM_GRAPH=0: 可以继续探索 LLM-boundary sync。
O5_LLM_GRAPH=1: TP2 同步必须包住 StreamDecoder.feed 所在的 model/backend step，
                 让每个 rank 都完整推进本地 graph/cache 状态。
```

这里不是要重构或隐藏 `LLMGraphRunner`。`LLMGraphRunner` 是既有加速能力，
TP2 当前目标是适配它的状态边界，而不是把它下沉到 `self.llm.model`。

如果 LLM 层暂时做不到覆盖 graph 路径，退一步可以封装在 backend/model public API 层：

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

## 当前分支的边界选择

当前 TP2 分支相对于 `single_opt` 已经基本把大块新增放在 `core/deploy` 和 `tools/o5deploy`。
边界选择分两档：

```text
O5_LLM_GRAPH=0
  使用 DistributedTPLLM，在 self.llm / self.llm.model 边界同步 forward/generate 调用。

O5_LLM_GRAPH=1
  使用 outer SpmdMirror，镜像 backend/model public calls。
  这是为了保留 Kaiqi 的 LLMGraphRunner 位置，确保 rank0/rank1 都执行完整
  StreamDecoder.feed()，从而同步推进 graph/cache/static_pos/mask 状态。
```

这不是简单退回 `tp2-encapsulation`。`DistributedTPLLM` 仍然固化了 LLM contract，
并服务于无 graph 路径和后续更细边界探索；但生产 graph 路径当前以正确性为先，
把 TP2 放在 `LLMGraphRunner` 的外层。

仍需关注的泄漏点：

1. `core/processors/unified.py` 里有 `_mcall()`，并且 `duplex_prepare/prefill/generate/finalize` 显式通过 mirror 调用。
2. 单工 chat 路径也被调整为更靠近 `prepare + generate`，以便 serving mirror。
3. `py_backend/server.py` 有 backend-level `_spmd_call()`，用于 mirror `chat_complete`、`duplex_prefill` 等 backend 方法。
4. `MiniCPMO45/modeling_navit_siglip_fast.py` 和 `MiniCPMO45/utils.py` 有少量兼容修复。这些应尽量作为通用修复进入 `single_opt` base，而不是 TP2 专属差异。

这些改法能跑通，但说明 TP2 runtime 仍然泄漏到了 processor/server 层。

当前 `wt/o5-tp2-llm-wrapper-2026-07-15` 进一步把 TP2 backbone 显式包成
`core.deploy.llm_wrapper.DistributedTPLLM`。它目前是行为代理，目的是先把
LLM contract 固化为代码边界。`O5_LLM_GRAPH=1` 时，backend public API mirror
仍保留且是有意选择，用来保证 rank1 和 rank0 进入相同的 `StreamDecoder.feed()`
调用序列。

真正把 mirror 下沉到 LLM 层时，rank0 不能只广播“调用了 forward”这个事件。
还必须把 `inputs_embeds`、`position_ids`、`attention_mask`、`cache_position`
等本次 LLM 调用输入同步到 worker rank，并保证每个 rank 维护自己的
KV cache / StaticCache 状态。否则 worker rank 没有音频、视觉、TTS 前序逻辑
产生的中间张量，无法独立进入等价的 tensor-parallel forward。

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
6. 当前阶段不重构 `LLMGraphRunner`。优化工作聚焦于：TP2 如何按 graph/no-graph 两档选择正确同步边界，并把这个选择集中在 `core/deploy`。

## 本次迭代

上一轮先完成一层 backend public API 封装，不触碰更底层的 LLM forward 封装：

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

本轮开始建立 LLM-compatible wrapper：

1. 新增 `core/deploy/llm_wrapper.py`。
2. TP2 builder 不再把 raw HF TP backbone 直接挂到 `model.llm`，而是挂
   `DistributedTPLLM(tp)`。
3. `DistributedTPLLM` 透传当前 MiniCPMO 依赖的 LLM contract，包括
   `forward`、`generate`、embedding/head、`config`、`generation_config`、
   `.model`、`.lm_head`、`device/dtype` 和必要属性。
4. 这一步保持行为不变，不移除 backend mirror；它为后续把 SPMD 同步从
   backend public API 下沉到 LLM wrapper 提供明确落点。

随后新增实验模式 `tp2_llm`：

1. `tp2` 仍是当前可用路径：rank1 镜像 backend/model public method。
2. `tp2_llm` 改为 rank1 进入 `DistributedTPLLM.worker_loop()`，rank0 每次
   `self.llm.forward()` / `self.llm.generate()` 广播调用描述和输入 tensor。
3. worker rank 不再运行 audio/vision/TTS 外层流程，而是在 LLM wrapper 内维护
   自己的 `past_key_values`。
4. 当前已用 CPU/gloo 小型分布式脚本验证两次 forward 的 tensor 同步和 worker
   cache 递进；真实 O5 端到端还需要 GPU 验证。

这使代码具备目标形态的落点，但真实 O5 graph 路径暴露了一个重要边界：
`LLMGraphRunner` 的状态在 `StreamDecoder.feed()`，不在裸 `self.llm.model` 内。
因此 `tp2_llm` 当前按 `O5_LLM_GRAPH` 分档：

```text
O5_LLM_GRAPH=0
  rank1 进入 DistributedTPLLM.worker_loop()，同步 LLM-boundary 调用。

O5_LLM_GRAPH=1
  rank1 进入 SpmdMirror.worker_loop()，镜像 backend/model public calls；
  token_broadcast=True；每个 rank 本地 capture/replay LLMGraphRunner。
```

验证过的可用组合：`.venv-accel + O5_ATTN_IMPLEMENTATION=auto -> flash_attention_2 +
O5_LLM_GRAPH=1 + O5_LLM_CACHE=32768`，audio/video realtime probe 均通过，
日志显示 LLM/TTS graph capture OK。

## 验收标准

1. `single_opt -> tp2` 只需改 config 和 launcher。
2. processor 中没有 rank / mirror / `_mcall`。
3. modeling 中没有 TP2-specific branch。
4. rank1 在 server bind 前进入 worker loop。
5. 所有 public backend/model API 在 single_opt 和 tp2 下语义一致。
6. rank0/rank1 出错时能一起退出，不会单边挂死。
7. 32K cache、token broadcast、SPMD descriptor 只存在于 deploy/runtime 层。
8. `O5_LLM_GRAPH=1` 时，TP2 同步边界必须覆盖 `LLMGraphRunner` 所在的
   `StreamDecoder.feed()` 调用序列；不要求本阶段把 `LLMGraphRunner` 下沉到
   `self.llm.model`。
