# O5 Demo 使用交付说明 2026-07-28

## 交付分支

- 代码目录：`/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26-session-replay-canonical-2026-07-26`
- 分支：`wt/o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26-session-replay-canonical-2026-07-26`
- 代码基线 commit：`30007a5bcf3ef42509183f8221372fc08fc36e96`
- Codeup remote：
  - `git@codeup.aliyun.com:thunlp/demo/MiniCPM-o-Demo.git`
  - `git@codeup.aliyun.com:modelbest/multi-modal/MiniCPM-o-Demo.git`

## 当前可用服务

- cctl job id：`631499`
- cctl task：`tasks/631499`
- 状态：`Running`
- 页面地址：`https://82.157.64.212:8040/omni`
- 健康检查：`https://82.157.64.212:8040/health`
- health 返回：`{"status":"healthy"}`
- cctl 查看启动参数：`cctl job get 631499 -o json`

## 运行环境

- project：`o5`
- cluster：`langfang_train`
- resource pool：`deploy`
- resourcePoolId：`534`
- image：`cybertron/minicpmo:202506140642-bb9a49`
- GPU：`2 x A100`
- CPU：`32`
- memory：`256Gi`
- venv：`/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel`

## 模型与权重路径

- `MODEL_PATH=/user/weihongliang/MiniCPM-o-4_6`
  - 模型 trusted-code / config / processor / tokenizer / assets 目录。
- `PT_PATH=/user/weihongliang/o5_weights/houyueran_o5_MB_omni-sft2-8k-hyr_a2_i2_0722_iter_0002800.pt`
  - O5 checkpoint 权重。
- `BACKBONE_DIR=/user/weihongliang/o5_weights/o5_backbone_hf_houyueran_o5_MB_omni-sft2-8k-hyr_a2_i2_0722_iter_0002800`
  - TP2 使用的 HF / safetensors backbone 目录。

## 关键启动参数

- `O5_DEPLOY_MODE=tp2_llm`
  - 双卡 TP2 LLM 部署模式。
- `O5_LLM_GRAPH=1`
  - 开启 LLM graph 优化路径。
- `O5_LLM_CACHE=32768`
  - LLM StaticCache 使用 32K cache。
- `O5_ATTN_IMPLEMENTATION=auto`
  - attention 实现由运行环境自动选择。
- `O5_DETERMINISTIC_REPLAY=1`
  - 开启可复现实验相关默认值。
- `O5_SESSION_SEED=0`
  - session 默认 seed 固定为 0。
- `O5_TTS_ARGMAX=1`
  - TTS token 使用 argmax，减少采样随机性。
- `ENABLE_FRP=1`
  - 启动 frp，把服务透出到 `82.157.64.212:8040`。

## 端口

- public URL：`https://82.157.64.212:8040/omni`
- frp 对外端口：`8040`
- gateway port：`8009`
- gateway internal port：`8010`
- backend port：`22510`
- worker port：`22410`

## 完整 cctl 启动命令

```bash
cctl job create \
  --project o5 \
  --cluster langfang_train \
  --resource-pool deploy \
  --image cybertron/minicpmo:202506140642-bb9a49 \
  --billing-account-id N00002 \
  --gpu 2 \
  --gpu-model A100 \
  --cpu 32 \
  --memory 256 \
  --description "o5 original model_path TP2 8040 20260728" \
  --entry 'bash -lc "cd /user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26-session-replay-canonical-2026-07-26 && MODEL_PATH=/user/weihongliang/MiniCPM-o-4_6 PT_PATH=/user/weihongliang/o5_weights/houyueran_o5_MB_omni-sft2-8k-hyr_a2_i2_0722_iter_0002800.pt VENV_DIR=/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-accel BACKBONE_DIR=/user/weihongliang/o5_weights/o5_backbone_hf_houyueran_o5_MB_omni-sft2-8k-hyr_a2_i2_0722_iter_0002800 O5_DEPLOY_MODE=tp2_llm O5_LLM_GRAPH=1 O5_LLM_CACHE=32768 O5_ATTN_IMPLEMENTATION=auto O5_DETERMINISTIC_REPLAY=1 O5_SESSION_SEED=0 O5_TTS_ARGMAX=1 ENABLE_FRP=1 FRPC_BIN=/user/weihongliang/frp_0.65.0_linux_amd64/frpc FRPC_CONFIG=/user/weihongliang/frp_0.65.0_linux_amd64/frpc_o5_session_replay_tp2_8009_8040.toml GATEWAY_PORT=8009 GATEWAY_INTERNAL_PORT=8010 BACKEND_PORT=22510 WORKER_PORT=22410 LOG_DIR=/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26-session-replay-canonical-2026-07-26/run-logs/o5_original_modelpath_8040_$(date +%Y%m%d_%H%M%S) bash scripts/start_o5_tp2_cctl_service.sh"'
```

## 日志

- 当前 job 日志目录：
  `/user/weihongliang/MiniCPM-o-Demo-wt-o5-no-fc-speedup-tp2-thin-unified-2026-07-24-api-canonical-align-2026-07-26-session-replay-canonical-2026-07-26/run-logs/o5_original_modelpath_8040_20260728_070421`
- cctl 日志命令：
  `cctl job logs 631499`

启动成功日志里可以看到：

```text
Loading model from /user/weihongliang/MiniCPM-o-4_6
[deploy] built via framework: mode=tp2_llm world=2 rank=0 ...
Backend server ready
```
