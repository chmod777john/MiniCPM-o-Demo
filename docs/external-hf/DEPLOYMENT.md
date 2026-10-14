# 外部 HF + accel + O5 Demo 启动

本分支基于 `o5-fc-dev@732b0758`。保留原 gateway、worker、API、runtime 和 demos
组织，只让 O5 从外部 HF 和原始 checkpoint 加载；没有应用目录迁移。

## 三物料

|物料|路径|基线/作用|
|---|---|---|
|HF|`/user/weihongliang/worktrees/minicpm-o-5-fix-3d-rope-o5fc-20261013-hf-chat-20261014`|`5cccbf7`：3D-RoPE + 固定源 FC 提取 + 完整历史单工|
|accel|`/user/weihongliang/worktrees/hf-accel-no-fc-3d-rope-20261009-canvas-window-20261010-combined-batch-tts-20261010-exact-window-20261010-bulk-window-20261010-paired-rotation-20261010-direct-table-20261010-o5fc`|独立 attach/patch，含 graph；可不加载|
|Demo|`/user/weihongliang/worktrees/o5-fc-dev-external-hf-20261013-hf-chat-20261014`|`47cb5315`：外部HF单工、普通双工和移动端桥接|

三者是独立 Git worktree。HF 的 FC 提取入口为 `fc/migration/extract.py`，
`--check` 可重现源码与适配清单；HF 不承担 graph 或服务 warm 策略。

## 环境

使用 `/user/weihongliang/environments/o5-3drope-fc-20261013/.venv`。
它继承 high-cu128-flashattn 计算依赖，并在项目 venv 安装官方 SDK0.0.6、
内部 causal minicpmo-utils1.0.7；具体 wheel 哈希和依赖声明例外见同目录 README。
不要直接用上游 requirements.txt 覆盖这个已验证组合。

## 双卡 A100 启动示例

```bash
export PROJECT_DIR=/user/weihongliang/worktrees/o5-fc-dev-external-hf-20261013-hf-chat-20261014
export VENV_DIR=/user/weihongliang/environments/o5-3drope-fc-20261013/.venv
export O5_CANONICAL_PATH=/user/weihongliang/worktrees/minicpm-o-5-fix-3d-rope-o5fc-20261013-hf-chat-20261014
export O5_CHECKPOINT_PATH=/backup/user/chenmoye/checkpoints/minicpm-5o-moe/0930_final_merge/0930_omni_baseline_mergeAll_agent_128k_176gpu/iter_300.pt
export O5_ACCELERATION_PATH=/user/weihongliang/worktrees/hf-accel-no-fc-3d-rope-20261009-canvas-window-20261010-combined-batch-tts-20261010-exact-window-20261010-bulk-window-20261010-paired-rotation-20261010-direct-table-20261010-o5fc
export O5_ASSETS_DIR=/user/shangchuxiang/code/MiniCPM-o-4_6/local_assets/token2wav_causal_vocoder
export CUDA_VISIBLE_DEVICES=0,1 O5_MOE_PARALLEL=tp O5_DEPLOY_MODE=tp2
export O5_ACCELERATION_HARDWARE=a100 O5_ATTN_IMPLEMENTATION=sdpa
export DUPLEX_SLOT_PROTOCOL=1
export O5_LLM_GRAPH=1 O5_TTS_GRAPH=1 O5_VOCODER_GRAPH=1
export CANONICAL_UNIT_PREFILL_BATCH=1 O5_COMPACT_LOGITS=1
export O5_TTS_FA2_WHOLE_DECODE=1 O5_CANVAS_WINDOW_EXACT_CHAIN=1
export BACKEND_HOST=127.0.0.1 BACKEND_PORT=23514
export WORKER_HOST=127.0.0.1 WORKER_PORT=23414
export GATEWAY_HOST=0.0.0.0 GATEWAY_PORT=18141 GATEWAY_INTERNAL_PORT=18142 GATEWAY_HTTPS=0
export ENABLE_FRP=0 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
export LOG_DIR=/path/to/isolated/run/logs
export O5_DEMO_CONFIG_PATH=/path/to/isolated/run/config.json
export REF_AUDIO_PATH="$O5_CANONICAL_PATH/assets/system_ref_audio.wav"
unset PYTHONPATH FC_DEPLOYMENT_PROFILE O5_MODEL_PROFILE
bash "$PROJECT_DIR/scripts/start_o5_tp2_cctl_service.sh"
```

config.json 至少配置 model.model_path、model.pt_path、model.deployment_mode="tp2"、
model.compile=false 和独立 service.data_dir。完整已运行脚本参考
`/user/weihongliang/analysis-runs/20261014-hf-simplex/job_entry.sh`
及同目录 config.json；最终参考 job **1218128**，agent-dev 双A100。
该任务用于隔离验收；运行状态请通过cctl查询，不能将Pod地址当作持久公网服务。CUDA compat 与 allocator 设置以该
脚本和对应集群环境为准。不要另加 torchrun 两进程，accel TP 在单进程内管理双卡。

FC checkpoint 可替换为：
`/user/chenjinpeng/Checkpoints/job_1054182/checkpoints/iter_0001275/job_1054182_iter_0001275.pt`。
SDK registry 解析 slot token；不改 tokenizer.json、不随机 resize 权重。
HF 在请求需要 slot/FC 时核对已训练 embedding/lm_head 行数。

## 会话和验证

- 普通视频用 `mode=video`、普通音频用 `mode=audio`；FC 用 API v4 `mode=audio`，音频路径按 v4 转成 base64 WAV。
- 同一服务可以顺序运行普通和 FC session，不由启动选项二选一。
- 普通滑窗、LS、HD、max speak 等选项见 `ORDINARY_OPTIONS.md`。当前普通页面已移除
  Length Penalty / Draft 无效控件；旧 API 客户端传 length_penalty 会警告并回传 inactive。
  未知字段不再静默丢弃。
- `docs/external-hf/evidence/` 分开记录普通、FC、tool result、模式复位实测。
- native 对 accel 精度比较必须使用同一 attention 后端策略；在本环境测试显式关闭
  cuDNN SDPA，与 accel 内部已有策略一致。合并 prefill 是已记录的 BF16 数值不同路径，
  不能将 ordered feed 对拍结论当作 merge-unit 逐位一致。
- 本轮代表性短上下文验收不能替代 64K/128K 全性能矩阵，也不是完整评测质量分数。

## 普通音视频双工页面（2026-10-13 迁移）

- `/omni`：普通音视频双工，摄像头/麦克风或本地视频。
- `/audio_duplex`：普通音频双工，麦克风或本地音频。
- 新版 Gallery / demos 仍使用本分支原有实现，本次未引入旧 FC demo 应用。
- 页面部署需关闭 `GATEWAY_API_ONLY`（unset 或设为 0），且 app registry 对应页面 enabled。
  API-only 部署按原规则不提供这两个页面。

两页默认 Basic（8000 / 6000），MaxKV 自动停止阈值默认和上限均为 256K。
MaxKV 与滑窗水位是独立选项。可在 gateway 的环境中设置：

```bash
export O5_FRONTEND_SLIDING_WINDOW_MODE=basic
export O5_BASIC_WINDOW_HIGH_TOKENS=131072
export O5_BASIC_WINDOW_LOW_TOKENS=98304
# 可选：限定普通页面可选的模式，默认模式应包含在内
export O5_FRONTEND_SLIDING_WINDOW_MODES=off,basic,kv_surgery
```

这只是页面启动默认值，后端由每次 session.init 的 config 配置，不改变 HF 源码。
普通页面的长会话不再应用 gateway 默认 20/40 分钟上限；仍可通过
`FC_API_SESSION_MAX_DURATION_S` 显式配置时长。断线恢复限同一 gateway 进程，
30 秒/16 MiB，细节和边界见 ORDINARY_OPTIONS.md。

本分支最新单工/移动API验收见 `SIMPLEX.md`、`MOBILE_NOFC_ALIGNMENT.md`；
原始三物料普通/FC精度及API范围见 `ACCEPTANCE.md`。当前短unit集成测试不替代长对话性能验收。

## 单工与移动端

- `/turnbased`：桌面单工；流式/非流式、文字/音频都由外部HF的 `chat` 实现。
- `/mobile/`：React移动单工与音频双工入口。
- `/mobile-omni/`：移动视频双工，复用桌面Omni运行时。

部署前用项目声明的Node/Bun在 `frontend/mobile` 下安装本地依赖并运行 `npm run build:static`。
它生成被Git忽略的 `static/mobile/`，同时更新Omni壳和widget；只checkout不会自动生成移动聊天页。
当前测试使用项目本地依赖和Node22.23.3。不要用系统Node16。

单工输入默认8192上限；语音单工要求enable_thinking=false。详细选项、取消协议和验收限制见 `SIMPLEX.md`。
