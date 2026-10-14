# Root接收：37构建与权重传输

本次合入打包分支 `b30b5aa68a03bdbb9bef2c440129c9c52244e49f`，目标为现有Demo工作分支 `o5-fc-dev-external-hf-20261013-hf-chat-20261014`。保留root自身worktree-readme；HF与accel计算源码没有修改。

## 已确认

- 新venv内pip安装137个固定版本包；FA2/causal/dense来自固定源码新编译，原始来源和SHA已记录。
- 镜像构建commit固定 `79177c66`，后续文档和本次合入不改变已构建镜像身份。
- 既有执行证据：14项CPU门、Gateway健康/静态页、1444文件与137包版本一致性、四归档完整SHA及blob检查通过。
- Root直接检查37专用Pod中的交付文件：model.pt为74606011097B，四归档和证据包在位，vocoder8文件，.part与锁不存在。
- checkpoint完整SHA和vocoder逐文件SHA以packager持久记录为准；root本轮未重复读取74.6GB重新计算。
- 两项明确metadata例外仍保留，pip check真实exit1；不能称依赖零冲突。

## 验收边界

新镜像GPU算子A/B、双卡模型加载、真实TP2 API和性能尚未验收。此次合入确认打包实现与构建/传输交付，不批准将旧镜像GPU结果继承为本轮结果。未推送registry、未部署LIS；同事Pod/GPU未动。

专用BuildKit Pod `eval/o5-pip-buildkit-20261014` 保留，由root管理；交付目录 `/data/weihongliang/o5-pip-image-20261014`。下一步是双卡资源协调完成后执行新镜像GPU/API验收。

## 证据

本目录 `FINAL_HANDOFF.md`、`DELIVERY_MANIFEST.json`、`checkpoint-final-state.json`。完整root审查位于 `/user/weihongliang/analysis-runs/20261014-docker-pip-install/ROOT_ENV_REVIEW.md`。本轮仅源码/记录审查及远端文件状态核对，没有重新运行模型测试。
