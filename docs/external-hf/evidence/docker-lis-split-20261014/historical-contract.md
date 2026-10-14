# LIS历史限制：只读核实

2026-10-14，docker_lis_package。未登录、注册、推送或创建任何LIS资源。

## 原始证据

会话 `/user/weihongliang/codex_home/sessions/2026/08/28/rollout-2026-08-28T06-20-21-01a04706-c7fd-7850-bb05-d2496e6c3869.jsonl`：

- line95280/95281，2026-08-28 06:39UTC：原始平台拓扑、TP2/env和MODEL_PATH差异分析。
- line95342，07:29UTC用户问“不改LIS、也不记录session”的打包法；line95358/95359，07:31UTC答固定配置镜像+单一ModelWeight。
- line96316，14:00UTC：真实失败为旧镜像找固定/models/o5-full-bundle、实际注入MODEL_PATH；并发现8个单卡worker与TP2不符、生产注册403。

本地LIS源码 `/user/weihongliang/lis-source` HEAD仍为历史固定`08f73fd63a0da5c8fac12813079f6dddd937d7e2`，不是当前远端最新版。

## 当时确认的契约

- `realtime_base_contract.py:394`拒绝gateway_command/gateway_args/worker_command/worker_args非空：publisher runtime不能覆盖image entrypoint。builder虽有这几个分支，但canonical契约不允许调用者使用。不能说Kubernetes本身不支持command。
- contract字段白名单无任意extra env；worker builder只注入POD_NAME/NAMESPACE/INDEX、WORKER_ID/ENDPOINT/GPU_GROUP、GATEWAY_REGISTRY_URL、GPU_ID、WORKER_PORT、BACKEND_PORT、MODEL_PATH。O5_DEPLOY_MODE、加速组合需由镜像ENV/入口固定。
- ModelWeight由init model-downloader下载，挂一个统一根目录；`MODEL_PATH=/mnt/model-cache/<artifact cache key>/model`。历史错误是镜像忽略此变量，绝非必须safetensors；当前可以在该根下放raw model.pt和token2wav_causal_vocoder。
- gateway Deployment + worker StatefulSet。每worker Pod `gpu_count=2`、worker_count=1；并非2个单卡worker。当前三物料TP2在一个进程中使用两卡，不沿用旧torchrun实现。
- gateway HTTP，TLS上游终止；历史仅路由精确/v1/realtime，页面不自动公开。recording=false，gateway数据emptyDir。
- worker endpoint使用Pod名+headless Service+namespace FQDN；registry URL使用gateway ClusterIP DNS及内部注册端口。历史模板不是依赖任意容器hostname可解析。

## 当前v9与“可直接LIS部署”的差距

镜像已经固化外部HF/accel、TP2/graph等ENV，读取MODEL_PATH，worker角色可读WORKER_ID/WORKER_ENDPOINT/GATEWAY_REGISTRY_URL；本轮仅standalone同容器的API已验。

尚不能宣称直接LIS部署：

1. 当前LIS在线契约/权限未查（只读auth401），旧源码不能代表当前平台。
2. 当前entrypoint注册只做一次PUT，没有历史contract的retry/backoff/定时refresh；gateway重启后的重注册未验。
3. 当前gateway角色不读取历史挂载`/app/config.json`/`/etc/lis/publisher-runtime`，而自写O5_DEMO_CONFIG_PATH；recording=false及平台端口/状态契约未接入。历史builder给gateway追加CLI args，当前entrypoint不解析argv，只读ENV，因此非默认端口存在缺口。
4. 当前backend绑定127.0.0.1；历史contract `WORKER_BACKEND_BIND_HOST=0.0.0.0`。同容器worker能访问，但模板探针/状态服务是否要求Pod级backend可达须核实。
5. 当前v9是local worker image，默认worker。需另建同v9的固定gateway角色薄层，分别绑定template；旧v6 gateway不能默认为当前最终产物。
6. LIS模型artifact上传、模板2GPU资源、注册/服务启动和真实跨Pod连通均未做；v9没推送。

应先拿当前template/runtime契约验证以上连接与生命周期点，再以固定gateway/worker两镜像+一个raw模型artifact交付。不能把34standalone成功等同LIS多Pod启动成功。
