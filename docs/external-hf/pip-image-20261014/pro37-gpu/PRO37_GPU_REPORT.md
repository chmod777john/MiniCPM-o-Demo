# 37新镜像GPU/API验收

## 接手

用户已明确授权正常停止eval/dev-8gpu-37；root任务见PRO37_GPU_HANDOFF_TASK.md。实施owner docker_lis_package，沿用pip-wheel隔离WT。先保存指定Pod配置/UID/owner，再正常UID限定终止；不删数据/其他Pod，不修改宿主runtime，不push registry/LIS。使用现有79177c66新pip镜像和已验原始权重，独立2GPU。

目前开始只读核现场，尚未关闭Pod或启动GPU工作。

## 指定Pod身份确认

已保存pro37-dev8gpu-before.json和restore.json：eval/dev-8gpu-37 UID cffaf7f3-6e60-4165-ae82-17e186dd325c，node10.20.1.37，创建2026-09-22，占8GPU，ownerReferences为空，无自动重建控制器。挂载eval-pvc到/workspace，删除Pod不删除PVC。完整运行进程/GPU状态保存pro37-dev8gpu-runtime-before.txt。按原30s grace与UID precondition进行API正常DELETE。

指定Pod已通过UID前置条件DELETE正常终止，kubectl wait成功且重新list确认dev-8gpu-37不存在。未修改/删除eval-pvc及其他Pod。此前唯一GPU进程为该Pod的llama-server，后续独立2GPU分配时确认设备空闲。

## 新镜像装载现场

节点runtime=containerd1.7.28-tke.2，当前节点cached images没有本次worker/gateway。镜像目前只有/work/output归档；容器内直接load不能使kubelet可见。现授权禁止host runtime socket和外部registry/LIS推送，不能凭归档直接创建可运行Pod。调查最小方案：在本任务范围临时提供节点loopback只读registry，仅服务已有OCI blobs，让kubelet经正常CRI pull获取；无需改宿主服务或公开push，完成后可删除。实施前给root确认。

## 节点真实CRI拉取/CPU入口通过

loopback registry Pod已启动，只读GET/HEAD且固定digest白名单；PUT405、任意路径404。节点containerd从127.0.0.1:18537成功按gateway固定OCI manifest拉取，25.633秒；实际gateway Pod Running，imageID与已验manifest一致，两health及omni/mobile/mobile-omni均200。准备同一镜像层worker独立申请2GPU，先执行native A/B后启动原镜像入口。详pro37-gateway-events.json、running.json、health.json。

## 网络分配阻塞

worker已调度37，但CNI实际报FailedCreatePodSandBox：tke-route-eni failed to get nominated ip for claim eval/o5-pip-worker-20261014, maybe no available ip in subnet。这是新增Pod IP不足，尚未运行镜像/GPU。Gateway已占10.20.1.74，原8GPU Pod删除后只释放的IP已被Gateway使用。最小范围方案是停止自己刚起的gateway和pending worker，改同一个普通网络Pod中两个独立容器：gateway无GPU/worker仅2GPU，仍用各自固定镜像与入口，共享一个Pod IP，保留正常CNI/CRI、不改宿主网络或集群策略。该方案不增加镜像构建，也不影响其他任务。正在通知root。

## 双卡真实镜像Pod运行

替代Pod o5-pip-validation-20261014 UID 1a0f3098-540c-462d-bf69-5a83082d6e0a 已Running，worker/gateway两个独立容器均按固定OCI digest正常CRI运行。只使用一个Pod IP 10.20.1.74；worker GPU limit=2，实际设备为两张NVIDIA RTX PRO5000 72GB（UUID4fc46…与d90ec…），初始0MiB且无compute进程，Torch确认2设备。原model.pt只读挂载，未修改。开始镜像内native新旧独立进程A/B，API尚未启动。

Native初次尝试未触及GPU输出：capabilities.drop=ALL使容器root不能写属主1000/mode755的/validation，Torch dynamo建立cache PermissionError。只将本任务gpu-validation根目录改mode777，子目录按各自创建者权限，模型/镜像不变。重跑新native侧成功保存17组输出，输入SHA5046dcad8f17f5f930f75e858aad8d7aae5456cc0a739971df16be62b42360d5；旧native独立进程运行中。reference二进制仅放独立诊断目录，不安装进venv或image。

## 新native GPU A/B通过

实际Pro5000 72GB、实际固定worker镜像内，旧native仅通过独立进程按绝对路径作为诊断reference，未替换新venv。17组输出全部逐位相同且finite，relative_l2/max_abs全部0：FA2 5种q/k/维度（1～3000 token），causal prefill 6长度+update输出/状态，dense causal4组（q1～953，k512～4096）。同一输入SHA5046dcad…，native hashes见两个运行日志，精确结果pro37-native-comparison.json。该门不等于整模型hidden精度或性能验收。已批准触发原image entrypoint开始TP2模型加载。

## TP2模型加载与注册通过

原image entrypoint正常启动backend/worker，model.pt加载成功；TP2/pro5000各加速开关均为既有固定配置（LLM/TTS/vocoder graph、merge-unit、retained KV等），启动grouped C++扩展在真实容器成功编译加载。2026-10-14 13:24 UTC backend和worker均health200并按20s幂等刷新向gateway注册成功。模型启动约92s，包括49.94s warmup。开始同历史脚本的12个单工API请求；未改推理实现。
