# Docker API 验收：2026-10-14

## 验收结论与范围

34机器上的真实Docker镜像已跑通单工、普通音视频双工和FC音频API。最终v6包装层的文字/语音代表请求、生成前端文件散列、进程关闭故障检查、gateway健康/关闭及静态页面再次通过。

本轮使用一张空闲RTX PRO6000 Blackwell Server Edition 96GB，`single_eager`，不调用accel.attach；另一张卡既有VLLM服务未动。镜像包含Pro5000 TP2运行依赖与启动默认值，但本轮不是Pro5000双卡加速性能/数值验收。普通双工短样例均listen；不代表实际窗口裁剪。FC使用录制工具结果，不是在线天气后端验收。

## 三物料及运行环境

|组成|镜像内路径|实际应用版本|
|---|---|---|
|HF|`/opt/o5/hf`|`5cccbf73e8ca7bc01b0377eff299891e9dd45248`|
|accel|`/opt/o5/accel`|`6e33931cdf7e1fcff2f896d706bb73b949fc4a39`|
|Demo|`/opt/o5/demo`|`08c22855e618e8b979d1b11a496ee9551146fd18`|
|包装代码|`/opt/o5/package`|`01716c8290fea71a1f47b2a2b02fb699de06e3c9`（root合入为`341eca22`）|

应用版本在完整API矩阵后未修改；后续包装修正处理ninja可执行文件、关闭竞态、构建清单以及旧前端文件残留。source-manifest与release-input-manifest分开记录应用来源和包装层，不把包装commit误记为推理源码commit。

使用Pro5000实际项目venv快照（source Python3.10.21，容器Python3.10.22，同cp310 ABI）：Torch2.14.0+cu130、Transformers5.13.0、Triton3.8、FlashAttention2.8.3.post1、FLA0.5.1。单独安装SDK0.0.6与minicpmo-utils1.0.7，不用旧requirements重新解析数值运行库。实际native CUDA primitive调用及Torch C++扩展编译通过。

## 权重

直接使用FC原始checkpoint，未导出或重新打包权重：

- 源：`/user/chenjinpeng/Checkpoints/job_1054182/checkpoints/iter_0001275/job_1054182_iter_0001275.pt`。
- 34：`/home/weihongliang/o5-docker-lis-20261014/model/model.pt`；同目录包含`token2wav_causal_vocoder/`。
- bytes：74606011097。
- SHA256：`0789c953677dc8ea71b6085543e1a946ae64bddecd2f6b911ac6380376ef01ff`，源/目的完整文件一致。
- 只删除了用户指定、确认无容器挂载/进程使用的旧legacy FC目录，其他权重/服务未动。allmerge未传到34。

## 真实API矩阵

|范围|结果|说明|
|---|---|---|
|单工7项|全部PASS|文字流/非流、语音流/非流、HD图片、音频输入、视频+语音|
|普通双工|音频2unit+视频2unit均PASS|实际移动helper请求、finalize+close；均listen|
|FC音频|8unit PASS|模型生成3次tool call，返回录制结果3次、后续unit消费2次；末unit之后未再推进|
|FC后回归|文字/语音2项PASS|同一模型服务切回chat|
|最终v6代表API|文字/语音2项PASS|推理应用源码未改，包装与静态文件修正后复测|
|最终镜像内容/关闭|PASS|package/mobile逐文件散列、CPU关闭竞态、gateway health200/stop exit0、页面HTTP200|

语音PCM已检查finite且非零。这里只确认API功能；未做人耳音质、真机浏览器、完整FC质量评测或实时性验收。原始响应的seconds是请求时间，含队列/冷编译等因素，不能当双工纯推理budget。

## 交付与记录

构建和运行方式见`docker/external-hf/README.md`。worker与gateway已推送Docker Hub新dev标签，见下文不可变digest。用户已明确暂不推送LIS；此前只读认证查询401，未做LIS注册或部署。Docker API通过不代表LIS已部署。自签开发TLS证书不作为生产身份，当前容器API使用HTTP，公网由平台TLS入口处理。

精简证据在本目录`evidence/20261014-docker/`。完整原始日志、事件和音频在`/user/weihongliang/analysis-runs/20261014-docker-lis/`。主任务A100单工/移动验收另见`SIMPLEX.md`及`MOBILE_NOFC_ALIGNMENT.md`；不要混淆硬件、checkpoint和版本范围。

## 已发布镜像

共同标签：`external-hf-pro5000-20261014-01716c82-dev`。标签中的pro5000表示运行环境和默认加速配置目标，已测硬件范围仍是上文的单Pro6000 native。

- Worker：`device0/minicpm-o-worker-backend@sha256:a7eddb53f66672160f6e71912b3b0893838806fc1968d3bf11a5efbb2b6e9647`。
- Gateway：`device0/minicpm-o-gateway@sha256:63a49213ad74b4f3838247a5054689c529d3dea983096c7a67b0d8ee07835cfd`。

Worker镜像默认role=worker，不包含对外gateway进程。单容器API需显式`O5_ROLE=standalone`；独立gateway镜像已固化role=gateway。

## 新增双卡验收要求

用户要求继续完成镜像的Pro5000双卡加速验证。2026-10-14只读核对：34为双Pro6000，GPU1有既有VLLM；118.89.233.148为8张Pro5000，但目前全部属于caitianchi的VLLM任务，没有可停止的本任务GPU进程。其docker命令实际为Podman兼容包装。后续必须确认容器权限和空闲双卡，不能占用或停止他人服务来完成测试。历史2026-10-01 Pro5000双卡API/精度报告使用旧HF/Demo，只能作为环境和实现参考，不能替代本镜像验收。
