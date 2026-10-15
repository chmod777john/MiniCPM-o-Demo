# 37新pip镜像GPU/API验收完成

固定build79177c66，真实Pro5000 72GB双卡Kubernetes Pod，worker/gateway分别使用原OCI digest；原始checkpoint只读挂载。HF/accel和镜像内容没有为本轮验收修改。

## 通过范围

- native新旧独立进程17输出全部逐位一致、finite，误差0：FA2、causal-conv1d、dense causal。
- TP2模型加载、grouped扩展真实编译、既有全部加速开关、worker注册正常。
- 单工12/12：文本/语音流与非流、history、高清图像、输入音频、cap1、实际视频解码、长语音、thinking功能冒烟。thinking以48token截断，不作为质量分数。
- 普通双工初始音频2/视频2 unit；补充说话音频12/视频12 unit全部完成并有有效PCM，basic8k/6k和slice=2传递正确。
- FC录制basic1 8/8unit、3实际工具调用、3结果回送、2在停止前消费、5个PCM片段，errors0；工具结果为fixture，不调用外部服务。
- 多模式后单工text/audio2/2，模型状态可回收。

## 本次耗时范围

| 场景 | 范围 | 时间口径 | p50 | max |
| --- | --- | --- | ---: | ---: |
| 普通音频说话 | 12unit，KV107～334 | backend wall_clock_ms | 567.05ms | 608.8ms |
| 普通视频说话 | 12unit，KV316～2863 | backend wall_clock_ms | 650.7ms | 698.3ms |
| FC录制basic1 | 8unit | 客户端提交到commit | 见原JSON | 1017.86ms |

backend指标不含客户端排队/网络；单工视频首请求含cold编译12.426s，其后非流1.537s，不能和双工unit混用。未新增64k/128k、全hardcase、长时滑窗或整模型hidden paired精度验收。

## 过程与资源

临时只读registry只监听节点127.0.0.1:18537，白名单GET/HEAD；kubelet正常CRI pull成功，不改宿主runtime、不外部push。TKE子网IP不足时改同一个普通网络Pod两容器，只有worker2GPU。自己的Pod/registry现均UID限定正常删除，backend/worker日志正常shutdown。BuildKit保留root管理。最终同名dev-8gpu-37为他方新UID、6GPU，未干预。

探针缺本地fixture路径、非法threshold=0、错误期望kind=speak均明确记录并修正客户端；服务源码没有为探针修改。最终事件/native输出/音频归档39,557,120B，SHA4297806c560934d8603f858272a8229f685630cea9401cbdf3f9be213f1c0b38，已完整回共享。

完整证据：/user/weihongliang/analysis-runs/20261014-docker-pip-install/pro37-gpu-api-evidence.tar。详细经过PRO37_GPU_REPORT.md，结构化结论pro37-gpu-api-summary.json，资源状态pro37-resource-release.json。未推registry/LIS。
