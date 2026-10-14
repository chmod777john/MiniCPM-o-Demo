# 37新pip镜像GPU验收：阶段记录

固定build79177c66，真实Pro5000 72GB双卡Pod。native新旧独立进程17输出逐位一致；TP2模型已加载/注册。API范围正在运行，未在本阶段宣称通过。

registry-pod为临时仅loopback18537只读OCI源，使用静态readonly_registry.c；正常CRI拉取成功。输出只白名单GET/HEAD，禁止上传/删除/任意路径。合并gateway/worker两容器到一个Pod缘于TKE子网IP不足，原阻塞见pro37-worker-events。只有worker申请2GPU，model只读。

完整过程和后续结果位于 /user/weihongliang/analysis-runs/20261014-docker-pip-install/PRO37_GPU_REPORT.md 。
