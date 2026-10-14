# pip镜像CPU验收证据

镜像输入代码为79177c66；此目录是其构建后的证据，后续文档commit不属于镜像内source identity。

- 原始wheel/source源码来源锁见docker/external-hf/pip；native-wheel-inputs.json保存新源码构建二进制来源。
- worker/gateway image verification核每OCI blob、角色和image内报告；image-input-consistency核实际137包版本、1444个文件与输入完全相同。
- cpu-diagnostic为最终image构建中14项CPU门；gateway-cpu-result为实际gateway进程HTTP/static/退出门。
- 两个metadata例外经root审核，pip check仍真实exit1，不称零冲突。
- GPU算子输出、TP2真实模型API、Pro5000性能尚未覆盖；未push/未部署LIS。
- checkpoint 74606011097B完整SHA通过并原子命名model.pt；vocoder逐8文件SHA完成。原始结果和最终进程/锁状态见checkpoint-transfer-result与checkpoint-final-state。
- worker/gateway的OCI和Docker四个归档均完整备份且全SHA/逐blob校验通过；具体路径见DELIVERY_MANIFEST和FINAL_HANDOFF。
- BUILD_COMMANDS记录复现命令；原始镜像build固定79177c66，docs-only提交不重标镜像。

完整原始日志和归档：/user/weihongliang/analysis-runs/20261014-docker-pip-install/。
