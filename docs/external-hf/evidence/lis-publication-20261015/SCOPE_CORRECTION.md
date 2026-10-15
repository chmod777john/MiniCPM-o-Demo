# 当前有效范围更正：直接上传原已验收镜像

2026-10-15 用户指出：“为什么又在验证新镜像而不是把原来的 push”。

Root承认把上传原已验收镜像擅自扩大为更新代码再验收；该新版发布路线立即停止。

## 唯一有效交付

- 原20261014双镜像：worker o5-external-hf-pip-20261014，gateway o5-external-hf-gateway-pip-20261014。
- 镜像内HF 5fea76841a268720e83ce68bd14f8c064ad72bde，accel 337ac3e84b69dde135efaad7f4113bc244fe7bd4，Demo 79177c666a0bbafe5b0ed360bde836254d5cd09c。
- 复用原Pro5000双卡验收，无需新GPU/API验收，不升级代码，不创建LIS推理服务。
- 权重job af790c8d-bb41-4a9e-8233-f64559520950继续完成，禁止取消或重复上传。

## 分工和授权

simplex_final_review：停止新版formal session的complete，先查真实状态。对仅属于本任务的未完成新版sessions安全取消并留证据；若已完成或无明确安全取消方式，保留但标记不交付，不删除别人的资产。以pip-20261014版本、o5-external-hf-pip-20261014 release ID上传原镜像并登记。核原OCI/config/layers与registry相同。仅修改自身images/和IMAGE_PUBLICATION.md，使用已有BuildKit；不得新占GPU。原验收报告为20261014-docker-pip-install/PRO37_GPU_REPORT.md。

hf_minimal_simplex_rewrite：停止新增验证和自身probe；保存已有日志；查真实UID后清理自有validation Pod o5-lis-publish-validation-20261015（afc32f4b-dc19-440f-9f2c-382097235840）与registry Pod o5-lis-publish-registry-20261015（0833191e-713b-4c4a-aa33-af5c19c84c4e）。不得碰dev-8gpu-37或BuildKit Pod，权重任务和原镜像push需要后者。仅收尾，不继续跑测试。

root：维护当前发布manifest与更正记录，持续跟进权重上传、核对源hash和平台资产；最终发布证据及时提交。RELEASE_PLAN.md之前的新版输入和GPU门全部被本文件替代。已产生新版日志保留为范围外记录，不可冒充原镜像验收。
