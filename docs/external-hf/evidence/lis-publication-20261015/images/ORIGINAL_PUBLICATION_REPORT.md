# 原20261014已验双镜像发布完成

当前交付仅为原始镜像，未更新源码/依赖；无新增推理部署。发布版本 `pip-20261014`，release `o5-external-hf-pip-20261014`，保留180天。

## 源码和验收

- HF `5fea76841a268720e83ce68bd14f8c064ad72bde`
- Demo `79177c666a0bbafe5b0ed360bde836254d5cd09c`
- accel `337ac3e84b69dde135efaad7f4113bc244fe7bd4`
- 复用 `/user/weihongliang/analysis-runs/20261014-docker-pip-install/PRO37_GPU_REPORT.md` 与 `pro37-gpu-api-summary.json`，summary `passed=true`。单工12项、普通audio/video各12unit、FC8unit、跨模式恢复及native门见原报告。此次没有重新跑GPU/API。

## LIS资产

### worker

- asset ID: `5bf9a66a-f9b9-40e0-a8f6-ef3c77c95f11`
- upload job: `eb000755-37de-463b-9bcd-25377791794e`
- URI: `modelbest-kt8zv1.swr-pro.myhuaweicloud.com/lis-dev-images/o5-external-hf-worker@sha256:c0463e5ea82eba878b8d111460fc7f2350ec61c6d76eac54749346909a4659eb`
- 原OCI manifest: `sha256:c35318d4cc66ffc9d15a6d21cf69f8c5d46526ebd7ff1b1144f8a2604aa3f8f6`
- config: `sha256:b094e9eea2ad246129947c9f940df7881328c2ee5a47305d87742d6453ff7c7b`
- 最终tag真实GET与原OCI比较：config及全部17层的digest/size/顺序完全相同，仅manifest media type转为Docker schema2。

### gateway

- asset ID: `aeedb241-c4a9-4cb8-bc7d-07982d6d66fa`
- upload job: `593c2d96-cd1b-47e8-9a60-82ed6001bb75`
- URI: `modelbest-kt8zv1.swr-pro.myhuaweicloud.com/lis-dev-images/o5-external-hf-gateway@sha256:6f88f803f5ecbfa68ab1068eb5aface102c5d11fa243df7607aabd4e27a5783a`
- 原OCI manifest: `sha256:143bdde56a917513253998c8ccfa032fc84ffcb72e7c38b1da95bbbcd7a5e3cd`
- config: `sha256:48d6767f7b03f8beee76e2fcfebaa48999eb7372eed7c05d0f51c4025c7532bf`
- 最终tag真实GET与原OCI比较：config及全部17层的digest/size/顺序完全相同，仅manifest media type转为Docker schema2。

## 路线与范围纠正

使用LIS正式push-session临时授权目标，37既有BuildKit固定FROM原OCI、无RUN/COPY、无Docker daemon、无公共registry，推送后complete再按asset ID和最终tag读回。CLI1.2.35仍要求Docker Engine，因此通过相同正式API控制面加BuildKit数据面完成。

此前误扩范围的formal20261015两个session从未complete，已核owner/PendingStaging后非force取消；平台无formal资产。暂存registry内容不交付，未扩展做仓库删除。

平台asset scan_status目前Pending，只声明上传登记及内容一致性完成，不声明安全扫描已结束。原验收的长上下文/完整质量覆盖边界仍见原报告。

## 证据

`publication-summary-original.json`为机器可读总览；`original/*-asset-readback.json`与`*-final-registry-identity.json`为最终平台/镜像证据。安全提交仅按`SAFE_COMMIT_WHITELIST_ORIGINAL.json`，排除所有private凭据、原始session response、二进制及范围外新镜像记录。

上传完成后已删除本任务本地8个raw session/docker auth文件、37的4个DOCKER_CONFIG文件；保留安全元数据。root LIS认证、模型上传和BuildKit未动。
