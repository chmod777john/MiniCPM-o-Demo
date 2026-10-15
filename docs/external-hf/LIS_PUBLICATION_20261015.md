# 原已验收双镜像及权重的 LIS 发布

## 本次范围

直接上传2026-10-14已经在37双Pro5000验收的Worker/Gateway镜像，以及相同FC原始权重和causal vocoder。镜像代码、环境、启动入口均保持原样。权重不转换格式，也不放进镜像。

曾误扩为更新HF/Demo代码并重新验证；用户纠正后停止该路线。新版两个未完成上传session已取消，验证Pod与临时registry已按UID清理；正式三物料实现未改。当前身份以PUBLISH_MANIFEST.json和ORIGINAL_ACCEPTANCE_REFERENCE.json为准。

## 固定版本

|物料|原镜像内commit|
|---|---|
|HF|5fea76841a268720e83ce68bd14f8c064ad72bde|
|accel|337ac3e84b69dde135efaad7f4113bc244fe7bd4|
|Demo/打包|79177c666a0bbafe5b0ed360bde836254d5cd09c|

这些是原验收镜像的版本，不代表当前正式分支HEAD。环境通过固定wheel及源码构建后pip装入新venv；没有复制本地site-packages。torch2.14.0+cu130、transformers5.13.0、FA2 2.8.3.post1、FLA0.5.1、minicpmo-utils1.0.7、minicpm-o5-sdk0.0.6。

## 发布方式

37现有BuildKit读取固定原OCI镜像，经LIS授权push session直接推送SWR。只转换OCI到Docker schema2 manifest；image config和17个layer的digest、大小及顺序必须与原OCI一致。通过后complete登记并读回平台image记录。没有重新构建应用层或运行新的推理验收。

权重目录共9文件，75,920,482,457bytes：model.pt为74,606,011,097bytes，另有8个token2wav_causal_vocoder文件。model.pt SHA256为0789c953677dc8ea71b6085543e1a946ae64bddecd2f6b911ac6380376ef01ff。CLI从37上传至OBS后由平台转存SWR，最终必须Complete且9文件SHA/大小与源一致才算完成。

发布保留180天，asset environment为dev。只发布资产，尚未创建LIS推理服务。最终asset ID、digest与平台返回有效期见PUBLISH_MANIFEST.json。

## 已有验收范围

复用原37真实双RTX PRO5000 72GB验收：17组native输出逐位一致；单工12请求；普通双工音频和视频各12个说话unit；FC录制样例8unit；跨模式后单工2请求。原始证据引用见ORIGINAL_ACCEPTANCE_REFERENCE.json。

该范围不包括新的全模型hidden paired、64K/128K或全部hardcase，也不包括真实外部工具质量、浏览器人工体验或LIS在线部署。此次只核对上传身份和文件完整性。

## 上传兼容记录

使用任务目录内lisctl1.2.35，未全局安装。旧1.2.9缺少当前服务要求的保留期限参数。模型首次并发文件上传出现multipart upload_id变化，已取消唯一失败任务，改file-concurrency=1、upload-concurrency=8后全部源数据成功上传OBS；平台并发metadata覆盖只是源码机制推断，未宣称线上根因已经证实。详MODEL_MULTIPART_NOTE.md。

## 完成结果

原双镜像上传job均Completed，平台资产和最终registry manifest已读回。两镜像config与全部17层digest/大小/顺序和原验收OCI一致。

|角色|LIS asset ID|版本|
|---|---|---|
|Worker|5bf9a66a-f9b9-40e0-a8f6-ef3c77c95f11|pip-20261014|
|Gateway|aeedb241-c4a9-4cb8-bc7d-07982d6d66fa|pip-20261014|

共同release为o5-external-hf-pip-20261014。资产scan_status当前Pending；上传完成不代表扫描完成。

权重job af790c8d-bb41-4a9e-8233-f64559520950仍在平台OBS到SWR阶段，8个vocoder文件SHA/大小已核对，model.pt尚未完成，不能视为权重已交付。当前状态见PUBLISH_MANIFEST.json和model-watch-status.json。

本仓库精简证据见 `evidence/lis-publication-20261015/`。完整原记录位于 `/user/weihongliang/analysis-runs/20261015-lis-upload-capability/`。
