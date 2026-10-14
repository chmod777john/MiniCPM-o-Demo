# 最小单工整合与验收（2026-10-14）

## 交付与历史

- HF：`19e87201fb5f094df134a84171a35715bc6e5a69`，分支/目录 `minicpm-o-5-fix-3d-rope-o5fc-20261013-minimal-simplex`。唯一父提交 `335bfddaf1c32d455f87d2e5f699b0f71c6d65a4`，单工只有一个提交。运行时与实际GPU验证的 `6139d9d` 完全相同，最后 amend 只更新本文所述验收文档。
- HF旧版 `5fea76841a268720e83ce68bd14f8c064ad72bde` 保留为 `archive/minicpm-o-5-fix-3d-rope-o5fc-20261013-hf-chat-20261014-before-minimal`；旧服务目录/分支没有 reset/rebase。
- Demo：本分支基于正式 `d036b7a6`，追加运行时 `4a5e59d0`、测试 `952102a9` 及本次验收记录。只包含单工清理/桌面取消；没有合入尚待用户审核的录屏候选。
- accel固定 `337ac3e84b69dde135efaad7f4113bc244fe7bd4`，无修改。

用户要求的HF接线保持为两运行时文件81增/9删、新1函数/改5函数。撤回本轮新增length penalty、strict cap、首EOS补丁、文字mask/输出过滤/producer取消及join、额外TTS normalize。保留完整输入接入旧chunk/TTS/vocoder、3D positions、独立TTS参考、已有greedy开关和实际出现过的rope配置兼容。旧chunk内部原有length_penalty默认1.0仍属上游代码，不通过chat新增暴露。

## 实际环境与输入

34机器两张96GB RTX Pro6000，Pro5000 TP2预设；现有worker v11镜像 `e2a1f41bd37d50343b3882a4fb7f62fe7aae45b1adf4e30166f257c83af3962b` 的环境。候选HF/Demo/accel以固定Git archive只读挂载，运行后逐文件核对HF63/Demo1288/accel86份源码一致。Python3.10.22、Torch2.14.0+cu130、Transformers5.13.0、Triton3.8.0、FA2.8.3.post1、FLA0.5.1、minicpm-o5-sdk0.0.6、minicpmo-utils1.0.7。allmerge iter300原始权重，旧完整SHA `0789c953677dc8ea71b6085543e1a946ae64bddecd2f6b911ac6380376ef01ff`；本轮未重拷或改权重。

原始证据目录：`/user/weihongliang/analysis-runs/20261014-hf-simplex-minimal`。`source34-manifest.json`、`host34/environment34.json`、`host34/container-inspect-running.json` 固定源与环境；`probe_api.py`、`probe_cancel_followup.py` 固定输入与断言。计时包括真实API请求链，不能用作模型模块纯推理计时或A100/Pro5000性能结论。

## 功能API11项

| 样例 | 请求至done | 结果范围 |
| --- | ---: | --- |
| 非流文字 |0.471s|你好，正常done|
| 流文字 |0.598s|你好并保留上游im_end|
| 非流语音 |4.227s|文本+1PCM块|
| 流语音 |2.903s|文本+1PCM块|
| 完整历史追问 |0.639s|回答此前暗号红苹果|
| 图片HD文字流 |0.669s|512×384红图，slice2，回答红色|
| 输入音频文字流 |2.000s|参考音频判断为人声，保留上游tts_eos|
| 视频语音流 |12.395s|既有red-audio.mp4合成红色画面，2PCM块|
| 长语音流 |6.125s|cap80/sample=true，北京介绍，15PCM块|
| thinking文字流 |1.051s|cap48停止在思考内容；只证明路径，不是完整最终数学答案|
| 视频非流语音 |3.418s|相同简单合成视频，1PCM块|

上述有音频输出均finite且非零。不是音质/复杂视频理解/全模型精度验收。原始结果在 `api/probe-summary.json` 与 `api-multimodal/probe-summary.json`。

## 取消与追问的准确边界

桌面Cancel立即冻结收到的部分文字/历史、停止PCM播放，并通过`session.close`取消活动请求；排队/连接阶段直接关传输；旧request的晚到delta/done/error不得写进下一问。HF仍无文字producer取消机制。Demo文字流静默消耗至自然结束才释放worker；语音流完成当前chunk再收尾；非流完成当前计算。这意味着追问可能排队。

严格验证脚本检查关闭原因必须为`client_cancel`，再检查下一请求回答2：

| 取消样例 | 关闭耗时 | 下一问ready | 下一问done |
| --- | ---: | ---: | ---: |
| 长文字流，cap256 |3.639s|4.315s|4.943s|
| 长语音流，cap256 |0.451s|1.170s|1.759s|
| 非流文字，cap256 |3.831s|4.507s|5.091s|
| **短**非流语音“请只说一句你好” |2.397s|3.073s|3.650s|

**仍失败的样例：**最初要求500字省份介绍、非流语音cap256，请求被取消后自然计算达到cap但没有TTS结束标记，上游非流TTS抛`TTS region end is invalid`，关闭原因是`backend_error`。下一问成功，但不能称正常取消通过。最初脚本只检查closed，误报pass；已修严格reason断言，保留原始 `api-cancel/cancel_nonstream_audio.events.json` 与backend日志。上述短样例的通过不覆盖此失败，没有增加HF边界补丁。`api-cancel-v2/cancel-followup-summary.json` 是正常取消结果。取消时worker内有底层close-frame错误日志，公开会话正确关闭及后问门已核验；不宣称完整传输异常恢复。

## CPU与静态产物

- actual AST ChatView桥接、backend文字drain/语音close、桌面request隔离/排队取消/部分历史/PCM停止均通过。
- runtime protocol pytest3项、mobile Node contracts24项通过；TypeScript/Vite build通过。
- 双工相关View AST未改；HF六个原逻辑方法与335bfdd字面一致。没有新做hidden/logits paired或双工GPU回归。
- `static/mobile`本来就被Git忽略。此候选已`npm run build`并单独`node scripts/publish-static.mjs`生成3份资产，和dist逐SHA一致；34实际HTTP index/JS/CSS均200且SHA与`mobile-assets-manifest.json`相同。没有运行会覆写共享omni/widget的build:static。
- 最终交互由用户验收；本轮没有浏览器点击测试。

## 下一次部署如何切换

当前公网1224474直接使用旧HF和旧Demo目录，因此本次整合交付使用独立分支/目录；公网没有更新。未来由root在明确部署切换时设置：

```bash
export PROJECT_DIR=/user/weihongliang/worktrees/o5-fc-dev-external-hf-20261013-hf-chat-20261014-minimal-simplex
export O5_CANONICAL_PATH=/user/weihongliang/worktrees/minicpm-o-5-fix-3d-rope-o5fc-20261013-minimal-simplex
```

accel/权重/环境继续按实际部署设置。本地候选static/mobile已生成；新checkout或镜像必须从该源码生成资产，不能仅用git archive后假定mobile已带入：

```bash
cd "$PROJECT_DIR/frontend/mobile"
# 已安装该项目依赖、使用Node>=22.12时：
npm run build
node scripts/publish-static.mjs
```

需要固定产物时用证据目录`mobile-assets.tar`及`mobile-assets-manifest.json`核SHA。旧已打包镜像含旧HF，不算本次交付已重打包。不要merge旧HF单工两提交回新分支，不需要force push；旧ref/旧目录留作服务和回溯。

## 资源收尾

agent-dev测试job1224535因同节点无双卡而未运行，替代启动后已停止为Killed。34仅本任务容器`o5-minimal-simplex-20261014`已SIGTERM停止（exit143、OOMfalse），双卡恢复3MiB/no compute；独占隧道已停。其他服务/镜像/任务未改。保留只读候选快照与证据，不再占GPU。
