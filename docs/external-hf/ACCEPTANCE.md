# 三物料阶段验收（2026-10-13）

## 当前范围与结果

按本轮收窄范围，已以固定来源 `minicpm-o-5-fix-3d-rope@e43c73a` 和
`o5-fc-dev@732b0758` 构建并跑通 HF / accel / Demo。保留现有 demos、gateway、
worker、runtime 和页面组织，没有进行原第6步的应用拆分或大量目录迁移。
直接使用原始 `.pt`，没有导出新权重包。所有 GPU 验收为 agent-dev A100 双卡 TP。

## 结构

- HF 独立 `fc/` 提取包：固定 Git blob + 显式 patch + AST/hash manifest。
  38个方法中28个原样、8个具名适配、2个旧warm移除；`--check` 再次通过。
  普通文件只涉及cache公共API兼容、processor注册与SDK slot registry。
- accel 承担graph缓存/hidden所有权与已有算子加速；没有把服务warm/graph放入HF。
- Demo以外部路径载入HF和可选accel，同一服务顺序选择普通/FC；外部TP为单进程双卡。
  普通会话参数从保留的API/View透传到HF，每个会话重新初始化。
- SDK0.0.6解析slot ID，权重embedding/lm_head覆盖检查，不修改tokenizer.json，
  不随机resize权重。环境utils1.0.7为内部causal实现，依赖声明例外详见环境README。

## 实测与证据

|验收|结果|job|
|---|---|---|
|FC native→TP fixed-feed|39次feed hidden/logits逐位一致|1209233|
|普通no-slot native→TP|22次feed逐位一致，位置/KV长度/argmax一致|1210239|
|普通slot native→TP|48次feed逐位一致，位置/KV长度/argmax一致|1210239|
|allmerge普通→FC→普通|4/8/46/24 unit链路完成，普通24有TTS|1209341|
|FC ckpt真实工具API|basic-1 8/8；3次真实tool call，前2个result被后续unit消费，返回语音|1209574|
|普通basic+binary选项|24/24；basic1024/512发生3次KV裁剪；下个无config会话恢复默认|1210050|
|最终同进程模式切换|参考音频chat返回文本+48960采样点→普通8→FC8→普通4，无后端错误|1210413|

API短会话时延：ordinary24（binary/basic）的server wall p50=727.15ms/max1207.3ms；
最终ordinary8 p50=324.8ms/max1138.6ms；FC basic-1 client elapsed最大1.229s。
这些口径不同，不混为纯GPU时间。普通API没有完整unit提交事件，不把first output
当作finalize完成。

## 排查与边界

1. 初始普通TP对拍失败：native默认cuDNN SDPA与accel既有排除cuDNN策略不同。
   在两侧统一策略后no-slot和slot均逐位一致，无accel产品修改；失败证据保留。
2. 原Board未触发工具，不伪造通过。改用历史成功basic-1原始prompt/audio/policy，
   仅适配v4传输，实测模型自主tool call并使用实际ID回填录制结果；非真实工具联网执行。
3. API参数原本静默丢失，已修sliding/LS/max-speak/force-listen/TTS透传及会话复位。
   原页面length_penalty在外部HF普通生成中没有语义，保留接收并明确日志提示无效，
   未复制生成循环或擅自改变HF逻辑。其余不支持字段明确拒绝。
4. 参考音频chat发现View根据类模块猜utils路径失败，改由facade提供HF原生采样类型，
   真实GPU/API重新返回文本/音频。只看到response.done不等于推理成功，本次还核查输出与日志。
5. 固定feed精度不包含merge-unit自身数值等价；merge已有数学/位置审计和BF16偏差报告，
   本轮API使用merge。KV对拍测长度和位置，不宣称对所有KV tensor做逐位比较。
6. 未重新跑全量质量评测、所有硬件/单卡/EP矩阵或64K/128K。当前是三物料迁移
   代表性验收。应用拆分按用户要求暂缓；不在本次收尾时暗中扩大范围。
7. subagent续跑和新建最小健康检查均报encrypted function output解码失败；root工具
   正常，已接管残留代码/任务，未依赖任何已失败agent继续执行。

## 版本与操作

HF335bfdd；accel6e33931（产品逻辑f06e7d7）；Demo最终提交见本报告所在Git HEAD。
Demo部署入口：`docs/external-hf/DEPLOYMENT.md`，含三个绝对路径、环境与参考job。
原始统一证据：`/user/weihongliang/analysis-runs/20261013-o5-three-materials/`。
重大阶段提交已完成；未推送/提交MR/合并上游。正式旧工作树和公开服务未替换。
自己的验收服务释放状态写入STATUS.json，未停止未知owner任务1201912。
