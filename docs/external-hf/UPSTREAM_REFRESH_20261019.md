# O5 FC 上游同步 — 2026-10-19

## 版本组合

本次为源码整合记录，尚未进行新版组合的推理、API、精度或性能验证。现有服务与 LIS 镜像继续使用各自历史发布记录。

| 物料 | 本地正式分支 | 固定实现 commit |
|---|---|---|
| HF | fc | `db7541095e458bf71269a49fafea508369556574` |
| accel | hf-fc-accel | `337ac3e84b69dde135efaad7f4113bc244fe7bd4` |
| Demo | demo-adapter-fc | `79c08878016f35aa63360ef35ff857fd6275011c` |
| 上游输入 | o5-fc-dev | `e0f2ada084d395a00d1956957d345e82b8a5f8f7` |

本说明和机器可读版本记录位于实现提交之后的文档提交中，不改变上述运行代码。见 [版本记录](releases/upstream-refresh-20261019.json)。运行时应把 `O5_CANONICAL_PATH` 与 `O5_ACCELERATION_PATH` 指向上述固定提交的实际 checkout，不能仅根据目录名或可移动的分支名判断版本。

## HF 更新方式与范围

从当前 `fc@19e87201` 继续更新 `fc/migration` 配方，再重新生成 FC 文件；没有回退或重建 HF 底座。源固定为 `e0f2ada0`。

- 收入上游的 spoken_turn_eos TTS condition 和轮末独立 text_eos。
- 收入 FC audio/omni、单 Unit 一帧视频、视频指标与几何校验。
- 视频映射至现有 CanvasImageSpec/feed_image_group，使用原生位置和图像元信息记录。
- 保留 causal Token2Wav prompt 所有权、无三个4218前缀、unit force-flush、最终 flush 和训练权重行数检查。
- HF 普通双工、canonical decoder/processor/model、minimal simplex 均保持原文件。

更新前后，生成器内置检查均能重现7个受管输出。新版39个上游方法中28个AST一致、9个有记录的适配、2个服务warm方法按原边界移除；manifest位于HF的 `fc/migration/manifest.json`。这些是生成与源码审计，不是模型推理测试。

## Demo 合并方式与手动范围

真正合并本地 `e7604219` 与上游 `e0f2ada0`，保留本地三条LIS文档提交。上游71个变更文件中69个无冲突，只有gateway和启动脚本各一个冲突块。66个仅上游改动文件与上游字节一致。

- gateway组合普通长会话/重连与audio/omni时限分类，并给retained-socket调用传入input_mode。
- 启动脚本保留外部HF可省略backbone的分支，吸收上游vendored-prefill配置。
- 命名端点使用v5 audio/omni；默认普通audio/video、单工chat保留。v4 FC按上游保留audio语义。
- 外部loader、facade、ordinary reconnect模块、桌面/移动端普通功能实现保留。
- 上游vendored canvas/Graph/prefill源码随merge入库，外部HF运行路径仍选指定HF和accel，不执行这些vendored实现。

## accel 与边界

accel没有源码变更。其现有FC挂接、Merge-unit的位置/spec账本和模型提供的RoPE position embeddings接口与此次HF适配相接。复用依据来自源码检查，尚未执行新FC视频组合的Graph/Merge-unit数值或性能验证。

Omni要求canvas配置资产，每Unit恰好一帧、不切片，同会话处理后的图像几何固定。记录图像元信息不等于已接通并验收FC滑窗触发/单元登记；本次不新增这项声明。合并feed下视频cost也不能当独立GPU计算耗时。

## 验证范围与部署状态

完成：Git差异与来源核对、生成器内置AST/重现检查、加载和参数调用链源码复核。

未执行：测试套件、运行时导入验证、前端构建、GPU推理、WebSocket/API probe、精度/音质/速度验收。没有重启服务、构建镜像、替换LIS资产或更换checkpoint。

本次TTS条件语义已经变化，后续评估accel精度应对比同一版更新后的HF；不能将相对旧TTS口径的输出变化直接归为加速误差。
