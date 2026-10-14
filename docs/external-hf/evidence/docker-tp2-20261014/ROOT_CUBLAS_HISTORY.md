# cuBLAS C++ 扩展兼容回归：三个物料阶段的历史

日期：2026-10-14 UTC。Owner：root。只读核对 Git blob、profile 和已存运行日志；未重新运行历史模型。

## 确认事实

1. hf-fc 阶段 2026-10-01 Pro5000 验收使用 accel07ea4ba，streamed_blas_grouped.cpp 的 BF16 reduction 判断为 enum 比较。docs/ACCEPTANCE_PROVENANCE_20261001.md 记录其 Torch2.14.0+cu130 与已验证输入范围。因此原 Pro5000 成功并非虚构，也不是使用当前6e33931。
2. 2026-10-02 03:17:53 UTC，573bf7413cece1f9668bac3175716a14034e4f19（fix(compat): support torch bool BF16 reduction API）将 direct/streamed/batched 三个 C++ 源文件从 enum 比较改为 !getter()。parallel_streamed 未改。此修改支持旧 bool API，却破坏这些文件在 enum API 下的编译。当前 high-cu128 Torch2.11 也已有 enum，不能说只有2.14新增。
3. 3drope accel worktree 2026-10-09 从hf-accel@d425a69建立；worktree-readme.md记录来源。git merge-base证实101fea7、b26e98b、6e33931都继承573bf74；逐blob证实streamed保持bool写法，parallel_streamed保持enum写法。3drope及后来的o5-fc-dev迁移没有引入这次回归，也没有修复它。
4. a100-tp profile具有OPT_PARALLEL_GEMM_SUBMIT=1，pro5000-tp未设置该项，默认0。streamed_blas_grouped.py据此选parallel_streamed或streamed两个不同扩展。Pro5000架构capability>=12即使native_grouped标志为真，parallel_experts.project也会选本地grouped实现。
5. 2026-10-13 gpu-first/fc-tp-replay.log第6/7行实际记录parallel_streamed的C++编译/链接成功；因此A100成功有明确路径解释，不依赖“可能用了旧缓存”的猜测。三物料FINAL_REPORT.md明确本阶段GPU验证是A100 TP，未重跑全硬件矩阵。3drope旧阶段ALLMERGE与MERGE_UNIT验收也为双A100。
6. 本次34 Docker v7实际失败文件为streamed_blas_grouped.cpp:61，operator!不接受at::CuBLASReductionOption。此前native single_eager不attach加速，不覆盖该路径。

## 三阶段结论

- hf-fc：10月1日Pro5000验收版没有当前回归；10月2日在此阶段末追加兼容提交后，引入潜在enum环境编译失败。
- hf-fc-3drope：继承不一致；A100 TP的parallel_streamed路径可用。没有依据把A100通过解释为该阶段Pro5000路径已复验通过。
- hf-o5fcdev-3drope：继续继承不一致；A100 TP真实通过。本次首次把这版Pro5000配置放入干净Docker构建时暴露streamed的失败。

## 独立包装问题

enum兼容修复后还暴露缺libcublas.so开发链接名；镜像已有libcublas.so.13运行库。2c07b81c仅在CUDA_HOME/lib64创建指向冻结运行库的链接。此问题属于本轮镜像构建，不能说三代HF都有此缺项。

## 当前修复范围

accel66c4d07以类型分派兼容bool/enum，四个调用点统一语义；splitK=false及未知enum明确拒绝，保留GEMM与stream参数。v9真实Torch2.14四扩展编译/load通过；两张Pro6000、direct/streamed/parallel、reduction true/false共12矩阵对照torch.mm逐位exact/max_abs0，6次splitK=false明确拒绝。batched仅编译，不启用。核对本地grouped-build-v9.log、grouped-gpu-v9.log。模型TP2 API仍由packager继续，不能将模块门当全流程通过。

## 证据路径

- /user/weihongliang/hf-accel/docs/ACCEPTANCE_PROVENANCE_20261001.md
- /user/weihongliang/hf-accel-no-fc-3d-rope-20261009/worktree-readme.md
- /user/weihongliang/analysis-runs/20261013-o5-three-materials/FINAL_REPORT.md
- /user/weihongliang/analysis-runs/20261013-o5-three-materials/gpu-first/fc-tp-replay.log
- 当前目录tp2-v7-full.log、grouped-build-v9.log、grouped-gpu-v9.log
