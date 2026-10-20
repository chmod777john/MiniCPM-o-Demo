# OpenAI 原生单工：评测种类核对与完整脚本验证

日期2026-10-20 UTC；owner root。

## 报告目录实际包含什么

在呼和浩特以只读任务1303202检查 `/user/lipengyue/opencompass_results/outputs/minicpm-o5`，6个运行目录、5份summary、5份meta均只涉及 `AI2D_TEST`。另核对52条同事的opencompass-platform作业记录，7条含 `--dataset` 的评测命令也都指定AI2D_TEST，其余为排查任务。

因此，这个指定目录的“所有benchmark种类”目前只有AI2D_TEST。15个`category`属于同一数据集的题目类别，不能写成15种benchmark。目录外的其他报告、音频/视频或其他评测框架未由此覆盖。

## 评测代码版本与上一轮结果更正

此前本地 `/user/luozhen/VLMEvalKit` 是7e9310a7，和同事指定分支不同。这次通过只读任务1303221取得 `ue2-integration-20260928` 当前commit `3d3fcbfee182cc9bebab90272525b2fb47aaeeb6`；导出的API wrapper、图像数据集、判分器、run_eval.sh、eval_runner.sh文件哈希逐一一致，源码未修改。

同事报告meta的backend_version是1f409531。这一短SHA在该分支完整仓库中无法解析（任务1303240/1303246记录失败），所以本轮准确口径是同名分支当前版本，不能称为完全复现历史commit。

将此前8条已保存原始回答交给当前分支原版 `ImageMCQDataset.evaluate(model="exact_matching")`，得到8/8；旧本地版本是2/8。没有重新推理、没有修改答案、没有自定义判分。当前分支增加了AI2D答案抽取及`extract_final_answer`，能处理解释中列举多个选项的回答。此前6条无法抽取的结论仅适用于旧版本。

## 本轮验证设置

Demo运行时代码 `c0cd6099871d46d8548f5d27c41768929f113eec`，分支 `demo-adapter-fc-openai-simplex-20261020`。HF528cb102、accel9c6f336保持原commit。本阶段未修改接口或模型代码。

独立任务1303263：langfang_train / agent-train，节点10.156.1.177，2×A100、16CPU、256GiB。Checkpoint及三物料配置沿用上一轮验收。

真实执行原版 `run_eval.sh → eval_runner.sh → run.py → VLLMOpenAIWrapper → 本接口 → HF chat → 原判分 → UE2格式summary/meta`。`max_tokens=2048`、`temperature=0`、`api-nproc=4`，原图像编码与prompt保持不变。

原框架 `--sample 2 --seed 0`按类别分层，保证每组至少1条，实际得到15题，覆盖全部15个类别。没有使用答案筛选题目。所有图片缓存、日志、结果在自有证据目录；`SEND_FEISHU=false`、`SEND_DATABASE=false`、`--no-send-feishu`，没有发送通知或写评测数据库。

## 证据路径

`/user/weihongliang/analysis-runs/20261020-openai-benchmark-coverage`

- `inventory.json`：完整报告目录清单、summary/meta和导出版本。
- `colleague-job-inventory.json`：52条作业的只读清单。
- `evaluator-current/`、`evaluator-sha256.json`：实际使用的评测源码与哈希。
- `rejudge-original8/`：上一轮原始8题在当前判分器下的结果。
- `gpu-run/benchmark.sh`：完整调用命令。
- `gpu-run/job.json`、`job_entry.sh`、`run_job.py`：资源、模型环境和有限任务生命周期。

## 最终结果与任务对齐范围

任务1303263已Succeeded，owned服务全部退出，GPU资源释放。8项SDK功能请求通过；随后实际run_eval.sh评测15题：15/15非空成功，推理失败0、判分失败0；原判分器14/15，Overall93.3333%。框架记录推理107.641秒、判分0.035秒。status.json、逐题TSV、acc.csv、UE2格式summary/meta均生成。

唯一判错题index395903，类别partsOfA；正确选项B，模型最终选择D。该题为内容错误，未重问或修改答案。每类仅一题，小样本分数不能当作全量成绩或优于同事模型的证据。

与同事任务1298230一致：直接以Cybertron任务启动VLMEvalKit仓库的run_eval.sh，使用同名ue2-integration-20260928分支、AI2D_TEST、VLLMOpenAIWrapper、api-nproc4、max_tokens2048、temperature0、timeout600、exact_matching，输出UE2要求的summary/meta。

差异：同事使用呼和浩特全量/续跑及双工模拟HTTP适配器；本次使用廊坊A100上的15题新跑与原生HF单工接口，关闭通知/数据库上报，独立路径。评测源码使用该分支当前3d3fcbfe，并非已确认等同于其历史1f409531。没有从UltraEval2平台ue job submit入口提交，也没有验证该平台调度/看板写入；同事文档和实际任务1298230同样直接cctl启动run_eval.sh。

导出源码不含.git，框架自动commit字段为空；真实来源commit和文件哈希保存在inventory.json/evaluator-sha256.json。缺.env提示不影响测试，所需参数均显式提供。

最终结果：`gpu-run/COVERAGE_SUMMARY.json`、`gpu-run/framework-results/minicpm-o5/T20261020-052011/`、`gpu-run/framework-results/summary/summary_20261020_052203.csv`、`gpu-run/logs/benchmark.log`。命令为`gpu-run/benchmark.sh`。所有结果均在本页证据根目录下。
