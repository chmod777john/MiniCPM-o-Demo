# 原生 HF 单工 OpenAI 兼容接口：验收记录

日期：2026-10-20 UTC。Owner：root。

## 结论与代码边界

接口最小集已实现，真实 A100 TP2 调用和 AI2D 小样本链路已完成。所有实现变更位于 Demo 独立分支 `demo-adapter-fc-openai-simplex-20261020`；HF、accel 工作树均干净且 commit 未改变。正式 Demo 与 MAP 应用分支未合入本候选。

| 物料 | 验证使用的 commit |
| --- | --- |
| Demo 基线 | `c4a72a68d078b32fbe33ae60c160294af7d72100` |
| Demo 接口实现 | `eab947916bd85f4b795cb877b88bef2d4e22c456` |
| Demo 实际 GPU 验证版本，含会话收尾修正 | `486dc4d7f236cb86b508ecdd1c78a4a976d58975` |
| HF | `528cb102868278f276d99a562a67c8912eee8996` |
| accel | `9c6f33619b7580b96fdd211d3fc736a75909398e` |

验收记录和 CPU 依赖清单在后续提交中补入，GPU 使用的运行时代码如上。

调用链：`openai_app.py` → Gateway `mode=chat` → Worker → `py_backend/server.py` → `core/processors/pytorch_backend.py` → `ChatView` → Demo 的 `modeling/external_o5_facade.py` → 外部 HF `chat()`。

Demo facade 的新增观察代码只读取 HF 实际 processor 输入长度与生成 token、EOS，生成方式仍调用原 `super().generate()`。它用于返回真实 `usage` 和 `finish_reason`。不涉及 HF 源文件、FC 状态机或 accel 算法修改。

## 已实现范围

- `/health`、`GET /v1/models`、`POST /v1/chat/completions`。
- 非流式文本结果、文本/内联 PNG/JPEG 输入，按顺序转交历史消息。
- `temperature=0` 为单 beam 贪心；正温度和 top_p 透传。
- `max_tokens` / `max_completion_tokens`，实际 `stop` / `length` 与 token 用量。
- 可选 Bearer 鉴权、错误格式、现有 Gateway 排队。
- 超出范围的参数明确报错。未实现音频 HTTP 端点、流式 SSE、工具调用、远程图片 URL、严格 JSON 等。完整支持矩阵见 `OPENAI_SIMPLEX.md`。

## 环境与任务

成功任务：`1303141`，`langfang_train / agent-train`，节点 `10.156.1.181`，2×A100、16 CPU、256GiB。任务状态 Succeeded；所有本任务服务退出，GPU 资源已释放。未操作既有9007/9009或他人的服务。

Checkpoint：`/backup/user/chenmoye/checkpoints/minicpm-5o-moe/0930_final_merge/0930_omni_baseline_mergeAll_agent_128k_176gpu/iter_300.pt`。

HF/accel 读取既有9009发布目录的冻结物料。推理 venv：`/user/weihongliang/environments/o5-3drope-fc-20261013/.venv`。HTTP bridge 使用候选目录的 `.venv`；具体包版本保存在证据目录的 `adapter-environment.lock`。评测侧使用 `/user/luozhen/miniforge3/envs/evalhub-vlmevalkit`（OpenAI SDK2.15.0），SDK 功能探针使用3.35.1。

沿用 TP2 加速配置。启动日志报告 dense causal prefill 的预编译产物不可用，保留原 attention；本次未改变该回退。不能把配置开关当作每条推理路径都使用了对应优化。

## 检查结果

| 验证 | 结果 |
| --- | --- |
| HTTP/参数/图像/历史/元数据/本地 WS 单测 | 26 passed；精简 venv 缺 pytest-asyncio，出现一个配置警告 |
| 现有 external HF chat contract 回归 | PASS |
| 现有 chat cancel/drain 回归 | PASS |
| 真实 OpenAI SDK 文本 | `2+2=4.`，stop，prompt22 / completion7 |
| 真实 token 上限 | max_tokens1，length，实际生成1 token |
| 真实历史消息 | 正确回忆 `ZXQ729` |
| 真实 PNG 图像 | 红色图片回答 `Red` |
| 四路 HTTP 并发进入单 Worker 排队 | 4/4非空成功；各请求约0.246/0.441/0.639/0.836秒，包含排队 |
| 不支持的 stream=true | HTTP400 |
| AI2D_TEST 原顺序前8题，四路请求 | 8/8 API成功、无空回复，约21.86秒 |
| 原始 exact_matching 判分 | 2/8；完整 `dataset.evaluate()` 复核 Overall0.25 |

前4个顺序功能请求约0.638、0.229、0.331、0.432秒，仅用于复现链路正常；不作为性能基准。

### 首次失败与修正

任务1303092先通过文本/图片/历史/上限，但 HTTP 桥接层在 response.done 后直接断开，下一请求偶尔被分配给尚在收尾的 Worker，触发403、10秒健康检查等待，最终并发出现502。

修正在 `openai_app.py`：拿到回答后发送已有 `session.close`，等待 `session.closed` 和连接关闭，再完成 HTTP。未修改 Worker/Gateway。第二轮所有16次推理（8功能探针+8AI2D）成功，Gateway/Worker日志无403、unreachable或session_failed。原失败证据保留。

## Benchmark 的真实范围与判分限制

使用 `/user/luozhen/VLMEvalKit`：Git HEAD `7e9310a7c713745ef5e604beea6814c9cb526817`，`vlmeval/api/vllm_openai.py` SHA256 `a9caac62c38acd09542ab4f9eba72a8d3d4004cfe0feeb0f5d0828b3077941a9`。该目录有与此次调用无关的既有 videomme.py 修改和未跟踪脚本；未修改它们。没有确认它就是同事呼和浩特作业的精确分支版本。

数据：`/user/luozhen/datasets/AI2D_TEST.tsv`，共3088题；固定前8题ID：345802、345801、345800、266902、266901、266904、266903、266900。

实际调用原 `ImageMCQDataset.build_prompt()`、`VLLMOpenAIWrapper.generate_inner()`，max_tokens2048、temperature0，使用原图片编码和prompt，四个客户端线程共享一个TP2模型Worker。使用原 `extract_answer_from_item(None, ...)`，并以 `dataset.evaluate(..., model="exact_matching")` 对保存结果复核。未运行整套 Cybertron run_eval.sh 提交包装，也未跑全量3088题。

6条判为Z（无法抽取），模型在解释中列举多个选项或使用较长回答，原规则无法唯一匹配。已保存原始全文；接口没有裁剪答案、修改prompt、空答重问、工具抑制prompt或自定义判分。2/8是该小样本在该判分器下的实际结果，不能写成内容质量已通过，也不能与同事77.88的全量结果直接比较。正式评测需对齐prompt/模型版本和判分策略。

## 证据与复现

成功证据根目录：`/user/weihongliang/analysis-runs/20261020-openai-simplex-close-v2`。

- `job.json`、`job_entry.sh`、`run_job.py`：资源、环境变量、启动与有限任务收尾。
- `probe.py`、`probe.json`：真实SDK功能请求与完整响应。
- `benchmark.py`、`judge_saved.py`：原框架调用与判分复核。
- `AI2D_SUMMARY.json`、`AI2D_TEST_8_predictions.tsv`、`AI2D_TEST_8_predictions_acc.csv`、`ai2d_*.json`：结果。
- `logs/`、`contract-tests.log`、`external-hf-chat.log`、`cancel-drain.log`：日志。
- `validated-materials.json`、`adapter-environment.lock`、`evidence-sha256.json`：版本与证据索引。

首次失败证据：`/user/weihongliang/analysis-runs/20261020-openai-simplex`。

启动方式见 `OPENAI_SIMPLEX.md`。本轮只保留代码与验收证据，没有保留一个长期运行的HTTP端点。
