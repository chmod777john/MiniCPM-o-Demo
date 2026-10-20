# Native simplex representative benchmark validation

Owner `/root/simplex_benchmark_suite`; date 2026-10-20 UTC. This is an independent candidate based on Demo `76e895eb428ba1e4701c2727837030cf1d877e57`. HF `528cb102868278f276d99a562a67c8912eee8996` and accel `9c6f33619b7580b96fdd211d3fc736a75909398e` remain unchanged.

## Scope and method

Run the original colleague-specified `ue2-integration-20260928` evaluator revision `3d3fcbfee182cc9bebab90272525b2fb47aaeeb6`, through `run_eval.sh → eval_runner.sh → run.py → VLLMOpenAIWrapper → OpenAI HTTP → Gateway chat → native HF chat → original per-dataset evaluation → UE2-format summary/meta`. No `ue job submit`, dashboard/database or notification coverage. AI2D reuses the prior completed 15-sample result, task1303263.

Evidence root: `/user/weihongliang/analysis-runs/20261020-openai-multibench`. Immutable evaluator copy has the same Python source hashes as the exported original. All image TSV MD5s match their evaluator-declared MD5s; Video-MME TSV MD5 matches `VideoMME.MD5`.

Task1303424 is a finite, owned langfang_train/agent-train TP2 A100 job. agent-dev had only one free A100 and deploy none; authorized agent-train had free quota. Public services and other jobs are untouched. Model/checkpoint and model environment follow previous successful task1303263. The CPU HTTP interpreter reuses the already provisioned base candidate project venv read-only; no global install.

## Sampling fixed before inference

`sample-manifest.json` records exact source, MD5, sample IDs and metadata. The original evaluator's stratified sampler is used without modifications. No answers or predictions were consulted to choose samples.

| Dataset | Sample/seed | Actual rows | Intended coverage |
|---|---:|---:|---|
| MMBench_DEV_EN_V11 | 2 / 0 | 20 | All20 categories; English |
| MMBench_DEV_CN_V11 | 2 / 0 | 20 | All20 categories; Chinese |
| MMMU_DEV_VAL | 2 / 4 | 6 | Six broad subjects, multiple choice/open questions, one/five images |
| TextVQA_VAL | 6 / 0 | 6 | Random real scene-text questions |
| DocVQA_VAL | 6 / 0 | 6 | Random document questions |
| ChartQA_TEST | 6 / 0 | 6 | Both original splits, three rows each |
| OCRBench | 2 / 0 | 10 | All ten categories |
| MathVista_MINI | 6 / 361 | 6 | Both categories, MCQ/free form, text/integer/float/list answer types |
| Video-MME_8frame | 2 / 0 | 12 | All12 task types; short/medium/long; real videos, original8-frame no-subtitle profile |

Seeds4 and361 are the first seeds meeting specified input/question-type coverage; chosen by metadata only. Video frames use original wrapper `img_size=448` to bound image contexts, a declared evaluation configuration rather than default full-resolution video-quality certification. Images elsewhere retain the original resolution behavior. `max_tokens=2048`, `temperature=0`, `timeout=600`, `api-nproc=4` match colleague settings.

## Judging boundaries known before the run

- MMBench/MMMU/Video-MME use each dataset's existing `exact_matching` route. MMBench's stock row-wise sampler may select circular rotations without their base rows: preserve its primary circular output, and request the existing `PRINT_VANILLA=1` diagnostic to report all sampled rows. Partial rotation coverage is not full MMBench circular consistency certification.
- TextVQA, DocVQA, ChartQA and OCRBench use their own default heuristic graders (VQA scoring, ANLS, relaxed accuracy and OCR scoring). No custom answer extraction or normalization is added.
- MathVista's original evaluator requires a working independent GPT judge even when some answers could be extracted heuristically. No authorized judge credential/base is configured in this task environment. Run actual inference and the original judge path; retain any judge failure explicitly. Do not substitute the tested model as judge or rewrite scores.
- OCRBench's original normalized score assumes the full1000-row benchmark; preserve that raw field but do not interpret it as percentage accuracy on ten samples.

## Results

In progress. Per-dataset exit codes, original status files, raw predictions, HTTP responses, timings and outputs will be added after the owned task completes. A successful API call is not proof that its answer is correct; a small-sample score is not a full benchmark claim.
