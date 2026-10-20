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

Completed:92/92 new benchmark requests returned nonempty successful responses. Eight new dataset configurations completed the original judge and UE2 exports; MathVista inference completed but its independent judge is blocked. See final table below. A successful API call is not proof that its answer is correct; a small-sample score is not a full benchmark claim.

## Image-stage result (preserved history)

All80 selected image benchmark requests returned nonempty HTTP200 responses: MMBench EN20, CN20, MMMU6, TextVQA6, DocVQA6, ChartQA6, OCRBench10, MathVista6. Seven image configurations finished the original judge and generated status/summary/meta; MathVista awaits failure/availability of the independent judge. The five-image MMMU request was observed in raw HTTP capture and succeeded. No runtime source changes were needed.

| Dataset | API | Original observed grade | Inference seconds |
|---|---:|---|---:|
| MMBench EN V1.1 dev |20/20|Vanilla diagnostic17/20; original partial circular3/3|89.010|
| MMBench CN V1.1 dev |20/20|Vanilla diagnostic17/20; original partial circular2/3|60.288|
| MMMU dev/val |6/6|4/6|74.104|
| TextVQA val |6/6|VQA score100|2.316|
| DocVQA val |6/6|ANLS94.0476|3.615|
| ChartQA test |6/6|Relaxed accuracy83.3333 (5/6)|2.343|
| OCRBench |10/10|7 hits; original score7, full-set normalized0.7|3.830|
| MathVista mini |6/6|Independent GPT judge unavailable; no valid grade|58.4 approx|

These numbers are tiny-sample integration observations only. HTTP capture `gpu-run/http-transactions.jsonl` preserves complete returned text/usage/finish reason and request text; image bodies are represented by SHA256 and length, with original data and decoded images retained separately. This transparent ASGI capture wraps the unchanged app and does not rewrite requests or responses.

## Reproduction and evidence

```sh
# Finite owned model job; submit only when resources are available.
cctl job create -f /user/weihongliang/analysis-runs/20261020-openai-multibench/gpu-run/job.json
# The job starts unchanged services and executes every full run_eval.sh command:
# gpu-run/run_suite.py; exact executed argv and wall time: gpu-run/suite-execution.json
```

`gpu-run/job_entry.sh` contains the exact model, TP2, runtime environment, isolated ports and HF/accel paths. `gpu-run/job.json`, `submission.json`, `diagnosis-live.json` identify resources and node10.156.1.177. Runtime baseline is76e895eb (printed by the task), even though later documentation commits were made while it ran; `runtime-version-check.json` confirms runtime source equality.

`sample-manifest.json` fixes samples before inference. `prepare_data.py` reconstructs seed selection solely from question/input metadata and read-only official TSVs. `prepare_video.py` invokes the unchanged VideoMME.build_prompt on the selected videos, saving96frames in the owned LMUData cache. `video-input-files.json`/`video-frame-manifest.json` record source/video/frame hashes. GPU inference still runs the actual original framework and its own prompt path.

`RESULTS.json` summarizes original status files; `gpu-run/framework-results/minicpm-o5/T*/` contains predictions, original judge outputs, and `status.json`. `gpu-run/framework-results/summary/` and the adjacent timestamped meta JSONs are the original run_eval.sh UE2 exports. Individual benchmark logs are in`gpu-run/logs/`. `evaluator-hashes.json` and`postrun-data-integrity.json` pin source/data and establish that shared TSVs were unchanged.

Two outputs reached the configured2048-token limit: MMMU index9929 and MathVista index641; HTTP`finish_reason=length` was correctly recorded. Their raw answers remain untouched. The endpoint does not promise a full answer when the caller's token limit is exhausted.

### MMBench group-count audit

Both EN/CN samples have20rows from20distinct group IDs, with no repeated rotations and no complete four-rotation group. Only three selected rows are the base rotation (`index == index % 1e6`):3318,1392,3385. The original circular evaluator considers these three single-row groups, yielding EN3/3 and CN2/3; this does not measure complete circular consistency. The unchanged evaluator's `PRINT_VANILLA=1` diagnostic considers all20selected rows and yields17/20 for each language. `mmbench-group-coverage.json` records the audit.

### External judge interruption and video continuation

MathVista produced6/6 responses in the first GPU task. Its original independent judge health check did not return after11+minutes and no authorized judge configuration was available. At root's direction, owned task1303424 was stopped to avoid spending GPU time on that external wait. Its terminal platform state isKilled; the original MathVista status remains`eval`, accurately showing interruption. No judge result or completed summary is fabricated. Original health-check logs, predictions and `mathvista-interruption.json` document this gap.

Video-only owned task1303538 starts the same unchanged runtime and runs only Video-MME; it does not repeat the80completed image requests. It prints Demo1c548e78, which adds documentation only over76e895eb. Runtime source remains identical.

## Final acceptance table

All sample IDs matched the manifest frozen before inference. No runtime source edits were necessary; no HF, accel or API code changed.

| Benchmark/configuration | API success/failure | Original grade observed | Infer time | Original judge/export | Task |
|---|---:|---|---:|---|---:|
| AI2D_TEST (prior evidence reused) |15/0|14/15|107.641s|Complete|1303263|
| MMBench_DEV_EN_V11 |20/0|Vanilla17/20; partial circular3/3, no complete rotation groups|89.010s|Complete|1303424|
| MMBench_DEV_CN_V11 |20/0|Vanilla17/20; partial circular2/3, no complete rotation groups|60.288s|Complete|1303424|
| MMMU_DEV_VAL |6/0|4/6; includes one/five images and MCQ/open questions|74.104s|Complete|1303424|
| TextVQA_VAL |6/0|VQA score100|2.316s|Complete|1303424|
| DocVQA_VAL |6/0|ANLS94.0476|3.615s|Complete|1303424|
| ChartQA_TEST |6/0|Relaxed accuracy83.3333 (5/6)|2.343s|Complete|1303424|
| OCRBench |10/0|7hits; stock raw score7, full-set normalized0.7|3.830s|Complete|1303424|
| MathVista_MINI |6/0|No grade: independent judge unavailable|58.4s approx (inference log)|Inference saved; original eval interrupted|1303424|
| Video-MME_8frame |12/0|5/12; stock rounded Overall Score41.7|5.818s|Complete|1303538|

The new run has92benchmark calls plus9original framework API health probes,101HTTP transactions total, all200. Input counts observed in HTTP capture:79single-image questions, one5-image question,12eight-frame video questions, and9text-only probes. Together with the prior15AI2D cases, the evidence covers107benchmark requests, without re-running AI2D. There are no HTTP failures or empty model answers in these selected samples.

Video uses real short/medium/long files, the original12task-type stratification, eight uniformly sampled frames, no subtitles/audio, and the original wrapper's448-pixel image size setting. Raw prediction and score tables confirm5correct of12; the stock report rounds0.41666 to0.417 before presenting41.7. This is video-as-multiple-images compatibility, not native video file upload/audio input or full video quality certification.

The parameter/API coverage is shared text/image Chat Completions with nonstreaming output. Platform submission was direct`cctl → run_eval.sh`, matching the colleague's documented actual route. UltraEval2 platform`ue job submit`, database ingestion, notifications and UI were not tested.

## Terminal resource state and follow-up

- Task1303424:Killed at our request after preserving all80image outputs and seven complete judge results; no stale owned GPU service remains.
- Task1303538:Succeeded. `video-run/CLEANUP.json`:suite0, service143 and HTTP wrapper-15 (deliberate final termination). All owned services exited and twoA100resources released.
- No formal branch merge or push. Only this isolated candidate's documentation/worktree ownership changed.
- Remaining external requirement:authorized independent MathVista judge endpoint/key. This can be completed CPU-only from saved predictions using the original evaluator; no further model inference is required.

Final machine-readable evidence:`RESULTS.json`, `sample-manifest.json`, `mmbench-group-coverage.json`, `mathvista-interruption.json`, `video-frame-manifest.json`, and`evidence-sha256.json`. Commands:`gpu-run/suite-execution.json` and`video-run/suite-execution.json`; final platform states:`gpu-run/job-final.json`, `video-run/job-final.json`.
