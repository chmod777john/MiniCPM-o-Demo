#!/usr/bin/env python3
"""Profile pure Qwen3.5 MoE backbone prefill/decode.

This script intentionally excludes MiniCPM-o outer logic, audio, TTS, duplex,
and serving code. It reads the HF backbone directory as a normal Transformers
model and records both coarse module CUDA-event timing and PyTorch profiler
operator/kernel summaries.
"""

import json
import os
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM
from transformers.cache_utils import StaticCache


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/user/weihongliang/wangkaiqi/o5_backbone_hf"))
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", "/user/weihongliang/MiniCPM-o-4_6"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/user/weihongliang/o5_pure_moe_profile"))
PROMPT = os.environ.get("PROMPT", "请用中文简要介绍西安的历史、地理、文化和旅游特色。")
PROMPT_REPEAT = int(os.environ.get("PROMPT_REPEAT", "4"))
DECODE_STEPS = int(os.environ.get("DECODE_STEPS", "8"))
WARMUP_PREFILL_STEPS = int(os.environ.get("WARMUP_PREFILL_STEPS", "1"))
WARMUP_DECODE_STEPS = int(os.environ.get("WARMUP_DECODE_STEPS", "2"))
ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPLEMENTATION", "sdpa")
EXPERTS_IMPLEMENTATION = os.environ.get("EXPERTS_IMPLEMENTATION", "auto")
PROFILE = os.environ.get("PROFILE", "1").lower() in {"1", "true", "yes", "on"}
MODULE_TIMING = os.environ.get("MODULE_TIMING", "1").lower() in {"1", "true", "yes", "on"}
RECORD_SHAPES = os.environ.get("RECORD_SHAPES", "1").lower() in {"1", "true", "yes", "on"}
PROFILE_MEMORY = os.environ.get("PROFILE_MEMORY", "1").lower() in {"1", "true", "yes", "on"}
EXPORT_TRACE = os.environ.get("EXPORT_TRACE", "0").lower() in {"1", "true", "yes", "on"}
PROFILER_ROW_LIMIT = int(os.environ.get("PROFILER_ROW_LIMIT", "80"))
ENABLE_COMPILE = os.environ.get("ENABLE_COMPILE", "0").lower() in {"1", "true", "yes", "on"}
COMPILE_TARGET = os.environ.get("COMPILE_TARGET", "model")
COMPILE_MODE = os.environ.get("COMPILE_MODE", "default")
COMPILE_DYNAMIC = os.environ.get("COMPILE_DYNAMIC", "1").lower() in {"1", "true", "yes", "on"}
DECODE_ATTENTION_MASK = os.environ.get("DECODE_ATTENTION_MASK", "1").lower() in {"1", "true", "yes", "on"}
CUDA_PROFILER_RANGE = os.environ.get("CUDA_PROFILER_RANGE", "0").lower() in {"1", "true", "yes", "on"}
NVTX_RANGES = os.environ.get("NVTX_RANGES", "1").lower() in {"1", "true", "yes", "on"}
NVTX_MODULES = os.environ.get("NVTX_MODULES", "0").lower() in {"1", "true", "yes", "on"}
SYNC_EACH_DECODE = os.environ.get("SYNC_EACH_DECODE", "1").lower() in {"1", "true", "yes", "on"}
LOGITS_TO_KEEP = int(os.environ.get("LOGITS_TO_KEEP", "1"))
PATCH_B1_EXPERTS = os.environ.get("PATCH_B1_EXPERTS", "0").lower() in {"1", "true", "yes", "on"}
USE_STATIC_CACHE = os.environ.get("USE_STATIC_CACHE", "0").lower() in {"1", "true", "yes", "on"}
DTYPE = torch.bfloat16


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def summarize(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "total_ms": sum(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 90),
        "p99_ms": percentile(values, 99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def sync_time(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


@contextmanager
def nvtx_range(name):
    if not NVTX_RANGES or not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def load_model():
    model = Qwen3_5MoeForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        attn_implementation=ATTN_IMPLEMENTATION,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    if EXPERTS_IMPLEMENTATION != "auto":
        model.config._experts_implementation = EXPERTS_IMPLEMENTATION
    model.to("cuda")
    if PATCH_B1_EXPERTS:
        patch_batch1_experts(model)
    if NVTX_MODULES:
        patch_nvtx_modules(model)
    return model


def batch1_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Specialized inference path for decode hidden_states.shape == (1, hidden).

    This keeps the same math as the batched_mm experts path for the single-token
    case, but avoids repeat_interleave and the second gather of hidden states.
    It is intentionally local to this profiling script.
    """

    if hidden_states.shape[0] != 1 or self.has_bias or not self.has_gate or self.is_transposed:
        return self.__base_forward(hidden_states, top_k_index, top_k_weights)

    expert_ids = top_k_index.reshape(-1).clamp(0, self.num_experts - 1)
    weights = top_k_weights.reshape(-1).to(hidden_states.dtype)
    hidden = hidden_states.expand(expert_ids.shape[0], -1)

    gate_up = self.gate_up_proj[expert_ids]
    proj = torch.bmm(gate_up, hidden.unsqueeze(-1)).squeeze(-1)
    proj = self._apply_gate(proj)

    down = self.down_proj[expert_ids]
    proj = torch.bmm(down, proj.unsqueeze(-1)).squeeze(-1)
    return (proj * weights.unsqueeze(-1)).sum(dim=0, keepdim=True).to(hidden_states.dtype)


def patch_batch1_experts(model):
    for layer in model.model.layers:
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if experts is None or hasattr(experts, "__base_forward"):
            continue
        experts.__base_forward = experts.forward
        experts.forward = batch1_experts_forward.__get__(experts, experts.__class__)


def wrap_forward_with_nvtx(module, label):
    if module is None or hasattr(module, "__nvtx_original_forward"):
        return
    module.__nvtx_original_forward = module.forward

    def wrapped_forward(*args, **kwargs):
        with nvtx_range(label):
            return module.__nvtx_original_forward(*args, **kwargs)

    module.forward = wrapped_forward


def patch_nvtx_modules(model):
    for layer_idx, layer in enumerate(model.model.layers):
        wrap_forward_with_nvtx(layer, f"layer.{layer_idx:02d}")
        for name in ("self_attn", "linear_attn", "input_layernorm", "post_attention_layernorm"):
            wrap_forward_with_nvtx(getattr(layer, name, None), f"layer.{layer_idx:02d}.{name}")
        mlp = getattr(layer, "mlp", None)
        wrap_forward_with_nvtx(mlp, f"layer.{layer_idx:02d}.mlp")
        if mlp is not None:
            for name in ("gate", "experts", "shared_expert", "shared_expert_gate"):
                wrap_forward_with_nvtx(getattr(mlp, name, None), f"layer.{layer_idx:02d}.mlp.{name}")


def maybe_compile_model(model):
    if not ENABLE_COMPILE:
        return {"enabled": False}
    kwargs = {"mode": COMPILE_MODE, "dynamic": COMPILE_DYNAMIC}
    start = time.perf_counter()
    if COMPILE_TARGET == "model":
        model.model = torch.compile(model.model, **kwargs)
        compiled = "model.model"
    elif COMPILE_TARGET == "full":
        model = torch.compile(model, **kwargs)
        compiled = "full_model"
    else:
        raise ValueError(f"Unsupported COMPILE_TARGET={COMPILE_TARGET!r}")
    return {
        "enabled": True,
        "target": compiled,
        "mode": COMPILE_MODE,
        "dynamic": COMPILE_DYNAMIC,
        "wrap_s": time.perf_counter() - start,
    }


class ModuleTimer:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.records = []
        self.phase = "unknown"

    def _pre(self, label):
        def hook(_module, _inputs):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            return start

        return hook

    def _post(self, label):
        def hook(_module, _inputs, _outputs):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.records.append((self.phase, label, _module.__class__.__name__, _module.__timer_start, end))

        return hook

    def _wrap_pre(self, label):
        def hook(module, inputs):
            module.__timer_start = self._pre(label)(module, inputs)

        return hook

    def register(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            candidates = {
                f"layer.{layer_idx:02d}.self_attn": getattr(layer, "self_attn", None),
                f"layer.{layer_idx:02d}.mlp": getattr(layer, "mlp", None),
                f"layer.{layer_idx:02d}.input_layernorm": getattr(layer, "input_layernorm", None),
                f"layer.{layer_idx:02d}.post_attention_layernorm": getattr(layer, "post_attention_layernorm", None),
            }
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                for sub_name in ("gate", "experts", "shared_expert", "shared_expert_gate", "gate_proj", "up_proj", "down_proj"):
                    sub = getattr(mlp, sub_name, None)
                    if sub is not None:
                        candidates[f"layer.{layer_idx:02d}.mlp.{sub_name}"] = sub
            for label, module in candidates.items():
                if module is None:
                    continue
                self.handles.append(module.register_forward_pre_hook(self._wrap_pre(label)))
                self.handles.append(module.register_forward_hook(self._post(label)))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @contextmanager
    def scoped_phase(self, phase):
        old = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = old

    def summary(self):
        torch.cuda.synchronize()
        by_phase_label = defaultdict(list)
        by_phase_kind = defaultdict(list)
        by_label = defaultdict(list)
        by_kind = defaultdict(list)
        for phase, label, class_name, start, end in self.records:
            ms = start.elapsed_time(end)
            by_phase_label[(phase, label)].append(ms)
            by_label[label].append(ms)
            kind = label.split(".")[-1]
            if ".mlp." in label:
                kind = "mlp." + kind
            elif label.endswith("self_attn"):
                kind = "self_attn"
            by_phase_kind[(phase, kind)].append(ms)
            by_kind[kind].append(ms)
        label_summary = {label: summarize(times) for label, times in sorted(by_label.items())}
        kind_summary = {label: summarize(times) for label, times in sorted(by_kind.items())}
        phase_kind_summary = {}
        for (phase, kind), times in sorted(by_phase_kind.items()):
            phase_kind_summary.setdefault(phase, {})[kind] = summarize(times)
        phase_label_summary = {}
        for (phase, label), times in sorted(by_phase_label.items()):
            phase_label_summary.setdefault(phase, {})[label] = summarize(times)
        top_labels = sorted(
            ((label, data["total_ms"], data) for label, data in label_summary.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:30]
        return {
            "by_kind": kind_summary,
            "by_label": label_summary,
            "by_phase_kind": phase_kind_summary,
            "by_phase_label": phase_label_summary,
            "top_labels": [{"label": label, **data} for label, _total, data in top_labels],
        }


@contextmanager
def maybe_profiler(out_dir):
    if not PROFILE:
        yield None
        return
    trace_dir = out_dir / "torch_profiler"
    trace_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "activities": [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        "record_shapes": RECORD_SHAPES,
        "profile_memory": PROFILE_MEMORY,
        "with_stack": False,
    }
    if EXPORT_TRACE:
        kwargs["on_trace_ready"] = torch.profiler.tensorboard_trace_handler(str(trace_dir))
    try:
        with torch.profiler.profile(acc_events=True, **kwargs) as prof:
            yield prof
    except TypeError:
        with torch.profiler.profile(**kwargs) as prof:
            yield prof


@torch.inference_mode()
def run_profile(model, input_ids, out_dir):
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device="cuda")
    initial_cache = None
    if USE_STATIC_CACHE:
        max_cache_len = input_ids.shape[1] + WARMUP_DECODE_STEPS + DECODE_STEPS + 8
        initial_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)

    def prefill_call():
        with nvtx_range("prefill_forward"):
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=initial_cache,
                use_cache=True,
                return_dict=True,
                logits_to_keep=LOGITS_TO_KEEP,
            )

    def decode_call(token, decode_mask, past_key_values, step_label):
        with nvtx_range(step_label):
            return model(
                input_ids=token,
                attention_mask=decode_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=LOGITS_TO_KEEP,
            )

    for _ in range(WARMUP_PREFILL_STEPS):
        out, _ = sync_time(prefill_call)
        del out

    timer = ModuleTimer(model)
    if MODULE_TIMING:
        timer.register()
    profiler = None
    try:
        with maybe_profiler(out_dir) as prof:
            if CUDA_PROFILER_RANGE:
                torch.cuda.cudart().cudaProfilerStart()
            with timer.scoped_phase("prefill") if MODULE_TIMING else nullcontext():
                outputs, prefill_s = sync_time(prefill_call)
            if prof:
                prof.step()

            past = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

            for _ in range(WARMUP_DECODE_STEPS):
                cache_len = past.get_seq_length() if hasattr(past, "get_seq_length") else attention_mask.shape[1]
                decode_mask = torch.ones((1, cache_len + 1), dtype=torch.bool, device="cuda") if DECODE_ATTENTION_MASK else None
                with timer.scoped_phase("warmup_decode") if MODULE_TIMING else nullcontext():
                    out, _ = sync_time(lambda: decode_call(next_token, decode_mask, past, "warmup_decode_forward"))
                past = out.past_key_values
                next_token = out.logits[:, -1:, :].argmax(dim=-1)

            decode_times = []
            generated = []
            with nvtx_range("decode_loop"):
                if SYNC_EACH_DECODE:
                    for step in range(DECODE_STEPS):
                        cache_len = past.get_seq_length() if hasattr(past, "get_seq_length") else attention_mask.shape[1]
                        decode_mask = torch.ones((1, cache_len + 1), dtype=torch.bool, device="cuda") if DECODE_ATTENTION_MASK else None
                        with timer.scoped_phase("decode") if MODULE_TIMING else nullcontext():
                            out, dt = sync_time(lambda: decode_call(next_token, decode_mask, past, f"decode_forward_{step:03d}"))
                        if prof:
                            prof.step()
                        decode_times.append(dt)
                        generated.append(int(next_token.item()))
                        past = out.past_key_values
                        next_token = out.logits[:, -1:, :].argmax(dim=-1)
                else:
                    torch.cuda.synchronize()
                    loop_start = time.perf_counter()
                    for step in range(DECODE_STEPS):
                        cache_len = past.get_seq_length() if hasattr(past, "get_seq_length") else attention_mask.shape[1]
                        decode_mask = torch.ones((1, cache_len + 1), dtype=torch.bool, device="cuda") if DECODE_ATTENTION_MASK else None
                        with timer.scoped_phase("decode") if MODULE_TIMING else nullcontext():
                            out = decode_call(next_token, decode_mask, past, f"decode_forward_{step:03d}")
                        if prof:
                            prof.step()
                        generated.append(int(next_token.item()))
                        past = out.past_key_values
                        next_token = out.logits[:, -1:, :].argmax(dim=-1)
                    torch.cuda.synchronize()
                    loop_s = time.perf_counter() - loop_start
                    decode_times = [loop_s / DECODE_STEPS] * DECODE_STEPS

            if CUDA_PROFILER_RANGE:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()

            profiler = prof
    finally:
        timer.close()

    profiler_tables = {}
    if PROFILE and profiler:
        key_averages = profiler.key_averages(group_by_input_shape=RECORD_SHAPES)
        profiler_tables["cuda_time_top"] = key_averages.table(
            sort_by="cuda_time_total", row_limit=PROFILER_ROW_LIMIT
        )
        profiler_tables["self_cuda_time_top"] = key_averages.table(
            sort_by="self_cuda_time_total", row_limit=PROFILER_ROW_LIMIT
        )
        profiler_tables["cpu_time_top"] = key_averages.table(
            sort_by="cpu_time_total", row_limit=PROFILER_ROW_LIMIT
        )
        profiler_tables["self_cpu_time_top"] = key_averages.table(
            sort_by="self_cpu_time_total", row_limit=PROFILER_ROW_LIMIT
        )
        if EXPORT_TRACE:
            profiler.export_chrome_trace(str(out_dir / "chrome_trace.json"))

    return {
        "prefill_s": prefill_s,
        "decode_times_s": decode_times,
        "decode_summary_s": summarize([v * 1000 for v in decode_times]),
        "decode_tokens_per_s": len(decode_times) / sum(decode_times) if decode_times else None,
        "generated_token_ids": generated,
        "module_timing_ms": timer.summary() if MODULE_TIMING else {},
        "profiler_tables": profiler_tables,
    }


def main():
    torch.cuda.set_device(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model = load_model()
    load_s = time.perf_counter() - t0
    compile_info = maybe_compile_model(model)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    prompt = "\n".join([PROMPT] * PROMPT_REPEAT)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to("cuda")

    result = run_profile(model, input_ids, OUT_DIR)
    text = tokenizer.decode(result["generated_token_ids"], skip_special_tokens=True)

    record = {
        "model_path": str(MODEL_PATH),
        "tokenizer_path": str(TOKENIZER_PATH),
        "model_class": model.__class__.__name__,
        "model_type": model.config.model_type,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "experts_implementation": getattr(model.config, "_experts_implementation", None),
        "has_functional_grouped_mm": hasattr(torch.nn.functional, "grouped_mm"),
        "has_private_grouped_mm": hasattr(torch, "_grouped_mm"),
        "prompt": PROMPT,
        "prompt_repeat": PROMPT_REPEAT,
        "prompt_tokens": int(input_ids.shape[1]),
        "decode_steps": DECODE_STEPS,
        "warmup_prefill_steps": WARMUP_PREFILL_STEPS,
        "warmup_decode_steps": WARMUP_DECODE_STEPS,
        "profile": PROFILE,
        "module_timing": MODULE_TIMING,
        "record_shapes": RECORD_SHAPES,
        "profile_memory": PROFILE_MEMORY,
        "export_trace": EXPORT_TRACE,
        "profiler_row_limit": PROFILER_ROW_LIMIT,
        "compile": compile_info,
        "decode_attention_mask": DECODE_ATTENTION_MASK,
        "logits_to_keep": LOGITS_TO_KEEP,
        "patch_b1_experts": PATCH_B1_EXPERTS,
        "use_static_cache": USE_STATIC_CACHE,
        "cuda_profiler_range": CUDA_PROFILER_RANGE,
        "nvtx_ranges": NVTX_RANGES,
        "nvtx_modules": NVTX_MODULES,
        "sync_each_decode": SYNC_EACH_DECODE,
        "load_s": load_s,
        "cuda_max_mem_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "generated_text": text,
        **result,
    }
    (OUT_DIR / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "profiler_cuda_top.txt").write_text(
        record["profiler_tables"].get("cuda_time_top", ""), encoding="utf-8"
    )
    (OUT_DIR / "profiler_self_cuda_top.txt").write_text(
        record["profiler_tables"].get("self_cuda_time_top", ""), encoding="utf-8"
    )
    (OUT_DIR / "profiler_cpu_top.txt").write_text(
        record["profiler_tables"].get("cpu_time_top", ""), encoding="utf-8"
    )
    (OUT_DIR / "profiler_self_cpu_top.txt").write_text(
        record["profiler_tables"].get("self_cpu_time_top", ""), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in record.items() if k != "module_timing_ms" and k != "profiler_tables"}, ensure_ascii=False), flush=True)
    if MODULE_TIMING:
        print("module_timing_top_labels", json.dumps(record["module_timing_ms"]["top_labels"][:10], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
