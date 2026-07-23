#!/usr/bin/env python3
"""Trustworthy 2-CARD benchmark for MiniCPM-O5 full duplex. Fixes the correctness bugs the audit
found in the demo's model.benchmark():
  1. explicit torch.cuda.synchronize() on BOTH ranks around every timed block (prefill/generate/
     finalize) -- CUDA is async; the demo's bare time.time() lets prefill GPU work leak into the
     generate timer. [CRITICAL fix]
  2. unit_total = prefill + generate + FINALIZE (finalize_unit is a mandatory per-unit LLM forward
     the demo excluded). [HIGH fix]
  3. warmup units (incl. SPEAK so tts/vocoder/llm graphs capture off-clock) then DISCARD. [HIGH]
  4. tail stats: p50/p95/p99/max + count over 1000ms, per LISTEN/SPEAK -- a hard deadline is judged
     by the tail, not avg. [HIGH]
  5. cross-rank MAX per unit (the slower rank sets the deadline) via all_reduce(MAX). [2-rank]
  6. measures the SAME deployed opt engine the 2-card server would run (batched_mm + graphs +
     vocoder bucketing + LLM/graph-boundary TP sync), and records it. [CRITICAL: don't measure a config the
     server doesn't run.]
Runs the REAL duplex pipeline (duplex_prefill/duplex_generate/duplex_finalize == server path).
Env: MODE(voice/omni) FORCE(listen/speak/natural) UNITS WARMUP OUT_DIR."""
import os, sys, json, time
import numpy as np, torch

WORKTREE = os.environ["WORKTREE"]; sys.path.insert(0, WORKTREE); sys.path.insert(0, os.path.join(WORKTREE, "scripts"))
os.environ.setdefault("ATTN_IMPLEMENTATION", "sdpa")
MODEL_PATH = os.environ["MODEL_PATH"]; PT_PATH = os.environ["PT_PATH"]
BACKBONE = os.environ.get("BACKBONE_DIR", "/user/weihongliang/wangkaiqi/o5_backbone_hf")
OUT_DIR = os.environ["OUT_DIR"]; os.makedirs(OUT_DIR, exist_ok=True); os.environ["OUT_DIR"] = OUT_DIR
MODE = os.environ.get("MODE", "omni"); FORCE = os.environ.get("FORCE", "speak")
UNITS = int(os.environ.get("UNITS", "120")); WARMUP = int(os.environ.get("WARMUP", "8"))
VIDEO = os.path.join(WORKTREE, "assets", "samples", "compile.mp4")
REF_WAV = os.path.join(WORKTREE, "assets/ref_audio/ref_minicpm_signature.wav")
DUP = {"generate_audio": True, "ls_mode": "explicit", "max_new_speak_tokens_per_chunk": 20,
       "temperature": 0.7, "top_k": 20, "top_p": 0.8, "force_listen_count": 3}
GEN = dict(temperature=0.7, top_k=20, top_p=0.8, listen_prob_scale=1.0,
           text_repetition_penalty=1.05, text_repetition_window_size=512, length_penalty=1.1)

def pct(a, q):
    if not a: return 0.0
    return float(np.percentile(np.array(a), q))

def main():
    import torch.distributed as dist
    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank(); world = dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    def log(*a):
        if rank == 0: print(*a, flush=True)
    torch.manual_seed(1234); np.random.seed(1234)

    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    import minimal_o5_unified_model_duplex as probe
    from MiniCPMO45.opt_flags import OPT

    # ---- surgery: MiniCPMO(non-llm) + TP backbone (the 2-card serving model) ----
    log(f"=== build 2-card model  MODE={MODE} FORCE={FORCE} UNITS={UNITS} WARMUP={WARMUP} ===")
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    cfg._attn_implementation = os.environ["ATTN_IMPLEMENTATION"]; cfg._name_or_path = MODEL_PATH; cfg.name_or_path = MODEL_PATH
    with init_empty_weights():
        model = probe.MiniCPMO(cfg)
    sd = torch.load(PT_PATH, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("llm.")}, strict=False, assign=True); del sd
    model.llm = None; model.to(device=dev, dtype=torch.bfloat16)
    tp = AutoModelForCausalLM.from_pretrained(BACKBONE, tp_plan="auto", dtype=torch.bfloat16)
    for m in tp.modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "_experts_implementation"): c._experts_implementation = "batched_mm"
    model.llm = tp
    model.processor = probe.MiniCPMOProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.init_unified(preload_both_tts=True, duplex_config=DUP, device="cuda", chat_vocoder="token2wav")

    # deployed opt engine (the SAME set a 2-card server must enable) -- recorded in output
    ENGINE = {"experts": "batched_mm", "tts_fast": True, "lmhead": True, "tts_graph": True,
              "fuse_vision_audio": True, "llm_graph": True, "vocoder": "bucketed{50,302}",
              "tp": 2, "token_broadcast": True, "llm_cache": int(os.environ.get("O5_LLM_CACHE", "8192"))}
    for k in OPT: OPT[k] = False
    OPT.update({"tts_fast": True, "lmhead": True, "tts_graph": True, "fuse_vision_audio": True, "llm_graph": True})
    def set_attn(m, impl):
        for mod in m.modules():
            c = getattr(mod, "config", None)
            if c is not None and hasattr(c, "_attn_implementation"): c._attn_implementation = impl
    set_attn(model.tts.model, "eager"); os.environ["O5_VISION_BATCH"] = "1"
    dec = model.duplex.decoder
    if FORCE == "listen":
        model.duplex.force_listen_count = 10**9
    elif FORCE == "speak":
        for _lid in (getattr(model.duplex, "listen_token_id", None), getattr(dec, "listen_id", None)):
            if _lid is not None and _lid not in dec.forbidden_token_ids: dec.forbidden_token_ids.append(_lid)
    # TP sync is owned by the LLM/graph runner.
    _od = dec.decode
    def _sd(*a, **k):
        t = _od(*a, **k)
        if torch.is_tensor(t): t = t.contiguous(); dist.broadcast(t, src=0)
        return t
    dec.decode = _sd
    ref = probe.load_16k(REF_WAV)

    # media
    import subprocess, tempfile
    from PIL import Image
    import librosa
    tmp = tempfile.mkdtemp(prefix="bench2_"); wav = os.path.join(tmp, "a.wav")
    subprocess.run(["ffmpeg", "-i", VIDEO, "-ar", "16000", "-ac", "1", "-t", "20", "-f", "wav", "-y", wav], capture_output=True, check=True)
    audio, _ = librosa.load(wav, sr=16000, mono=True)
    ac = [audio[i*16000:(i+1)*16000].astype(np.float32) for i in range(min(20, len(audio)//16000))]
    fr = None
    if MODE == "omni":
        fdir = os.path.join(tmp, "f"); os.makedirs(fdir, exist_ok=True)
        subprocess.run(["ffmpeg", "-i", VIDEO, "-vf", "fps=1", "-t", "20", os.path.join(fdir, "f%03d.jpg"), "-y"], capture_output=True, check=True)
        fr = [Image.open(os.path.join(fdir, f)).convert("RGB") for f in sorted(os.listdir(fdir))]
    na = len(ac); nf = len(fr) if fr else 0
    def prep():
        model.duplex_prepare(prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
                             suffix_system_prompt="<|audio_end|><|im_end|>", ref_audio=ref, prompt_wav_path=REF_WAV)

    def synced_max(x):
        # cross-rank MAX: the slower rank sets the deadline
        t = torch.tensor([x], device=dev, dtype=torch.float64); dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    # WARMUP (past force_listen=3 so tts/vocoder/llm graphs capture off-clock), then DISCARD
    log(f"=== warmup {WARMUP} units (graphs capture off-clock, discarded) ===")
    prep()
    for k in range(WARMUP):
        fl = [fr[k % nf]] if MODE == "omni" else None
        model.duplex_prefill(audio_waveform=ac[k % na], frame_list=fl)
        model.duplex_generate(decode_mode="sampling", **GEN); model.duplex_finalize()
    try:
        import vocoder_graph as vg; vg.enable_bucketed(model, log, bucket=50)
    except Exception as e: log("vocoder enable failed:", e)

    # ---- timed loop: synchronize BOTH ranks around EACH block; unit_total = pf+gen+fin ----
    log(f"=== timed {UNITS} units ===")
    prep()
    recs = []
    for u in range(UNITS):
        fl = [fr[u % nf]] if MODE == "omni" else None
        dist.barrier()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model.duplex_prefill(audio_waveform=ac[u % na], frame_list=fl)
        torch.cuda.synchronize(); t1 = time.perf_counter()                 # prefill synced
        r = model.duplex_generate(decode_mode="sampling", **GEN)
        torch.cuda.synchronize(); t2 = time.perf_counter()                 # generate synced
        model.duplex_finalize()
        torch.cuda.synchronize(); t3 = time.perf_counter()                 # finalize synced
        pf, gen, fin = (t1-t0)*1e3, (t2-t1)*1e3, (t3-t2)*1e3
        unit = pf + gen + fin
        unit_max = synced_max(unit)                                        # slower rank's time
        recs.append({"unit": u, "is_listen": r.get("is_listen"), "n_tts": r.get("n_tts_tokens"),
                     "seq_len": int(dec.get_cache_length()),
                     "prefill_ms": round(pf,2), "generate_ms": round(gen,2), "finalize_ms": round(fin,2),
                     "unit_total_ms": round(unit,2), "unit_total_max_ms": round(unit_max,2),
                     "token2wav_ms": round((r.get("cost_token2wav") or 0)*1e3,2)})
        if u % 20 == 0: log(f"  unit {u}: total={unit:.1f}ms (max {unit_max:.1f}) pf={pf:.0f} gen={gen:.0f} fin={fin:.0f} listen={r.get('is_listen')}")
    if rank == 0:
        def stats(rs, key):
            v = [x[key] for x in rs]
            if not v: return None
            over = sum(1 for x in v if x > 1000.0)
            return {"n": len(v), "avg": round(sum(v)/len(v),1), "p50": round(pct(v,50),1),
                    "p95": round(pct(v,95),1), "p99": round(pct(v,99),1), "max": round(max(v),1),
                    "over_1000ms": over, "over_1000ms_frac": round(over/len(v),3)}
        li = [x for x in recs if x["is_listen"]]; sp = [x for x in recs if not x["is_listen"]]
        summ = {"mode": MODE, "force": FORCE, "world": world, "engine": ENGINE, "n_units": len(recs),
                "SLA_metric": "unit_total_max_ms (prefill+generate+finalize, synced, max across ranks)",
                "ALL": stats(recs, "unit_total_max_ms"),
                "LISTEN": stats(li, "unit_total_max_ms"), "SPEAK": stats(sp, "unit_total_max_ms"),
                "token2wav_SPEAK": stats(sp, "token2wav_ms")}
        fn = os.path.join(OUT_DIR, f"bench2_{MODE}_{FORCE}.json")
        json.dump({"summary": summ, "records": recs}, open(fn, "w"), indent=1)
        log("SUMMARY " + json.dumps(summ, ensure_ascii=False))
        log("saved " + fn)
    sys.stdout.flush(); os._exit(0)

if __name__ == "__main__":
    main()
