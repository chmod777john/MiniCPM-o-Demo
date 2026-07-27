#!/usr/bin/env python3
"""2-CARD budget-calibration GRID. Full duplex on the TP backbone (SPMD, token-broadcast synced),
run to 32K seq length, N trials, recording per-unit budget + seq_length for 4 scenarios x 2 paths.

budget (== user's "extra decodes until the unit's 1s timer") is computed ANALYTICALLY as
floor((1s - t_base)/per_tok): the number of extra single-token decodes that still fit before the
1s deadline. per_tok (one graphed/raw decode) is remeasured every PROBE_EVERY units via a fixed-N
decode probe (rolled back) -- constant on the graph path, growing with context on the raw path.
The old wall-clock probe-until-deadline loop desyncs the TP ranks (per-rank clock -> different
all_reduce counts -> NCCL hang); the analytic form is the same quantity, SPMD-safe.

Env: MODE(voice/omni) FORCE(listen/speak) PATH_KIND(graph/raw) TARGET_SEQ UNIT_CAP NTRIAL OUT_DIR.
Graph path = deployed StaticCache(O5_LLM_CACHE) + llm_graph -> Family A (budget vs unit_idx).
Raw path   = DynamicCache + eager TP forward (real O(ctx) attention) -> Family B (budget vs seq)."""
import os, sys, json, time
import numpy as np, torch

WORKTREE = os.environ["WORKTREE"]; sys.path.insert(0, WORKTREE); sys.path.insert(0, os.path.join(WORKTREE, "scripts"))
os.environ.setdefault("ATTN_IMPLEMENTATION", "sdpa")
MODEL_PATH = os.environ["MODEL_PATH"]; PT_PATH = os.environ["PT_PATH"]
BACKBONE = os.environ.get("BACKBONE_DIR", "/user/weihongliang/wangkaiqi/o5_backbone_hf")
OUT_DIR = os.environ["OUT_DIR"]; os.makedirs(OUT_DIR, exist_ok=True); os.environ["OUT_DIR"] = OUT_DIR
MODE = os.environ.get("MODE", "omni"); FORCE = os.environ.get("FORCE", "speak")
PATH = os.environ.get("PATH_KIND", "graph")
TARGET_SEQ = int(os.environ.get("TARGET_SEQ", "32768"))
UNIT_CAP = int(os.environ.get("UNIT_CAP", "3000"))
NTRIAL = int(os.environ.get("NTRIAL", "6"))
START_TRIAL = int(os.environ.get("START_TRIAL", "0"))
PROBE_N = 12; PROBE_EVERY = 15; DEADLINE = 1.0
VIDEO = os.path.join(WORKTREE, "assets", "samples", "compile.mp4")
REF_WAV = os.path.join(WORKTREE, "assets/ref_audio/ref_minicpm_signature.wav")
DUP = {"generate_audio": True, "ls_mode": "explicit", "max_new_speak_tokens_per_chunk": 20,
       "temperature": 0.7, "top_k": 20, "top_p": 0.8, "force_listen_count": 3}
GEN = dict(temperature=0.7, top_k=20, top_p=0.8, listen_prob_scale=1.0,
           text_repetition_penalty=1.05, text_repetition_window_size=512, length_penalty=1.1)

def main():
    import torch.distributed as dist
    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank(); torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    def log(*a):
        if rank == 0: print(*a, flush=True)

    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    import minimal_o5_unified_model_duplex as probe
    from modeling.o5.opt_flags import OPT

    # --- surgery: MiniCPMO(non-llm) + TP backbone ---
    log(f"=== build MiniCPMO + TP backbone  MODE={MODE} FORCE={FORCE} PATH={PATH} ===")
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    cfg._attn_implementation = os.environ["ATTN_IMPLEMENTATION"]
    cfg._name_or_path = MODEL_PATH; cfg.name_or_path = MODEL_PATH
    with init_empty_weights():
        model = probe.MiniCPMO(cfg)
    sd = torch.load(PT_PATH, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("llm.")}, strict=False, assign=True)
    del sd
    model.llm = None; model.to(device=dev, dtype=torch.bfloat16)
    tp = AutoModelForCausalLM.from_pretrained(BACKBONE, tp_plan="auto", dtype=torch.bfloat16)
    for mod in tp.modules():
        c = getattr(mod, "config", None)
        if c is not None and hasattr(c, "_experts_implementation"): c._experts_implementation = "batched_mm"
    model.llm = tp
    model.processor = probe.MiniCPMOProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    model.init_unified(preload_both_tts=True, duplex_config=DUP, device="cuda", chat_vocoder="token2wav")
    for k in OPT: OPT[k] = False
    if PATH == "graph":
        OPT.update({"tts_fast": True, "lmhead": True, "tts_graph": True, "fuse_vision_audio": True, "llm_graph": True})
    else:
        OPT.update({"tts_fast": True, "lmhead": True, "tts_graph": True, "fuse_vision_audio": True})  # raw: no llm_graph
    def set_attn(m, impl):
        for mod in m.modules():
            c = getattr(mod, "config", None)
            if c is not None and hasattr(c, "_attn_implementation"): c._attn_implementation = impl
    set_attn(model.tts.model, "eager"); os.environ["O5_VISION_BATCH"] = "1"
    dec = model.duplex.decoder
    if FORCE == "listen":
        model.duplex.force_listen_count = 10**9
    else:
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
    tmp = tempfile.mkdtemp(prefix="grid_")
    wav = os.path.join(tmp, "a.wav")
    subprocess.run(["ffmpeg", "-i", VIDEO, "-ar", "16000", "-ac", "1", "-t", "20", "-f", "wav", "-y", wav],
                   capture_output=True, check=True)
    audio, _ = librosa.load(wav, sr=16000, mono=True)
    ac = [audio[i*16000:(i+1)*16000].astype(np.float32) for i in range(min(20, len(audio)//16000))]
    fr = None
    if MODE == "omni":
        fdir = os.path.join(tmp, "f"); os.makedirs(fdir, exist_ok=True)
        subprocess.run(["ffmpeg", "-i", VIDEO, "-vf", "fps=1", "-t", "20", os.path.join(fdir, "f%03d.jpg"), "-y"],
                       capture_output=True, check=True)
        fr = [Image.open(os.path.join(fdir, f)).convert("RGB") for f in sorted(os.listdir(fdir))]
    na = len(ac); nf = len(fr) if fr else 0

    def prep():
        model.duplex_prepare(prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
                             suffix_system_prompt="<|audio_end|><|im_end|>", ref_audio=ref, prompt_wav_path=REF_WAV)

    # warmup 8 (past force_listen=3 so TTS/vocoder graphs capture on real speak) -- graph path only
    log("=== warmup 8 ===")
    prep()
    for k in range(8):
        fl = [fr[k % nf]] if MODE == "omni" else None
        model.duplex_prefill(audio_waveform=ac[k % na], frame_list=fl)
        model.duplex_generate(decode_mode="sampling", **GEN); model.duplex_finalize()
    if PATH == "graph":
        try:
            import vocoder_graph as vg; vg.enable_bucketed(model, log, bucket=50)
        except Exception as e: log("vocoder enable failed:", e)
    try: dummy = dec.embed_token(1000)
    except Exception: dummy = dec.embed_tokens([1000])

    # per_tok probe (fixed N decodes, rolled back) -- graph StaticCache or raw DynamicCache
    def snap_graph():
        r = dec._llm_runner; ls = []
        for lyr in r.cache.layers:
            s = {}
            if getattr(lyr, "cumulative_length", None) is not None: s["cum"] = lyr.cumulative_length.clone()
            if getattr(lyr, "conv_states", None) is not None: s["conv"] = lyr.conv_states.clone()
            if getattr(lyr, "recurrent_states", None) is not None: s["rec"] = lyr.recurrent_states.clone()
            ls.append(s)
        return (dec._static_pos, ls)
    def rest_graph(sn):
        pos, ls = sn; dec._static_pos = pos; r = dec._llm_runner
        with torch.inference_mode():
            for lyr, s in zip(r.cache.layers, ls):
                if "cum" in s: lyr.cumulative_length.copy_(s["cum"])
                if "conv" in s: lyr.conv_states.copy_(s["conv"])
                if "rec" in s: lyr.recurrent_states.copy_(s["rec"])
    def measure_per_tok():
        # both ranks do exactly PROBE_N decodes -> lock-step; median seconds/token. Rolls back so the
        # real trajectory is unaffected. Graph: restore StaticCache conv/recurrent/cumulative. Raw:
        # clone+restore every tensor in each DynamicLayer (crop corrupts the hybrid cache -> assert).
        if PATH == "graph":
            if dec._llm_runner is None or not dec._llm_runner._captured: return None
            sn = snap_graph()
        else:
            sn = [{k: v.clone() for k, v in lyr.__dict__.items() if torch.is_tensor(v)} for lyr in dec.cache.layers]
        tks = []
        for _ in range(PROBE_N):
            torch.cuda.synchronize(); tt = time.perf_counter()
            dec.feed(dummy, return_logits=True); torch.cuda.synchronize()
            tks.append(time.perf_counter() - tt)
        if PATH == "graph":
            rest_graph(sn)
        else:
            for lyr, d in zip(dec.cache.layers, sn):
                for k, v in d.items(): setattr(lyr, k, v)
        tks.sort(); return tks[len(tks) // 2]

    all_records = []
    for trial in range(START_TRIAL, START_TRIAL + NTRIAL):
        torch.manual_seed(1000 + trial); np.random.seed(1000 + trial)
        prep()
        per_tok = None; recs = []; unit = 0
        while unit < UNIT_CAP:
            fl = [fr[unit % nf]] if MODE == "omni" else None
            dist.barrier()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            model.duplex_prefill(audio_waveform=ac[unit % na], frame_list=fl)
            r = model.duplex_generate(decode_mode="sampling", **GEN)
            torch.cuda.synchronize()
            t_base = time.perf_counter() - t0
            seq_len = int(dec.get_cache_length())
            if unit % PROBE_EVERY == 0 or per_tok is None:
                pt = measure_per_tok()
                if pt: per_tok = pt
            budget = max(0, int((DEADLINE - t_base) / per_tok)) if per_tok else 0
            recs.append({"trial": trial, "unit": unit, "seq_len": seq_len, "t_base_s": round(t_base, 4),
                         "budget": budget, "per_tok_ms": round(per_tok * 1e3, 3) if per_tok else None,
                         "is_listen": r.get("is_listen"), "n_tts": r.get("n_tts_tokens")})
            model.duplex_finalize()
            unit += 1
            if seq_len >= TARGET_SEQ: break
            if unit % 50 == 0: log(f"  t{trial} unit {unit} seq {seq_len} budget {budget} t_base {t_base:.3f} per_tok {per_tok*1e3:.1f}ms")
        log(f"=== trial {trial} done: {len(recs)} units, final_seq {recs[-1]['seq_len']} ===")
        all_records += recs
        if rank == 0:
            fn = os.path.join(OUT_DIR, f"grid_{MODE}_{FORCE}_{PATH}_t{trial}.json")
            json.dump({"mode": MODE, "force": FORCE, "path": PATH, "trial": trial,
                       "n_units": len(recs), "final_seq": recs[-1]["seq_len"], "records": recs}, open(fn, "w"))
        # reset session for next trial
        if PATH == "graph" and dec._llm_runner is not None:
            dec._llm_runner.reset(); dec._static_pos = 0
        dec.reset() if hasattr(dec, "reset") else None
    log("DONE ALL TRIALS")
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
