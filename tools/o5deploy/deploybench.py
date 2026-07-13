#!/usr/bin/env python3
"""Framework-driven multi-mode benchmark: builds the model THROUGH core.deploy (the pluggable
deployment-mode framework) and runs the real duplex pipeline with trustworthy timing (synced,
finalize included, warmup discarded, p50/p95/p99/over-1000ms, cross-rank max for SPMD modes).
Proves the framework can construct AND serve any registered mode.
Env: MODE(single_eager/single_opt/tp2), FORCE(listen/speak), UNITS, WARMUP, OUT_DIR."""
import os, sys, json, time
import numpy as np, torch

WORKTREE = os.environ["WORKTREE"]; sys.path.insert(0, WORKTREE); sys.path.insert(0, os.path.join(WORKTREE, "scripts"))
os.environ.setdefault("ATTN_IMPLEMENTATION", "sdpa")
MODE = os.environ.get("MODE", "single_opt"); FORCE = os.environ.get("FORCE", "speak")
UNITS = int(os.environ.get("UNITS", "80")); WARMUP = int(os.environ.get("WARMUP", "8"))
OUT_DIR = os.environ["OUT_DIR"]; os.makedirs(OUT_DIR, exist_ok=True)
VIDEO = os.path.join(WORKTREE, "assets", "samples", "compile.mp4")
REF_WAV = os.path.join(WORKTREE, "assets/ref_audio/ref_minicpm_signature.wav")
GEN = dict(temperature=0.7, top_k=20, top_p=0.8, listen_prob_scale=1.0,
           text_repetition_penalty=1.05, text_repetition_window_size=512, length_penalty=1.1)

def pct(a, q): return float(np.percentile(np.array(a), q)) if a else 0.0

def main():
    import core.deploy as deploy
    mode = deploy.get_mode(MODE)
    cfg = {"model_path": os.environ["MODEL_PATH"], "pt_path": os.environ["PT_PATH"],
           "backbone_dir": os.environ.get("BACKBONE_DIR", "/user/weihongliang/wangkaiqi/o5_backbone_hf"),
           "chat_vocoder": "token2wav", "attn_implementation": "sdpa",
           "llm_cache_len": int(os.environ.get("O5_LLM_CACHE", "8192")), "gpu_id": 0}
    torch.manual_seed(1234); np.random.seed(1234)
    br = mode.build(cfg)                              # <-- build THROUGH the framework
    model = br.model; dec = model.duplex.decoder
    is_driver = br.is_driver; world = br.world_size
    def log(*a):
        if is_driver: print(*a, flush=True)
    log(f"=== framework build OK: mode={MODE} world={world} rank={br.rank} engine={json.dumps(br.engine)} ===")

    if FORCE == "listen":
        model.duplex.force_listen_count = 10**9
    elif FORCE == "speak":
        for _lid in (getattr(model.duplex, "listen_token_id", None), getattr(dec, "listen_id", None)):
            if _lid is not None and _lid not in dec.forbidden_token_ids: dec.forbidden_token_ids.append(_lid)
    ref = __import__("minimal_o5_unified_model_duplex").load_16k(REF_WAV)

    import subprocess, tempfile, librosa
    from PIL import Image
    tmp = tempfile.mkdtemp(); wav = os.path.join(tmp, "a.wav")
    subprocess.run(["ffmpeg","-i",VIDEO,"-ar","16000","-ac","1","-t","20","-f","wav","-y",wav], capture_output=True, check=True)
    audio,_ = librosa.load(wav, sr=16000, mono=True)
    is_omni = os.environ.get("OMNI", "1") == "1"
    ac = [audio[i*16000:(i+1)*16000].astype(np.float32) for i in range(min(20, len(audio)//16000))]
    fr = None
    if is_omni:
        fdir = os.path.join(tmp,"f"); os.makedirs(fdir, exist_ok=True)
        subprocess.run(["ffmpeg","-i",VIDEO,"-vf","fps=1","-t","20",os.path.join(fdir,"f%03d.jpg"),"-y"], capture_output=True, check=True)
        fr = [Image.open(os.path.join(fdir,f)).convert("RGB") for f in sorted(os.listdir(fdir))]
    na=len(ac); nf=len(fr) if fr else 0
    def prep():
        model.duplex_prepare(prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
                             suffix_system_prompt="<|audio_end|><|im_end|>", ref_audio=ref, prompt_wav_path=REF_WAV)
    def barrier():
        if world > 1:
            import torch.distributed as dist; dist.barrier()
    def smax(x):
        if world == 1: return x
        import torch.distributed as dist
        t=torch.tensor([x],device=f"cuda:{br.rank}",dtype=torch.float64); dist.all_reduce(t,op=dist.ReduceOp.MAX); return float(t.item())

    log(f"=== warmup {WARMUP} (discarded) ===")
    prep()
    for k in range(WARMUP):
        fl=[fr[k%nf]] if is_omni else None
        # exercise SPMD input broadcast on the driver path
        idx = k
        if br.broadcast_input: idx = br.broadcast_input(idx)
        model.duplex_prefill(audio_waveform=ac[idx%na], frame_list=fl)
        model.duplex_generate(decode_mode="sampling", **GEN); model.duplex_finalize()
    deploy.modes._enable_vocoder_bucket(model)

    log(f"=== timed {UNITS} units ===")
    prep(); recs=[]
    for u in range(UNITS):
        fl=[fr[u%nf]] if is_omni else None
        idx=u
        if br.broadcast_input: idx=br.broadcast_input(idx)
        barrier()
        torch.cuda.synchronize(); t0=time.perf_counter()
        model.duplex_prefill(audio_waveform=ac[idx%na], frame_list=fl)
        torch.cuda.synchronize(); t1=time.perf_counter()
        r=model.duplex_generate(decode_mode="sampling", **GEN)
        torch.cuda.synchronize(); t2=time.perf_counter()
        model.duplex_finalize()
        torch.cuda.synchronize(); t3=time.perf_counter()
        unit=(t3-t0)*1e3; um=smax(unit)
        recs.append({"unit":u,"is_listen":r.get("is_listen"),"n_tts":r.get("n_tts_tokens"),
                     "prefill_ms":round((t1-t0)*1e3,2),"generate_ms":round((t2-t1)*1e3,2),
                     "finalize_ms":round((t3-t2)*1e3,2),"unit_total_max_ms":round(um,2)})
        if u%20==0: log(f"  unit {u}: total={unit:.0f}ms(max {um:.0f}) pf={(t1-t0)*1e3:.0f} gen={(t2-t1)*1e3:.0f} fin={(t3-t2)*1e3:.0f} listen={r.get('is_listen')}")
    if is_driver:
        def st(rs):
            v=[x["unit_total_max_ms"] for x in rs]
            if not v: return None
            over=sum(1 for x in v if x>1000)
            return {"n":len(v),"avg":round(sum(v)/len(v),1),"p50":round(pct(v,50),1),"p95":round(pct(v,95),1),
                    "p99":round(pct(v,99),1),"max":round(max(v),1),"over_1000ms":over}
        sp=[x for x in recs if not x["is_listen"]]; li=[x for x in recs if x["is_listen"]]
        summ={"mode":MODE,"force":FORCE,"omni":is_omni,"world":world,"engine":br.engine,
              "ALL":st(recs),"LISTEN":st(li),"SPEAK":st(sp)}
        fn=os.path.join(OUT_DIR,f"deploybench_{MODE}_{FORCE}.json")
        json.dump({"summary":summ,"records":recs},open(fn,"w"),indent=1)
        log("SUMMARY "+json.dumps(summ,ensure_ascii=False)); log("saved "+fn)
    sys.stdout.flush()
    if world>1: os._exit(0)

if __name__=="__main__":
    main()
