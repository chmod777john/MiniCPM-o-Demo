#!/usr/bin/env python3
"""Live client for the 2-card SPMD server smoke test. Connects over the real WebSocket backend
protocol (runtime.backend_client.RemoteBackendSession), inits a full_duplex session, pushes N
1-second audio+video chunks, and pulls the server's response deltas. Proves the tp2 server
(rank0 uvicorn HTTP + rank1 worker_loop) serves a real network client end-to-end."""
import asyncio, base64, json, os, sys, time
import numpy as np

WORKTREE = os.environ["WORKTREE"]; sys.path.insert(0, WORKTREE)
PORT = int(os.environ.get("PORT", "22500")); N = int(os.environ.get("N_UNITS", "10"))
REF = os.path.join(WORKTREE, "assets/ref_audio/ref_minicpm_signature.wav")
V = os.path.join(WORKTREE, "assets/samples/compile.mp4")

def prep_media():
    import subprocess, tempfile, librosa
    from PIL import Image
    tmp = tempfile.mkdtemp(); wav = os.path.join(tmp, "a.wav")
    subprocess.run(["ffmpeg","-i",V,"-ar","16000","-ac","1","-t","20","-f","wav","-y",wav], capture_output=True, check=True)
    audio,_ = librosa.load(wav, sr=16000, mono=True)
    ac = [audio[i*16000:(i+1)*16000].astype(np.float32) for i in range(min(20,len(audio)//16000))]
    fdir = os.path.join(tmp,"f"); os.makedirs(fdir, exist_ok=True)
    subprocess.run(["ffmpeg","-i",V,"-vf","fps=1","-t","20",os.path.join(fdir,"f%03d.jpg"),"-y"], capture_output=True, check=True)
    jpgs = [open(os.path.join(fdir,f),"rb").read() for f in sorted(os.listdir(fdir))]
    ref,_ = librosa.load(REF, sr=16000, mono=True)
    return ac, jpgs, ref.astype(np.float32)

async def main():
    from runtime.backend_client import RemoteBackendSession
    ac, jpgs, ref = prep_media()
    ref_b64 = base64.b64encode(ref.tobytes()).decode()
    sess = RemoteBackendSession(base_url=f"http://127.0.0.1:{PORT}", mode="full_duplex")
    print("[client] connecting + init full_duplex ...", flush=True)
    created = await asyncio.wait_for(sess.init({
        "mode": "full_duplex",
        "config": {"generate_audio": True, "ls_mode": "explicit", "max_new_speak_tokens_per_chunk": 20,
                   "temperature": 0.7, "top_k": 20, "top_p": 0.8, "force_listen_count": 3},
        "voice": {"ref_audio_base64": ref_b64},
        "system_prompt": "Streaming Omni Conversation.",
    }), timeout=600)   # init triggers duplex_prepare (mirrored build path) -> allow model warm time
    print(f"[client] session created: {created.get('type')} id={sess.session_id}", flush=True)

    kinds = {}; texts = []; per_unit_ms = []
    for u in range(N):
        aud = base64.b64encode(ac[u % len(ac)].tobytes()).decode()
        frm = base64.b64encode(jpgs[u % len(jpgs)]).decode()
        t0 = time.perf_counter()
        await sess.push({"audio": aud, "frames": [frm], "input_id": f"in{u}"})
        # pull deltas until this unit's response arrives (listen delta, or text delta(s))
        got_unit = False; deadline = time.perf_counter() + 15
        while not got_unit and time.perf_counter() < deadline:
            ev = await asyncio.wait_for(sess.pull(), timeout=15)
            t = ev.get("type", "")
            if t == "response.output.delta":
                k = ev.get("kind", "?"); kinds[k] = kinds.get(k, 0) + 1
                if k == "text" and ev.get("text"): texts.append(ev["text"])
                # a "listen" delta ends the unit; for speak, the delta carrying metrics ends it
                if k == "listen" or ev.get("metrics"):
                    got_unit = True
            elif t in ("response.completed", "response.done", "error", "session.error"):
                got_unit = True
        per_unit_ms.append(round((time.perf_counter()-t0)*1e3, 1))
        if u % 3 == 0: print(f"[client] unit {u}: {per_unit_ms[-1]}ms kinds={kinds}", flush=True)

    await sess.close()
    ok = sum(kinds.values()) > 0 and all(m < 3000 for m in per_unit_ms)  # network RTT included, generous
    summ = {"port": PORT, "units": N, "delta_kinds": kinds, "sample_text": "".join(texts)[:120],
            "per_unit_ms": per_unit_ms, "ok": ok}
    json.dump(summ, open(os.path.join(os.environ["OUT_DIR"], "live_client.json"), "w"), indent=1)
    print("[client] LIVE-CLIENT-RESULT " + json.dumps(summ, ensure_ascii=False), flush=True)
    print("[client] PASS: real network client served by 2-card tp2 server (rank0 HTTP + rank1 worker)." if ok
          else "[client] FAIL: no response deltas / timeout", flush=True)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    asyncio.run(main())
