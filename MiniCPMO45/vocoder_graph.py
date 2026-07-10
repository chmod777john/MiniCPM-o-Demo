"""Enable the dormant CosyVoice2 DiT CUDA-graph in the vocoder, VRAM-safely.

The stepaudio2 lib ships a hand-rolled CUDA-graph path for the diffusion DiT but
only captures chunk_size in [30,48,96] and is never turned on. The unified 25-token
path uses chunk_size = frames (~50), which isn't captured. This module:
  - spies on the real runtime chunk sizes,
  - captures graphs for ONLY those sizes (each ~2GB VRAM),
  - enables use_cuda_graph,
with try/except so any failure falls back to eager (correct, just no speedup).
Replay is bit-identical (same FP32 kernels) => output-neutral.
"""
import torch


def get_dit(model):
    # flow.decoder = CausalConditionalCFM (flow_matching), .estimator = DiT (decoder_dit)
    return model.tts.audio_tokenizer.flow.decoder.estimator


def install_spy(model, seen):
    dit = get_dit(model)
    orig = dit.forward_chunk
    def spy(x, *a, **k):
        try:
            seen.add(int(x.shape[2]))  # chunk_size = frames = x.shape[2] at entry
        except Exception:
            pass
        return orig(x, *a, **k)
    dit.forward_chunk = spy
    return orig


def remove_spy(model):
    dit = get_dit(model)
    if "forward_chunk" in dit.__dict__:
        del dit.__dict__["forward_chunk"]  # fall back to the class method (with graph logic)


def _capture_one(dit, chunk_size, log, max_size=None):
    dtype, device = dit.cnn_cache_buffer.dtype, dit.cnn_cache_buffer.device
    if max_size is None:
        max_size = 1024 if chunk_size <= 60 else 1536
    for attr in ("graph_chunk", "inference_buffers_chunk", "max_size_chunk"):
        if not hasattr(dit, attr) or getattr(dit, attr) is None:
            setattr(dit, attr, {})
    dit.max_size_chunk[chunk_size] = max_size
    # capture under inference_mode so the static-buffer in-place updates don't hit the
    # "inplace update to inference tensor outside InferenceMode" error when co-existing
    # with the TTS graph (whose forwards run under torch.inference_mode()).
    with torch.inference_mode():
        static_x1 = torch.zeros((2, 320, chunk_size), dtype=dtype, device=device)
        static_t1 = torch.zeros((2, 1, 512), dtype=dtype, device=device)
        static_mask1 = torch.ones((2, chunk_size, max_size + chunk_size), dtype=torch.bool, device=device)
        static_att_cache = torch.zeros((16, 2, 8, max_size, 128), dtype=dtype, device=device)
        static_cnn_cache = torch.zeros((16, 2, 1024, 2), dtype=dtype, device=device)
        static_new_cnn_cache = torch.zeros((16, 2, 1024, 2), dtype=dtype, device=device)
        static_new_att_cache = torch.zeros((16, 2, 8, max_size + chunk_size, 128), dtype=dtype, device=device)
        # warmup (outside graph) then capture
        dit.blocks_forward_chunk(static_x1, static_t1, static_mask1, static_cnn_cache,
                                 static_att_cache, static_new_cnn_cache, static_new_att_cache)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out1 = dit.blocks_forward_chunk(static_x1, static_t1, static_mask1, static_cnn_cache,
                                                   static_att_cache, static_new_cnn_cache, static_new_att_cache)
    dit.inference_buffers_chunk[chunk_size] = {
        "static_inputs": [static_x1, static_t1, static_mask1, static_cnn_cache, static_att_cache],
        "static_outputs": [static_out1, static_new_cnn_cache, static_new_att_cache],
    }
    dit.graph_chunk[chunk_size] = g
    log(f"[vocoder] captured graph chunk_size={chunk_size} max_size={max_size} "
        f"free_gb={torch.cuda.mem_get_info()[0]/1e9:.1f}")


def enable(model, sizes, log):
    dit = get_dit(model)
    ok = []
    for cs in sorted(sizes):
        try:
            _capture_one(dit, cs, log)
            ok.append(cs)
        except Exception as e:
            log(f"[vocoder] capture chunk_size={cs} FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
    if ok:
        dit.use_cuda_graph = True
        log(f"[vocoder] cuda graph ENABLED for {ok}")
        return True
    dit.use_cuda_graph = False
    log("[vocoder] no graphs captured; staying eager")
    return False


def install_diag(model, log):
    """Diagnostic: wrap forward_chunk to log every call that would MISS the CUDA-graph fast path
    (uncaptured chunk_size OR att context too long for the captured max_size), so we know exactly
    why a token2wav unit spikes to eager. Records (chunk_size, last_att_len, reason)."""
    dit = get_dit(model)
    orig = dit.forward_chunk
    stats = {"calls": 0, "graph": 0, "miss": []}
    import torch as _t
    def diag(x, *a, **k):
        cs = int(x.shape[2])
        att_cache = k.get("att_cache", a[5] if len(a) > 5 else (a[-1] if a else None))
        last_att_len = 0
        try:
            if att_cache is not None and att_cache[0] is not None:
                last_att_len = int(att_cache.shape[3])
        except Exception:
            last_att_len = -1
        stats["calls"] += 1
        reason = None
        if not getattr(dit, "use_cuda_graph", False):
            reason = "use_cuda_graph=False"
        elif last_att_len == 0:
            reason = "first_chunk(att_cache None)"
        elif cs not in getattr(dit, "graph_chunk", {}):
            _cap = sorted(getattr(dit, "graph_chunk", {}).keys())
            reason = "uncaptured_size cs=%d captured=%s" % (cs, _cap)
        elif last_att_len > dit.max_size_chunk.get(cs, 0):
            reason = f"att_too_long cs={cs} last_att_len={last_att_len} max_size={dit.max_size_chunk.get(cs)}"
        else:
            stats["graph"] += 1
        if reason is not None:
            stats["miss"].append({"cs": cs, "last_att_len": last_att_len, "reason": reason})
        return orig(x, *a, **k)
    dit.forward_chunk = diag
    dit._diag_stats = stats
    log("[vocoder_diag] installed")
    return stats


def install_counter(model, log):
    """Cheap size-frequency probe: wrap forward_chunk to COUNT chunk_size occurrences (no capture,
    no graph). Tells us the real flush-size distribution so we can pick a bounded pre-capture set."""
    dit = get_dit(model)
    orig = dit.forward_chunk
    from collections import Counter as _C
    freq = _C()
    freq_att = _C()  # sizes only for calls WITH att_cache (the graphable ones)
    def counter(x, *a, **k):
        cs = int(x.shape[2])
        freq[cs] += 1
        att_cache = k.get("att_cache", a[-1] if a else None)
        if att_cache is not None and hasattr(att_cache, "shape"):
            freq_att[cs] += 1
        return orig(x, *a, **k)
    dit.forward_chunk = counter
    dit._size_freq = freq
    dit._size_freq_att = freq_att
    log("[vocoder_counter] installed")
    return freq, freq_att


def enable_lazy(model, log, max_size_small=1024, max_size_large=1536, warm_sizes=(30, 50, 96, 302)):
    """Robust vocoder graphing: PRE-CAPTURE the known chunk sizes at enable time (off the measured
    path, during warmup) so steady-state has no capture-cost spikes, PLUS wrap forward_chunk so any
    NOVEL chunk_size is still captured on first sight (one-time small hitch, then fast). Root cause of
    the >1s omni tail was cs=30, which the fixed spy never discovered so it always fell to eager
    (token2wav ~530ms). Deployable straight from model init (no discovery pass). First chunk of a turn
    (att_cache None) stays eager (cheap, small). Numerically identical replay (same FP32 DiT kernels)."""
    dit = get_dit(model)
    for attr in ("graph_chunk", "inference_buffers_chunk", "max_size_chunk"):
        if not hasattr(dit, attr) or getattr(dit, attr) is None:
            setattr(dit, attr, {})
    # pre-capture the known sizes now (during warmup), so they never capture inside a measured turn
    for cs in warm_sizes:
        if cs in dit.graph_chunk:
            continue
        try:
            _capture_one(dit, cs, log, max_size=(max_size_small if cs <= 60 else max_size_large))
        except Exception as e:
            log("[vocoder] pre-capture cs=%d FAILED: %s: %s" % (cs, type(e).__name__, e))
            torch.cuda.empty_cache()
    orig = type(dit).forward_chunk  # class method (has the graph-dispatch logic)

    def lazy_forward_chunk(x, mu, t, spks, cond, cnn_cache=None, att_cache=None):
        try:
            cs = int(x.shape[2])
            has_att = att_cache is not None and hasattr(att_cache, "shape")
            if has_att:
                last_att_len = int(att_cache.shape[3])
                ms = max_size_small if cs <= 60 else max_size_large
                need = (cs not in dit.graph_chunk) or (last_att_len > dit.max_size_chunk.get(cs, 0))
                if need and last_att_len <= ms:
                    _capture_one(dit, cs, log, max_size=ms)
        except Exception as e:
            log("[vocoder] lazy capture failed (staying eager for this call): %s: %s" % (type(e).__name__, e))
            torch.cuda.empty_cache()
        return orig(dit, x, mu, t, spks, cond, cnn_cache, att_cache)

    dit.forward_chunk = lazy_forward_chunk
    dit.use_cuda_graph = True
    log("[vocoder] lazy capture-on-demand ENABLED (pre-warmed %s + robust to novel sizes)" % (sorted(dit.graph_chunk.keys()),))
    return True


def _bucketed_replay(dit, x, mu, t, spks, cond, cnn_cache, att_cache, bucket):
    """Pad a flush chunk (cs<bucket) to `bucket`, custom-mask so real frames attend only
    [real chunk frames + cache], replay the bucket graph, slice output+att_cache back to cs. Handles
    att_cache=None (first chunk of an utterance -> last_att_len=0). Returns (out, new_cnn, new_att) or
    None if not applicable (caller falls back to eager)."""
    import torch as _t
    import torch.nn.functional as _F
    from einops import pack as _pack, repeat as _repeat
    cs = int(x.shape[-1])            # x/mu/cond are (b, c, dt); time = LAST dim (flow.py transposes)
    if cs in dit.graph_chunk or cs >= bucket:
        return None
    if att_cache is not None and hasattr(att_cache, "shape"):
        last_att_len = int(att_cache.shape[3]); adtype = att_cache.dtype
    else:
        last_att_len = 0; adtype = dit.cnn_cache_buffer.dtype   # first chunk: no cache
    if last_att_len > dit.max_size_chunk.get(bucket, 0):
        return None
    B = bucket
    xp = _F.pad(x, (0, B - cs)); mup = _F.pad(mu, (0, B - cs))
    condp = _F.pad(cond, (0, B - cs)) if cond is not None else None
    te = dit.t_embedder(t).unsqueeze(1)
    xc = _pack([xp, mup], "b * t")[0]
    if spks is not None:
        spks_r = _repeat(spks, "b c -> b c t", t=xc.shape[-1]); xc = _pack([xc, spks_r], "b * t")[0]
    if condp is not None:
        xc = _pack([xc, condp], "b * t")[0]
    maxs = dit.max_size_chunk[B]
    # KEY ORDER inside the graph is [chunk(B) ++ cache(maxs)] (attn does k=cat([chunk_k, k_cache])).
    # Real query attends real chunk keys [0:cs] + cache keys [B:B+last_att_len]; NEVER padding [cs:B].
    pmask = _t.zeros((2, B, maxs + B), dtype=_t.bool, device=xc.device)
    pmask[:, :, :cs] = True                        # real chunk keys
    if last_att_len > 0:
        pmask[:, :, B:B + last_att_len] = True      # cache keys
    patt = _t.zeros((16, 2, 8, maxs, 128), dtype=adtype, device=xc.device)
    if last_att_len > 0:
        patt[:, :, :, :last_att_len, :] = att_cache
    bufs = dit.inference_buffers_chunk[B]
    bufs["static_inputs"][0].copy_(xc); bufs["static_inputs"][1].copy_(te)
    bufs["static_inputs"][2].copy_(pmask); bufs["static_inputs"][3].copy_(cnn_cache)
    bufs["static_inputs"][4].copy_(patt)
    dit.graph_chunk[B].replay()
    out = bufs["static_outputs"][0][:, :, :cs]
    new_cnn = bufs["static_outputs"][1]
    _oa = bufs["static_outputs"][2]                 # new att_cache layout also [chunk(B) ++ cache]
    if last_att_len > 0:
        new_att = _t.cat([_oa[:, :, :, :cs, :], _oa[:, :, :, B:B + last_att_len, :]], dim=3)
    else:
        new_att = _oa[:, :, :, :cs, :]
    return out, new_cnn, new_att


def enable_bucketed(model, log, bucket=50, max_size=1024, warm_sizes=(50, 302)):
    """Graph the SHORT flush chunks (cs<bucket) by padding to `bucket` (float-reorder ~0.003 mel, gate-
    verified) AND pre-capture the stable exact sizes in `warm_sizes` as plain graphs (bit-identical, e.g.
    the dominant cs=50 and the large first/prompt cs=302). cs in warm_sizes -> exact graph (0.0); cs<bucket
    -> bucketed; other cs -> eager. Ship only after install_bucket_diff confirms equivalence."""
    dit = get_dit(model)
    for attr in ("graph_chunk", "inference_buffers_chunk", "max_size_chunk"):
        if not hasattr(dit, attr) or getattr(dit, attr) is None:
            setattr(dit, attr, {})
    for cs in warm_sizes:
        if cs not in dit.graph_chunk:
            try:
                _capture_one(dit, cs, log, max_size=(max_size if cs <= 60 else 1024))
            except Exception as e:
                log("[vocoder] warm capture cs=%d FAILED: %s" % (cs, e)); import torch as _t; _t.cuda.empty_cache()
    if bucket not in dit.graph_chunk:
        _capture_one(dit, bucket, log, max_size=max_size)
    orig = type(dit).forward_chunk
    from collections import Counter as _Counter
    dit._eager_sizes = _Counter()
    def bucketed_forward_chunk(x, mu, t, spks, cond, cnn_cache=None, att_cache=None):
        if getattr(dit, "use_cuda_graph", False):
            try:
                r = _bucketed_replay(dit, x, mu, t, spks, cond, cnn_cache, att_cache, bucket)
                if r is not None:
                    return r
            except Exception as e:
                log("[vocoder] bucketed fallback to eager: %s: %s" % (type(e).__name__, e))
        cs = int(x.shape[-1])
        if cs not in dit.graph_chunk:
            dit._eager_sizes[cs] += 1        # residual: not a captured size, not buckettable
        return orig(dit, x, mu, t, spks, cond, cnn_cache, att_cache)
    dit.forward_chunk = bucketed_forward_chunk
    dit.use_cuda_graph = True
    log("[vocoder] BUCKETED forward_chunk ENABLED (exact graphs %s + flush cs<%d padded to %d)"
        % (sorted(dit.graph_chunk.keys()), bucket, bucket))
    return True


def install_bucket_diff(model, log, bucket=50, max_size=1024):
    """AUDIO-EQUIVALENCE GATE: for every FLUSH chunk in a real run, compute BOTH eager and bucketed mel
    on the SAME inputs and record max|Δ| (uses eager for the actual run so it proceeds normally).
    Bit-identical (~0) across all flush chunks => pad+mask preserves real-frame output => safe to ship."""
    dit = get_dit(model)
    for attr in ("graph_chunk", "inference_buffers_chunk", "max_size_chunk"):
        if not hasattr(dit, attr) or getattr(dit, attr) is None:
            setattr(dit, attr, {})
    if bucket not in dit.graph_chunk:
        _capture_one(dit, bucket, log, max_size=max_size)
    dit.use_cuda_graph = True
    orig = type(dit).forward_chunk
    diffs = []
    from collections import Counter as _Counter
    eager_sizes = _Counter()
    def diff_forward_chunk(x, mu, t, spks, cond, cnn_cache=None, att_cache=None):
        eager = orig(dit, x, mu, t, spks, cond, cnn_cache, att_cache)   # real run uses eager (unaffected)
        cs = int(x.shape[-1])
        has_att = att_cache is not None and hasattr(att_cache, "shape")
        if cs not in dit.graph_chunk and cs < bucket:
            try:
                r = _bucketed_replay(dit, x, mu, t, spks, cond, cnn_cache, att_cache, bucket)
                if r is not None:
                    d = float((eager[0] - r[0]).abs().max())
                    da = float((eager[2] - r[2]).abs().max()) if eager[2].shape == r[2].shape else -1.0
                    diffs.append({"cs": cs, "first_chunk": (not has_att), "last_att_len": (int(att_cache.shape[3]) if has_att else 0),
                                  "mel_max_abs_diff": round(d, 6), "att_cache_max_abs_diff": round(da, 6)})
            except Exception as e:
                diffs.append({"cs": cs, "error": "%s: %s" % (type(e).__name__, e)})
        elif cs not in dit.graph_chunk:            # cs >= bucket -> still eager; record for diagnosis
            eager_sizes[cs] += 1
        return eager
    dit.forward_chunk = diff_forward_chunk
    dit._bucket_diffs = diffs
    dit._eager_sizes = eager_sizes
    log("[vocoder] bucket DIFF gate installed")
    return diffs
