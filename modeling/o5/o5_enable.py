"""One-call enable for the O5 inference optimizations. Default is OFF everywhere; the demo turns
this on via config `model.optimize=true` or env `O5_OPTIMIZE=1` (see core/processors/unified.py and
INTEGRATION.md). Idempotent-ish and safe: any sub-step that fails logs a warning and falls back to
eager (correct, just slower). Call ONCE, right after `model.init_unified(...)`, and do NOT combine
with torch.compile.

Levers enabled (all numerically equivalent = argmax-preserving float reorder / bit-identical; see
INTEGRATION.md for the equivalence evidence):
  - MoE  eager -> batched_mm   (config._experts_implementation, per-forward dispatch)
  - TTS backbone attn -> eager (required so the TTS CUDA graph can capture)
  - OPT flags: tts_fast, lmhead, tts_graph, fuse_vision_audio, llm_graph  (opt_flags.py)
  - batch_vision_feed via env O5_VISION_BATCH=1
  - vocoder DiT CUDA graph for the dominant chunk size (50)
The TTS/LLM CUDA graphs capture lazily on the FIRST request (~1-2s one-time); send a warmup request
at startup if you want the first real request to be fast.
"""
import os


def enable_o5_optimizations(model, logger=None, vocoder_chunk_sizes=(50,)):
    """Enable the O5 inference optimizations on an already-init_unified'd model. Returns a dict of
    what was turned on. Safe to call once after init_unified; do not combine with torch.compile."""
    def _log(msg):
        if logger is not None:
            logger.info("[o5opt] %s", msg)
        else:
            print("[o5opt] " + msg, flush=True)

    from .opt_flags import OPT
    from . import vocoder_graph as _vg

    # 1) MoE: eager -> batched_mm (zero host-syncs; no H100 needed). per-forward dispatch.
    seen = set(); n_moe = 0
    for m in model.modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "_experts_implementation") and id(c) not in seen:
            c._experts_implementation = "batched_mm"; seen.add(id(c)); n_moe += 1

    # 2) TTS backbone -> eager attn (needed for the TTS CUDA-graph capture)
    n_tts = 0
    try:
        for m in model.tts.model.modules():
            c = getattr(m, "config", None)
            if c is not None and hasattr(c, "_attn_implementation"):
                c._attn_implementation = "eager"; n_tts += 1
    except Exception as e:
        _log("could not force TTS eager attn (continuing): %s" % e)

    # 3) runtime flags (opt_flags.OPT is read per-forward inside the model)
    OPT.update({"tts_fast": True, "lmhead": True, "tts_graph": True,
                "fuse_vision_audio": True, "llm_graph": True})
    os.environ["O5_VISION_BATCH"] = "1"   # batch the vision token feeds into the fused forward

    # 4) vocoder DiT CUDA graph: exact graphs for the stable sizes {50,302} (bit-identical) PLUS
    #    pad-to-50 bucketing for the short flush chunks cs<50 (gate-verified float-reorder ~0.003 mel).
    #    Eliminates the token2wav eager spikes that otherwise cost 200-500ms on utterance-boundary units.
    voc = False
    try:
        voc = _vg.enable_bucketed(model, lambda *a: _log(" ".join(str(x) for x in a)), bucket=50)
    except Exception as e:
        _log("vocoder bucketed graph enable failed (eager fallback): %s" % e)

    _log("ENABLED: batched_mm(%d configs) + tts_eager(%d) + tts_fast/lmhead/tts_graph/fuse/batch/"
         "llm_graph + vocoder_graph=%s. TTS/LLM graphs capture on first request." % (n_moe, n_tts, voc))
    return {"moe_batched_mm": n_moe, "tts_eager": n_tts, "vocoder_graph": voc,
            "flags": {k: OPT[k] for k in ("tts_fast", "lmhead", "tts_graph", "fuse_vision_audio", "llm_graph")}}
