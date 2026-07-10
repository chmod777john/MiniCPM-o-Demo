"""LLM decode CUDA-graph for the hybrid Qwen3.5-MoE backbone (30 linear-attn + 10 full-attn).

Graphs the single-token decode step (StreamDecoder.feed with L==1), which is the launch-bound
hot path: a speak unit runs it once per generated text token (~5-15x), each ~100ms host-bound.
Same technique as TTSGraphRunner, adapted for the HYBRID cache and a PERSISTENT conversation cache:

  - StaticCache(config) is hybrid-aware: LinearAttentionLayer (conv+recurrent, .copy_ on static
    addresses) for the 30 linear layers, StaticLayer (index_copy_ at self.cumulative_length, which
    auto-increments in-place) for the 10 full-attn layers. Both are CUDA-graph-safe by design.
  - Force EAGER on the full-attn configs + pass an explicit static 4D additive mask, to bypass
    create_causal_mask's dynamic build (which blocks capture) — exactly like TTS.
  - Capture the decode graph at init against the SAME StaticCache buffers that will hold the real
    conversation KV (the graph replays fixed buffer addresses), then reset the cache so the real
    prefill starts clean. Prefill (variable L>1) runs the NON-graph StaticCache path; decode (L==1)
    replays the graph.
  - lm_head is applied OUTSIDE the graph on the last hidden state (cheap, keeps the graph to the
    backbone; full-vocab head each replay would bloat capture).

Numerically: same kernels as the llm_static StaticCache path (gate-verified argmax-identical on real
omni context), just replayed. Position/mask are updated in-place before each replay (graph reads
current tensor values).
"""
import torch
from transformers.cache_utils import StaticCache


class LLMGraphRunner:
    def __init__(self, model, lm_head, max_cache_len=8192):
        self.model = model            # hybrid backbone (self.m.model)
        self.lm_head = lm_head        # self.m.lm_head
        self.cfg = model.config
        self.max_cache_len = max_cache_len
        p = next(model.parameters())
        self.device = p.device
        self.dtype = p.dtype
        self.neg = torch.finfo(self.dtype).min
        # force eager on every attention config (linear-attn/GatedDeltaNet ignores this)
        n = 0
        for mod in model.modules():
            c = getattr(mod, "config", None)
            if c is not None and hasattr(c, "_attn_implementation"):
                c._attn_implementation = "eager"; n += 1
        try: self.cfg._attn_implementation = "eager"
        except Exception: pass
        print(f"[llm_graph] forced eager attention on {n} configs", flush=True)
        self.cache = None
        self.graph = None
        self._emb = self._pos = self._cpos = self._mask = self._hidden = None
        self._captured = False
        self._failed = False
        H = self.cfg.hidden_size
        self._init(H)
        self._capture(H)

    def _init(self, H):
        # Force inference_mode(True) for ALL buffer ops in this runner (init/capture/decode/prefill),
        # mirroring the working vocoder graph. The duplex pipeline touches these StaticCache buffers
        # under BOTH @inference_mode (streaming feed) AND @no_grad (finalize_unit feeds </unit>). A
        # single consistent inference_mode context makes the buffers inference tensors mutated ONLY
        # inside inference mode, regardless of the caller's context — otherwise you hit "Inplace update
        # to inference tensor outside InferenceMode" (finalize's no_grad feed touching inference-tensor
        # buffers) OR, with no_grad buffers, "outside InferenceMode" when a pre-existing inference tensor
        # (created during the demo's inference-mode init) is touched under our no_grad.
        with torch.inference_mode():
            self.cache = StaticCache(config=self.cfg, max_cache_len=self.max_cache_len)
            self._emb = torch.zeros(1, 1, H, device=self.device, dtype=self.dtype)
            self._pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            self._cpos = torch.zeros(1, dtype=torch.long, device=self.device)
            self._mask = torch.full((1, 1, 1, self.max_cache_len), self.neg, dtype=self.dtype, device=self.device)

    def _fwd(self):
        return self.model(inputs_embeds=self._emb, position_ids=self._pos, cache_position=self._cpos,
                          attention_mask=self._mask, past_key_values=self.cache, use_cache=True,
                          return_dict=True).last_hidden_state

    def _capture(self, H):
        try:
            with torch.inference_mode():
                # prime: two eager fwds so the linear layers have has_previous_state=True and the
                # full-attn cumulative_length>0, i.e. capture the steady-state decode op-graph.
                self._pos.fill_(0); self._cpos.fill_(0)
                self._mask.fill_(self.neg); self._mask[..., 0] = 0.0
                _ = self._fwd()                       # writes pos 0 (cumulative 0->1)
                self._pos.fill_(1); self._cpos.fill_(1); self._mask[..., 1] = 0.0
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _ = self._fwd()               # more warmup on side stream
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._hidden = self._fwd()
                self.graph = g
                self.cache.reset()
                self._mask.fill_(self.neg)
            self._captured = True
            print("[llm_graph] CUDA graph captured OK", flush=True)
        except Exception as e:
            import traceback
            print("[llm_graph] capture FAILED, eager fallback:", type(e).__name__, str(e)[:200], flush=True)
            print("[llm_graph] TRACEBACK:\n" + traceback.format_exc(), flush=True)
            self._failed = True
            try:
                with torch.no_grad():
                    self.cache.reset(); self._mask.fill_(self.neg)
            except Exception:
                pass

    def reset(self):
        # inference_mode: the buffers are inference tensors (created under inference_mode); the demo
        # calls decoder.reset() between videos OUTSIDE inference_mode, and StaticCache.reset() does
        # conv_states.zero_()/keys.zero_() in-place -> must be inside inference_mode.
        with torch.inference_mode():
            if self.cache is not None:
                self.cache.reset()
            if self._mask is not None:
                self._mask.fill_(self.neg)

    def _warn_overflow(self, pos):
        if not getattr(self, "_overflowed", False):
            self._overflowed = True
            print(f"[llm_graph] WARNING: LLM position {pos} >= max_cache_len {self.max_cache_len}; "
                  f"clamping. A >{self.max_cache_len}-token session: reset sessions or raise max_cache_len "
                  f"(the demo sliding_window does NOT trim a StaticCache). No crash; tail may degrade.",
                  flush=True)

    def prefill(self, inputs_embeds, start_pos):
        """Variable-length prefill via the non-graph StaticCache path. inputs_embeds: [1,L,H]."""
        with torch.inference_mode():
            L = inputs_embeds.shape[1]
            if L > self.max_cache_len:              # pathological single prefill > ceiling: keep last window
                self._warn_overflow(L); inputs_embeds = inputs_embeds[:, -self.max_cache_len:, :]
                L = self.max_cache_len; start_pos = 0
            end = start_pos + L
            if end > self.max_cache_len:
                self._warn_overflow(end); end = self.max_cache_len; start_pos = max(0, end - L)
            cpos = torch.arange(start_pos, start_pos + L, device=self.device)
            mask2d = torch.zeros(1, self.max_cache_len, dtype=torch.long, device=self.device)
            mask2d[:, :start_pos + L] = 1
            h = self.model(inputs_embeds=inputs_embeds, position_ids=cpos.unsqueeze(0),
                           cache_position=cpos, attention_mask=mask2d,
                           past_key_values=self.cache, use_cache=True, return_dict=True).last_hidden_state
            # mark all prefilled positions valid for the graph's additive mask
            self._mask[..., :start_pos + L] = 0.0
        return h

    def decode(self, inputs_embeds, pos):
        """Single-token decode at absolute position `pos` (== cache length before this token)."""
        with torch.inference_mode():
            if pos >= self.max_cache_len:
                self._warn_overflow(pos); pos = self.max_cache_len - 1
            self._emb.copy_(inputs_embeds)
            self._pos.fill_(pos); self._cpos.fill_(pos)
            self._mask[..., pos] = 0.0
            if self._failed or not self._captured:
                return self._fwd()
            self.graph.replay()
            return self._hidden
