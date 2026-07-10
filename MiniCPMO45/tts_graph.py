"""TTS decode CUDA-graph: StaticCache + EAGER attn + EXPLICIT static 4D mask (bypasses
create_causal_mask's dynamic build, which blocks capture). Numerically exact replay.
t==0 prefill eager (2D mask); t>0 replays captured single-token decode with a static
additive 4D mask buffer (valid=0, masked=-inf), flipping position `pos` valid before replay.
"""
import torch
from transformers.cache_utils import StaticCache


class TTSGraphRunner:
    def __init__(self, llama, max_cache_len=8192):
        self.llama = llama
        self.cfg = llama.config
        self.max_cache_len = max_cache_len
        self.device = next(llama.parameters()).device
        self.dtype = next(llama.parameters()).dtype
        self.neg = torch.finfo(self.dtype).min
        n = 0
        for mod in llama.modules():
            c = getattr(mod, "config", None)
            if c is not None and hasattr(c, "_attn_implementation"):
                c._attn_implementation = "eager"; n += 1
        try: self.cfg._attn_implementation = "eager"
        except Exception: pass
        print(f"[tts_graph] forced eager attention on {n} configs", flush=True)
        self.cache = None; self.graph = None
        self._emb = self._pos = self._cpos = self._mask = self._hidden = None
        self._captured = False; self._failed = False
        # capture the decode graph NOW (before any real prefill/decode), with dummy KV.
        # Doing it lazily on the first decode would let the post-capture cache.reset()
        # wipe the real prefill KV -> corrupt output.
        self._init(self.cfg.hidden_size)
        self._capture()

    def _init(self, H):
        # normal tensors (force inference_mode OFF) so the buffers survive being touched under
        # both @inference_mode (streaming TTS decode) and @no_grad contexts — same fix as llm_graph.
        with torch.inference_mode():
            self.cache = StaticCache(config=self.cfg, max_cache_len=self.max_cache_len)
            self._emb = torch.zeros(1, 1, H, device=self.device, dtype=self.dtype)
            self._pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            self._cpos = torch.zeros(1, dtype=torch.long, device=self.device)
            self._mask = torch.full((1, 1, 1, self.max_cache_len), self.neg, dtype=self.dtype, device=self.device)

    def reset(self):
        with torch.inference_mode():
            if self.cache is not None:
                self.cache.reset()
            if self._mask is not None:
                self._mask.fill_(self.neg)

    def _fwd(self):
        return self.llama(inputs_embeds=self._emb, position_ids=self._pos, cache_position=self._cpos,
                          attention_mask=self._mask, past_key_values=self.cache, use_cache=True).last_hidden_state

    def _capture(self):
        try:
          with torch.inference_mode():
            self._pos.fill_(1); self._cpos.fill_(1)
            self._mask.fill_(self.neg); self._mask[..., :2] = 0.0
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    _ = self._fwd()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._hidden = self._fwd()
            self.graph = g
            self.cache.reset(); self._mask.fill_(self.neg)
            self._captured = True
            print("[tts_graph] CUDA graph captured OK", flush=True)
        except Exception as e:
            print(f"[tts_graph] capture FAILED, eager fallback: {type(e).__name__}: {str(e)[:200]}", flush=True)
            self._failed = True
            try: self.cache.reset(); self._mask.fill_(self.neg)
            except Exception: pass

    def _warn_overflow(self, pos):
        if not getattr(self, "_overflowed", False):
            self._overflowed = True
            print(f"[tts_graph] WARNING: TTS position {pos} >= max_cache_len {self.max_cache_len}; "
                  f"clamping (continuous-speech turn too long). Audio may degrade at the tail; no crash.",
                  flush=True)

    def prefill(self, inputs_embeds, cache_position):
        if self.cache is None:
            self._init(inputs_embeds.shape[-1])
        with torch.inference_mode():
            mv = int(cache_position[-1]) + 1
            if mv > self.max_cache_len:
                self._warn_overflow(mv); mv = self.max_cache_len
            mask2d = torch.zeros(1, self.max_cache_len, dtype=torch.long, device=self.device)
            mask2d[:, :mv] = 1
            h = self.llama(inputs_embeds=inputs_embeds, position_ids=cache_position.unsqueeze(0),
                           cache_position=cache_position, attention_mask=mask2d,
                           past_key_values=self.cache, use_cache=True).last_hidden_state
            self._mask[..., :mv] = 0.0   # mark prefilled positions valid for subsequent decode
        return h

    def decode(self, inputs_embeds, pos):
        if self.cache is None:
            self._init(inputs_embeds.shape[-1])
        with torch.inference_mode():
            if pos >= self.max_cache_len:
                self._warn_overflow(pos); pos = self.max_cache_len - 1
            self._emb.copy_(inputs_embeds); self._pos.fill_(pos); self._cpos.fill_(pos)
            self._mask[..., pos] = 0.0   # this token's position becomes valid; [0..pos] now 0, rest -inf
            if self._failed or not self._captured:
                return self._fwd()
            self.graph.replay()
            return self._hidden
