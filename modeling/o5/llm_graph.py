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
import os

import torch
from transformers.cache_utils import StaticCache

from .cache_limits import CacheLimitExceeded


_NO_DIST_CALL = object()


class LLMGraphRunner:
    def __init__(self, model, lm_head, max_cache_len=8192, distributed=None):
        self.model = model            # hybrid backbone (self.m.model)
        self.lm_head = lm_head        # self.m.lm_head
        self.distributed = distributed
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
        self.trace_layers = os.environ.get("O5_LAYER_TRACE", "0") == "1"
        self._layer_outputs = None
        self._graph_layer_outputs = None
        self._trace_hook_outputs = {}
        self._trace_hook_handles = []
        if self.trace_layers:
            self._install_layer_trace_hooks()
        H = self.cfg.hidden_size
        self._init(H)
        self._capture(H)

    @property
    def _dist_enabled(self):
        return (
            self.distributed is not None
            and bool(getattr(self.distributed, "sync_calls", False))
            and int(getattr(self.distributed, "world_size", 1)) > 1
        )

    @property
    def _dist_driver(self):
        return bool(getattr(self.distributed, "is_driver", True))

    def shutdown_worker(self):
        if self._dist_enabled and self._dist_driver:
            with self.distributed._call_lock:
                self.distributed._broadcast_object(("graph_shutdown", None))

    def noop(self):
        if self._dist_enabled and self._dist_driver:
            with self.distributed._call_lock:
                self.distributed._broadcast_object(("graph_noop", None))

    def worker_loop(self):
        assert self._dist_enabled and not self._dist_driver, "worker_loop() is worker-rank only"
        rank = getattr(self.distributed, "rank", "?")
        print(f"[llm_graph_spmd] rank={rank} entering graph worker_loop", flush=True)
        while True:
            method, payload = self.distributed._broadcast_object(None)
            if method in {"graph_shutdown", "__shutdown__"}:
                return
            if method in {"graph_noop", "noop"}:
                continue
            if method in {"forward", "model", "generate"}:
                args, kwargs = self.distributed._materialize_payload(payload)
                self.distributed._run_local(method, args, kwargs)
                continue
            if method == "graph_reset":
                self._reset_local()
                continue
            if method == "graph_prefill":
                meta, tensor_payload = payload
                tensor = self._empty_like(tensor_payload)
                self._broadcast_tensor(tensor)
                self._prefill_local(tensor, int(meta["start_pos"]))
                continue
            if method == "graph_decode":
                meta, tensor_payload = payload
                tensor = self._empty_like(tensor_payload)
                self._broadcast_tensor(tensor)
                self._decode_local(tensor, int(meta["pos"]))
                continue
            raise RuntimeError(f"unknown graph worker method: {method}")

    def _tensor_payload(self, tensor):
        return {
            "shape": tuple(tensor.shape),
            "dtype": tensor.dtype,
        }

    def _empty_like(self, payload):
        return torch.empty(payload["shape"], dtype=payload["dtype"], device=self.device)

    def _broadcast_tensor(self, tensor):
        import torch.distributed as dist

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        dist.broadcast(tensor, src=0)
        return tensor

    def _driver_graph_call(self, method, tensor=None, **meta):
        if not self._dist_enabled or not self._dist_driver:
            return _NO_DIST_CALL
        with self.distributed._call_lock:
            if method == "graph_reset":
                self.distributed._broadcast_object((method, None))
                self._reset_local()
                return True
            payload = (meta, self._tensor_payload(tensor))
            self.distributed._broadcast_object((method, payload))
            self._broadcast_tensor(tensor)
            if method == "graph_prefill":
                return self._prefill_local(tensor, int(meta["start_pos"]))
            if method == "graph_decode":
                return self._decode_local(tensor, int(meta["pos"]))
            raise RuntimeError(f"unknown graph driver method: {method}")

    @staticmethod
    def _first_tensor(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            return next((item for item in output if torch.is_tensor(item)), None)
        return None

    def _install_layer_trace_hooks(self):
        layers = getattr(self.model, "layers", None)
        if layers is None:
            layers = getattr(getattr(self.model, "model", None), "layers", None)
        if layers is None or len(layers) != self.cfg.num_hidden_layers:
            raise RuntimeError("could not locate all decoder layers for LLM graph tracing")

        for index, layer in enumerate(layers):
            def capture(_module, _inputs, output, index=index):
                tensor = self._first_tensor(output)
                if tensor is not None:
                    self._trace_hook_outputs[index] = tensor.detach()

            self._trace_hook_handles.append(layer.register_forward_hook(capture))

    def _begin_layer_trace(self):
        if self.trace_layers:
            self._trace_hook_outputs.clear()

    def _finish_layer_trace(self):
        if not self.trace_layers:
            return
        missing = [
            index
            for index in range(self.cfg.num_hidden_layers)
            if index not in self._trace_hook_outputs
        ]
        if missing:
            raise RuntimeError(f"LLM layer trace missing decoder outputs: {missing}")
        self._layer_outputs = tuple(
            self._trace_hook_outputs[index]
            for index in range(self.cfg.num_hidden_layers)
        )

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
        self._begin_layer_trace()
        outputs = self.model(
            inputs_embeds=self._emb,
            position_ids=self._pos,
            cache_position=self._cpos,
            attention_mask=self._mask,
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
        )
        self._finish_layer_trace()
        return outputs.last_hidden_state

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
                if self.trace_layers:
                    self._graph_layer_outputs = self._layer_outputs
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

    def _reset_local(self):
        # inference_mode: the buffers are inference tensors (created under inference_mode); the demo
        # calls decoder.reset() between videos OUTSIDE inference_mode, and StaticCache.reset() does
        # conv_states.zero_()/keys.zero_() in-place -> must be inside inference_mode.
        with torch.inference_mode():
            if self.cache is not None:
                self.cache.reset()
            if self._mask is not None:
                self._mask.fill_(self.neg)

    def reset(self):
        if self._driver_graph_call("graph_reset") is not _NO_DIST_CALL:
            return
        self._reset_local()

    def _warn_overflow(self, pos):
        raise CacheLimitExceeded("llm", max(int(pos) - 1, 0), int(pos), self.max_cache_len)

    def _prefill_local(self, inputs_embeds, start_pos):
        """Variable-length prefill via the non-graph StaticCache path. inputs_embeds: [1,L,H]."""
        with torch.inference_mode():
            L = inputs_embeds.shape[1]
            if L > self.max_cache_len:
                self._warn_overflow(L)
            end = start_pos + L
            if end > self.max_cache_len:
                self._warn_overflow(end)
            cpos = torch.arange(start_pos, start_pos + L, device=self.device)
            mask2d = torch.zeros(1, self.max_cache_len, dtype=torch.long, device=self.device)
            mask2d[:, :start_pos + L] = 1
            self._begin_layer_trace()
            h = self.model(inputs_embeds=inputs_embeds, position_ids=cpos.unsqueeze(0),
                           cache_position=cpos, attention_mask=mask2d,
                           past_key_values=self.cache, use_cache=True, return_dict=True)
            self._finish_layer_trace()
            h = h.last_hidden_state
            # mark all prefilled positions valid for the graph's additive mask
            self._mask[..., :start_pos + L] = 0.0
        return h

    def prefill(self, inputs_embeds, start_pos):
        out = self._driver_graph_call("graph_prefill", inputs_embeds, start_pos=int(start_pos))
        if out is not _NO_DIST_CALL:
            return out
        return self._prefill_local(inputs_embeds, start_pos)

    def _decode_local(self, inputs_embeds, pos):
        """Single-token decode at absolute position `pos` (== cache length before this token)."""
        with torch.inference_mode():
            if pos >= self.max_cache_len:
                self._warn_overflow(pos + 1)
            self._emb.copy_(inputs_embeds)
            self._pos.fill_(pos); self._cpos.fill_(pos)
            self._mask[..., pos] = 0.0
            if self._failed or not self._captured:
                return self._fwd()
            self.graph.replay()
            if self.trace_layers:
                self._layer_outputs = self._graph_layer_outputs
            return self._hidden

    @property
    def layer_outputs(self):
        """Raw decoder-layer outputs from the latest prefill or graph replay."""
        return self._layer_outputs

    def decode(self, inputs_embeds, pos):
        out = self._driver_graph_call("graph_decode", inputs_embeds, pos=int(pos))
        if out is not _NO_DIST_CALL:
            return out
        return self._decode_local(inputs_embeds, pos)
