"""SPMD serving mirror for multi-rank deployment modes (e.g. tp2).

Under torchrun, every rank runs the model; only rank 0 (the driver) has the gateway/network.
The other ranks must run the *same* model calls in lockstep so the per-layer tensor-parallel
NCCL all_reduce (and the in-graph token broadcast) stay matched. SpmdMirror lets the driver call
model methods normally while transparently replaying them on the worker ranks:

    driver:  mirror.call("duplex_prefill", audio_waveform=chunk, frame_list=frames)   # broadcasts + runs
             out = mirror.call("duplex_generate", decode_mode="sampling", ...)
             mirror.call("duplex_finalize")
             ...
             mirror.shutdown()        # tells workers to exit

    worker:  mirror.worker_loop()     # blocks: receive (method,args)->run, until shutdown

The gateway integration is: on a non-driver rank, call mirror.worker_loop() after the model is
built (instead of starting the async server); on the driver rank, route every backend model call
through mirror.call(). Idle gaps need a heartbeat call (mirror.call("noop")) so the group stays in
lockstep when no client request is pending.
"""
from __future__ import annotations
from typing import Any


class SpmdMirror:
    def __init__(self, model: Any, is_driver: bool, rank: int, world_size: int):
        self.model = model
        self.is_driver = is_driver
        self.rank = rank
        self.world_size = world_size
        import torch
        self._device = torch.device(f"cuda:{rank}")

    def _bcast(self, obj):
        import torch.distributed as dist
        box = [obj]
        dist.broadcast_object_list(box, src=0, device=self._device)
        return box[0]

    # ── driver side ──
    def call(self, method: str, *args, **kwargs):
        """Driver: broadcast (method,args) to workers, then run locally and return the result."""
        assert self.is_driver, "call() is driver-only"
        if self.world_size > 1:
            self._bcast((method, args, kwargs))
        if method == "noop":
            return None
        return getattr(self.model, method)(*args, **kwargs)

    def shutdown(self):
        if self.is_driver and self.world_size > 1:
            self._bcast(("__shutdown__", (), {}))

    # ── worker side ──
    def worker_loop(self):
        """Worker rank: mirror the driver's model calls until shutdown. Never returns to serving."""
        assert not self.is_driver, "worker_loop() is worker-only"
        while True:
            method, args, kwargs = self._bcast(None)
            if method == "__shutdown__":
                return
            if method == "noop":
                continue
            getattr(self.model, method)(*args, **kwargs)
