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
import inspect
from typing import Any

from MiniCPMO45.cache_limits import CacheLimitExceeded


class SpmdMirror:
    def __init__(self, model: Any, is_driver: bool, rank: int, world_size: int):
        self.model = model
        self.is_driver = is_driver
        self.rank = rank
        self.world_size = world_size
        import threading
        self._call_lock = threading.Lock()
        self._control_group = None
        if world_size > 1:
            import torch.distributed as dist
            self._control_group = dist.new_group(backend="gloo")

    def _bcast(self, obj):
        import torch.distributed as dist
        box = [obj]
        # Broadcast only small Python call descriptors here. Keeping object
        # broadcast on the default CPU path avoids torch 2.8 CUDA object-tensor
        # SymInt failures; actual model tensors still use CUDA collectives in
        # the mirrored model methods themselves.
        dist.broadcast_object_list(box, src=0, group=self._control_group)
        return box[0]

    # ── driver side ──
    def call(self, method: str, *args, **kwargs):
        """Driver: broadcast (method,args) to workers, then run locally and return the result."""
        assert self.is_driver, "call() is driver-only"
        # Rank 1 executes mirrored calls strictly in broadcast order. The serving
        # path may issue real duplex calls from request threads while the SPMD
        # heartbeat issues noop calls from a background thread; serialize them so
        # NCCL collectives cannot be interleaved in different orders across ranks.
        with self._call_lock:
            if self.world_size > 1:
                self._bcast((method, args, kwargs))
            if method == "noop":
                return None
            result = getattr(self.model, method)(*args, **kwargs)
            if inspect.isgenerator(result):
                return self._locked_generator(result)
            return result

    def _locked_generator(self, generator):
        while True:
            with self._call_lock:
                try:
                    yield next(generator)
                except StopIteration:
                    return

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
            try:
                result = getattr(self.model, method)(*args, **kwargs)
            except CacheLimitExceeded:
                # The driver closes the session and mirrors duplex_stop and
                # duplex_cleanup next. Keep this rank alive until those
                # commands arrive; an uncaught exception here would make
                # torchrun tear down the whole TP2 process group.
                continue
            if inspect.isgenerator(result):
                for _ in result:
                    pass
