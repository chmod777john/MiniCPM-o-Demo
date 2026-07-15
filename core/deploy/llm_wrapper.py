"""LLM-compatible wrappers for deployment runtimes.

The serving/modeling code should treat ``MiniCPMO.llm`` as a normal
HuggingFace causal LM.  Deployment-specific behavior, such as TP2 tensor
parallelism and future rank synchronization, belongs behind this object rather
than in chat/duplex/TTS call sites.
"""

from __future__ import annotations

import inspect
import logging
import threading
from typing import Any

import torch


_CACHE_SENTINEL = {"__tp2_cache__": True}
logger = logging.getLogger("deploy.llm_wrapper")


class DistributedTPLLM(torch.nn.Module):
    """Transparent wrapper for a tensor-parallel HuggingFace CausalLM.

    This class deliberately starts as a behavioral proxy: it preserves the
    current TP2 model behavior while making the contract explicit.  Future work
    can move LLM-level SPMD synchronization into this class without changing
    MiniCPMO chat/streaming/duplex code.
    """

    def __init__(
        self,
        inner: torch.nn.Module,
        *,
        is_driver: bool = True,
        rank: int = 0,
        world_size: int = 1,
        sync_calls: bool = False,
    ):
        super().__init__()
        self.inner = inner
        self.is_driver = is_driver
        self.rank = rank
        self.world_size = world_size
        self.sync_calls = sync_calls and world_size > 1
        self._call_lock = threading.Lock()
        self._control_group = None
        self._worker_past_key_values = None
        if self.sync_calls:
            import torch.distributed as dist

            self._control_group = dist.new_group(backend="gloo")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.sync_calls and self.is_driver:
            return self._driver_call("forward", args, kwargs)
        return self.inner(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if self.sync_calls and self.is_driver:
            return self._driver_call("generate", args, kwargs)
        return self.inner.generate(*args, **kwargs)

    def noop(self) -> None:
        if self.sync_calls and self.is_driver:
            self._driver_call("noop", (), {})

    def shutdown_worker(self) -> None:
        if self.sync_calls and self.is_driver:
            self._broadcast_object(("__shutdown__", None))

    def worker_loop(self) -> None:
        assert self.sync_calls and not self.is_driver, "worker_loop() is worker-rank only"
        logger.info("[tp2-llm] rank=%s entering LLM worker_loop", self.rank)
        while True:
            method, payload = self._broadcast_object(None)
            logger.info("[tp2-llm] rank=%s recv method=%s", self.rank, method)
            if method == "__shutdown__":
                return
            if method == "noop":
                continue
            args, kwargs = self._materialize_payload(payload)
            result = self._run_local(method, args, kwargs)
            if inspect.isgenerator(result):
                for _ in result:
                    pass

    def _driver_call(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        with self._call_lock:
            if method == "noop":
                self._broadcast_object((method, None))
                return None
            logger.info("[tp2-llm] rank=%s driver begin method=%s", self.rank, method)
            payload = self._make_payload(args, kwargs)
            self._broadcast_object((method, payload))
            self._broadcast_payload_tensors(payload, (args, kwargs))
            output = self._run_local(method, args, kwargs)
            logger.info("[tp2-llm] rank=%s driver end method=%s", self.rank, method)
            return output

    def _run_local(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if method == "forward":
            output = self.inner(*args, **kwargs)
        elif method == "generate":
            output = self.inner.generate(*args, **kwargs)
        else:
            raise AttributeError(method)
        self._remember_cache(output)
        if not self.is_driver:
            logger.info("[tp2-llm] rank=%s worker end method=%s", self.rank, method)
        return output

    def _remember_cache(self, output: Any) -> None:
        cache = None
        if isinstance(output, dict):
            cache = output.get("past_key_values")
        else:
            cache = getattr(output, "past_key_values", None)
        if cache is not None:
            self._worker_past_key_values = cache

    def _broadcast_object(self, obj: Any) -> Any:
        import torch.distributed as dist

        box = [obj]
        dist.broadcast_object_list(box, src=0, group=self._control_group)
        return box[0]

    def _make_payload(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        return self._encode_value(args, key=None), self._encode_value(kwargs, key=None)

    def _materialize_payload(self, payload: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args_payload, kwargs_payload = payload
        args = self._decode_value(args_payload, key=None)
        kwargs = self._decode_value(kwargs_payload, key=None)
        self._broadcast_payload_tensors(payload, (args, kwargs))
        return args, kwargs

    def _encode_value(self, value: Any, *, key: str | None) -> Any:
        if key == "past_key_values":
            return _CACHE_SENTINEL
        if torch.is_tensor(value):
            return {
                "__tp2_tensor__": True,
                "shape": tuple(value.shape),
                "dtype": value.dtype,
                "requires_grad": bool(value.requires_grad),
            }
        if isinstance(value, tuple):
            return {"__tp2_tuple__": [self._encode_value(v, key=None) for v in value]}
        if isinstance(value, list):
            return [self._encode_value(v, key=None) for v in value]
        if isinstance(value, dict):
            return {k: self._encode_value(v, key=str(k)) for k, v in value.items()}
        return value

    def _decode_value(self, value: Any, *, key: str | None) -> Any:
        if value == _CACHE_SENTINEL or key == "past_key_values":
            return self._worker_past_key_values
        if isinstance(value, dict) and value.get("__tp2_tensor__"):
            return torch.empty(value["shape"], dtype=value["dtype"], device=self.device)
        if isinstance(value, dict) and "__tp2_tuple__" in value:
            return tuple(self._decode_value(v, key=None) for v in value["__tp2_tuple__"])
        if isinstance(value, list):
            return [self._decode_value(v, key=None) for v in value]
        if isinstance(value, dict):
            return {k: self._decode_value(v, key=str(k)) for k, v in value.items()}
        return value

    def _broadcast_payload_tensors(self, payload: Any, source: Any | None = None) -> None:
        if source is None:
            args_source = None
            kwargs_source = None
        else:
            args_source, kwargs_source = source
        args_payload, kwargs_payload = payload
        self._broadcast_encoded_tensors(args_payload, args_source)
        self._broadcast_encoded_tensors(kwargs_payload, kwargs_source)

    def _broadcast_encoded_tensors(self, encoded: Any, source: Any) -> None:
        import torch.distributed as dist

        if isinstance(encoded, dict) and encoded.get("__tp2_tensor__"):
            tensor = source
            if not torch.is_tensor(tensor):
                raise TypeError("TP2 LLM tensor payload lost its source tensor")
            dist.broadcast(tensor, src=0)
            return
        if encoded == _CACHE_SENTINEL:
            return
        if isinstance(encoded, dict) and "__tp2_tuple__" in encoded:
            source_items = source if source is not None else [None] * len(encoded["__tp2_tuple__"])
            for child, child_source in zip(encoded["__tp2_tuple__"], source_items):
                self._broadcast_encoded_tensors(child, child_source)
            return
        if isinstance(encoded, list):
            source_items = source if source is not None else [None] * len(encoded)
            for child, child_source in zip(encoded, source_items):
                self._broadcast_encoded_tensors(child, child_source)
            return
        if isinstance(encoded, dict):
            source_dict = source if isinstance(source, dict) else {}
            for key, child in encoded.items():
                self._broadcast_encoded_tensors(child, source_dict.get(key))

    def get_input_embeddings(self) -> Any:
        return self.inner.get_input_embeddings()

    def set_input_embeddings(self, value: Any) -> None:
        self.inner.set_input_embeddings(value)

    def get_output_embeddings(self) -> Any:
        return self.inner.get_output_embeddings()

    def set_output_embeddings(self, value: Any) -> None:
        self.inner.set_output_embeddings(value)

    @property
    def config(self) -> Any:
        return self.inner.config

    @property
    def generation_config(self) -> Any:
        return self.inner.generation_config

    @generation_config.setter
    def generation_config(self, value: Any) -> None:
        self.inner.generation_config = value

    @property
    def model(self) -> Any:
        return self.inner.model

    @model.setter
    def model(self, value: Any) -> None:
        self.inner.model = value

    @property
    def lm_head(self) -> Any:
        return self.inner.lm_head

    @lm_head.setter
    def lm_head(self, value: Any) -> None:
        self.inner.lm_head = value

    @property
    def device(self) -> torch.device:
        try:
            return next(self.inner.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype | None:
        try:
            return next(self.inner.parameters()).dtype
        except StopIteration:
            return None

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = super().__getattr__("inner")
            return getattr(inner, name)


def unwrap_llm(llm: Any) -> Any:
    """Return the underlying HF model when a deployment wrapper is present."""

    return getattr(llm, "inner", llm)
