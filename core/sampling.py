"""Shared sampling controls used by inference and replay instrumentation."""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Iterator
from typing import Any

import torch


_TRUE_VALUES = {"1", "true", "yes", "on"}
_TTS_ARGMAX_SCOPE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "o5_tts_argmax_scope", default=False
)
_SAMPLING_AWARE = "_o5_sampling_aware"


def tts_argmax_enabled() -> bool:
    """Return whether TTS multinomial decisions must use argmax."""

    value = os.environ.get("O5_TTS_ARGMAX", "0").strip().lower()
    return _TTS_ARGMAX_SCOPE.get() or value in _TRUE_VALUES


def argmax_multinomial(
    input_tensor: torch.Tensor,
    num_samples: int,
    replacement: bool = False,
    *,
    generator: Any = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Implement the narrow multinomial form used by O5 TTS sampling."""

    del replacement, generator
    if num_samples != 1:
        raise RuntimeError("O5_TTS_ARGMAX only supports num_samples=1")
    result = torch.argmax(input_tensor, dim=-1, keepdim=True)
    if out is not None:
        out.copy_(result)
        return out
    return result


@contextlib.contextmanager
def tts_argmax_scope(enabled: bool | None = None) -> Iterator[None]:
    """Apply argmax without bypassing an installed trace sampling wrapper.

    The backend can enter this scope around generation. When the trace
    controller is installed, it observes the scope through
    :func:`tts_argmax_enabled` and continues recording every decision. When no
    trace wrapper is installed, this context supplies the temporary global
    ``torch.multinomial`` override required by the backend.
    """

    requested = tts_argmax_enabled() if enabled is None else bool(enabled)
    token = _TTS_ARGMAX_SCOPE.set(requested)
    original = torch.multinomial
    installed = False
    if requested and not getattr(original, _SAMPLING_AWARE, False):
        def configured_multinomial(
            input_tensor: torch.Tensor,
            num_samples: int,
            replacement: bool = False,
            *,
            generator: Any = None,
            out: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return argmax_multinomial(
                input_tensor,
                num_samples,
                replacement=replacement,
                generator=generator,
                out=out,
            )

        setattr(configured_multinomial, _SAMPLING_AWARE, True)
        torch.multinomial = configured_multinomial
        installed = True
    try:
        yield
    finally:
        if installed:
            torch.multinomial = original
        _TTS_ARGMAX_SCOPE.reset(token)


def mark_sampling_aware(function: Any) -> Any:
    """Mark a multinomial wrapper as scope-aware for nested generation."""

    setattr(function, _SAMPLING_AWARE, True)
    return function
