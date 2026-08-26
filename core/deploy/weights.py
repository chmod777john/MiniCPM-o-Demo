"""Streaming loaders for the Demo's safetensors artifact layouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import torch


def _weight_map(directory: Path) -> dict[str, str]:
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    single = directory / "model.safetensors"
    if single.is_file():
        from safetensors import safe_open

        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return {key: single.name for key in handle.keys()}
    raise FileNotFoundError(f"no safetensors index or model.safetensors in {directory}")


def _matches(
    key: str,
    *,
    include_prefixes: Optional[Iterable[str]],
    exclude_prefixes: Optional[Iterable[str]],
) -> bool:
    includes = tuple(include_prefixes or ())
    excludes = tuple(exclude_prefixes or ())
    if includes and not key.startswith(includes):
        return False
    return not excludes or not key.startswith(excludes)


def _artifact_key_candidates(key: str, expected: set[str]) -> tuple[str, ...]:
    """Return exact and complete-bundle aliases for one artifact key.

    The public complete bundle normally stores the LLM below ``llm.*`` while
    the outer MiniCPMO module exposes the same names.  Older/internal bundles
    may store native Qwen keys (``model.*`` and ``lm_head.*``); support that
    one-way alias for compatibility.  The alias is deliberately limited to
    keys that exist in the target module, so unrelated checkpoints cannot be
    silently remapped.
    """
    candidates = [key]
    if not key.startswith("llm."):
        candidates.append(f"llm.{key}")
    return tuple(candidate for candidate in candidates if candidate in expected)


def _maybe_resize_llm_vocab(
    model: torch.nn.Module,
    root: Path,
    weight_map: dict[str, str],
    safe_open,
    exclude_prefixes: Optional[Iterable[str]],
) -> None:
    """Match a complete bundle's LLM vocabulary before ``assign=True`` load.

    FC checkpoints may contain the 24 SDK protocol rows (248168) while an
    older code/config directory still declares the base 248144 rows.  The
    extra rows belong to the LLM embedding and lm_head only; TTS keeps its
    independent text embedding size.  TP2 skips this because it excludes the
    outer ``llm.*`` namespace and constructs the native LLM from ``llm/``.
    """
    excludes = tuple(exclude_prefixes or ())
    if any(prefix == "llm." or prefix.startswith("llm.") for prefix in excludes):
        return

    artifact_key = next(
        (
            key
            for key in (
                "llm.model.embed_tokens.weight",
                "model.embed_tokens.weight",
            )
            if key in weight_map
        ),
        None,
    )
    if artifact_key is None:
        return
    shard_path = root / weight_map[artifact_key]
    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        checkpoint_vocab_size = int(handle.get_slice(artifact_key).get_shape()[0])

    llm = getattr(model, "llm", None)
    embeddings = getattr(llm, "get_input_embeddings", lambda: None)()
    if embeddings is None or not hasattr(embeddings, "weight"):
        return
    configured_vocab_size = int(embeddings.weight.shape[0])
    if checkpoint_vocab_size == configured_vocab_size:
        return
    resize = getattr(llm, "resize_token_embeddings", None)
    if not callable(resize):
        raise RuntimeError(
            "complete safetensors bundle requires LLM vocabulary resize, but "
            "the target model does not expose resize_token_embeddings"
        )
    resize(checkpoint_vocab_size, mean_resizing=False)
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "vocab_size"):
        config.vocab_size = checkpoint_vocab_size


def load_safetensors_into(
    model: torch.nn.Module,
    directory: str | Path,
    *,
    include_prefixes: Optional[Iterable[str]] = None,
    exclude_prefixes: Optional[Iterable[str]] = None,
) -> tuple[list[str], list[str]]:
    """Load matching tensors shard-by-shard into a meta-constructed model.

    Loading one shard at a time keeps CPU memory bounded by the largest shard
    and avoids materializing a second full state dict.  ``assign=True`` is
    required because deployment builders construct the outer model on meta.
    """
    root = Path(directory).expanduser().resolve()
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required for the HF weight bundle") from exc

    weight_map = _weight_map(root)
    _maybe_resize_llm_vocab(
        model,
        root,
        weight_map,
        safe_open,
        exclude_prefixes,
    )
    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    used_artifact_keys: set[str] = set()
    shard_names = list(dict.fromkeys(weight_map.values()))
    for shard_name in shard_names:
        shard_path = root / shard_name
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            chunk = {}
            for key in handle.keys():
                target_keys = _artifact_key_candidates(key, expected)
                target_key = next(
                    (
                        candidate
                        for candidate in target_keys
                        if _matches(
                            candidate,
                            include_prefixes=include_prefixes,
                            exclude_prefixes=exclude_prefixes,
                        )
                    ),
                    None,
                )
                if target_key is None:
                    continue
                chunk[target_key] = handle.get_tensor(key)
                loaded.add(target_key)
                used_artifact_keys.add(key)
        if chunk:
            model.load_state_dict(chunk, strict=False, assign=True)

    selected_expected = {
        key
        for key in expected
        if _matches(key, include_prefixes=include_prefixes, exclude_prefixes=exclude_prefixes)
    }
    missing = sorted(selected_expected - loaded)
    unexpected = sorted(
        key
        for key in weight_map
        if key not in used_artifact_keys
        and not (
            _artifact_key_candidates(key, expected)
            and not any(
                _matches(
                    candidate,
                    include_prefixes=include_prefixes,
                    exclude_prefixes=exclude_prefixes,
                )
                for candidate in _artifact_key_candidates(key, expected)
            )
        )
    )
    if missing:
        raise RuntimeError(
            f"safetensors bundle {root} is missing {len(missing)} expected tensors; "
            f"first keys: {missing[:5]}"
        )
    return missing, unexpected
