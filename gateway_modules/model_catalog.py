"""Public model IDs are independent of the internal pool names and worker addresses."""
from __future__ import annotations

import json
from typing import Any


class ModelCatalogError(ValueError):
    pass


def model_signature(info: dict[str, Any]) -> str:
    # Include defaults/runtime as well as weights: replicas must advertise the same contract.
    return json.dumps(info, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def pool_model_id(pool: Any) -> str | None:
    explicit = getattr(pool, "public_model_id", None)
    identities = [w.model_info for w in pool.workers.values()]
    if not identities or any(not info for info in identities):
        return None
    signatures = {model_signature(info) for info in identities}
    if len(signatures) != 1:
        raise ModelCatalogError("Pool contains inconsistent model identities/defaults")
    info = identities[0]
    return explicit or info.get("model_id") or info.get("profile_id")


def catalog_entries(registry: Any) -> list[tuple[str, Any, dict[str, Any]]]:
    entries = []
    seen = set()
    if registry is None:
        raise ModelCatalogError("Model registry is not ready")
    for pool_name, pool in registry.items():
        model_id = pool_model_id(pool)
        if not model_id:
            continue  # Unidentified workers are never advertised or selected by public APIs.
        if model_id in seen:
            raise ModelCatalogError(f"Model ID {model_id!r} is assigned to multiple pools")
        seen.add(model_id)
        workers = list(pool.workers.values())
        capabilities = set.intersection(*(set(w.capabilities) for w in workers))
        modes = (["chat"] if "chat" in capabilities else [])
        if "audio_duplex" in capabilities:
            modes += ["audio"]
        if "omni_duplex" in capabilities:
            modes += ["video", "omni"]
        entries.append((pool_name, pool, {
            "id": model_id,
            "object": "model",
            "created": int(min(w.registered_at for w in workers).timestamp()),
            "owned_by": "minicpm",
            "loaded_model": dict(workers[0].model_info),
            "capabilities": {"modes": modes, "request_types": sorted(capabilities)},
            "status": "available" if pool.idle_count else "busy" if any(w.is_busy for w in workers) else "unavailable",
        }))
    return entries
