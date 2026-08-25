from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from modeling.o5.modeling_minicpmo import _audio_cache_length


def _dynamic_cache(length: int) -> DynamicCache:
    cache = DynamicCache()
    key = torch.zeros(1, 2, length, 4)
    value = torch.zeros_like(key)
    cache.update(key, value, layer_idx=0)
    return cache


def test_audio_cache_length_accepts_empty_and_encoder_decoder_cache():
    assert _audio_cache_length(None) == 0
    assert _audio_cache_length(EncoderDecoderCache(DynamicCache(), DynamicCache())) == 0
    assert _audio_cache_length(EncoderDecoderCache(_dynamic_cache(7), DynamicCache())) == 7


def test_audio_cache_length_keeps_legacy_tuple_compatibility():
    cache = ((torch.zeros(1, 2, 5, 4), torch.zeros(1, 2, 5, 4)),)
    assert _audio_cache_length(cache) == 5
