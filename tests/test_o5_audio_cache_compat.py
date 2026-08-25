from __future__ import annotations

import torch
from transformers import PretrainedConfig
from transformers import WhisperConfig
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from modeling.o5.modeling_minicpmo import MiniCPMWhisperEncoder, _audio_cache_length
from tools.o5replay.loaders import _patch_canonical_attention_config


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


def test_canonical_attention_patch_exposes_private_field_to_constructor_copy():
    config = PretrainedConfig()
    _patch_canonical_attention_config(config, "eager")
    assert config._attn_implementation == "eager"
    assert config.to_dict()["_attn_implementation"] == "eager"


def test_whisper_encoder_returns_cache_object_for_incremental_calls():
    config = WhisperConfig(
        d_model=16,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=32,
        max_source_positions=100,
        _attn_implementation="eager",
    )
    encoder = MiniCPMWhisperEncoder(config).eval()
    features = torch.randn(1, 80, 20)
    first = encoder(
        features,
        use_cache=True,
        output_hidden_states=True,
        attention_mask=torch.zeros(1, 1, 10, 10),
        return_dict=True,
    )
    assert isinstance(first.past_key_values, EncoderDecoderCache)
    assert _audio_cache_length(first.past_key_values) == 10

    second = encoder(
        features,
        past_key_values=first.past_key_values,
        use_cache=True,
        output_hidden_states=True,
        attention_mask=torch.zeros(1, 1, 10, 20),
        return_dict=True,
    )
    assert isinstance(second.past_key_values, EncoderDecoderCache)
    assert _audio_cache_length(second.past_key_values) == 20
