"""Session options supported by the externally supplied O5 HF capability."""
from typing import Literal
from pydantic import ConfigDict, Field
from .duplex import DuplexConfig


class ExternalO5DuplexConfig(DuplexConfig):
    model_config = ConfigDict(extra="forbid")

    ls_mode: Literal["explicit", "implicit"] = "explicit"
    decode_mode: Literal["greedy", "sampling"] = "sampling"
    sample_rate: Literal[16000] = 16000
    sliding_window_mode: Literal["off", "basic", "context", "kv_surgery"] = "off"
    basic_window_high_tokens: int = Field(8000, gt=0, le=262144)
    basic_window_low_tokens: int = Field(6000, gt=0, le=262144)
    context_previous_max_tokens: int = Field(500, ge=0)
    context_max_units: int = Field(24, gt=0)
    text_sink: int = Field(512, ge=0)
    text_sliding_window: int = Field(512, ge=0)
    verify_mode: bool = False
    ls_decision_mode: Literal["legacy", "binary"] = "legacy"
    speak_threshold: float = Field(0.5, gt=0, lt=1)
    ls_strict_unit: bool = False
    ls_log_prob: bool = False
    tts_repetition_penalty: float = Field(1.05, ge=1)

    max_slice_nums: int = Field(1, ge=1, le=9)
    strategy_hd: bool = False
    strategy_hd_max_slice_nums: int = Field(4, ge=1, le=9)
