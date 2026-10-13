# External HF ordinary session options

The retained gateway/worker/View routes stay in place. Before DuplexConfig
normalization, the external facade validates the raw config and passes HF
window/LS options to the external capability. No inference loop is copied.

Supported extensions include sliding_window_mode, basic_window_high_tokens,
basic_window_low_tokens, context_previous_max_tokens, context_max_units,
text_sink, text_sliding_window, verify_mode, ls_decision_mode, speak_threshold,
ls_strict_unit, ls_log_prob and tts_repetition_penalty. Existing DuplexConfig
sampling/max speak/force listen/TTS options are also applied. Unknown keys
are rejected rather than silently dropped. External HF does not implement
length_penalty; non-default requests fail clearly (the retained View always
passes its default 1.1, which does not change canonical generation).

Session prepare reinitializes config even when omitted; window defaults are
restored and LS overrides applied after HF prepare resets its policy. Binary
mode sets the decoder's fix_enabled token-ID switch. CPU contract command:

```bash
python -m scripts.check_external_duplex_options
```

This checks plumbing and lifecycle using fake capabilities, not GPU inference.
GPU basic-window verification is recorded separately when complete.
