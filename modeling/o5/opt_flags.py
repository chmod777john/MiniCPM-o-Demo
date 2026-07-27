"""Runtime-mutable inference-optimization switches. All default False = original behavior.
Enabled by core/processors/unified.py when model.optimize (config) or env O5_OPTIMIZE=1 is set.
See INTEGRATION.md."""
OPT = {"tts_fast": False, "lmhead": False, "vocoder_graph": False, "tts_static": False,
       "tts_graph": False, "tts_greedy": False, "fuse_vision_audio": False,
       "llm_static": False, "llm_graph": False}
