"""CPU contract for the external facade, without allocating a real model."""
import ast
import json
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
import logging
import torch
from core.schemas.external_o5_duplex import ExternalO5DuplexConfig

@dataclass
class Window:
    sliding_window_mode: str = 'off'
    basic_window_high_tokens: int = 8000
    basic_window_low_tokens: int = 6000
    context_previous_max_tokens: int = 500
    context_max_units: int = 24
    text_sink: int = 512
    text_sliding_window: int = 512
    verify_mode: bool = False

@dataclass
class LS:
    fix_enabled: bool = False
    mode: str = 'legacy'
    speak_threshold: float = 0.5
    log_prob: bool = False
    strict_unit: bool = False

source = Path(__file__).parents[1] / 'modeling/external_o5_facade.py'
tree = ast.parse(source.read_text())
klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'MiniCPMO')
selected = {'configure_duplex_session', 'duplex_prepare', 'duplex_generate'}
methods = [x for x in klass.body if isinstance(x, ast.FunctionDef) and x.name in selected]
assert len(methods) == len(selected), 'duplicate method definitions'
for method in methods:
    method.decorator_list = []
ns = {'deepcopy': deepcopy, 'replace': replace, 'fields': fields, 'torch': torch,
      'canonical_module': lambda name: SimpleNamespace(DuplexWindowConfig=Window),
      '_plain_duplex_system_prompt': lambda x: x, 'logger': logging.getLogger(__name__)}
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + methods, type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), ns)
Facade = type('Facade', (), {name: ns[name] for name in selected})
class Decoder:
    _window_config = Window()
    def set_window_config(self, c): self._window_config = c
    def set_window_enabled(self, e): self.enabled = e
class Duplex:
    def __init__(self):
        for key, value in ExternalO5DuplexConfig().model_dump().items(): setattr(self, key, value)
        self.tts_temperature = torch.tensor([0.8])
        self.decoder = Decoder()
        self._streaming_generate_count = 9
        self.ls_decision_config = LS()
    def prepare(self, **kwargs): self.ls_decision_config = LS(); return 'prepared'
    def set_ls_decision_config(self, value): self.ls_decision_config = value
    def streaming_generate(self, **kwargs):
        self.last_kwargs = kwargs
        self.force_seen = self.force_listen_count
        return kwargs
f = Facade(); f.duplex = Duplex()
f.configure_duplex_session({'sliding_window_mode':'basic', 'basic_window_high_tokens':2048,
    'basic_window_low_tokens':1024, 'ls_decision_mode':'binary', 'speak_threshold':0.2,
    'max_new_speak_tokens_per_chunk':7, 'tts_temperature':0.3, 'force_listen_count':2})
assert f.duplex.decoder.enabled and f.duplex.decoder._window_config.basic_window_high_tokens == 2048
assert f.duplex_prepare() == 'prepared'
assert f.duplex.ls_decision_config.mode == 'binary' and f.duplex.ls_decision_config.fix_enabled
assert abs(float(f.duplex.tts_temperature[0]) - 0.3) < 1e-6
result = f.duplex_generate(force_listen_override=True)
assert result['max_new_speak_tokens_per_chunk'] == 7 and result['speak_threshold'] == 0.2
assert f.duplex.force_seen == 10 and f.duplex.force_listen_count == 2
f.configure_duplex_session({}); f.duplex_prepare(); result=f.duplex_generate()
assert f.duplex.decoder._window_config.sliding_window_mode == 'off'
assert not f.duplex.decoder.enabled and f.duplex.ls_decision_config.mode == 'legacy'
assert result['max_new_speak_tokens_per_chunk'] == 20 and 'speak_threshold' not in result
for bad in ({'sliding_window_mode':'typo'}, {'speak_threshold':2}, {'basic_window_high_tokens':1},
            {'unknown':True}, {'sample_rate':24000}):
    try: f.configure_duplex_session(bad)
    except ValueError: pass
    else: raise AssertionError(bad)
for legacy_default in (1.0, 1.05, 1.1):
    f.configure_duplex_session({'length_penalty': legacy_default})
    assert 'length_penalty' not in f.duplex_generate()
print(json.dumps({'pass':True, 'scope':'CPU facade contract: config, LS post-prepare, generation kwargs, per-session reset, invalid options; not GPU inference'}))
