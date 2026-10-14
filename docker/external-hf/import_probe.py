import importlib,json,os,sys
from importlib import metadata
sys.path[:0]=['/opt/o5/demo','/opt/o5/accel/src']
versions={n:metadata.version(n) for n in ['torch','transformers','triton','flash-attn','flash-linear-attention','causal-conv1d','minicpmo-utils','minicpm-o5-sdk']}
print(json.dumps({'versions':versions,'python':sys.version}),flush=True)
for module in ['torch','torchvision','torchaudio','flash_attn','causal_conv1d','fla','stepaudio2.token2wav','py_backend.chat_util','core.external_o5']:
 try:
  obj=importlib.import_module(module)
  print('IMPORT',module,'OK',getattr(obj,'__file__',None),flush=True)
 except Exception as exc:
  print('IMPORT',module,'FAIL',repr(exc),flush=True)
  raise
import torch
print('GPU',torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0),flush=True)
from core.external_o5 import canonical_module
canonical_module('modeling_minicpmo')
print('CANONICAL_IMPORT_OK',flush=True)
from flash_attn import flash_attn_func
q=torch.randn(1,16,4,64,device='cuda',dtype=torch.bfloat16)
y=flash_attn_func(q,q,q,causal=True)
assert torch.isfinite(y).all()
from causal_conv1d import causal_conv1d_fn
x=torch.randn(1,64,16,device='cuda',dtype=torch.bfloat16)
w=torch.randn(64,4,device='cuda',dtype=torch.bfloat16)
y=causal_conv1d_fn(x,w,activation='silu');assert torch.isfinite(y).all()
from minicpm_o_accel.llm.ops.dense_causal import artifact
backend,status=artifact.load(); print('DENSE_ARTIFACT',json.dumps(status,default=str),flush=True)
assert backend is not None,status
q=torch.randn(1,8,16,256,device='cuda',dtype=torch.bfloat16)
y=backend.call(q,q,q);assert torch.isfinite(y).all()
torch.cuda.synchronize()
print('NATIVE_PRIMITIVES_OK',flush=True)
