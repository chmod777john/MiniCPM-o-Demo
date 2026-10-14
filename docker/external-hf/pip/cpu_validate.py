import hashlib,importlib,importlib.metadata,json,os,runpy,subprocess,sys,traceback
from pathlib import Path
output=Path('/opt/o5/cpu-validation');output.mkdir(exist_ok=True)
result={'scope':'CPU environment/native imports and original decord functionality; no GPU/model inference','checks':{}}
def check(name,fn):
    try:
        info=fn();result['checks'][name]={'pass':True,'details':info};print(name,'PASS',info,flush=True)
    except Exception as exc:
        result['checks'][name]={'pass':False,'error':repr(exc),'traceback':traceback.format_exc()};print(name,'FAIL',traceback.format_exc(),flush=True)
    (output/'cpu-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
sys.path[:0]=['/opt/o5/demo','/opt/o5/accel/src']
os.environ.update(O5_CANONICAL_PATH='/opt/o5/hf',O5_ACCELERATION_PATH='/opt/o5/accel',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def native():
    import torch,flash_attn_2_cuda,causal_conv1d_cuda
    from minicpm_o_accel.llm.ops.dense_causal import artifact
    backend,status=artifact.load();assert backend is not None,status
    return {'torch':str(torch.__version__),'fa2':flash_attn_2_cuda.__file__,'causal':causal_conv1d_cuda.__file__,'dense':status}
check('native',native)
def decord_test():
    import decord,numpy as np
    from decord._ffi.function import _FunctionBase
    path='/tmp/decord-probe.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=red:s=32x32:r=4','-frames:v','4','-threads','1','-pix_fmt','yuv420p','-y',path],check=True)
    reader=decord.VideoReader(path,ctx=decord.cpu(0));frames=reader.get_batch([0,1,2,3]).asnumpy();assert frames.shape==(4,32,32,3)
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',path,'-f','rawvideo','-pix_fmt','rgb24','-threads','1','-']);other=np.frombuffer(raw,dtype=np.uint8).reshape(frames.shape);delta=np.abs(frames.astype(np.int16)-other.astype(np.int16));assert int(delta.max())==0
    return {'version':decord.__version__,'frames':len(reader),'shape':list(frames.shape),'ffi':_FunctionBase.__module__,'ffmpeg_max_abs':int(delta.max()),'frames_sha256':hashlib.sha256(frames.tobytes()).hexdigest()}
check('decord_video',decord_test)
for name in ['flash_attn','causal_conv1d','fla','minicpmo','minicpm_o5_sdk','py_backend.server','gateway','worker']:
    check('import_'+name,lambda name=name: {'file':getattr(importlib.import_module(name),'__file__',None)})
def canonical():
    from core.external_o5 import canonical_module
    model=canonical_module('modeling_minicpmo');return {'file':model.__file__,'class':model.MiniCPMO.__name__}
check('canonical_import',canonical)
for name in ['test_entrypoint.py','test_lis_contract.py','test_launcher_image.py']:
    check(name,lambda name=name: subprocess.check_output([sys.executable,'/opt/o5/package/'+name],text=True))
print(json.dumps(result,indent=2),flush=True)

assert all(c['pass'] for c in result['checks'].values()), result['checks']
