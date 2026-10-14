"""Compare source-rebuilt native packages with old binaries in separate processes.

The optional reference directory is diagnostic input only; never copy it into
wheelhouse or final images. Run each side in a fresh process, then --compare.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reference-binaries', type=Path)
    parser.add_argument('--accel-path', type=Path, default=Path('/opt/o5/accel/src'))
    parser.add_argument('--compare', type=Path, nargs=2)
    args = parser.parse_args()
    import torch
    if args.compare:
        left, right = [torch.load(p, map_location='cpu', weights_only=True) for p in args.compare]
        assert left['inputs_sha256'] == right['inputs_sha256'], 'Input mismatch'
        results = {}
        assert left['outputs'].keys() == right['outputs'].keys()
        for name, reference in left['outputs'].items():
            actual = right['outputs'][name]
            diff = actual.float() - reference.float()
            results[name] = {'exact': torch.equal(reference, actual),
                'finite': bool(torch.isfinite(actual).all()),
                'relative_l2': (diff.norm()/reference.float().norm().clamp_min(1e-30)).item(),
                'max_abs': diff.abs().max().item()}
        print(json.dumps(results, indent=2))
        if args.output:
            args.output.write_text(json.dumps(results, indent=2)+'\n')
        if not all(r['exact'] and r['finite'] for r in results.values()):
            raise SystemExit('Native comparison failed')
        return
    if args.output is None:
        parser.error('--output is required for a GPU run')
    sys.path.insert(0, str(args.accel_path))
    loaded = {}
    if args.reference_binaries:
        for name in ('flash_attn_2_cuda', 'causal_conv1d_cuda'):
            paths = list(args.reference_binaries.glob(name+'*.so'))
            assert len(paths) == 1, (name, paths)
            spec = importlib.util.spec_from_file_location(name, paths[0])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = sha(paths[0])
    from flash_attn import flash_attn_func
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from minicpm_o_accel.llm.ops.dense_causal import artifact
    if args.reference_binaries:
        name = artifact.module_name()
        path = args.reference_binaries/(name+'.so')
        spec = importlib.util.spec_from_file_location(name, path)
        dense = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dense)
        loaded[name] = sha(path)
    else:
        backend, status = artifact.load()
        assert backend is not None, status
        dense = backend.module
        loaded[artifact.module_name()] = status['binary_sha256']
    for name in ('flash_attn_2_cuda', 'causal_conv1d_cuda'):
        loaded[name] = sha(Path(sys.modules[name].__file__))
    generator = torch.Generator(device='cpu').manual_seed(20261014)
    inputs_sha = hashlib.sha256()
    def tensor(*shape):
        value = torch.randn(shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)
        inputs_sha.update(value.view(torch.uint8).numpy().tobytes())
        return value.to('cuda')
    outputs = {}
    with torch.inference_mode():
        for qlen, klen, dim in [(1, 128, 64), (14, 512, 64), (200, 200, 128),
                                (953, 2048, 128), (3000, 3000, 64)]:
            q, k, v = tensor(1, qlen, 8, dim), tensor(1, klen, 8, dim), tensor(1, klen, 8, dim)
            outputs[f'fa2-{qlen}-{klen}-{dim}'] = flash_attn_func(q, k, v, causal=True).cpu()
        for length in (1, 2, 14, 200, 953, 3000):
            x, w = tensor(1, 128, length), tensor(128, 4)
            outputs[f'causal-prefill-{length}'] = causal_conv1d_fn(x, w, activation='silu').cpu()
        state, w, x = tensor(1, 128, 4), tensor(128, 4), tensor(1, 128)
        outputs['causal-update-output'] = causal_conv1d_update(x, state, w, activation='silu').cpu()
        outputs['causal-update-state'] = state.cpu()
        for qlen, klen in [(1, 512), (14, 512), (200, 2048), (953, 4096)]:
            q, k, v = tensor(1, qlen, 8, 256), tensor(1, klen, 8, 256), tensor(1, klen, 8, 256)
            empty = torch.empty(0, device='cuda', dtype=q.dtype)
            outputs[f'dense-{qlen}-{klen}'] = dense.forward(q, k, v, empty, True, .0625).cpu()
    torch.cuda.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    info = {'inputs_sha256': inputs_sha.hexdigest(), 'outputs': outputs,
            'native_sha256': loaded, 'gpu': torch.cuda.get_device_name(),
            'torch': str(torch.__version__)}
    torch.save(info, args.output)
    print(json.dumps({k:v for k,v in info.items() if k != 'outputs'}, indent=2))
    print('NATIVE_OUTPUTS_SAVED', len(outputs))


if __name__ == '__main__':
    main()
