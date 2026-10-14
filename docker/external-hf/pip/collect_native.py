"""Freeze source-built native wheels and their build logs as installation inputs."""
import argparse
import hashlib
from email import message_from_bytes
import json
from pathlib import Path
from zipfile import ZipFile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.build_root.resolve()
    plans = [('flash-attn', 'flash_attn-2.8.3.post1', 'fa2-build.log', 'build_fa2_cpu.sh'),
             ('causal-conv1d', 'causal_conv1d-1.7.0', 'causal-build.log', 'build_causal_cpu.sh')]
    records = []
    for name, source, log_name, script_name in plans:
        wheel_paths = list((root/'native-wheels').glob(source+'-*.whl'))
        if len(wheel_paths) != 1:
            raise RuntimeError(f'Expected one new wheel for {name}')
        wheel = wheel_paths[0]
        if not wheel.name.endswith('-cp310-cp310-linux_x86_64.whl'):
            raise RuntimeError('Unexpected native ABI/platform: '+wheel.name)
        with ZipFile(wheel) as archive:
            metadata = message_from_bytes(archive.read(next(n for n in archive.namelist()
                                                          if n.endswith('.dist-info/METADATA'))))
            binaries = {n: hashlib.sha256(archive.read(n)).hexdigest()
                        for n in archive.namelist() if n.endswith('.so')}
        ninja = list((root/'sources'/source/'build').rglob('build.ninja'))
        if len(ninja) != 1:
            raise RuntimeError('Need one real build.ninja for '+name)
        records.append({'name': name, 'version': metadata['Version'], 'filename': wheel.name,
                        'sha256': sha(wheel), 'bytes': wheel.stat().st_size,
                        'source': {'kind': 'official sdist rebuilt in fresh project venv',
                            'filename': source+'.tar.gz', 'sha256': sha(root/(source+'.tar.gz')),
                            'setup_py_sha256': sha(root/'sources'/source/'setup.py'),
                            'script': script_name, 'script_sha256': sha(root/script_name),
                            'build_log': log_name, 'build_log_sha256': sha(root/log_name),
                            'build_ninja_sha256': sha(ninja[0]),
                            'kernel_sources_modified': False},
                        'native_sha256': binaries})
    dense_provenance = json.loads((root/'dist/dense-wheel-provenance.json').read_text())
    wheel = root/'dist'/dense_provenance['wheel']
    with ZipFile(wheel) as archive:
        metadata = message_from_bytes(archive.read(next(n for n in archive.namelist()
                                                      if n.endswith('.dist-info/METADATA'))))
    records.append({'name': metadata['Name'], 'version': metadata['Version'],
                    'filename': wheel.name, 'sha256': sha(wheel), 'bytes': wheel.stat().st_size,
                    'source': dense_provenance})
    args.output.write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
