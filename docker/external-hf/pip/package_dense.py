"""Package a newly compiled dense operator as a platform wheel.

Run only after dense_causal.build in a fresh project venv. The artifact is
installed under sys.prefix/var/cache/minicpm-o-accel as expected by its loader.
This does not build or accept binaries copied from an installed environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--build-log', required=True, type=Path)
    parser.add_argument('--source-archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    folder = args.manifest.parent.resolve()
    binary = folder / manifest['binary']
    if binary.parent != folder or sha(binary) != manifest['binary_sha256']:
        raise RuntimeError('Binary disagrees with the new build manifest')
    if sha(folder / 'build.ninja') != manifest['ninja_sha256']:
        raise RuntimeError('Build description disagrees with manifest')
    if not args.build_log.is_file() or not args.source_archive.is_file():
        raise RuntimeError('Fresh build log and source archive are required')
    names = ['manifest.json', 'build.ninja', 'dense_kernel_forward.h',
             'native_header.diff', binary.name]
    provenance = {
        'schema': 1,
        'installation': 'platform wheel generated from fixed-source compilation',
        'source_archive_sha256': sha(args.source_archive),
        'build_log_sha256': sha(args.build_log),
        'artifact_key': folder.name,
        'files': {n: sha(folder / n) for n in names},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dense-wheel-') as tmp:
        stage = Path(tmp)
        data = stage / 'artifact'
        data.mkdir()
        for name in names:
            shutil.copyfile(folder / name, data / name)
        (data / 'build-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        files = sorted(str(p.relative_to(stage)) for p in data.iterdir())
        setup = '''from setuptools import setup, Distribution
class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True
setup(name="minicpm-o-accel-dense-pro5000", version="1.0.0+%s",
      description="Fixed-source Pro5000 dense causal operator",
      data_files=[(%r, %r)], distclass=NativeDistribution)
''' % (folder.name, 'var/cache/minicpm-o-accel/' + folder.name, files)
        (stage / 'setup.py').write_text(setup)
        subprocess.run([sys.executable, 'setup.py', 'bdist_wheel', '--dist-dir',
                        str(args.output.resolve())], cwd=stage, check=True)
    wheels = sorted(args.output.glob('minicpm_o_accel_dense_pro5000-*.whl'))
    if len(wheels) != 1 or wheels[0].name.endswith('none-any.whl'):
        raise RuntimeError('Expected exactly one platform-specific native wheel')
    provenance['wheel'] = wheels[0].name
    provenance['wheel_sha256'] = sha(wheels[0])
    (args.output / 'dense-wheel-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
