"""Record installed distributions and fail on unexpected dependency conflicts."""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    packages = []
    for dist in metadata.distributions():
        native = {}
        for file in dist.files or []:
            if file.name.endswith('.so') or '.so.' in file.name:
                path = dist.locate_file(file)
                if path.is_file():
                    h = hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                            h.update(block)
                    native[str(file)] = h.hexdigest()
        packages.append({'name': dist.metadata['Name'], 'version': dist.version,
                         'requires': dist.requires or [], 'native_sha256': native})
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'],
                           capture_output=True, text=True)
    exception = json.loads(Path(__file__).with_name('decord-metadata-exception.json').read_text())
    decord = metadata.distribution('decord')
    assert decord.version == '0.6.0'
    assert hashlib.sha256(decord.read_text('WHEEL').encode()).hexdigest() == exception['internal_wheel_sha256']
    assert hashlib.sha256(Path(decord.locate_file('decord/libdecord.so')).read_bytes()).hexdigest() == exception['native_sha256']
    expected = ['minicpmo-utils 1.0.7 has requirement librosa==0.9.0, but you have librosa 0.11.0.',
                'decord 0.6.0 is not supported on this platform']
    actual = [line.strip() for line in check.stdout.splitlines() if line.strip()]
    passed = (check.returncode == 1 and sorted(actual) == sorted(expected) and not check.stderr.strip())
    result = {'python': sys.version, 'prefix': sys.prefix,
              'packages': sorted(packages, key=lambda p: p['name'].lower()),
              'pip_check': {'returncode': check.returncode, 'stdout': check.stdout,
                            'stderr': check.stderr},
              'accepted_metadata_exceptions': expected,
              'decord_metadata_evidence': exception,
              'no_unexpected_dependency_conflicts': passed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'distributions': len(packages), 'pip_check': result['pip_check'],
                      'no_unexpected_dependency_conflicts': passed}, indent=2))
    if not passed:
        raise SystemExit('Dependency audit differs from the explicit two recorded metadata exceptions')


if __name__ == '__main__':
    main()
