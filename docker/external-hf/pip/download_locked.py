"""Fetch exact manifest URLs with SHA256 verification into an offline input tree."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    for manifest_name, directory in [('public-wheel-inputs.json', 'wheelhouse'),
                                     ('source-inputs.json', 'sources')]:
        target = args.output / directory
        target.mkdir(parents=True, exist_ok=True)
        for record in json.loads((here / manifest_name).read_text()):
            if record.get('status', 'downloaded') != 'downloaded':
                continue
            path = target / record['filename']
            if path.parent != target:
                raise RuntimeError('Invalid manifest filename')
            if not path.is_file() or sha(path) != record['sha256']:
                tmp = path.with_suffix(path.suffix + '.part')
                with urllib.request.urlopen(record['url'], timeout=120) as src, tmp.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                if sha(tmp) != record['sha256']:
                    raise RuntimeError(f'Hash mismatch for {path.name}')
                tmp.replace(path)
            print(f'verified {directory}/{path.name}', flush=True)
    shutil.copyfile(here / 'requirements-public.lock', args.output / 'install-public.lock')


if __name__ == '__main__':
    main()
