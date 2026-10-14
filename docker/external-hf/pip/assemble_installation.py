"""Verify actual wheels and produce the complete installation lock/manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
from email import message_from_bytes


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--native-manifest', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    records = []
    for name in ('public-wheel-inputs.json', 'internal-wheel-inputs.json'):
        records.extend(r for r in json.loads((here / name).read_text())
                       if r.get('status', 'downloaded') == 'downloaded')
    records.extend(json.loads(args.native_manifest.read_text()))
    names = set()
    for record in records:
        path = args.context / 'wheelhouse' / record['filename']
        if sha(path) != record['sha256']:
            raise RuntimeError(f'Wheel hash mismatch: {path}')
        with ZipFile(path) as archive:
            meta = message_from_bytes(archive.read(next(n for n in archive.namelist()
                                                       if n.endswith('.dist-info/METADATA'))))
        normalize = lambda s: s.lower().replace('_', '-').replace('.', '-')
        if normalize(meta['Name']) != normalize(record['name']) or meta['Version'] != record['version']:
            raise RuntimeError(f'Wheel metadata mismatch: {path}')
        if normalize(record['name']) in names:
            raise RuntimeError('Duplicate distribution '+record['name'])
        names.add(normalize(record['name']))
    for record in json.loads((here / 'source-inputs.json').read_text()):
        if sha(args.context / 'sources' / record['filename']) != record['sha256']:
            raise RuntimeError('Compiler/source hash mismatch: '+record['filename'])
    records.sort(key=lambda r: r['name'].lower())
    (args.context / 'requirements-install.lock').write_text('\n'.join(
        f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}" for r in records)+'\n')
    manifest = {'schema': 1, 'installation': 'pip --no-index --no-deps --require-hashes in a new venv',
                'wheels': records, 'source_inputs': json.loads((here/'source-inputs.json').read_text()),
                'dependency_exceptions': [{'distribution': 'minicpmo-utils==1.0.7',
                    'declared': 'librosa==0.9.0', 'actual': 'librosa==0.11.0'}]}
    (args.context / 'installation-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'wheels': len(records), 'context': str(args.context)}, indent=2))


if __name__ == '__main__':
    main()
