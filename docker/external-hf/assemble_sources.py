"""Freeze clean Git inputs and freshly built mobile assets for a Docker context."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def tree_record(root):
    files = {}
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        content = path.readlink().as_posix().encode() if path.is_symlink() else path.read_bytes()
        files[rel] = hashlib.sha256(content).hexdigest()
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return {'tree_sha256': hashlib.sha256(encoded).hexdigest(), 'files': files}


def main():
    p = argparse.ArgumentParser()
    for name in ('hf', 'accel', 'demo', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    package = Path(__file__).resolve().parent
    if git(package, 'status', '--porcelain', '--', '.'):
        raise SystemExit('Commit packaging changes before assembly')
    record = {}
    for name in ('hf', 'accel', 'demo'):
        root = getattr(a, name).resolve()
        if git(root, 'status', '--porcelain'):
            raise SystemExit(f'{name} has uncommitted inputs; commit before assembly')
        record[name] = {'path': str(root), 'sha': git(root, 'rev-parse', 'HEAD')}
    # Build from the exact clean Demo checkout; an old static/ tree is insufficient.
    mobile_root = a.demo / 'frontend/mobile'
    tools = {name: subprocess.check_output([name, '--version'], text=True).strip()
             for name in ('node', 'npm')}
    subprocess.run(['npm', 'run', 'build:static'], cwd=mobile_root, check=True)
    if git(a.demo, 'status', '--porcelain', '--untracked-files=no'):
        raise SystemExit('Mobile build changed tracked files; commit them and repeat assembly')
    dest = a.output / 'materials'
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('hf', 'accel', 'demo'):
        root = getattr(a, name).resolve()
        cmd = ['git', '-C', str(root), 'archive', record[name]['sha']]
        if name == 'accel':
            cmd += ['src', 'pyproject.toml', 'README.md']
        with (dest / ('accel-runtime.tar' if name == 'accel' else name + '.tar')).open('wb') as f:
            subprocess.run(cmd, stdout=f, check=True)
    for name in ('mobile', 'mobile-omni'):
        target = dest / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(a.demo / 'static' / name, target)
    record['packaging'] = {'commit': git(package, 'rev-parse', 'HEAD'), **tree_record(package)}
    record['mobile_build'] = {
        'input_commit': record['demo']['sha'],
        'command': 'npm run build:static', 'cwd': 'frontend/mobile', 'tools': tools,
        'outputs': {name: tree_record(dest / name) for name in ('mobile', 'mobile-omni')},
    }
    target = a.output / 'docker/external-hf'
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(package, target, ignore=shutil.ignore_patterns('__pycache__'))
    (dest / 'source-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v.get('sha', v.get('commit')) for k, v in record.items()}, indent=2))


if __name__ == '__main__':
    main()
