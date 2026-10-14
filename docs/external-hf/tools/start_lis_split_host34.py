"""Restart the accepted host34 split fixture; never stop or reuse occupied resources."""
import argparse
import json
import socket
import subprocess
from pathlib import Path

WORKER_IMAGE = 'sha256:e2a1f41bd37d50343b3882a4fb7f62fe7aae45b1adf4e30166f257c83af3962b'
GATEWAY_IMAGE = 'sha256:d7e3b5ed6619a78b9e0d9e327cf507ef3830681b4c9df692cd35c75d11c0e3dd'
NETWORK = 'o5-lis-split-20261014'
CLI = ['sudo', '-n', 'docker']


def output(args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/weihongliang/o5-docker-lis-20261014'))
    args = parser.parse_args()
    commands = json.loads((args.root / 'split-v11/commands.json').read_text())
    names = output(CLI + ['ps', '-a', '--format', '{{.Names}}']).splitlines()
    for role, image in [('worker', WORKER_IMAGE), ('gateway', GATEWAY_IMAGE)]:
        command = commands[role]
        assert command[:5] == CLI + ['run', '-d']
        assert image in command
        assert command[command.index('--name') + 1] == f'o5-lis-split-{role}-20261014'
        assert command[command.index('--network') + 1] == NETWORK
        if command[command.index('--name') + 1] in names:
            raise SystemExit('Owned fixture name already exists; inspect it before another launch')
        assert output(CLI + ['image', 'inspect', image, '--format', '{{.Id}}']) == image
    if NETWORK in output(CLI + ['network', 'ls', '--format', '{{.Name}}']).splitlines():
        raise SystemExit('Fixture network exists; inspect it before another launch')
    if output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']):
        raise SystemExit('GPU workload exists; do not stop another owner')
    memory = output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'])
    if len(memory.splitlines()) != 2 or any(int(row.split(',')[1]) > 1024 for row in memory.splitlines()):
        raise SystemExit('Requires two currently free host34 GPUs')
    for port in (18024, 18025):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    if not (args.root / 'model/model.pt').is_file() or not (args.root / 'model/token2wav_causal_vocoder/flow.pt').is_file():
        raise SystemExit('Missing previously verified raw model assets')
    print(json.dumps(commands, indent=2), flush=True)
    subprocess.run(CLI + ['network', 'create', NETWORK], check=True)
    for role in ('worker', 'gateway'):
        subprocess.run(commands[role], check=True)
    print('Started owned split fixture. Model loading and registration may take up to 1200s.')


if __name__ == '__main__':
    main()
