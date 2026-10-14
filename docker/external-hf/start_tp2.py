"""Launch the immutable TP2 image only after two explicitly selected GPUs are free."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess


def output(*cmd):
    return subprocess.check_output(cmd, text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', choices=('docker', 'podman'), required=True)
    p.add_argument('--sudo', action='store_true')
    p.add_argument('--devices', required=True, help='Two host GPU indices, e.g. 0,1')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--logs-dir', type=Path, required=True)
    p.add_argument('--image', required=True, help='Immutable repository@sha256:...')
    p.add_argument('--name', default='o5-external-hf-tp2-20261014')
    p.add_argument('--port', type=int, default=18016)
    a = p.parse_args()
    ids = a.devices.split(',')
    if len(ids) != 2 or len(set(ids)) != 2 or not all(x.isdigit() for x in ids):
        p.error('Select exactly two distinct host GPU indices')
    if '@sha256:' not in a.image:
        p.error('Use the immutable image digest from delivery-manifest.json')
    devices = {}
    for row in csv.reader(output('nvidia-smi', '--query-gpu=index,uuid,name,memory.used',
                                 '--format=csv,noheader,nounits').splitlines()):
        index, uuid, name, used = [x.strip() for x in row]
        devices[index] = dict(uuid=uuid, name=name, memory_mib=int(used))
    processes = list(csv.reader(output('nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                      '--format=csv,noheader,nounits').splitlines()))
    occupied = {r[0].strip() for r in processes if r}
    for index in ids:
        gpu = devices.get(index)
        if gpu is None or gpu['uuid'] in occupied or gpu['memory_mib'] > 1024:
            raise SystemExit(f'GPU {index} is unavailable; never stop another workload')
    model = a.model_dir.resolve()
    if not (model / 'model.pt').is_file() or not (model / 'token2wav_causal_vocoder/flow.pt').is_file():
        raise SystemExit('model-dir must contain raw model.pt and token2wav_causal_vocoder/flow.pt')
    cli = (['sudo', '-n'] if a.sudo else []) + [a.engine]
    names = output(*cli, 'ps', '-a', '--format', '{{.Names}}').splitlines()
    if a.name in names:
        raise SystemExit(f'Container already exists: {a.name}')
    # Pro5000 Podman uses task-local CONTAINERS_CONF with generated CDI metadata.
    args = cli + ['run', '-d', '--name', a.name, '--ipc=host']
    if a.engine == 'docker':
        args += ['--gpus', '"device=' + ','.join(ids) + '"']
    else:
        if not os.environ.get('CONTAINERS_CONF'):
            raise SystemExit('Set task-local CONTAINERS_CONF for Podman CDI lookup')
        args += ['--security-opt=label=disable']
        for index in ids:
            args += ['--device', 'nvidia.com/gpu=' + index]
    env = dict(O5_ROLE='standalone', O5_DEPLOY_MODE='tp2', O5_MOE_PARALLEL='tp',
               O5_ACCELERATION_HARDWARE='pro5000', O5_LLM_GRAPH='1', O5_TTS_GRAPH='1',
               O5_VOCODER_GRAPH='1', CANONICAL_UNIT_PREFILL_BATCH='1', O5_COMPACT_LOGITS='1',
               O5_TTS_FA2_WHOLE_DECODE='1', O5_CANVAS_WINDOW_EXACT_CHAIN='1',
               O5_DATA_DIR='/run-evidence/data', TRITON_CACHE_DIR='/run-evidence/triton',
               TORCHINDUCTOR_CACHE_DIR='/run-evidence/inductor')
    for key, value in env.items():
        args += ['-e', key + '=' + value]
    a.logs_dir.mkdir(parents=True, exist_ok=True)
    args += ['-p', f'127.0.0.1:{a.port}:8006', '-v', str(model) + ':/models/o5:ro',
             '-v', str(a.logs_dir.resolve()) + ':/run-evidence', a.image]
    print(json.dumps({'selected_devices': {k: devices[k] for k in ids}, 'command': args}, indent=2), flush=True)
    subprocess.run(args, check=True)


if __name__ == '__main__':
    main()
