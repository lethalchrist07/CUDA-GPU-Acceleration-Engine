"""Read-only baseline inspection; writes only a new, timestamped Phase 10 JSON file."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
    return {'command': args, 'exit_code': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def main():
    from torch.utils import cpp_extension
    manifest = json.loads((ROOT / 'results/phase2_verification_20261016/manifest.json').read_text())
    integrity = {name: {'expected': meta['sha256'], 'actual': digest(ROOT / name)}
                 for name, meta in manifest.items()}
    checkpoints = {}
    for path in sorted((ROOT / 'checkpoints/resnet18_cifar10').glob('*.pt')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        checkpoints[str(path.relative_to(ROOT))] = {
            'sha256': digest(path), 'bytes': path.stat().st_size,
            'epoch': checkpoint.get('epoch'), 'keys': list(checkpoint), 'readable': True}
        del checkpoint
    try:
        cpp_extension._check_cuda_version('cl', cpp_extension.TorchVersion('19.51'))
        compatibility = {'status': 'passed'}
    except RuntimeError as error:
        compatibility = {'status': 'blocked', 'error': str(error),
                         'scope': 'Installed PyTorch build-extension version guard; no compilation attempted'}
    candidates = command(['git', 'ls-files', '--others', '--exclude-standard'])
    source_hashes = {name: digest(ROOT / name) for name in candidates['stdout'].splitlines()
                     if (ROOT / name).is_file() and not name.startswith('results/phase10/')}
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'cuda_home': cpp_extension.CUDA_HOME, 'disk_bytes': shutil.disk_usage(ROOT)._asdict(),
        'git_head': command(['git', 'rev-parse', '--verify', 'HEAD']),
        'git_status': command(['git', 'status', '--short']),
        'nvcc': command(['nvcc', '--version']),
        'telemetry': command(['nvidia-smi']),
        'gpu': str(torch.cuda.get_device_properties(0)),
        'gpu_free_total_bytes': torch.cuda.mem_get_info(),
        'extension_compatibility': compatibility, 'historical_integrity': integrity,
        'checkpoints': checkpoints, 'untracked_file_sha256': source_hashes,
    }
    folder = ROOT / 'results/phase10/audit'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(path)
    assert all(item['expected'] == item['actual'] for item in integrity.values())
    assert len(checkpoints) == 3


if __name__ == '__main__':
    main()
