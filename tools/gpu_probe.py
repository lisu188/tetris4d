from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description='Report GPU/CUDA availability without installing or changing drivers.')
    parser.add_argument('--output', type=Path, default=Path('gpu-info.json'))
    args = parser.parse_args()
    result = {'platform': platform.platform(), 'python': platform.python_version(), 'torch_installed': False,
              'cuda_available': False, 'gpus': []}
    command = shutil.which('nvidia-smi')
    if command is None and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():
        command = '/usr/lib/wsl/lib/nvidia-smi'
    if command:
        try:
            probe = subprocess.run([command, '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                                   text=True, capture_output=True, timeout=15)
            result['nvidia_smi'] = probe.stdout.strip() if probe.returncode == 0 else probe.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as error:
            result['nvidia_smi_error'] = type(error).__name__
    if importlib.util.find_spec('torch') is not None:
        try:
            import torch
            result.update(torch_installed=True, torch_version=torch.__version__,
                          torch_cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
            if result['cuda_available']:
                for i in range(torch.cuda.device_count()):
                    device = torch.cuda.get_device_properties(i)
                    result['gpus'].append({'name': device.name, 'vram_bytes': device.total_memory,
                                           'compute_capability': [device.major, device.minor]})
        except (ImportError, RuntimeError, OSError) as error:
            result['torch_error'] = type(error).__name__
    result['note'] = 'Diagnostic only; GPU neural policy is proposed, not trained or selected by the game.'
    text = json.dumps(result, indent=2)
    args.output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
