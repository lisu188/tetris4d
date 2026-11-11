from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description='Build the optional strict-float native scorer for Linux/WSL or macOS.')
    parser.add_argument('--cc', default=os.environ.get('CC', 'cc'))
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    if sys.platform not in ('linux', 'darwin'):
        parser.error('Build inside Linux/WSL or macOS. Windows Python retains the NumPy/Python backend.')
    compiler = shlex.split(args.cc)
    if not compiler or not shutil.which(compiler[0]):
        parser.error('C compiler not found; install build-essential in Ubuntu/WSL.')
    source = ROOT / 'native' / 'tetris_native.c'
    output = source.with_name('libtetris4d_native' + ('.dylib' if sys.platform == 'darwin' else '.so'))
    flags = ['-std=c11', '-O3', '-fPIC', '-fno-fast-math', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror']
    if args.sanitize:
        flags += ['-fsanitize=undefined', '-fno-sanitize-recover=all', '-g']
    with tempfile.TemporaryDirectory(dir=source.parent) as temporary:
        staged = Path(temporary) / output.name
        command = compiler + flags + (['-dynamiclib'] if sys.platform == 'darwin' else ['-shared']) + [str(source), '-o', str(staged)]
        subprocess.run(command, check=True)
        library = ctypes.CDLL(str(staged))
        if library.t4_abi() != 1 or library.t4_shape_stride() != 182:
            raise RuntimeError('Native ABI self-check failed.')
        capabilities = library.t4_capabilities()
        staged.replace(output)
    result = {'library': str(output), 'abi': 1, 'platform': platform.platform(),
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'binary_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'command': command, 'avx2': bool(capabilities & 1), 'bmi2': bool(capabilities & 2),
              'sanitized': args.sanitize, 'compiler': subprocess.check_output(compiler + ['--version'], text=True).splitlines()[0]}
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f'Build failed: {error}', file=sys.stderr)
        raise SystemExit(1)
