"""Check executable exit codes without changing machine settings."""
import os
import subprocess
from pathlib import Path

audit = Path(__file__).resolve().parent
root = audit.parents[1]
exe = root / 'bin' / 'benchmark_suite.exe'
cases = [
    ('unknown option', ['--not-a-supported-option'], 2, None),
    ('missing output directory argument', ['--output-dir'], 2, None),
    ('output directory is an existing file', ['--output-dir', str(exe)], 1, None),
    ('no visible CUDA device', ['--output-dir', str(audit / 'no_device')], 1,
     {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}),
]
for label, arguments, expected, env in cases:
    command = [str(exe), *arguments]
    run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    print('COMMAND:', subprocess.list2cmdline(command))
    if env is not None:
        print('Child environment: CUDA_VISIBLE_DEVICES=-1')
    print(run.stdout + run.stderr, end='')
    print(f'{label}: exit={run.returncode}, expected={expected}')
    assert run.returncode == expected
    assert 'PHASE 2 STATUS: [PASS]' not in run.stdout
assert not list((audit / 'no_device').glob('benchmark_results.*'))
print('PASS: all four negative execution cases returned failure; no false PASS or no-device result files.')
