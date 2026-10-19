import difflib
import hashlib
import json
from pathlib import Path

audit = Path(__file__).resolve().parent
root = audit.parents[1]
patch = []
for original in sorted((audit / 'original_sources').rglob('*')):
    if original.is_file():
        relative = original.relative_to(audit / 'original_sources')
        current = root / relative
        patch.extend(difflib.unified_diff(original.read_text().splitlines(True), current.read_text().splitlines(True),
                                          fromfile=f'original/{relative.as_posix()}', tofile=relative.as_posix()))
regression = root / 'tests' / 'benchmark_regression.cu'
patch.extend(difflib.unified_diff([], regression.read_text().splitlines(True),
                                 fromfile='/dev/null', tofile='tests/benchmark_regression.cu'))
(audit / 'changes.patch').write_text(''.join(patch))
manifest = {}
paths = [root / 'bin' / 'benchmark_suite.exe', root / 'bin' / 'benchmark_regression.exe', root / 'CMakeLists.txt']
paths += list((root / 'src').glob('*')) + list((root / 'include').glob('*')) + [regression]
paths += list((audit / 'final_run').glob('*')) + list((audit / 'original_run').glob('*.exe'))
for path in paths:
    if path.is_file():
        manifest[str(path.relative_to(root)).replace('\\', '/')] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size,
        }
for original in (audit / 'prior_results').iterdir():
    current = root / 'results' / original.name
    assert original.read_bytes() == current.read_bytes(), f'Original artifact changed: {current}'
(audit / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Captured SHA-256 for {len(manifest)} source/binary/result files and changes.patch; all six original result files unchanged.')
