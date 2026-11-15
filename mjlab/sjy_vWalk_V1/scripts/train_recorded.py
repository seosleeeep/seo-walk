"""Run train.py with a clean Git snapshot and persist provenance before launch."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    commit = git('rev-parse', 'HEAD')
    dirty = git('status', '--porcelain', '--untracked-files=normal')
    if dirty:
        raise SystemExit('Commit source changes before training:\n' + dirty)
    args = sys.argv[1:]
    if args and args[0] == '--':
        args = args[1:]
    command = [sys.executable, '-u', str(root/'scripts/train.py'), *args]
    now = datetime.now(timezone.utc)
    output = root/'run_records'/(now.strftime('%Y%m%dT%H%M%S_%fZ')+'_'+commit[:8])
    output.mkdir(parents=True)
    versions = {}
    for name in ('mjlab', 'mujoco', 'mujoco-warp', 'torch', 'rsl-rl-lib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record = dict(commit=commit, start_utc=now.isoformat(), argv=command,
        cwd=str(root), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        packages=versions, status='running', source_sha256={})
    for file in sorted((root/'src').rglob('*')):
        if file.is_file() and file.suffix in ('.py', '.json', '.xml'):
            record['source_sha256'][file.relative_to(root).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest = output/'manifest.json'
    def save():
        manifest.write_text(json.dumps(record, indent=2), encoding='utf-8')
    save()
    print('Run record:', manifest, flush=True)
    try:
        result = subprocess.run(command, cwd=root)
        record.update(status='completed' if result.returncode == 0 else 'failed', exit_code=result.returncode)
    except BaseException as exc:
        record.update(status='interrupted', error=type(exc).__name__)
        raise
    finally:
        record['end_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
