"""Verify the extracted source manifest and print precise Git staging commands.

Read only: does not execute Git, stage, commit, fetch or push anything.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='UPLOAD-MANIFEST.json');args=p.parse_args()
    entries=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    names=[]
    for entry in entries:
        name=entry['path'];path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not (name.startswith('experiments/chloe/') or name.startswith('src/e2lora/adapters/')):
            raise ValueError('Unexpected manifest path: '+name)
        content=path.read_text(encoding='utf-8').replace('\r\n','\n').encode('utf-8')
        if hashlib.sha256(content).hexdigest()!=entry['sha256']:raise ValueError('Changed source; review/re-export: '+name)
        names.append(name)
    print('Verified',len(names),'source files. Review before running these commands:')
    # One source path per command: never a broad git add directory.
    for name in names:print('git add -- '+shlex.quote(name))
    print('git diff --cached --stat');print('git diff --cached --check')
    print('# No commit/push executed. Do not stage UPLOAD-MANIFEST.json or outputs.')


if __name__=='__main__':main()
