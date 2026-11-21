"""Allowlisted source-only archive; never stage, commit or push."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[2]
SHARED=['adaptive.py','lora_layers.py','ablation.py','grouping.py','partial_rank.py','preparation.py','similarity.py']
EXTENSIONS={'.py','.json','.md','.csv','.sh','.sbatch','.txt'}
SECRET=re.compile(r'(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def source_files(root):
    files=[p for p in (root/'experiments/chloe').rglob('*') if p.is_file() and p.suffix in EXTENSIONS
           and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
    return sorted(set(files+[root/'src/e2lora/adapters'/n for n in SHARED]))


def export(root,destination):
    root=Path(root).resolve();destination=Path(destination).resolve()
    destination.parent.mkdir(parents=True,exist_ok=True)
    payload={};manifest=[]
    for path in source_files(root):
        name=path.relative_to(root).as_posix();content=path.read_bytes()
        if len(content)>512000:raise ValueError('Unexpected large source file: '+name)
        text=content.decode('utf-8')
        if SECRET.search(text):raise ValueError('Possible credential found: '+name)
        content=text.replace('\r\n','\n').encode('utf-8');payload[name]=content
        manifest.append(dict(path=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for name,content in payload.items():z.writestr(name,content)
        z.writestr('UPLOAD-MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/chloe/review/chloe-source.zip');args=p.parse_args()
    from common import output_path
    path=output_path(args.output);manifest=export(ROOT,path)
    print(json.dumps(dict(files=len(manifest),zip=str(path),bytes=path.stat().st_size,git_actions=0)))


if __name__=='__main__':main()
