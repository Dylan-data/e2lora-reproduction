"""Compile JSON-driven math suite; submit only on explicit `submit` command.

No implicit seeds, account, partition or repeated failures. `resume` uses saved
plans/checkpoints and never recomputes gradients. Run from repository root.
"""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from common import digest, output_path, read_json, resolve, write_json
from suite_plan import null_paths, specifications, validate


def compile_suite(config, setup, directory):
    runs=specifications(config,setup)
    directory=output_path(directory)
    entries=[]
    for run in runs:
        path=directory/(run['run_id']+'.json')
        if path.exists() and read_json(path)!=run:
            raise ValueError(f'Preserve changed config: {path}')
        entries.append(dict(config=str(path.relative_to(ROOT)),run_id=run['run_id'],
                            method=run['run']['method'],rank=run['run']['rank'],
                            sampling_steps=run['run']['sampling_steps'],seed=run['run']['seed'],
                            studies=run['run']['studies'],config_sha256=digest(run)))
    manifest=dict(schema_version=2,entries=entries,source_config_sha256=digest(config),
                  reference_nulls=null_paths(setup),gpu_jobs_submitted=0,
                  explicit_resolution=config['suite'],seeds=config['seeds'])
    if (directory/'batch.json').exists() and read_json(directory/'batch.json')!=manifest:
        raise ValueError('Existing batch differs; select a new directory')
    for run,e in zip(runs,entries):write_json(resolve(e['config']),run)
    write_json(directory/'batch.json',manifest)
    return manifest


def command(entry, smoke=False, resume=False):
    argv=[sys.executable,'-u','experiments/chloe/suite_train.py','--config',entry['config'],
          '--method',entry['method'],'--rank',str(entry['rank']),'--seed',str(entry['seed']),
          '--sampling-steps',str(entry['sampling_steps'])]
    if smoke:argv.append('--smoke')
    if resume:argv.append('--resume')
    return argv


def run_directory(entry, smoke=False):
    cfg=read_json(entry['config'])
    if digest(cfg)!=entry['config_sha256']:raise ValueError('Compiled run changed')
    phase='smoke' if smoke else 'formal'
    return output_path(f"{cfg['output_root']}/{phase}/{entry['method']}-r{entry['rank']}-n{entry['sampling_steps']}/seed-{entry['seed']}")


def smoke_ok(result):
    return (result.get('status')=='completed' and result.get('smoke') is True
            and result.get('fixed_components_unchanged') is True
            and result.get('source_training_manifest_verified') is True
            and result.get('evaluation',{}).get('evaluation_examples')==8
            and result.get('evaluation',{}).get('empty_outputs')==0
            and result.get('evaluation',{}).get('unparseable_predictions')==0)


def execute(entry, smoke_only=False):
    """Smoke is separate from formal, including when either phase resumes."""
    smoke_dir=run_directory(entry,True)
    result_path=smoke_dir/'result.json'
    if not result_path.exists():
        run_child(command(entry,True,(smoke_dir/'checkpoint.pt').exists()))
    if not smoke_ok(read_json(result_path)):
        raise ValueError('Smoke failed structural/output checks; inspect generated examples before formal training')
    if smoke_only:return
    formal=run_directory(entry)
    if (formal/'result.json').exists():
        result=read_json(formal/'result.json')
        if result.get('status')=='completed' and result.get('smoke') is False:return
        raise ValueError('Existing result is not a completed formal run')
    run_child(command(entry,resume=(formal/'checkpoint.pt').exists()))


def run_child(argv):
    child=subprocess.Popen(argv,cwd=ROOT)
    previous={}
    def forward(signum,_frame):
        if child.poll() is None:child.send_signal(signum)
    for name in ['SIGTERM','SIGUSR1']:
        if hasattr(signal,name):
            signum=getattr(signal,name);previous[signum]=signal.signal(signum,forward)
    try:
        code=child.wait()
        if code:raise subprocess.CalledProcessError(code,argv)
    finally:
        for signum,handler in previous.items():signal.signal(signum,handler)


def submission(entry, batch, index, resources, dependency=None):
    if null_paths(resources):raise ValueError('Unresolved cluster resources: '+', '.join(null_paths(resources)))
    if not resources['inspection_file'] or not resolve(resources['inspection_file']).is_file():
        raise ValueError('Run slurm/inspect_cluster.sh and retain actual resource evidence first')
    argv=['sbatch','--parsable',f"--partition={resources['partition']}",f"--account={resources['account']}",
          f"--qos={resources['qos']}",f"--cpus-per-task={resources['cpus']}",f"--mem={resources['memory']}",
          f"--time={resources['time']}"]
    if dependency:argv.append('--dependency=afterok:'+dependency)
    argv+=['experiments/chloe/slurm/train.sbatch',str(resolve(batch)),str(index)]
    return argv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['compile','status','run','submit','resume'])
    p.add_argument('--config',default='experiments/chloe/configs/suite-math.json')
    p.add_argument('--setup',default='references/ablation-setup.json')
    p.add_argument('--directory',default='outputs/chloe/suite-configs')
    p.add_argument('--batch',default='outputs/chloe/suite-configs/batch.json')
    p.add_argument('--index',type=int)
    p.add_argument('--resources',default='experiments/chloe/configs/slurm-resources.json')
    p.add_argument('--smoke-only',action='store_true')
    args=p.parse_args()
    if args.action=='compile':
        result=compile_suite(read_json(args.config),read_json(args.setup),args.directory)
        print(json.dumps(dict(runs=len(result['entries']),seeds=result['seeds'],gpu_jobs_submitted=0,
                              unresolved_reference_fields=result['reference_nulls']),indent=2));return
    batch=read_json(args.batch)
    if args.index is not None and not 0<=args.index<len(batch['entries']):p.error('index out of range')
    entries=list(enumerate(batch['entries']))
    if args.index is not None:entries=[entries[args.index]]
    if args.action=='run':
        if args.index is None:p.error('run requires --index')
        try:execute(entries[0][1],args.smoke_only)
        except subprocess.CalledProcessError as error:raise SystemExit(error.returncode)
        return
    if args.action=='status':
        for i,e in entries:
            path=run_directory(e)
            state=read_json(path/'status.json') if (path/'status.json').exists() else {'stage':'not_started'}
            print(json.dumps(dict(index=i,run_id=e['run_id'],**state)))
        return
    # Serial submissions respect the observed one-GPU user allocation. Do not
    # auto-retry OOM/config errors; resume explicitly after inspecting accounting.
    resources=read_json(args.resources)
    if len(entries)>resources['max_submitted_jobs']:
        raise ValueError('Select --index: submitting this many exceeds observed user job quota')
    dependency=None
    for i,e in entries:
        directory=run_directory(e)
        if (directory/'result.json').exists():continue
        if args.action=='resume' and not any((run_directory(e,s)/'checkpoint.pt').exists() for s in [False,True]):
            raise ValueError('Resume requires a checkpoint')
        queue=subprocess.run(['squeue','-h','-u',os.environ['USER'],'-o','%j'],check=True,capture_output=True,text=True)
        if len(queue.stdout.splitlines())>=resources['max_submitted_jobs']:
            raise ValueError('User submission quota is full; no job submitted')
        if 'chloe-suite' in queue.stdout:
            raise ValueError('A suite job is already queued/running; prevent duplicate submissions')
        result=subprocess.run(submission(e,args.batch,i,resources,dependency),check=True,capture_output=True,text=True)
        dependency=result.stdout.strip().split(';')[0]
        if not dependency.isdecimal():raise ValueError('Unexpected sbatch response: '+result.stdout)
        output_path(args.directory).mkdir(parents=True,exist_ok=True)
        with output_path(args.directory).joinpath('submissions.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(job_id=dependency,index=i,run_id=e['run_id']))+'\n')
        print(json.dumps(dict(job_id=dependency,run_id=e['run_id'])))


if __name__=='__main__':main()
