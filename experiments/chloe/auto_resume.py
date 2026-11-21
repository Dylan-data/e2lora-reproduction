"""Remote lightweight supervisor for the two authorized seed42 baseline jobs.

Runs on the cluster, independent of the laptop/VS Code connection. Does not
modify the running trainer, cancel jobs, or retry code errors/OOM/cancellations.
"""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/chloe/ablations-math-v1'
DEFAULT_STATE=BASE/'controller/state.json'
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','RESIZING','REQUEUED','REQUEUE_FED','REQUEUE_HOLD'}
RETRY={'TIMEOUT','PREEMPTED','NODE_FAIL','BOOT_FAIL'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def command(args):
    result=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=40)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def job_state(job_id,run_command=command):
    pending=run_command(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T'])
    for row in pending.splitlines():
        fields=row.strip().split('|')
        if len(fields)==2 and fields[0]==str(job_id):
            return fields[1],None
    records=run_command(['sacct','-n','-X','-j',str(job_id),'--format=JobIDRaw,State,ExitCode','-P'])
    for row in records.splitlines():
        fields=row.strip().split('|')
        if len(fields)>=3 and fields[0]==str(job_id):
            return fields[1].split()[0].rstrip('+'),fields[2]
    return 'ACCOUNTING_PENDING',None


def retryable(state,exit_code):
    return state in RETRY or (state=='FAILED' and exit_code=='75:0')


def initial():
    return dict(schema_version=1,status='running',created_utc=now(),updated_utc=now(),
        scope='Only LoRA r4/r8 seed42, no additional experiments',max_resumes_per_run=8,
        runs=[dict(method='lora',rank=4,seed=42,job_id='56320',history=['56320'],resumes=0,completed=False),
              dict(method='lora',rank=8,seed=42,job_id='56321',history=['56321'],resumes=0,completed=False,depends_on='56320')],
        events=[])


def directory(run):
    return BASE/'formal'/f'{run["method"]}-r{run["rank"]}'/f'seed-{run["seed"]}'


def verify_sources(run):
    path=directory(run)/'run_config.json'
    if not path.exists():raise RuntimeError('No formal run manifest: refusing an unverified resume')
    manifest=read(path)
    for name,expected in manifest['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Training source changed since checkpoint: {name}')
    current=read(ROOT/'experiments/chloe/configs/math-ablation-v1.json')
    if current!=manifest['config']:raise RuntimeError('Training config changed since checkpoint')


def tick(state,persist,run_command=command,state_query=job_state):
    statuses={r['job_id']:state_query(r['job_id'],run_command) for r in state['runs'] if not r['completed']}
    for run in state['runs']:
        if not run['completed']:
            run['slurm_state'],run['exit_code']=statuses[run['job_id']]
    # If a resumed parent has a new job ID, repair the existing pending child's dependency.
    parent,child=state['runs']
    if (not child['completed'] and child.get('depends_on')!=parent['job_id']):
        if child['slurm_state']!='PENDING':
            raise RuntimeError('Downstream job is no longer pending; inspect before changing dependencies')
        run_command(['scontrol','update',f'JobId={child["job_id"]}',f'Dependency=afterok:{parent["job_id"]}'])
        child['depends_on']=parent['job_id']
        state['events'].append(dict(time=now(),action='repaired_dependency',child=child['job_id'],parent=parent['job_id']))
        persist(state)
    for run in state['runs']:
        if run['completed']:continue
        status,exit_code=run['slurm_state'],run['exit_code']
        if status in ACTIVE or status=='ACCOUNTING_PENDING':
            return
        result=directory(run)/'result.json'
        if status=='COMPLETED' and result.exists() and read(result).get('status')=='completed':
            run['completed']=True
            state['events'].append(dict(time=now(),action='completed',job_id=run['job_id'],rank=run['rank']))
            persist(state)
            continue
        if not retryable(status,exit_code):
            raise RuntimeError(f'Job {run["job_id"]}: {status}, exit {exit_code}. No automatic retry for code errors, OOM or cancellation.')
        checkpoint=directory(run)/'checkpoint.pt'
        if not checkpoint.is_file() or checkpoint.stat().st_size==0:
            raise RuntimeError('Interrupted job has no nonempty checkpoint; no automatic restart from scratch')
        if run['resumes']>=state['max_resumes_per_run']:
            raise RuntimeError('Automatic resume limit reached')
        modified=checkpoint.stat().st_mtime_ns
        if modified<=run.get('last_resume_checkpoint_mtime',0):
            raise RuntimeError('Checkpoint did not advance after previous resume; stopping retry loop')
        verify_sources(run)
        submitted=run_command(['sbatch','--parsable','experiments/chloe/slurm/math-ablation-v1.sbatch',
            run['method'],str(run['rank']),str(run['seed']),'--resume']).split(';')[0]
        if not submitted.isdigit():raise RuntimeError(f'Unexpected submission result: {submitted}')
        old=run['job_id'];run.update(job_id=submitted,slurm_state='PENDING',exit_code=None,
            resumes=run['resumes']+1,last_resume_checkpoint_mtime=modified)
        run['history'].append(submitted)
        state['events'].append(dict(time=now(),action='resume_submitted',old_job=old,new_job=submitted,rank=run['rank']))
        persist(state)  # Save immediately, before dependency mutation, to avoid duplicate submissions after restart.
        if run is parent and not child['completed']:
            if child['slurm_state']!='PENDING':raise RuntimeError('Pending successor was canceled or changed; inspect manually')
            run_command(['scontrol','update',f'JobId={child["job_id"]}',f'Dependency=afterok:{submitted}'])
            child['depends_on']=submitted
            persist(state)
        return
    state['status']='completed';persist(state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',type=Path,default=DEFAULT_STATE)
    p.add_argument('--once',action='store_true')
    args=p.parse_args()
    import fcntl
    args.state.parent.mkdir(parents=True,exist_ok=True)
    lock=(args.state.parent/'controller.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(args.state) if args.state.exists() else initial()
    if state['status']!='running':raise SystemExit(f'Controller status {state["status"]}; inspect existing report first')
    def persist(s):
        s['updated_utc']=now();write(args.state,s)
    query_failures=0
    while True:
        try:
            tick(state,persist)
            query_failures=0
        except (subprocess.TimeoutExpired,RuntimeError) as error:
            message=str(error)
            # Bounded transient SLURM command outages; never repeatedly submit on uncertain state.
            transient=isinstance(error,subprocess.TimeoutExpired) or message.startswith(('squeue failed:','sacct failed:'))
            query_failures+=1
            if transient and query_failures<10:
                state['last_query_error']=message;persist(state)
            else:
                state.update(status='needs_attention',error=message);persist(state)
                print(json.dumps(state),flush=True);raise SystemExit(1)
        persist(state)
        print(json.dumps(dict(time=now(),status=state['status'],jobs=[{k:r.get(k) for k in ['rank','job_id','slurm_state','resumes','completed']} for r in state['runs']])),flush=True)
        if args.once or state['status']=='completed':return
        time.sleep(60)


if __name__=='__main__':main()
