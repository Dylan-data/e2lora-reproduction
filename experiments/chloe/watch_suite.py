"""Optional serial SLURM controller. Run in tmux on the login node, never here.

Only TIMEOUT or exit75 with a checkpoint may resume automatically. Other
failures, user cancellations and lack of progress stop the controller.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from common import digest, output_path, read_json, write_json
from run_ablations import run_directory, submission


def retryable(state, exitcode, checkpoint_exists):
    return checkpoint_exists and (state=='TIMEOUT' or (state=='FAILED' and exitcode=='75:0'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',default='outputs/chloe/suite-configs/batch.json')
    p.add_argument('--resources',default='experiments/chloe/configs/slurm-resources.json')
    p.add_argument('--state',default='outputs/chloe/suite-controller/state.json')
    p.add_argument('--indices',nargs='+',type=int,required=True)
    p.add_argument('--max-resumes',type=int,default=20)
    p.add_argument('--poll-seconds',type=int,default=30)
    args=p.parse_args()
    if not 5<=args.poll_seconds<=60:p.error('poll-seconds must be 5..60')
    if args.max_resumes<0 or len(set(args.indices))!=len(args.indices):p.error('Invalid indices/resume limit')
    path=output_path(args.state);path.parent.mkdir(parents=True,exist_ok=True)
    # OS lock survives a controller crash without leaving an irrecoverable lock.
    import fcntl
    lock=path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    batch=read_json(args.batch);resources=read_json(args.resources)
    identity=digest(dict(batch=batch,indices=args.indices,resources=resources))
    state=read_json(path) if path.exists() else dict(identity=identity,position=0,job_id='',resumes=0,last_checkpoint='',phase='ready')
    if state['identity']!=identity:raise ValueError('Controller batch/resources changed; use a new state path')
    if state['phase']=='attention':raise ValueError('Previous failure needs inspection; preserve state and use a new controller state after resolving it')
    while state['position']<len(args.indices):
        index=args.indices[state['position']];entry=batch['entries'][index]
        directory=run_directory(entry)
        if (directory/'result.json').exists() and read_json(directory/'result.json').get('status')=='completed':
            if state['job_id']:
                pending=subprocess.run(['squeue','-h','-j',state['job_id'],'-o','%T'],check=True,capture_output=True,text=True)
                if pending.stdout.strip():time.sleep(args.poll_seconds);continue
            state.update(position=state['position']+1,job_id='',resumes=0,last_checkpoint='',phase='ready');write_json(path,state);continue
        if not state['job_id']:
            queued=subprocess.run(['squeue','-h','-u',os.environ['USER'],'-o','%j'],check=True,capture_output=True,text=True).stdout.splitlines()
            if any('chloe-suite' in name for name in queued):raise ValueError('Another suite job exists; inspect it before submitting')
            if len(queued)>=resources['max_submitted_jobs']:time.sleep(args.poll_seconds);continue
            result=subprocess.run(submission(entry,args.batch,index,resources),check=True,capture_output=True,text=True)
            job=result.stdout.strip().split(';')[0]
            if not job.isdecimal():raise ValueError('Invalid job ID')
            state.update(job_id=job,phase='submitted');write_json(path,state);print(json.dumps(state),flush=True)
        queue=subprocess.run(['squeue','-h','-j',state['job_id'],'-o','%T'],check=True,capture_output=True,text=True)
        if queue.stdout.strip():time.sleep(args.poll_seconds);continue
        accounting=subprocess.run(['sacct','-X','-n','-P','-j',state['job_id'],'--format=State,ExitCode'],check=True,capture_output=True,text=True).stdout.strip().splitlines()
        if not accounting:time.sleep(args.poll_seconds);continue
        status,exitcode=accounting[0].split('|')[:2];status=status.split()[0].rstrip('+')
        if status in ['RUNNING','PENDING','COMPLETING']:time.sleep(args.poll_seconds);continue
        # Result files are authoritative for completion; accounting cannot create a score.
        if (directory/'result.json').exists() and read_json(directory/'result.json').get('status')=='completed':continue
        checkpoints=[run_directory(entry,s)/'checkpoint.pt' for s in [False,True]]
        checkpoint=next((c for c in checkpoints if c.exists()),None)
        marker='' if checkpoint is None else f'{checkpoint}:{checkpoint.stat().st_mtime_ns}:{checkpoint.stat().st_size}'
        for phase in [False,True]:
            for prediction in sorted(run_directory(entry,phase).glob('predictions-epoch*.jsonl')):
                marker+=f':{prediction}:{prediction.stat().st_size}'
        if retryable(status,exitcode,checkpoint is not None) and state['resumes']<args.max_resumes and marker!=state['last_checkpoint']:
            state.update(job_id='',resumes=state['resumes']+1,last_checkpoint=marker,phase='resuming');write_json(path,state);continue
        state.update(phase='attention',slurm_state=status,exitcode=exitcode,
                     reason='No automatic retry for this failure, missing/unchanged checkpoint, or retry limit')
        write_json(path,state);raise RuntimeError(json.dumps(state))
    state.update(phase='completed',job_id='');write_json(path,state);print(json.dumps(state),flush=True)


if __name__=='__main__':main()
