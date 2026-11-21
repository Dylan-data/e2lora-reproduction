"""Measured results only: Table4/5, Figure5 and per-epoch validation curves."""
import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import statistics

from common import digest, output_path, read_json, write_json

PAPER4={
 'sharing_only':(3.51,69.45),'rank_allocation_only':(6.82,71.19),
 'cosine_with_rank_allocation':(3.75,66.67),'l1_with_rank_allocation':(3.75,68.48),
 'l2_with_rank_allocation':(3.75,67.93),'kl_with_rank_allocation':(3.43,67.57),'e2lora':(3.61,70.51)}
PAPER5={('lora',4):(3.41,67.58),('lora',8):(6.82,70.21),('e2lora',8):(3.61,70.51),('e2lora',16):(7.22,71.19)}


def collect(roots):
    rows=[];skipped=[];seen={}
    for root in roots:
        for path in sorted(Path(root).rglob('result.json')):
            r=json.loads(path.read_text(encoding='utf-8'))
            if r.get('status')!='completed' or r.get('smoke') is not False or r.get('eligible_for_formal_tables') is not True:
                skipped.append(dict(path=str(path),reason='not completed eligible formal result'));continue
            e=r.get('evaluation',{})
            score=e.get('accuracy')
            if e.get('evaluation_examples')!=1319 or not e.get('ordered_team_manifest_verified') or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
                skipped.append(dict(path=str(path),reason='incomplete or incompatible GSM8K evaluation'));continue
            method,rank,seed=r['method'],r['base_rank'],r['seed']
            # Legacy baselines are shown separately until protocol equivalence is
            # reviewed; filenames/rank labels alone never establish equivalence.
            protocol=r['protocol_sha256']
            steps=r.get('sampling_steps',0 if method=='lora' else 'legacy-unspecified')
            key=(protocol,method,rank,steps,seed)
            checkpoint=r['checkpoint_sha256']
            if key in seen:
                if seen[key]!=checkpoint:raise ValueError(f'Conflicting independent results for {key}')
                continue
            seen[key]=checkpoint
            rows.append(dict(protocol=protocol,method=method,rank=rank,sampling_steps=steps,seed=seed,
                accuracy_percent=score*100,parameters=r['parameters']['trainable_adapter_parameters'],
                empty_outputs=e['empty_outputs'],unparseable=e['unparseable_predictions'],
                studies=r.get('studies',['table5'] if method=='lora' else []),path=str(path)))
    return rows,skipped


def aggregate(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['protocol'],r['method'],r['rank'],r['sampling_steps'])].append(r)
    result=[]
    for (protocol,method,rank,steps),values in sorted(grouped.items(),key=lambda x:str(x[0])):
        scores=[v['accuracy_percent'] for v in values]
        result.append(dict(protocol=protocol,method=method,rank=rank,sampling_steps=steps,n=len(values),
            seeds=[v['seed'] for v in values],accuracy_percent=statistics.mean(scores),
            std_percent=statistics.stdev(scores) if len(scores)>1 else None,
            parameters_mean=statistics.mean(v['parameters'] for v in values),
            parameters_min=min(v['parameters'] for v in values),parameters_max=max(v['parameters'] for v in values),
            studies=sorted({s for v in values for s in v['studies']})))
    return result


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def report(rows,aggregates,directory,plots=True):
    lines=['# Chloe math ablations','',
        'Only completed full GSM8K runs appear below. Different protocol hashes remain separate. ',
        'Single-seed scores are n=1; no invented standard deviation. Missing entries stay pending.',
        'Partial freezing and exact budget correction are team reproduction choices. Paper values below follow the team report.','']
    for study,paper in [('table4',PAPER4),('table5',PAPER5)]:
        lines+=['## '+study,'','|Protocol|Method|Rank|n|Parameters (M), measured|GSM8K %, measured|Paper M / %|',
                '|---|---|---:|---:|---:|---:|---:|']
        selected=[r for r in aggregates if study in r['studies']]
        for r in selected:
            ref=paper.get(r['method'] if study=='table4' else (r['method'],r['rank']))
            refstr=f'{ref[0]:.2f} / {ref[1]:.2f}' if ref else '—'
            score=f"{r['accuracy_percent']:.2f}"
            if r['std_percent'] is not None:score+=f" ± {r['std_percent']:.2f}"
            lines.append(f"|{r['protocol'][:12]}|{r['method']}|{r['rank']}|{r['n']}|{r['parameters_mean']/1e6:.4f}|{score}|{refstr}|")
        if not selected:lines+=['','Pending: no completed eligible results.']
        lines+=['']
    comparisons=[]
    for protocol in sorted({r['protocol'] for r in rows}):
        for a,b in [(4,8),(8,16)]:
            left=[r for r in aggregates if r['protocol']==protocol and r['method']=='lora' and r['rank']==a and 'table5' in r['studies']]
            right=[r for r in aggregates if r['protocol']==protocol and r['method']=='e2lora' and r['rank']==b and 'table5' in r['studies']]
            if left and right:
                gap=abs(right[0]['parameters_mean']/left[0]['parameters_mean']-1)
                comparisons.append(dict(protocol=protocol,lora_rank=a,e2lora_rank=b,relative_parameter_gap=gap,
                    note='Measured gap disclosed; accuracy is not discarded to enforce a paper match'))
    write_json(directory/'table5-budget-gaps.json',comparisons)
    (directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_csv(directory/'runs.csv',[dict(r,studies=','.join(r['studies'])) for r in rows])
    write_json(directory/'aggregates.json',aggregates)
    if not plots:return
    os.environ.setdefault('MPLCONFIGDIR',str(directory/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for study,xkey in [('figure5_rank','rank'),('figure5_steps','sampling_steps')]:
        selected=[r for r in aggregates if study in r['studies']]
        if not selected:continue
        fig,ax=plt.subplots(figsize=(6,4))
        for protocol in sorted({r['protocol'] for r in selected}):
            group=sorted([r for r in selected if r['protocol']==protocol],key=lambda r:r[xkey])
            ax.errorbar([r[xkey] for r in group],[r['accuracy_percent'] for r in group],
                yerr=[r['std_percent'] or 0 for r in group],marker='o',label=protocol[:12],capsize=3)
        ax.set(xlabel='Base rank' if xkey=='rank' else 'Logical gradient sampling steps',ylabel='GSM8K accuracy (%)')
        ax.set_xscale('log',base=2);ax.legend(title='Protocol');ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(directory/(study+'.png'),dpi=200);fig.savefig(directory/(study+'.pdf'));plt.close(fig)
    # Per-epoch curves remain per-run and do not pool epochs as independent seeds.
    fig,ax=plt.subplots(figsize=(6,4));plotted=False
    for r in rows:
        points=[]
        for path in Path(r['path']).parent.glob('validation-epoch*.json'):
            value=json.loads(path.read_text(encoding='utf-8'))
            points.append((int(path.stem.split('epoch')[1]),value['loss']))
        if points:
            points.sort();ax.plot(*zip(*points),marker='o',label=f"{r['method']} r{r['rank']} seed{r['seed']} {r['protocol'][:6]}");plotted=True
    if plotted:
        ax.set(xlabel='Epoch',ylabel='Held-out token loss');ax.legend(fontsize=6);fig.tight_layout();fig.savefig(directory/'validation.png',dpi=200)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',nargs='+',default=['outputs/chloe/suite-v2','outputs/chloe/ablations-math-v1'])
    p.add_argument('--output',default='outputs/chloe/summary')
    p.add_argument('--no-plots',action='store_true')
    args=p.parse_args();directory=output_path(args.output);directory.mkdir(parents=True,exist_ok=True)
    rows,skipped=collect(args.roots);report(rows,aggregate(rows),directory,not args.no_plots)
    write_json(directory/'excluded.json',skipped)
    print(json.dumps(dict(eligible_runs=len(rows),excluded=len(skipped),report=str(directory/'report.md'))))


if __name__=='__main__':main()
