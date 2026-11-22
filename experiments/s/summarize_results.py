#!/usr/bin/env python3
"""Aggregate Table 2 / Table 14 / Table 15 run outputs across seeds."""
from __future__ import annotations
import argparse, csv, json, math, statistics
from pathlib import Path

TABLE2_METHODS=["fft_attn_only","dora_r4","dora_r8","adalora_r4","adalora_r8","lora_plus","lora_fa","vera","lora","e2lora","sharelora","e2sharelora"]
TABLE14_TASKS=["arc_challenge","arc_easy","boolq","openbookqa","piqa","siqa"]
TABLE15_MODELS=["llama2_13b","qwen3_8b","gpt_oss_20b"]
SEEDS=[0,18,42]
CONFIGS=Path(__file__).parent / "configs"

def load_config(name):
    path=CONFIGS/name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def metric_from_run(path, task):
    result=read_json(path/"result.json")
    if not result: return None, None
    final=result.get("final_evaluation")
    if not final:
        final=read_json(path/"final-evaluation"/"result.json")
    if not final: return None, result.get("parameter_counts",{}).get("trainable_total_params")
    if task=="code":
        # The official harness output can be a dict or JSON-encoded text depending on package version.
        harness=final.get("official_harness",{})
        raw=harness.get("official_harness_result") if isinstance(harness,dict) else None
        score=None
        if isinstance(raw,dict): score=raw.get("pass@1") or raw.get("pass_at_1")
        elif isinstance(raw,str):
            try:
                parsed=json.loads(raw); score=parsed.get("pass@1") or parsed.get("pass_at_1")
            except Exception: pass
        if score is None:
            score=final.get("pass@1") or final.get("pass_at_1")
        # Official HumanEval pass@1 is returned in [0,1]; paper tables show percent.
        return (100.0 * float(score) if score is not None else None,
                result.get("parameter_counts",{}).get("trainable_total_params"))
    score=final.get("accuracy")
    # Run metrics are stored as fractions; Table 2/14/15 paper values are percentages.
    return (100.0 * float(score) if score is not None else None,
            result.get("parameter_counts",{}).get("trainable_total_params"))

def mean_std(values):
    values=[float(x) for x in values if x is not None]
    if not values:return None,None,0
    return statistics.mean(values), (statistics.stdev(values) if len(values)>1 else 0.0), len(values)

def fmt(value): return "" if value is None else f"{value:.4f}"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",type=Path,default=Path("outputs/e2lora-all-suites"))
    p.add_argument("--output-dir",type=Path,default=Path("outputs/e2lora-all-suites/summary"))
    args=p.parse_args(); root=args.output_root; out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    ref2=load_config("table2_nlg.json").get("reference_table2", {})
    ref14=load_config("table14_commonsense.json")
    ref15=load_config("table15_larger_models.json").get("models", {})
    # Table 2
    t2=[]
    for task in ("math","code"):
        for method in TABLE2_METHODS:
            scores=[]; params=[]
            for seed in SEEDS:
                score,param=metric_from_run(root/"table2"/task/method/f"seed_{seed}",task)
                scores.append(score); params.append(param)
            avg,std,n=mean_std(scores); pavg,_,pn=mean_std(params)
            reference=ref2.get(task,{}).get(method,{})
            paper_score=reference.get("score")
            row={"table":"Table2","task":task,"method":method,"metric":"accuracy" if task=="math" else "pass@1",
                 "score_unit":"percent","score_mean":avg,"score_std":std,"score_n":n,
                 "paper_reference_score":paper_score,
                 "delta_vs_paper":(avg-float(paper_score)) if avg is not None and paper_score is not None else None,
                 "paper_reference_params":reference.get("params"),
                 "trainable_params_mean":pavg,"parameter_n":pn}
            t2.append(row);all_rows.append(row)
    # Table 14: per-task and arithmetic average of six subtask accuracies, computed per seed first.
    t14=[]
    for method in ("lora","e2lora"):
        seed_averages=[]; all_params=[]; task_detail={}
        for task in TABLE14_TASKS:
            scores=[]; params=[]
            for seed in SEEDS:
                score,param=metric_from_run(root/"table14"/task/method/f"seed_{seed}",task)
                scores.append(score);params.append(param)
            m,s,n=mean_std(scores);pm,_,pn=mean_std(params)
            task_cfg=ref14.get("tasks",{}).get(task,{})
            reference_score=task_cfg.get("reference_lora" if method=="lora" else "reference_e2lora")
            task_detail[task]={"score_mean":m,"score_std":s,"score_n":n,"score_unit":"percent",
                               "paper_reference_score":reference_score,
                               "delta_vs_paper":(m-float(reference_score)) if m is not None and reference_score is not None else None,
                               "params_mean":pm}
            all_params.extend([x for x in params if x is not None])
        for seed in SEEDS:
            one=[]
            for task in TABLE14_TASKS:
                score,_=metric_from_run(root/"table14"/task/method/f"seed_{seed}",task)
                if score is not None: one.append(score)
            if len(one)==len(TABLE14_TASKS): seed_averages.append(sum(one)/len(one))
        avg,std,n=mean_std(seed_averages);pavg,_,pn=mean_std(all_params)
        reference_avg=ref14.get("reference_average",{}).get("lora" if method=="lora" else "e2lora")
        row={"table":"Table14","method":method,"metric":"arithmetic mean accuracy across six tasks",
             "score_unit":"percent","score_mean":avg,"score_std":std,"score_n":n,
             "paper_reference_score":reference_avg,
             "delta_vs_paper":(avg-float(reference_avg)) if avg is not None and reference_avg is not None else None,
             "trainable_params_mean":pavg,"parameter_n":pn,"per_task":task_detail}
        t14.append(row);all_rows.append(row)
    # Table 15
    t15=[]
    for model in TABLE15_MODELS:
        for method in ("lora","e2lora"):
            scores=[];params=[]
            for seed in SEEDS:
                score,param=metric_from_run(root/"table15"/model/method/f"seed_{seed}","math")
                scores.append(score);params.append(param)
            avg,std,n=mean_std(scores);pavg,_,pn=mean_std(params)
            model_cfg=ref15.get(model,{})
            paper_score=model_cfg.get("reference_lora_accuracy" if method=="lora" else "reference_e2lora_accuracy")
            paper_params=model_cfg.get("reference_lora_params" if method=="lora" else "reference_e2lora_params")
            row={"table":"Table15","model":model,"method":method,"metric":"GSM8K accuracy",
                 "score_unit":"percent","score_mean":avg,"score_std":std,"score_n":n,
                 "paper_reference_score":paper_score,
                 "delta_vs_paper":(avg-float(paper_score)) if avg is not None and paper_score is not None else None,
                 "paper_reference_params":paper_params,
                 "trainable_params_mean":pavg,"parameter_n":pn}
            t15.append(row);all_rows.append(row)
    (out/"summary.json").write_text(json.dumps({"table2":t2,"table14":t14,"table15":t15},indent=2,ensure_ascii=False),encoding="utf-8")
    # Flat CSV omits nested per-task details.
    flat=[{k:v for k,v in r.items() if k!="per_task"} for r in all_rows]
    fields=sorted({k for r in flat for k in r})
    with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
    print(f"Wrote {out/'summary.json'} and {out/'summary.csv'}")
    print("Scores in summary.csv/json are percentages (0-100), matching the paper tables. Missing values mean runs/results are incomplete.")
if __name__=="__main__":main()
