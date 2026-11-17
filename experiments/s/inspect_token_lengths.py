#!/usr/bin/env python3
"""Measure prompt/answer token lengths before selecting max_seq_length."""
from __future__ import annotations
import argparse, json, math, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from transformers import AutoTokenizer
from e2lora.dataset.nlg_suites import load_metamath_records, load_codefeedback_records
from e2lora.dataset.commonsense_reasoning import DATASETS, load_commonsense_split, normalize_commonsense_row

MODELS = {
    "llama31_8b": ("meta-llama/Llama-3.1-8B", "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"),
    "llama2_13b": ("meta-llama/Llama-2-13b-hf", "5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"),
    "qwen3_8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "gpt_oss_20b": ("openai/gpt-oss-20b", "6cee5e81ee83917806bbde320786a8fb61efebee"),
}

def pct(xs, p):
    if not xs: return None
    ys = sorted(xs); x = (len(ys)-1)*p; lo=math.floor(x); hi=math.ceil(x)
    return ys[lo] if lo==hi else round(ys[lo]*(hi-x)+ys[hi]*(x-lo))

def stats(xs):
    return {"count":len(xs),"min":min(xs) if xs else None,"mean":statistics.mean(xs) if xs else None,
            "p50":pct(xs,.5),"p90":pct(xs,.9),"p95":pct(xs,.95),"p99":pct(xs,.99),"max":max(xs) if xs else None}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("math","code",*DATASETS), required=True)
    p.add_argument("--model-key",choices=tuple(MODELS),default="llama31_8b")
    p.add_argument("--code-jsonl",type=Path)
    p.add_argument("--cache-dir",type=Path,default=None)
    p.add_argument("--allow-unpinned-datasets",action="store_true")
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--candidates",default="512,1024,2048,4096")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    args=p.parse_args()
    if args.task=="math":
        rows, _, source=load_metamath_records(train_count=args.limit or 100000,validation_count=0,cache_dir=args.cache_dir)
    elif args.task=="code":
        if args.code_jsonl is None: raise ValueError("--code-jsonl is required for Code-Feedback length inspection")
        from e2lora.dataset.nlg_suites import _records_from_jsonl, _normalize_code_row
        raw=_records_from_jsonl(args.code_jsonl)
        rows=[x for r in raw if (x:=_normalize_code_row(r)) is not None]
        if args.limit: rows=rows[:args.limit]
        source={"dataset":"Code-Feedback local JSONL","path":str(args.code_jsonl),"rows":len(rows)}
    elif args.task in DATASETS:
        train=load_commonsense_split(args.task,"train",cache_dir=args.cache_dir,allow_unpinned=args.allow_unpinned_datasets)
        rows=[normalize_commonsense_row(args.task,dict(r)) for r in train]
        if args.limit: rows=rows[:args.limit]
        source={**DATASETS[args.task],"rows":len(rows)}
    else: raise ValueError(args.task)
    repo,revision=MODELS[args.model_key]
    tokenizer=AutoTokenizer.from_pretrained(repo,revision=revision,
        cache_dir=str(args.cache_dir) if args.cache_dir else None)
    prompt_lengths=[]; target_lengths=[]; total_lengths=[]
    for row in rows:
        prompt=tokenizer(row["prompt"],add_special_tokens=True,truncation=False)["input_ids"]
        target=tokenizer(row["target"],add_special_tokens=False,truncation=False)["input_ids"]
        if tokenizer.eos_token_id is not None and row["target"]: target=list(target)+[tokenizer.eos_token_id]
        prompt_lengths.append(len(prompt));target_lengths.append(len(target));total_lengths.append(len(prompt)+len(target))
    candidates=[int(x) for x in args.candidates.split(",") if x.strip()]
    report={"task":args.task,"model":{"repo":repo,"revision":revision},"source":source,
            "prompt_tokens":stats(prompt_lengths),"answer_tokens_including_eos":stats(target_lengths),
            "combined_tokens":stats(total_lengths),"candidate_checks":[]}
    for cap in candidates:
        answer_too_long=sum(x>=cap for x in target_lengths)
        report["candidate_checks"].append({"max_seq_length":cap,
            "answer_alone_too_long_to_fit":answer_too_long,
            "examples_requiring_prompt_left_truncation":sum(p+t>cap and t<cap for p,t in zip(prompt_lengths,target_lengths)),
            "naive_right_truncation_risk":sum(p+t>cap for p,t in zip(prompt_lengths,target_lengths))})
    report["selected_max_seq_length"]=None
    report["note"]="Selection is intentionally not automated; choose the smallest cap that preserves every target answer and has acceptable prompt truncation."
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
