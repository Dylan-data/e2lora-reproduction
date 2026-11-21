"""Chloe's independent Table 4/5 math runs, sharing one audited protocol.

Public E2 kernels and partial-rank adapters are reused. No base weights are saved.
The first release covers LoRA r4/r8 and team-partial E2 r8; other policies stay explicit.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from common import read_json, write_json, resolve, output_path, digest, file_sha256

STOP = False


def stop_requested(*_):
    global STOP
    STOP = True


def row_hash(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def normalized(row, index):
    return dict(prompt=str(row['query']).strip()+'\n', target=str(row['response']).strip(),
                sample_id=str(row.get('id',index)), source_index=index)


def encode(row, tokenizer, maximum):
    prompt = tokenizer.encode(row['prompt'], add_special_tokens=True)
    answer = tokenizer.encode(row['target'], add_special_tokens=False) + [tokenizer.eos_token_id]
    if not row['target'] or len(answer) >= maximum:
        raise ValueError(f'Empty/overlength response for {row["sample_id"]}; do not truncate answers')
    trim = max(0,len(prompt)+len(answer)-maximum)
    prompt = prompt[trim:]
    return dict(input_ids=prompt+answer, labels=[-100]*len(prompt)+answer, prompt_trimmed=trim)


def lr_multiplier(step, total_steps, warmup):
    # Matches member4 suite_train.py's recorded schedule: floor warmup, (step+1)/warmup.
    if warmup and step < warmup:
        return max(1e-8,(step+1)/warmup)
    progress = (step-warmup)/max(1,total_steps-warmup)
    return max(0.,.5*(1+math.cos(math.pi*min(1.,progress))))


def data(cfg, tokenizer, smoke):
    raw = read_json(cfg['data']['source'])
    manifest = read_json(cfg['data']['manifest'])
    if file_sha256(resolve(cfg['data']['source'])) != manifest['source_sha256']:
        raise ValueError('Training source differs from fixed manifest')
    ids = manifest['indices']
    expected = random.Random(cfg['data']['sampling_seed']).sample(range(len(raw)),100000)
    if ids != expected or manifest['source_revision'] != cfg['data']['revision']:
        raise ValueError('Fixed 100K selection differs from team sampling')
    records = [normalized(raw[i],i) for i in ids]
    fingerprint = hashlib.sha256(''.join(row_hash(r) for r in records).encode()).hexdigest()
    if fingerprint != cfg['data']['member4_normalized_train_sha256']:
        raise ValueError(f'Normalized data differs from uploaded member4 manifest: {fingerprint}')
    training_set = set(ids)
    remaining = [i for i in range(len(raw)) if i not in training_set]
    valid_ids = random.Random(cfg['data']['validation_seed']).sample(remaining,2000)
    validations = [normalized(raw[i],i) for i in valid_ids]
    if smoke:
        records = records[:128]
        validations = validations[:16]
    maximum = cfg['training']['max_length']
    train = [encode(row,tokenizer,maximum) for row in records]
    valid = [encode(row,tokenizer,maximum) for row in validations]
    metadata = dict(train_count=len(train),validation_count=len(valid),
        normalized_train_sha256=fingerprint,manifest_sha256=file_sha256(resolve(cfg['data']['manifest'])),
        train_indices=ids[:len(train)],validation_indices=valid_ids[:len(valid)],
        response_truncated=0,prompt_trimmed_examples=sum(x['prompt_trimmed']>0 for x in train),
        maximum_training_tokens=max(len(x['input_ids']) for x in train))
    return train, valid, metadata


def validation_loss(model, examples, tokenizer):
    import torch
    from common import collate
    model.eval()
    numerator=denominator=0
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for example in examples:
            batch={k:v.to('cuda') for k,v in collate([example],tokenizer.pad_token_id).items()}
            tokens=int(batch['labels'][:,1:].ne(-100).sum())
            loss=model(**batch).loss
            if not torch.isfinite(loss): raise ValueError('Nonfinite held-out loss')
            numerator+=loss.float().item()*tokens
            denominator+=tokens
    return numerator/denominator


def evaluate(model, tokenizer, cfg, directory, smoke, epoch):
    import torch
    import pyarrow.parquet as pq
    from transformers import GenerationConfig
    from evaluate import generate_completion,score_records
    e=cfg['evaluation']
    rows=pq.read_table(resolve(e['local_parquet'])).to_pylist()
    if len(rows)!=1319:raise ValueError('Expected 1319 pinned GSM8K test rows')
    # Check ordered normalized eval row hashes against member4's uploaded actual manifest.
    expected=read_json(e['member4_manifest'])['samples']
    normalized_eval=[dict(question=r['question'],answer=r['answer'],prompt=f'Question: {r["question"]}\nAnswer:',target=r['answer'])
                     for i,r in enumerate(rows)]
    actual=[row_hash(r) for r in normalized_eval]
    if actual != [r['row_sha256'] for r in expected]:
        raise ValueError('GSM8K content/order/prompt differs from uploaded team evaluation manifest')
    rows=rows[:8] if smoke else rows
    generation=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=e['max_new_tokens'],
        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,use_cache=True)
    model.eval()
    records=[]
    path=directory/f'predictions-epoch{epoch}.jsonl'
    with path.open('w',encoding='utf-8') as f:
        for i,row in enumerate(rows):
            completion=generate_completion(model,tokenizer,e['prompt'].format(question=row['question']),
                max_prompt_length=cfg['training']['max_length'],generation_config=generation)
            record=dict(source_index=i,gold=row['answer'],**completion)
            records.append(record)
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            if (i+1)%100==0:write_json(directory/'status.json',dict(stage='gsm8k',evaluated=i+1,total=len(rows)))
    scores,scored=score_records(records)
    scores.update(empty_outputs=sum(not r['prediction'].strip() for r in records),
        epoch=epoch,smoke=smoke,dataset_revision=e['revision'],
        ordered_team_manifest_verified=True,
        repeated_long_line_cases=[r['source_index'] for r in records if repeated_lines(r['prediction'])])
    write_json(directory/f'evaluation-epoch{epoch}.json',scores)
    write_json(directory/f'generated-examples-epoch{epoch}.json',scored[:8])
    return scores


def repeated_lines(text):
    from collections import Counter
    return any(n>1 for n in Counter(s.strip() for s in text.splitlines() if len(s.strip())>=80).values())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--method',choices=['lora','e2lora'],required=True)
    p.add_argument('--rank',type=int,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    import torch
    from common import load_local_model,seed_everything,collate
    from train import require_gpu_allocation,load_complete_adapter_state
    from e2lora.adapters.lora_layers import apply_plan
    from e2lora.adapters.partial_rank import adapter_state,count_adapters,active_rank_plan
    from team_rank_check import exact_ranks,attach,audit,fixed_snapshot,assert_fixed
    require_gpu_allocation()
    cfg=read_json(args.config)
    if args.method=='e2lora' and str(args.rank) not in cfg['partial_rank_budgets']:
        raise ValueError('No explicitly configured partial-rank budget for this base rank')
    phase='smoke' if args.smoke else 'formal'
    directory=output_path(f'{cfg["output_root"]}/{phase}/{args.method}-r{args.rank}/seed-{args.seed}')
    if directory.exists() and any(directory.iterdir()) and not args.resume:
        raise FileExistsError(f'Preserve existing run: {directory}')
    directory.mkdir(parents=True,exist_ok=True)
    source_files=['experiments/chloe/formal_math.py','experiments/chloe/common.py','experiments/chloe/evaluate.py',
        'experiments/chloe/team_rank_check.py','src/e2lora/adapters/adaptive.py',
        'src/e2lora/adapters/lora_layers.py','src/e2lora/adapters/partial_rank.py','src/e2lora/adapters/preparation.py']
    source_hashes={n:file_sha256(resolve(n)) for n in source_files}
    identity=digest(dict(config=cfg,method=args.method,rank=args.rank,seed=args.seed,smoke=args.smoke,sources=source_hashes))
    checkpoint_path=directory/'checkpoint.pt'
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False) if args.resume else None
    if saved and saved['identity']!=identity:raise ValueError('Resume protocol changed')
    write_json(directory/'status.json',dict(stage='loading',phase=phase))
    seed_everything(args.seed)
    model,tokenizer=load_local_model(cfg,'cuda:0')
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    train,valid,metadata=data(cfg,tokenizer,args.smoke)
    write_json(directory/'data-manifest.json',metadata)
    fs={p:[f'model.layers.{i}.self_attn.{p}' for i in range(model.config.num_hidden_layers)]
        for p in cfg['adapter']['target_modules']}
    plan=[];diagnostics={}
    if saved:
        plan=saved['plan']
    elif args.method=='lora':
        plan=[dict(family=p,layers=[n],rank=args.rank) for p,names in fs.items() for n in names]
    else:
        from e2lora.adapters.preparation import collect_family_gradients
        from e2lora.adapters.adaptive import gradient_similarity,greedy_intervals
        budget=cfg['partial_rank_budgets'][str(args.rank)]
        count=cfg['gradient_sample_count'] if not args.smoke else 2
        selected=random.Random(args.seed).sample(range(len(train)),count)
        write_json(directory/'gradient-samples.json',dict(sample_count=count,seed=args.seed,
            positions=selected,source_indices=[metadata['train_indices'][i] for i in selected]))
        batches=[collate([train[i]],tokenizer.pad_token_id) for i in selected]
        def loss(m,b):
            b={k:v.to('cuda') for k,v in b.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):value=m(**b).loss
            return value,int(b['labels'][:,1:].ne(-100).sum())
        for family,names in fs.items():
            gradients=collect_family_gradients(model,names,batches,loss)
            entropy,matrix=gradient_similarity([gradients[n] for n in names]);del gradients
            intervals,thresholds=greedy_intervals(matrix)
            ranks=exact_ranks([max(entropy[a:b]) for a,b in intervals],budget['total'])
            plan.extend(dict(family=family,layers=names[a:b],rank=r) for (a,b),r in zip(intervals,ranks))
            diagnostics[family]=dict(entropy=entropy,matrix=matrix,intervals=intervals,ranks=ranks,thresholds=thresholds)
            write_json(directory/f'diagnostics-{family}.json',diagnostics[family])
        plan=active_rank_plan(plan,{f:budget['active'] for f in fs})
    if args.method=='lora':
        apply_plan(model,plan,alpha=cfg['adapter']['alpha'])
        for g in plan:
            layer=model.get_submodule(g['layers'][0]);layer.down.float();layer.up.float()
        counts=count_adapters(model,plan)
        expected={4:3407872,8:6815744}[args.rank]
    else:
        attach(model,plan,cfg)
        counts=audit(model,plan)
        expected=cfg['partial_rank_budgets'][str(args.rank)]['expected_parameters']
    if counts['trainable_adapter_parameters']!=expected:raise ValueError(f'Unexpected adapter count: {counts}')
    write_json(directory/'run_config.json',dict(config=cfg,method=args.method,rank=args.rank,seed=args.seed,
        phase=phase,identity=identity,parameters=counts,plan=plan,source_hashes=source_hashes,
        effective_method=dict(sharing='independent' if args.method=='lora' else 'group A/B',
            rank_allocation='uniform' if args.method=='lora' else 'entropy, exact largest remainder',
            partial_rank_freezing=args.method=='e2lora'),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__))
    write_json(directory/'parameter_counts.json',counts)
    if not args.resume:
        torch.save({n:x.detach().cpu().clone() for n,x in adapter_state(model).items()},directory/'initial-adapter.pt')
        write_json(directory/'initialization.json',dict(seed=args.seed,
            A='public nn.Linear Kaiming uniform initialization',B_active='zero',
            B_fixed='uniform +/-0.01, buffers' if args.method=='e2lora' else 'none',
            state_sha256=file_sha256(directory/'initial-adapter.pt')))
    train_cfg=cfg['training'];batch_size=train_cfg['global_batch_size']
    per_epoch=math.ceil(len(train)/batch_size)
    epochs=1 if args.smoke else train_cfg['epochs']
    total=epochs*per_epoch;warmup=int(total*train_cfg['warmup_ratio'])
    write_json(directory/'effective-training.json',dict(epochs=epochs,total_steps=total,warmup_steps=warmup,
        micro_batch_size=1,gradient_accumulation=64,effective_batch=64,last_batch=len(train)%64 or 64,
        gradient_clipping=False,minimum_lr=0.0,trainable_dtype='float32',base_dtype='bfloat16'))
    parameters=[x for x in model.parameters() if x.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=train_cfg['learning_rate'],betas=tuple(train_cfg['betas']),weight_decay=train_cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:lr_multiplier(step,total,warmup))
    start_step=0;history=[]
    if saved:
        load_complete_adapter_state(model,saved['adapter'])
        opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        start_step=saved['step'];history=saved['history']
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    frozen=fixed_snapshot(model)
    segment_start=time.perf_counter();prior_seconds=saved.get('training_seconds',0.) if saved else 0.
    signal.signal(signal.SIGTERM,stop_requested);signal.signal(signal.SIGUSR1,stop_requested)
    def save(step):
        state=dict(identity=identity,plan=plan,adapter={n:x.detach().cpu().clone() for n,x in adapter_state(model).items()},
            optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),step=step,history=history,
            python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            training_seconds=prior_seconds+time.perf_counter()-segment_start)
        temporary=checkpoint_path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(checkpoint_path)
    active_epoch=None;order=None
    for step in range(start_step,total):
        epoch,offset=divmod(step,per_epoch)
        if epoch!=active_epoch:
            order=list(range(len(train)));random.Random(args.seed+epoch).shuffle(order);active_epoch=epoch
        selected=order[offset*batch_size:(offset+1)*batch_size]
        denominator=sum(sum(label!=-100 for label in train[i]['labels'][1:]) for i in selected)
        opt.zero_grad(set_to_none=True);model.train();weighted=0.
        lr_used=opt.param_groups[0]['lr']
        for i in selected:
            batch={k:v.to('cuda') for k,v in collate([train[i]],tokenizer.pad_token_id).items()}
            count=int(batch['labels'][:,1:].ne(-100).sum())
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch).loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            (loss*(count/denominator)).backward();weighted+=float(loss.detach())*count/denominator
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise FloatingPointError('Nonfinite gradients')
        opt.step();scheduler.step()
        record=dict(step=step+1,epoch=epoch+1,loss=weighted,lr_used=lr_used,examples=len(selected),supervised_tokens=denominator)
        history.append(record)
        with (directory/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        write_json(directory/'status.json',dict(stage='training',phase=phase,step=step+1,total_steps=total,loss=weighted,
            examples_seen=epoch*len(train)+min((offset+1)*batch_size,len(train)),elapsed_seconds=prior_seconds+time.perf_counter()-segment_start))
        if (step+1)%10==0:print(json.dumps(record),flush=True)
        if (step+1)%train_cfg['checkpoint_every']==0 or offset+1==per_epoch or STOP:
            assert_fixed(model,frozen);save(step+1)
        if STOP:
            write_json(directory/'status.json',dict(stage='checkpointed_for_preemption',step=step+1));raise SystemExit(75)
        if offset+1==per_epoch:
            heldout=validation_loss(model,valid,tokenizer)
            write_json(directory/f'validation-epoch{epoch+1}.json',dict(loss=heldout,examples=len(valid)))
    assert_fixed(model,frozen)
    heldout=read_json(directory/f'validation-epoch{epochs}.json')['loss'] if (directory/f'validation-epoch{epochs}.json').exists() else validation_loss(model,valid,tokenizer)
    write_json(directory/'status.json',dict(stage='gsm8k',evaluated=0,total=8 if args.smoke else 1319))
    evaluation=evaluate(model,tokenizer,cfg,directory,args.smoke,epochs)
    result=dict(status='completed',smoke=args.smoke,eligible_for_formal_tables=not args.smoke,
        method=args.method,base_rank=args.rank,seed=args.seed,parameters=counts,steps=total,
        evaluation=evaluation,heldout_validation_loss=heldout,protocol_sha256=digest(cfg),
        source_training_manifest_verified=True,fixed_components_unchanged=True,
        checkpoint_sha256=file_sha256(checkpoint_path),gpu=torch.cuda.get_device_name(),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),slurm_job_id=os.environ['SLURM_JOB_ID'])
    write_json(directory/'result.json',result)
    write_json(directory/'status.json',dict(stage='completed',accuracy=evaluation['accuracy'],phase=phase))


if __name__=='__main__':
    main()
