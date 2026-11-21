"""Team partial-rank acceptance check, separate from preserved historical smoke runs.

Reuses public gradient/entropy/group/partial-rank code and Chloe's trainer/scorer.
This is a team reproduction choice, not an assertion about released author code.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common import read_json, write_json, resolve, output_path, digest, file_sha256


def exact_ranks(scores, budget):
    """Team largest-remainder correction to public round-based allocation."""
    if not scores or type(budget) is not int or budget < len(scores):
        raise ValueError('Invalid exact total rank budget')
    if any(not math.isfinite(x) for x in scores) or abs(sum(scores)) < 1e-12:
        raise ValueError('Invalid unshifted entropy scores')
    values = [Fraction(x) for x in scores]
    quotas = [v * budget / sum(values) for v in values]
    if any(q < 0 for q in quotas):
        raise ValueError('Literal entropy ratios produced negative rank quotas')
    ranks = [int(q // 1) for q in quotas]
    order = sorted(range(len(ranks)), key=lambda i: (-(quotas[i] - ranks[i]), i))
    for i in order[:budget - sum(ranks)]:
        ranks[i] += 1
    if min(ranks) < 1 or sum(ranks) != budget:
        raise ValueError('Exact rank allocation failed; do not invent replacement entropies')
    return ranks


def sampling_positions(pool_size, sample_count, seed):
    if type(sample_count) is not int or not 0 < sample_count <= pool_size:
        raise ValueError('Explicit sample_count must fit the actual training pool')
    return random.Random(seed).sample(range(pool_size), sample_count)


def snapshot(model):
    from e2lora.adapters.partial_rank import adapter_state
    return {n: x.detach().cpu().clone() for n, x in adapter_state(model).items()}


def fixed_snapshot(model):
    from e2lora.adapters.partial_rank import adapter_state
    return {n: x.detach().cpu().clone() for n, x in adapter_state(model).items() if not x.requires_grad}


def assert_fixed(model, before):
    import torch
    after = fixed_snapshot(model)
    if before.keys() != after.keys() or any(not torch.equal(before[n], after[n]) for n in before):
        raise RuntimeError('Frozen adapter components changed')


def attach(model, plan, cfg, *, share_a=False):
    from e2lora.adapters.partial_rank import apply_partial_plan
    apply_partial_plan(model, plan, alpha=cfg['adapter']['alpha'],
        frozen_b_init_scale=cfg['team_rank_check']['frozen_b_init_scale'], global_down=share_a)
    # Same precision policy as the established Chloe trainer: BF16 base, FP32 adapters.
    for g in plan:
        layer = model.get_submodule(g['layers'][0])
        layer.down.float()
        layer.up.float()


def audit(model, plan, *, share_a=False):
    import torch
    from e2lora.adapters.partial_rank import count_adapters
    result = count_adapters(model, plan)
    result['other_trainable_parameters'] = 0  # count_adapters asserts this, rather than assuming it.
    result['projections'] = {}
    for family in sorted({g['family'] for g in plan}):
        groups = [g for g in plan if g['family'] == family]
        parameters, details, down_ids = {}, [], set()
        for g in groups:
            layer = model.get_submodule(g['layers'][0])
            down_ids.add(id(layer.down))
            for factor in (layer.down, layer.up):
                for parameter in factor.parameters():
                    if parameter.requires_grad:
                        parameters[id(parameter)] = parameter
            n = g['trainable_rank']
            fixed_a = layer.down.weight[n:g['rank']]
            fixed_b = layer.up.weight[:, n:]
            if fixed_b.numel():
                probe = fixed_b @ (fixed_a @ torch.ones(fixed_a.shape[1], device=fixed_a.device, dtype=fixed_a.dtype))
                if not torch.isfinite(probe).all() or not probe.abs().max().item() > 0:
                    raise RuntimeError('Frozen rank branch has no finite nonzero forward contribution')
            details.append(dict(layers=g['layers'], total_rank=g['rank'], trainable_rank=n))
        if share_a and len(down_ids) != 1:
            raise RuntimeError('E2ShareLoRA A is not shared across the entire projection family')
        result['projections'][family] = dict(groups=details, group_count=len(groups),
            total_rank=sum(g['rank'] for g in groups), trainable_rank=sum(g['trainable_rank'] for g in groups),
            unique_trainable_parameters=sum(p.numel() for p in parameters.values()))
    return result


def validation(model, tokenizer, cfg):
    import torch
    import pyarrow.parquet as pq
    from transformers import GenerationConfig
    from evaluate import generate_completion, score_records
    e = cfg['evaluation']
    rows = pq.read_table(resolve(e['local_parquet'])).to_pylist()
    if len(rows) != 1319:
        raise ValueError('Expected full pinned GSM8K source before selecting smoke prefix')
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=e['max_new_tokens'],
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    records = []
    model.eval()
    for i, row in enumerate(rows[:e['limit']]):
        prompt = cfg['data']['prompt_template'].format(query=row['question'])
        completion = generate_completion(model, tokenizer, prompt,
            max_prompt_length=cfg['training']['max_prompt_length'], generation_config=generation)
        records.append(dict(source_index=i, gold=row['answer'], **completion))
    scores, scored = score_records(records)
    scores.update(empty_outputs=sum(not r['prediction'].strip() for r in records),
        examples=scored, evaluation_source_sha256=file_sha256(resolve(e['local_parquet'])))
    return scores


def execute(config_path):
    import torch
    from common import load_local_model, load_math_examples, baseline_plan, collate, seed_everything
    from train import require_gpu_allocation, fit, load_complete_adapter_state
    from e2lora.adapters.preparation import collect_family_gradients
    from e2lora.adapters.adaptive import gradient_similarity, greedy_intervals
    from e2lora.adapters.partial_rank import active_rank_plan
    require_gpu_allocation()
    cfg = read_json(config_path)
    team = read_json('references/text-generation-setup.json')
    settings = team['partial_rank_sharing']['final_settings']
    check = cfg['team_rank_check']
    if cfg['model']['revision'] != team['model']['revision'] or cfg['run_type'] != 'smoke':
        raise ValueError('This bounded acceptance run requires the team model and smoke label')
    if check['total_rank_per_projection'] != settings['total_rank_per_projection'] or check['trainable_rank_per_projection'] != settings['trainable_rank_per_projection']:
        raise ValueError('Team budgets changed; do not silently continue')
    out = output_path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Use a new acceptance output directory; preserve existing evidence')
    write_json(out / 'run_config.json', cfg)
    write_json(out / 'status.json', dict(stage='loading', gpu=torch.cuda.get_device_name(), complete=False))
    seed_everything(cfg['seed'])
    model, tokenizer = load_local_model(cfg, 'cuda:0')
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    examples, data_stats = load_math_examples(cfg, tokenizer)
    if data_stats['response_truncated_examples']:
        raise ValueError('Short-run subset truncates answers; inspect lengths before running')
    positions = sampling_positions(len(examples), check['gradient_sample_count'], cfg['seed'])
    manifest = read_json(cfg['data']['manifest'])
    sampling = dict(seed=cfg['seed'], sample_count=len(positions), without_replacement=True,
        sample_count_independent_of_microbatch=True, positions=positions,
        source_row_indices=[manifest['indices'][i] for i in positions],
        rule='Random(seed).sample(actual_training_pool, explicit_sample_count)',
        loss='response-only; weight by labels[:,1:] != -100; eval dropout; no optimizer updates')
    write_json(out / 'sampling.json', sampling)
    batches = [collate([examples[i]], tokenizer.pad_token_id) for i in positions]
    batches = [{k: v.to('cuda') for k, v in b.items()} for b in batches]
    def loss(m, batch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            value = m(**batch).loss
        return value, int(batch['labels'][:, 1:].ne(-100).sum())
    fs = {p: [f'model.layers.{i}.self_attn.{p}' for i in range(model.config.num_hidden_layers)]
          for p in settings['target_projections']}
    plan, diagnostics = [], {}
    started = time.perf_counter()
    for family, names in fs.items():
        gradients = collect_family_gradients(model, names, batches, loss)
        entropy, matrix = gradient_similarity([gradients[n] for n in names])
        del gradients
        intervals, thresholds = greedy_intervals(matrix)
        ranks = exact_ranks([max(entropy[a:b]) for a,b in intervals], check['total_rank_per_projection'])
        plan += [dict(family=family, layers=names[a:b], rank=r) for (a,b),r in zip(intervals,ranks)]
        diagnostics[family] = dict(entropy=entropy, matrix=matrix, intervals=intervals, ranks=ranks, thresholds=thresholds)
        write_json(out / f'diagnostics-{family}.json', diagnostics[family])
        write_json(out / 'status.json', dict(stage='gradient_sampling', family_completed=family, complete=False))
    plan = active_rank_plan(plan, {p: check['trainable_rank_per_projection'] for p in fs})
    write_json(out / 'plan.json', dict(plan=plan, diagnostics=diagnostics,
        gradient_seconds=time.perf_counter()-started, rank_policy='paper plus team exact largest remainder'))
    originals = {n: model.get_submodule(n) for names in fs.values() for n in names}
    # Verify E2ShareLoRA on the same actual loaded architecture and plan, then restore the base.
    attach(model, plan, cfg, share_a=True)
    share_counts = audit(model, plan, share_a=True)
    if share_counts['trainable_adapter_parameters'] != settings['estimated_trainable_adapter_parameters']['e2sharelora']:
        raise RuntimeError(f'E2ShareLoRA count mismatch: {share_counts}')
    for name, layer in originals.items():
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent), child, layer)
    seed_everything(cfg['seed'])
    attach(model, plan, cfg)
    counts = audit(model, plan)
    write_json(out / 'parameter-audit.json', dict(e2lora=counts, e2sharelora=share_counts,
        expected=settings['estimated_trainable_adapter_parameters']))
    if counts['trainable_adapter_parameters'] != settings['estimated_trainable_adapter_parameters']['e2lora']:
        raise RuntimeError(f'E2LoRA count mismatch: {counts}')
    initial = snapshot(model)
    fixed = fixed_snapshot(model)
    torch.save(initial, out / 'initial-adapter.pt')
    write_json(out / 'initialization.json', dict(A='public nn.Linear initialization (uniform Kaiming), split along rank axis',
        B_active='zero', B_fixed=f'uniform +/- {check["frozen_b_init_scale"]}',
        frozen_storage='buffers, excluded from optimizer; frozen components contribute to forward',
        initial_state_sha256=file_sha256(out / 'initial-adapter.pt')))
    checkpoint = out / 'checkpoint.pt'
    fingerprint = digest(dict(config=cfg, manifest_sha256=data_stats['manifest_sha256'], plan=plan))
    rounds = []
    per_epoch = math.ceil(len(examples) / cfg['training']['global_batch_size'])
    for epoch in range(cfg['training']['epochs']):
        write_json(out / 'status.json', dict(stage='training', epoch=epoch+1, complete=False))
        trained = fit(model, examples, plan, cfg, pad_token_id=tokenizer.pad_token_id,
            checkpoint=checkpoint, fingerprint=fingerprint,
            resume=checkpoint if epoch else None,
            stop_after_step=per_epoch*(epoch+1), log_path=out / 'training.jsonl')
        assert_fixed(model, fixed)
        if not any(not torch.equal(initial[n], value) for n,value in snapshot(model).items() if n not in fixed):
            raise RuntimeError('No trainable adapter component updated')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        load_complete_adapter_state(model, state['complete_adapter_state'])
        assert_fixed(model, fixed)
        audit(model, plan)
        write_json(out / 'status.json', dict(stage='validation', epoch=epoch+1, complete=False))
        scores = validation(model, tokenizer, cfg)
        rounds.append(dict(epoch=epoch+1, training=trained, validation=scores))
        write_json(out / 'rounds.json', rounds)
        print(json.dumps(dict(epoch=epoch+1, accuracy=scores['accuracy'], empty=scores['empty_outputs'],
            unparseable=scores['unparseable_predictions'])), flush=True)
    result = dict(complete=True, run_type='smoke', eligible_for_formal_tables=False, config=cfg,
        data=data_stats, sampling=sampling, parameters=counts, e2sharelora_parameter_check=share_counts,
        fixed_components_unchanged=True, shared_objects_verified=True, resume_between_rounds=True,
        rounds=rounds, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        full_run_started=False, formal_protocol='Unresolved fields preserved; requires complete formal configuration')
    write_json(out / 'result.json', result)
    write_json(out / 'status.json', dict(stage='completed', complete=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    out = output_path(read_json(args.config)['output_dir'])
    try:
        execute(args.config)
    except Exception as error:
        write_json(out / 'failure.json', dict(error=f'{type(error).__name__}: {error}', full_run_started=False))
        raise


if __name__ == '__main__':
    main()
