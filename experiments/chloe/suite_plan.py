"""Math ablation planning. Policies here are explicit team choices, not author code."""
from copy import deepcopy
from fractions import Fraction
import math
import random

from common import digest, read_json
from e2lora.adapters.similarity import CONVENTIONS, similarity_matrix
from e2lora.adapters.grouping import search_intervals
from team_rank_check import exact_ranks

METHODS = {
    'lora': (False, False, 'rmi'),
    'sharing_only': (True, False, 'rmi'),
    'rank_allocation_only': (False, True, 'rmi'),
    'cosine_with_rank_allocation': (True, True, 'cosine'),
    'l1_with_rank_allocation': (True, True, 'l1'),
    'l2_with_rank_allocation': (True, True, 'l2'),
    'kl_with_rank_allocation': (True, True, 'kl'),
    'e2lora': (True, True, 'rmi'),
}


def null_paths(value, prefix=''):
    if value is None:
        return [prefix]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in null_paths(v, f'{prefix}.{k}'.strip('.'))]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in null_paths(v, f'{prefix}[{i}]')]
    return []


def validate(config):
    """No inferred Figure5 settings, seed numbers or rank budgets."""
    allowed = {'data.train_limit', 'training.max_steps', 'training.max_grad_norm', 'evaluation.limit'}
    bad = set(null_paths(config)) - allowed
    if bad:
        raise ValueError('Unresolved null settings: ' + ', '.join(sorted(bad)))
    if config['task'] != 'math':
        raise ValueError('Current approved scope is Math; Code/HumanEval is excluded, not a zero result')
    if not config['seeds'] or any(type(s) is not int for s in config['seeds']) or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('Supply distinct integer seeds; 43/44 are not mandatory')
    t = config['training']
    for key in ['learning_rate','weight_decay','warmup_ratio']:
        if type(t[key]) not in [int,float] or not math.isfinite(t[key]):
            raise ValueError(f'Invalid training.{key}')
    if t['learning_rate']<=0 or t['weight_decay']<0 or not 0<=t['warmup_ratio']<1:
        raise ValueError('Invalid optimizer values')
    if len(t['betas'])!=2 or any(not isinstance(x,(int,float)) or not 0<=x<1 for x in t['betas']):
        raise ValueError('Invalid AdamW betas')
    if (t['micro_batch_size'], t['gradient_accumulation'], t['global_batch_size']) != (1, 64, 64):
        raise ValueError('This audited runner implements microbatch1, accumulation64, effective64')
    if t['precision'] != 'bf16' or t['optimizer'] != 'AdamW' or t['scheduler'] != 'cosine':
        raise ValueError('Unsupported optimizer/precision/schedule')
    if t['max_steps'] is not None or t['max_grad_norm'] is not None:
        raise ValueError('Full epochs and no gradient clipping are the recorded protocol')
    if not t['gradient_checkpointing'] or t['epochs'] < 1 or t['checkpoint_every'] < 1:
        raise ValueError('Invalid epochs/checkpointing')
    if config['adapter']['dropout'] != 0 or config['adapter']['target_modules'] != ['q_proj','k_proj','v_proj','o_proj']:
        raise ValueError('Expected all four attention projections and zero dropout')
    if config['data']['loss'] != 'response_only_token_weighted' or config['data']['packing']:
        raise ValueError('Only response-token-weighted unpacked training is implemented')
    if config['data']['prompt_template']!='{query}\n' or config['data']['manifest_count']!=100000 or config['data']['train_limit'] is not None:
        raise ValueError('This runner uses the fixed 100K and query-newline prompt protocol')
    if config['adapter']['alpha']<=0 or t['max_length']<2:
        raise ValueError('Invalid alpha/sequence length')
    if config['evaluation']['do_sample'] or config['evaluation']['num_beams'] != 1 or config['evaluation']['limit'] is not None:
        raise ValueError('Formal GSM8K evaluation must be greedy and complete')
    if type(config['evaluation']['max_new_tokens']) is not int or config['evaluation']['max_new_tokens']<1:
        raise ValueError('Invalid generation limit')
    suite = config['suite']
    for key in ['table4_rank', 'main_sampling_steps', 'rank_sweep_fixed_steps', 'step_sweep_fixed_rank', 'examples_per_sampling_step']:
        if type(suite[key]) is not int or suite[key] < 1:
            raise ValueError(f'Invalid suite.{key}')
    if suite['adapter_policy'] not in ['all_trainable', 'team_partial_fraction']:
        raise ValueError('Explicit adapter_policy required')
    fraction = Fraction(suite['active_fraction'])
    if not 0 < fraction <= 1:
        raise ValueError('active_fraction must be in (0,1]')
    for metric, convention in CONVENTIONS.items():
        if suite['metric_conventions'][metric] != convention:
            raise ValueError(f'Unsupported {metric} convention')
    for rank, entry in suite['rank_budgets'].items():
        if int(rank) < 1 or type(entry['total']) is not int or entry['total'] < 1:
            raise ValueError('Invalid per-projection total rank budget')
    return config


def selection(pool_size, count, seed):
    """A single permutation gives nested prefixes even when sample counts differ."""
    if type(count) is not int or not 0 < count <= pool_size:
        raise ValueError('Gradient sample count exceeds actual training pool; no replacement/cycling')
    order = list(range(pool_size))
    random.Random(seed).shuffle(order)
    return order[:count]


def family_plan(gradients, names, family, rank, method, config):
    lss, hra, metric = METHODS[method]
    convention = None if metric == 'rmi' else config['suite']['metric_conventions'][metric]
    entropy, matrix = similarity_matrix([gradients[n] for n in names], metric=metric, convention=convention)
    if lss:
        intervals, grouping = search_intervals(matrix, method='greedy')
    else:
        intervals, grouping = [(i, i+1) for i in range(len(names))], {'method':'independent'}
    if hra:
        budget = config['suite']['rank_budgets'][str(rank)]['total']
        ranks = exact_ranks([max(entropy[a:b]) for a,b in intervals], budget)
    else:
        ranks = [rank] * len(intervals)
    plan = [dict(family=family, layers=names[a:b], rank=r) for (a,b),r in zip(intervals,ranks)]
    return plan, dict(entropy=entropy, matrix=matrix, intervals=intervals, ranks=ranks,
                      metric=metric, convention=convention, grouping=grouping,
                      lss=lss, hra=hra, total_rank=sum(ranks),
                      rank_policy='team_exact_largest_remainder' if hra else 'fixed_rank_per_group')


def active_budgets(plan, config):
    fraction = Fraction(config['suite']['active_fraction'])
    result = {}
    for family in {g['family'] for g in plan}:
        total = sum(g['rank'] for g in plan if g['family'] == family)
        # Explicit nearest integer, half up; group allocation uses public largest remainder.
        quota = total * fraction
        result[family] = math.floor(quota + Fraction(1,2))
    return result


def specifications(config, setup):
    validate(config)
    s = config['suite']
    rows = []
    def add(study, method, rank, steps):
        rows.append(dict(study=study, method=method, rank=rank, sampling_steps=0 if method=='lora' else steps))
    for method in METHODS:
        add('table4', method, s['table4_rank'], s['main_sampling_steps'])
    for pair in setup['parameter_budget_comparisons']:
        add('table5','lora',pair['lora_rank'],0)
        add('table5','e2lora',pair['e2lora_rank'],s['main_sampling_steps'])
    for rank in setup['rank_values']:
        add('figure5_rank','e2lora',rank,s['rank_sweep_fixed_steps'])
    for steps in setup['gradient_sampling_steps']:
        add('figure5_steps','e2lora',s['step_sweep_fixed_rank'],steps)
    # Identical settings reused across tables/curves, with all study labels retained.
    unique = {}
    for row in rows:
        key = row['method'], row['rank'], row['sampling_steps']
        unique.setdefault(key, dict(method=key[0], rank=key[1], sampling_steps=key[2], studies=[]))['studies'].append(row['study'])
    runs = []
    for row in unique.values():
        if row['method']!='lora' and METHODS[row['method']][1] and str(row['rank']) not in s['rank_budgets']:
            raise ValueError(f'No explicit budget for rank {row["rank"]}')
        for seed in config['seeds']:
            run = deepcopy(config)
            run['run'] = dict(row, seed=seed)
            run['run_id'] = f'{row["method"]}-r{row["rank"]}-n{row["sampling_steps"]}-seed{seed}'
            runs.append(run)
    return runs
