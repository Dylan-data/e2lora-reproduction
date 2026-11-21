from copy import deepcopy
import json
import random

import pytest
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

import common
from adaptive_smoke import attach_prepared, prepared_artifact, sampling_batches
from common import attach_trainable_plan, digest, parameter_report, validate_config, write_json
from e2lora.adapters.adaptive import (allocate_ranks, build_plan, collect_gradients,
                                     gradient_similarity, greedy_intervals)
from e2lora.adapters.lora_layers import apply_plan, trainable_count
from e2lora.adapters.preparation import collect_plan
from prepare_e2lora import probe_gradients
from train import fit
from test_math_baseline import tiny_run_setup


def e2_config():
    return common.read_json('experiments/chloe/configs/smoke-e2lora-r8.json')


def test_hra_uses_group_max_and_literal_rounding():
    assert allocate_ranks([1, 9, 3, 5], [(0, 2), (2, 4)], 2, rank_policy='paper') == [5, 3]
    # Two half-integer quotas round separately: no silent exact-budget repair.
    assert allocate_ranks([1, 1, 1], [(0, 1), (1, 3)], 1, rank_policy='paper') == [2, 2]
    # Negative differential entropy is retained, including its inverted ratios.
    assert allocate_ranks([-3, -1], [(0, 1), (1, 2)], 2, rank_policy='paper') == [3, 1]


@pytest.mark.parametrize('values', [[0, 0], [-2, 1]])
def test_hra_invalid_literal_scores_fail_without_shift(values):
    with pytest.raises(ValueError):
        allocate_ranks(values, [(0, 1), (1, 2)], 2, rank_policy='paper')


def test_lss_contiguous_anchor_threshold_and_rmi_not_clipped():
    matrix = [[1, .9, .1, .2], [.9, 1, .1, .2], [.1, .1, 1, .95], [.2, .2, .95, 1]]
    intervals, thresholds = greedy_intervals(matrix)
    assert intervals == [(0, 2), (2, 4)]
    assert thresholds[0] == pytest.approx(.4)
    _, rmi = gradient_similarity([torch.tensor([-.1, .1]), torch.tensor([9.9, 10.1])])
    assert rmi[0][1] > 1  # The entropy proxy does not guarantee a [0,1] range.


def toy_shared():
    torch.manual_seed(12)
    base = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4), nn.Tanh(),
                         nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4))
    plan = [{'family': 'linear', 'layers': ['0', '2'], 'rank': 2},
            {'family': 'linear', 'layers': ['4', '6'], 'rank': 3}]
    return base, plan


def test_shared_identity_unique_count_and_frozen_base():
    model, plan = toy_shared()
    inputs = torch.randn(3, 4)
    original = model(inputs).detach()
    attach_trainable_plan(model, plan, alpha=4)
    torch.testing.assert_close(model(inputs), original, atol=0, rtol=0)
    assert model[0].down is model[2].down and model[0].up is model[2].up
    assert model[4].down is model[6].down and model[4].up is model[6].up
    assert model[0].down is not model[4].down
    report = parameter_report(model, plan)
    assert report['total_trainable_parameters'] == trainable_count(model) == (2 + 3) * (4 + 4)
    assert all(not p.requires_grad for layer in [model[0], model[2], model[4], model[6]] for p in layer.base.parameters())
    assert all(g['frozen_rank'] == 0 for g in report['groups'])


def test_shared_gradients_sum_layer_uses_and_first_zero_B_is_expected():
    model, plan = toy_shared()
    attach_trainable_plan(model, plan, alpha=4)
    inputs = torch.randn(3, 4)
    model(inputs).square().mean().backward()
    assert model[0].down.weight.grad.abs().max() == 0
    assert model[0].up.weight.grad.abs().max() > 0
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model[0].up.weight.normal_(0, .1)
        model[4].up.weight.normal_(0, .1)
    untied = deepcopy(model)
    for name in ['0', '2', '4', '6']:
        layer = untied.get_submodule(name)
        layer.down = deepcopy(layer.down)
        layer.up = deepcopy(layer.up)
    model(inputs).square().mean().backward()
    untied(inputs).square().mean().backward()
    for group in plan:
        first = model.get_submodule(group['layers'][0])
        for factor in ['down', 'up']:
            expected = sum(getattr(untied.get_submodule(name), factor).weight.grad for name in group['layers'])
            actual = getattr(first, factor).weight.grad
            torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-5)
            assert actual.abs().max() > 0
    assert all(p.grad is None for name, p in model.named_parameters() if '.base.' in name)


def test_preparation_gradient_probe_performs_zero_updates():
    model, plan = toy_shared()
    x = torch.randn(3, 4)
    callback = lambda m, b: (m(b).square().mean(), len(b))
    with torch.no_grad():
        base_loss = callback(model, x)[0]
    attach_trainable_plan(model, plan, alpha=4)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    report = probe_gradients(model, plan, x, callback, base_loss)
    assert report['optimizer_updates'] == 0 and report['all_trainable_tensors_connected']
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name], atol=0, rtol=0)
        assert p.grad is None


def test_instantiated_llama_budget_is_not_halved_by_grouping_alone():
    config = e2_config()
    architecture = AutoConfig.from_pretrained(common.resolve(config['model']['path']), local_files_only=True)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(architecture, dtype=torch.bfloat16, attn_implementation='sdpa')
        # A synthetic two-group partition, not a claimed real-data grouping.
        plan = [{'family': p, 'layers': [f'model.layers.{i}.self_attn.{p}' for i in range(a, b)], 'rank': 128}
                for p in config['adapter']['target_modules'] for a, b in [(0, 16), (16, 32)]]
        attach_trainable_plan(model, plan, alpha=16)
    counts = parameter_report(model, plan)
    assert counts['adapter_parameters'] == counts['total_trainable_parameters'] == 6815744
    assert counts['total_trainable_parameters'] != 3700736


def test_sequential_sampling_matches_joint_collection_and_restores_flags():
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 3))
    model.requires_grad_(False)
    model.train()
    xs = torch.randn(3, 3)
    ys = torch.randn(3, 3)
    batches = [(xs[:2], ys[:2]), (xs[2:], ys[2:])]
    callback = lambda m, b: (((m(b[0]) - b[1]) ** 2).mean() * .01, len(b[0]))
    gradients = collect_gradients(model, ['0', '2'], batches, callback)
    reference, diagnostics = build_plan(gradients, {'linear': ['0', '2']}, base_rank=4, rank_policy='paper')
    actual, details, times = collect_plan(model, {'linear': ['0', '2']}, batches, callback, base_rank=4, rank_policy='paper')
    assert actual == reference
    torch.testing.assert_close(torch.tensor(details['linear']['rmi']), torch.tensor(diagnostics['linear']['rmi']))
    assert times['backward_passes'] == 4 and times['target_block_size'] == 1
    assert model.training and not any(p.requires_grad or p.grad is not None for p in model.parameters())


def test_collect_failure_restores_modes_and_flags():
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(.5))
    model.requires_grad_(False)
    model.train()
    with pytest.raises(RuntimeError, match='injected'):
        collect_gradients(model, ['0'], [None], lambda m, b: (_ for _ in ()).throw(RuntimeError('injected')))
    assert model.training and model[1].training
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_partial_rank_and_formal_e2_are_explicitly_blocked():
    config = e2_config()
    validate_config(config)
    config['adaptive']['partial_rank_sharing']['enabled'] = True
    with pytest.raises(ValueError, match='139-rank assumption'):
        validate_config(config)
    config = e2_config()
    config['run_type'] = 'formal'
    with pytest.raises(ValueError, match='not formal training'):
        validate_config(config)
    config = e2_config()
    config['output_dir'] = 'outputs/chloe/smoke/math-lora-r8-seed42'
    with pytest.raises(ValueError, match='passed LoRA baseline'):
        validate_config(config)


def test_sampling_reuses_baseline_order_prefix():
    config = e2_config()
    examples = [{'input_ids': [1, 3, 2], 'labels': [-100, 3, 2]} for _ in range(160)]
    batches, positions = sampling_batches(examples, config, 2)
    order = list(range(160))
    random.Random(42).shuffle(order)
    assert positions == order[:2] and len(batches) == 2


def test_shared_llama_resume_reconstructs_identity_and_matches_full_run(tmp_path):
    config = common.read_json('experiments/chloe/configs/smoke-lora-r8.json')
    config, examples, make_independent = tiny_run_setup(config, attention_dropout=.1)

    def make_shared():
        independent, _ = make_independent()
        # Recover the original base modules; preserve the same base initialization.
        for name, module in list(independent.named_modules()):
            if hasattr(module, 'base') and hasattr(module, 'down'):
                parent, _, child = name.rpartition('.')
                setattr(independent.get_submodule(parent), child, module.base)
        plan = [{'family': p, 'layers': [f'model.layers.{i}.self_attn.{p}' for i in range(2)], 'rank': 4}
                for p in config['adapter']['target_modules']]
        torch.manual_seed(51)
        attach_trainable_plan(independent, plan, alpha=16)
        return independent, plan

    full, plan = make_shared()
    frozen = {n: p.clone() for n, p in full.named_parameters() if not p.requires_grad}
    fit(full, examples, plan, config, pad_token_id=2, checkpoint=tmp_path/'full.pt', fingerprint='shared')
    part, plan = make_shared()
    path = tmp_path/'resume.pt'
    fit(part, examples, plan, config, pad_token_id=2, checkpoint=path, fingerprint='shared', stop_after_step=2)
    restored, plan = make_shared()
    fit(restored, examples, plan, config, pad_token_id=2, checkpoint=path, fingerprint='shared', resume=path)
    for (name, left), right in zip(restored.named_parameters(), full.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        if name in frozen:
            torch.testing.assert_close(left, frozen[name], atol=0, rtol=0)
    assert restored.model.layers[0].self_attn.q_proj.down is restored.model.layers[1].self_attn.q_proj.down


def test_prepared_plan_config_and_source_cannot_be_silently_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'OUTPUT_ROOT', tmp_path)
    config = e2_config()
    config['output_dir'] = str(tmp_path/'e2')
    config['adaptive']['prepared_plan'] = str(tmp_path/'e2/prepared-plan.json')
    artifact = {'status': 'prepared', 'config_sha256': digest(config), 'source_fingerprint': 'original',
                'optimizer_updates': 0, 'partial_rank_sharing_enabled': False}
    write_json(config['adaptive']['prepared_plan'], artifact)
    assert prepared_artifact(config, source_fingerprint='original')['optimizer_updates'] == 0
    with pytest.raises(ValueError, match='fingerprint changed'):
        prepared_artifact(config, source_fingerprint='different')
    config['seed'] = 43
    with pytest.raises(ValueError, match='unchanged Smoke config'):
        prepared_artifact(config)


def test_loading_prepared_plan_rebuilds_sharing_and_checks_actual_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'OUTPUT_ROOT', tmp_path)
    config = e2_config()
    config['output_dir'] = str(tmp_path/'e2')
    config['adaptive']['prepared_plan'] = str(tmp_path/'e2/prepared-plan.json')
    base = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2))
    original = deepcopy(base)
    plan = [{'family': p, 'layers': [f'model.layers.{i}.self_attn.{p}' for i in range(2)], 'rank': 16}
            for p in config['adapter']['target_modules']]
    attach_trainable_plan(base, plan, alpha=16)
    counts = parameter_report(base, plan)
    artifact = {'status': 'prepared', 'config_sha256': digest(config), 'source_fingerprint': 'same',
                'optimizer_updates': 0, 'partial_rank_sharing_enabled': False, 'plan': plan,
                'diagnostics': {p: {'ranks': [16]} for p in config['adapter']['target_modules']},
                'parameters': counts}
    write_json(config['adaptive']['prepared_plan'], artifact)
    attached, _ = attach_prepared(original, config, source_fingerprint='same')
    assert attached == plan
    assert original.model.layers[0].self_attn.q_proj.up is original.model.layers[1].self_attn.q_proj.up
    artifact['parameters']['total_trainable_parameters'] += 1
    write_json(config['adaptive']['prepared_plan'], artifact)
    with pytest.raises(ValueError, match='parameter count changed'):
        fresh = LlamaForCausalLM(base.config)
        attach_prepared(fresh, config, source_fingerprint='same')
