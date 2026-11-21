from copy import deepcopy
import json
import weakref

import pytest
import torch
from torch import nn

from e2lora.adapters.adaptive import collect_gradients
from e2lora.adapters.preparation import collect_family_gradients, collect_plan
import e2lora.adapters.preparation as preparation
from multi_target_preflight import probe_plan
from sampling_audit import backward_geometry, weight_inventory
import common


def frozen_model():
    torch.manual_seed(619)
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4), nn.Tanh(),
                          nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4))
    model.requires_grad_(False)
    model.train()
    return model


def test_loss_and_gradient_objects_are_released_before_next_forward(monkeypatch):
    model = frozen_model()
    batches = [torch.randn(2, 4) for _ in range(3)]
    references = []
    original = torch.autograd.grad

    def observed_grad(*args, **kwargs):
        assert kwargs["retain_graph"] is False and kwargs["create_graph"] is False
        result = original(*args, **kwargs)
        references.extend(weakref.ref(g) for g in result)
        return result

    monkeypatch.setattr(torch.autograd, "grad", observed_grad)

    def loss_for_batch(current, batch):
        assert all(ref() is None for ref in references)
        loss = current(batch).square().mean()
        references.append(weakref.ref(loss))
        return loss, len(batch)

    result = collect_gradients(model, ["0"], batches, loss_for_batch)
    assert all(ref() is None for ref in references)
    assert result["0"].device.type == "cpu" and result["0"].grad_fn is None
    assert model.training and not any(p.requires_grad or p.grad is not None for p in model.parameters())


def test_sequential_family_matches_independent_weighted_autograd_and_updates_no_weights():
    model = frozen_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    reference = deepcopy(model).eval()
    targets = ["0", "2", "4", "6"]
    weights = [reference.get_submodule(n).weight for n in targets]
    for p in weights:
        p.requires_grad_(True)
    x = torch.randn(4, 4)
    y = torch.randn(4, 4)
    expected = torch.autograd.grad((reference(x) - y).square().mean(), weights)
    batches = [(x[:1], y[:1]), (x[1:], y[1:])]
    callbacks = []

    def loss_for_batch(current, batch):
        active = [n for n, p in current.named_parameters() if p.requires_grad]
        assert len(active) == 1 and active[0].endswith(".weight")
        assert not current.training
        return (current(batch[0]) - batch[1]).square().mean(), len(batch[0])

    actual = collect_family_gradients(model, targets, batches, loss_for_batch, on_target_complete=callbacks.append)
    assert len(callbacks) == 4 and sum(row["backward_passes"] for row in callbacks) == 8
    json.dumps(callbacks)  # Observers cannot retain raw tensor/graph objects.
    assert [r["cpu_gradient_cache_bytes"] for r in callbacks] == [64, 128, 192, 256]
    for name, expected_gradient in zip(targets, expected):
        torch.testing.assert_close(actual[name], expected_gradient, rtol=2e-5, atol=1e-8)
        assert actual[name].dtype == torch.float32 and actual[name].grad_fn is None
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        assert parameter.grad is None and not parameter.requires_grad
    assert model.training


def test_previous_projection_cpu_gradients_die_before_next_family(monkeypatch):
    model = frozen_model()
    references = []
    original = preparation.collect_gradients

    def traced(current, targets, batches, callback):
        if targets == ["4"]:
            assert len(references) == 2 and all(ref() is None for ref in references)
        result = original(current, targets, batches, callback)
        references.extend(weakref.ref(g) for g in result.values())
        return result

    monkeypatch.setattr(preparation, "collect_gradients", traced)
    batches = [torch.randn(2, 4)]
    _, detail, timings = collect_plan(model, {"first": ["0", "2"], "second": ["4", "6"]}, batches,
        lambda current, batch: (current(batch).square().mean(), len(batch)), base_rank=4, rank_policy="paper",
        mechanism_options=dict(lss=False, hra=False, similarity="rmi", convention=None))
    assert timings["backward_passes"] == 4
    assert all(ref() is None for ref in references)
    json.dumps(detail)


def test_observer_failure_restores_model_mode_flags_and_grad_buffers():
    model = frozen_model()

    def failed(_):
        raise RuntimeError("stop at observed target")

    with pytest.raises(RuntimeError, match="observed target"):
        collect_family_gradients(model, ["0", "2"], [torch.randn(2, 4)],
            lambda current, batch: (current(batch).square().mean(), len(batch)), on_target_complete=failed)
    assert model.training and not any(p.requires_grad or p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("steps", [8, 16, 32, 64, 128, 256, 512])
def test_128_target_backward_counts_separate_steps_batch_and_microbatch(steps):
    single = backward_geometry(targets=128, steps=steps, batch_size=1, micro_batch_size=1)
    large = backward_geometry(targets=128, steps=steps, batch_size=64, micro_batch_size=1)
    assert single["autograd_grad_calls"] == 128 * steps
    assert large["autograd_grad_calls"] == 8192 * steps
    assert single["optimizer_updates"] == large["optimizer_updates"] == 0
    partial = backward_geometry(targets=4, steps=3, batch_size=5, micro_batch_size=2)
    assert partial["micro_batches_per_target"] == 9 and partial["autograd_grad_calls"] == 36


def test_real_weight_headers_imply_five_gib_total_and_two_gib_family_cache():
    config = common.read_json("experiments/chloe/configs/smoke-lora-r8.json")
    inventory = weight_inventory(common.resolve(config["model"]["path"]))
    assert inventory["target_count"] == 128
    assert inventory["all_targets_cpu_fp32_bytes_if_retained"] == 5 * 1024 ** 3
    assert inventory["sequential_peak_family_cpu_fp32_bytes"] == 2 * 1024 ** 3
    assert inventory["families"]["q_proj"]["targets"][0]["cpu_fp32_gradient_bytes"] == 64 * 1024 ** 2
    assert inventory["families"]["k_proj"]["targets"][0]["cpu_fp32_gradient_bytes"] == 16 * 1024 ** 2
    assert inventory["largest_alternative_metric_family_fp64_copies_bytes"] == 4 * 1024 ** 3


def test_multi_target_preflight_refuses_full128_and_more_than_two_samples():
    first = probe_plan("first-layer4", 2)
    spread = probe_plan("spread12", 2)
    assert first["geometry"]["autograd_grad_calls"] == 8
    assert spread["geometry"]["autograd_grad_calls"] == 24
    assert first["geometry"]["examples"] == spread["geometry"]["examples"] == 2
    assert not first["eligible_for_formal_tables"] and first["optimizer_updates"] == 0
    with pytest.raises(ValueError):
        probe_plan("full128", 2)
    with pytest.raises(ValueError):
        probe_plan("first-layer4", 512)
