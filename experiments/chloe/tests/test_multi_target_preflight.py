"""CPU orchestration tests; CUDA metrics/transfers are stubs, not GPU evidence."""
from contextlib import nullcontext
from itertools import chain, repeat
import json
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn

import e2lora.adapters.preparation as preparation
import multi_target_preflight as probe
import train


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict()
        for index in (0, 15, 31):
            layer = nn.Module()
            layer.self_attn = nn.ModuleDict({
                name: nn.Linear(4, 4, bias=False)
                for name in ("q_proj", "k_proj", "v_proj", "o_proj")})
            self.model.layers[str(index)] = layer

    def forward(self, input_ids, **_):
        hidden = input_ids[:, :4].float() / 20
        for layer in self.model.layers.values():
            for projection in layer.self_attn.values():
                hidden = hidden + projection(hidden) / 10
        return SimpleNamespace(loss=hidden.square().mean())


@pytest.fixture
def cpu_probe(monkeypatch, tmp_path):
    model = TinyModel()
    references, leaked = [], []
    calls = {"autograd": 0}
    original_grad = torch.autograd.grad
    original_to = torch.Tensor.to
    original_collector = preparation.collect_family_gradients

    def counted_grad(*args, **kwargs):
        calls["autograd"] += 1
        return original_grad(*args, **kwargs)

    def cpu_transfer(tensor, *args, **kwargs):
        if args and args[0] == "cuda:0":
            args = ("cpu", *args[1:])
        return original_to(tensor, *args, **kwargs)

    def collector(current, targets, batches, callback, *, on_target_complete):
        assert all(ref() is None for ref in references)

        def observed(detail):
            detail.update(cuda_allocated_after_bytes=1000, cuda_reserved_after_bytes=2000,
                          cuda_peak_allocated_bytes=1500, cuda_peak_reserved_bytes=2000)
            if calls.get("corrupt_cache"):
                detail["cpu_gradient_cache_bytes"] += 4
            on_target_complete(detail)

        result = original_collector(current, targets, batches, callback, on_target_complete=observed)
        references.extend(weakref.ref(value) for value in result.values())
        if calls.get("leak"):
            leaked.append(next(iter(result.values())))
        return result

    monkeypatch.setenv("SLURM_JOB_ID", "cpu-test-only")
    monkeypatch.setattr(train, "require_gpu_allocation", lambda: None)
    monkeypatch.setattr(probe, "output_path", lambda value: tmp_path / value)
    monkeypatch.setattr(probe, "load_local_model", lambda *_: (
        model, SimpleNamespace(encode=lambda value, **_: [int(value)])))
    monkeypatch.setattr(probe, "seed_everything", lambda _: None)
    monkeypatch.setattr(torch, "autocast", lambda *_, **__: nullcontext())
    monkeypatch.setattr(torch.Tensor, "to", cpu_transfer)
    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    monkeypatch.setattr(preparation, "collect_family_gradients", collector)
    for name, value in {"memory_allocated": 1000, "memory_reserved": 2000,
                        "max_memory_allocated": 1500, "max_memory_reserved": 2000}.items():
        monkeypatch.setattr(torch.cuda, name, lambda value=value: value)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "A40 CPU TEST STUB")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=3000))
    output = "outputs/chloe/preflight/multi-target2048/spread12-v1.json"
    monkeypatch.setattr(sys, "argv", ["multi_target_preflight.py", "--case", "spread12",
                                     "--sampling-steps", "2", "--output", output, "--execute"])
    return SimpleNamespace(path=tmp_path / output, model=model, calls=calls,
                           references=references, leaked=leaked)


def test_spread12_cpu_orchestration_accumulates_then_releases_each_family(cpu_probe):
    assert probe.main() == 0
    result = json.loads(cpu_probe.path.read_text())
    progress = [json.loads(line) for line in cpu_probe.path.with_suffix(".targets.jsonl").read_text().splitlines()]
    assert result["status"] == "MULTI_TARGET_PROBE_COMPLETED_REQUIRES_REVIEW"
    assert result["per_target"] == progress and len(progress) == 12
    assert result["measured_forward_calls"] == result["measured_autograd_grad_calls"] == cpu_probe.calls["autograd"] == 24
    assert result["measured_optimizer_updates"] == 0
    assert [row["cpu_gradient_cache_bytes"] for row in progress] == [64, 128, 192] * 4
    assert all(row["cpu_gradient_cache_bytes"] == row["expected_cpu_gradient_cache_bytes"] for row in progress)
    assert len(result["per_family"]) == 4
    for row in result["per_family"]:
        assert row["released_gradient_bytes"] == 192
        assert row["cpu_gradient_tensors_released"] and row["cpu_gradient_live_tensor_count"] == 0
        assert row["cuda_allocated_after_release_bytes"] == 1000
        assert row["cuda_reserved_after_release_bytes"] == 2000
    assert all(ref() is None for ref in cpu_probe.references)
    assert not any(p.requires_grad or p.grad is not None for p in cpu_probe.model.parameters())


def test_probe_stops_if_a_released_cpu_mean_is_still_referenced(cpu_probe):
    cpu_probe.calls["leak"] = True
    with pytest.raises(ValueError, match="still referenced"):
        probe.main()
    result = json.loads(cpu_probe.path.read_text())
    assert result["status"] == "FAILED" and len(result["per_target"]) == 3
    assert result["per_family"][0]["cpu_gradient_live_tensor_count"] == 1
    assert not result["per_family"][0]["cpu_gradient_tensors_released"]


def test_probe_stops_on_incorrect_cpu_cache_size(cpu_probe):
    cpu_probe.calls["corrupt_cache"] = True
    with pytest.raises(ValueError, match="actual target prefix"):
        probe.main()
    assert json.loads(cpu_probe.path.read_text())["status"] == "FAILED"
    assert cpu_probe.calls["autograd"] == 2


def test_probe_stops_at_first_target_boundary_over_budget(cpu_probe, monkeypatch):
    times = chain([0, 0, 0], repeat(601))
    monkeypatch.setattr(probe, "perf_counter", lambda: next(times))
    with pytest.raises(TimeoutError, match="target boundary"):
        probe.main()
    result = json.loads(cpu_probe.path.read_text())
    assert result["status"] == "FAILED" and len(result["per_target"]) == 1
    assert cpu_probe.calls["autograd"] == 2


@pytest.mark.parametrize("existing_suffix", [".json", ".targets.jsonl"])
def test_probe_preserves_existing_report_or_progress(cpu_probe, existing_suffix):
    existing = cpu_probe.path.with_suffix(existing_suffix)
    existing.parent.mkdir(parents=True)
    existing.write_text("protected existing result\n")
    with pytest.raises(FileExistsError, match="protected"):
        probe.main()
    assert existing.read_text() == "protected existing result\n"
    assert cpu_probe.calls["autograd"] == 0
