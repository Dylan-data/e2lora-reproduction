from copy import deepcopy
import itertools
import math
import random

import pytest
import torch
from torch import nn

from ablations import METHODS, compile_batch, generate, materialize, specifications
from alignment import compare_configs, protocol_issues
import common
from e2lora.adapters.ablation import build_ablation_plan
from e2lora.adapters.adaptive import allocate_ranks, gradient_similarity
from e2lora.adapters.grouping import search_intervals
from e2lora.adapters.lora_layers import apply_plan
from e2lora.adapters.partial_rank import (active_rank_plan, adapter_state, apply_partial_plan,
                                         count_adapters)
from e2lora.adapters.preparation import collect_plan
from e2lora.adapters.similarity import CONVENTIONS, similarity_matrix
from formal_runner import fixed_sampling_batches, install
from formal_runner import runtime_config, validate_resolved
from summarize_ablations import audit_budget_pairs, audit_run
from train import fit, load_complete_adapter_state
from test_math_baseline import tiny_run_setup


def exhaustive(matrix, penalty, policy, singleton):
    """Independent oracle: direct slices, enumerate every subset of N-1 cuts."""
    n = len(matrix)
    candidates = []
    for cuts in itertools.product([False, True], repeat=n - 1):
        ends = [0] + [i + 1 for i, cut in enumerate(cuts) if cut] + [n]
        intervals = list(zip(ends, ends[1:]))
        value = penalty * len(intervals)
        for start, end in intervals:
            entries = [matrix[i][j] for i in range(start, end) for j in range(start, end)
                       if policy == "include_diagonal" or i != j]
            value += -sum(entries) / len(entries) if entries else singleton
        candidates.append((value, len(intervals), intervals))
    optimum = min(value for value, _, _ in candidates)
    return optimum, candidates


@pytest.mark.parametrize("policy,singleton", [("include_diagonal", None), ("exclude_diagonal", 0.0),
                                            ("exclude_diagonal", -1.0)])
@pytest.mark.parametrize("penalty", [0.0, .25, 1.0, 3.0])
def test_dp_globally_optimal_against_every_partition_n1_to_n8(policy, singleton, penalty):
    for n in range(1, 9):
        rng = random.Random(913 + n)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                matrix[i][j] = matrix[j][i] = rng.choice([-2, -1, .25, .5, 1, 2])
        intervals, detail = search_intervals(matrix, method="dp", penalty_lambda=penalty,
                                             mean_policy=policy, singleton_cost=singleton)
        optimum, candidates = exhaustive(matrix, penalty, policy, singleton)
        assert detail["objective"] == pytest.approx(optimum, abs=1e-11)
        actual = next(value for value, _, candidate in candidates if candidate == intervals)
        assert actual == pytest.approx(optimum, abs=1e-11)
        assert [i for start, end in intervals for i in range(start, end)] == list(range(n))
        assert detail["grouping_seconds"] >= 0


def test_dp_ties_prefer_fewer_groups_deterministically_and_penalty_changes_partition():
    matrix = [[1.0] * 5 for _ in range(5)]
    assert search_intervals(matrix, method="dp", penalty_lambda=1, mean_policy="include_diagonal")[0] == [(0, 5)]
    assert search_intervals(matrix, method="dp", penalty_lambda=0, mean_policy="include_diagonal")[0] == [(i, i + 1) for i in range(5)]
    block = [[1, .9, .1, .1], [.9, 1, .1, .1], [.1, .1, 1, .9], [.1, .1, .9, 1]]
    assert search_intervals(block, method="greedy")[0] == [(0, 2), (2, 4)]


@pytest.mark.parametrize("matrix,options", [([], {}), ([[1, 2]], {}), ([[float('nan')]], {}),
    ([[1, 2], [0, 1]], {}), ([[1]], {"penalty_lambda": -1}),
    ([[1]], {"penalty_lambda": float('inf')}), ([[1]], {"penalty_lambda": True}),
    ([[1]], {"penalty_lambda": 0, "mean_policy": "exclude_diagonal"})])
def test_dp_invalid_or_ambiguous_inputs_rejected(matrix, options):
    with pytest.raises(ValueError):
        search_intervals(matrix, method="dp", **options)


def gradient_fixture():
    generator = torch.Generator().manual_seed(718)
    # Two distinct gradient distributions in two contiguous blocks.
    return {str(i): torch.randn(4, 4, generator=generator) * scale + mean
            for i, (scale, mean) in enumerate([(1, 0), (1.1, 0), (5, 8), (5.1, 8)])}


def test_switches_change_only_the_corresponding_plan_operation_and_real_sharing():
    grads = gradient_fixture()
    families = {"q": list(grads)}
    options = dict(base_rank=8, rank_policy="paper")
    plans, details = {}, {}
    for lss, hra in itertools.product([False, True], repeat=2):
        plans[lss, hra], details[lss, hra] = build_ablation_plan(grads, families, lss=lss, hra=hra, **options)
    assert len(plans[False, False]) == len(grads)
    assert len(plans[True, False]) < len(grads)
    for lss in [False, True]:
        assert [g["layers"] for g in plans[lss, False]] == [g["layers"] for g in plans[lss, True]]
        assert all(g["rank"] == 8 for g in plans[lss, False])
        assert [g["rank"] for g in plans[lss, True]] != [g["rank"] for g in plans[lss, False]]
        assert details[lss, True]["q"]["entropy"] == details[lss, False]["q"]["entropy"]
    base = nn.Sequential(*[nn.Linear(4, 4) for _ in grads])
    x = torch.randn(2, 4)
    original = base(x).detach()
    for flags, plan in plans.items():
        model = deepcopy(base)
        apply_plan(model, plan, alpha=16)
        torch.testing.assert_close(model(x), original, atol=0, rtol=0)
        for group in plan:
            layers = [model.get_submodule(name) for name in group["layers"]]
            assert all(layer.down is layers[0].down and layer.up is layers[0].up for layer in layers)
        assert count_adapters(model, plan)["trainable_adapter_parameters"] == sum(g["rank"] * 8 for g in plan)


@pytest.mark.parametrize("metric", list(CONVENTIONS))
def test_similarity_changes_metric_but_preserves_entropy_hra_and_inputs(metric):
    grads = gradient_fixture()
    before = {key: value.clone() for key, value in grads.items()}
    entropy, rmi = gradient_similarity(list(grads.values()))
    actual, matrix = similarity_matrix(list(grads.values()), metric=metric, convention=CONVENTIONS[metric])
    assert actual == entropy
    assert not torch.allclose(torch.tensor(matrix), torch.tensor(rmi))
    assert torch.isfinite(torch.tensor(matrix)).all()
    torch.testing.assert_close(torch.tensor(matrix), torch.tensor(matrix).T)
    plan, detail = build_ablation_plan(grads, {"q": list(grads)}, base_rank=8, lss=True, hra=True,
                                       similarity=metric, convention=CONVENTIONS[metric])
    d = detail["q"]
    assert d["ranks"] == allocate_ranks(entropy, d["intervals"], 8, rank_policy="paper")
    for key in grads:
        torch.testing.assert_close(grads[key], before[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="explicit convention"):
        similarity_matrix(list(grads.values()), metric=metric)


def test_alternative_metric_definitions_have_correct_direction_and_distinct_values():
    x, y = torch.tensor([0., 2., 0., 0.]), torch.tensor([2., 0., 2., 0.])
    assert similarity_matrix([x, y], metric="l1", convention=CONVENTIONS["l1"])[1][0][1] == -1.5
    assert similarity_matrix([x, y], metric="l2", convention=CONVENTIONS["l2"])[1][0][1] == pytest.approx(-math.sqrt(3))
    for metric, convention in CONVENTIONS.items():
        _, matrix = similarity_matrix([x, x, y], metric=metric, convention=convention)
        assert matrix[0][1] > matrix[0][2]
    _, matrix = similarity_matrix([x * 0, x * 0, y], metric="cosine", convention=CONVENTIONS["cosine"])
    assert matrix[0][1] == 1 and matrix[0][2] == 0


def test_formal_options_reach_actual_sequential_gradient_collector():
    torch.manual_seed(9)
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 4))
    model.requires_grad_(False)
    batches = [torch.randn(2, 4)]
    loss = lambda m, b: (m(b).square().mean(), len(b))
    plan, details, times = collect_plan(model, {"q": ["0", "2"]}, batches, loss,
        base_rank=4, rank_policy="paper", mechanism_options=dict(lss=False, hra=False))
    assert [g["layers"] for g in plan] == [["0"], ["2"]]
    assert [g["rank"] for g in plan] == [4, 4]
    assert times["backward_passes"] == 2
    assert details["q"]["method"] == "independent"


def test_sampling_steps_are_nested_logical_batches_and_microbatching_is_separate():
    examples = [{"input_ids": [1, i, 2], "labels": [-100, i, 2]} for i in range(3, 43)]
    small, a = fixed_sampling_batches(examples, steps=2, batch_size=4, micro_batch_size=1, seed=7, pad_token_id=2)
    big, b = fixed_sampling_batches(examples, steps=4, batch_size=4, micro_batch_size=2, seed=7, pad_token_id=2)
    assert a["selected_positions"] == b["selected_positions"][:8]
    assert len(small) == len(big) == 8 and a["logical_steps"] == 2 and b["logical_steps"] == 4
    with pytest.raises(ValueError, match="no implicit repetition"):
        fixed_sampling_batches(examples, steps=512, batch_size=4, micro_batch_size=1, seed=7, pad_token_id=2)


def test_partial_ranks_have_real_frozen_contribution_and_optimizer_cannot_change_buffers():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    plan = [{"family": "q", "layers": ["0", "1"], "rank": 4}]
    plan = active_rank_plan(plan, {"q": 2})
    apply_partial_plan(model, plan, alpha=8, frozen_b_init_scale=.1)
    assert model[0].up is model[1].up and model[0].down is model[1].down
    frozen = {key: value.clone() for key, value in model.named_buffers()}
    inputs = torch.randn(3, 4)
    first = model[0]
    contribution = (first.up.fixed @ first.down.fixed)
    assert contribution.abs().max() > 0
    torch.testing.assert_close(first(inputs) - first.base(inputs),
                               nn.functional.linear(inputs, contribution) * first.scale)
    assert count_adapters(model, plan)["trainable_adapter_parameters"] == 2 * (4 + 4)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model(inputs).square().mean().backward()
    optimizer.step()
    for key, value in model.named_buffers():
        torch.testing.assert_close(value, frozen[key], atol=0, rtol=0)
    saved = {key: value.clone() for key, value in adapter_state(model).items()}
    with torch.no_grad():
        model[0].up.fixed.zero_()
    load_complete_adapter_state(model, saved)
    torch.testing.assert_close(model[0].up.fixed, frozen["0.up.fixed"], atol=0, rtol=0)


def test_active_rank_largest_remainder_respects_budget_capacity_and_ties():
    plan = [{"family": "q", "layers": [str(i)], "rank": rank} for i, rank in enumerate([1, 1, 5])]
    for budget in range(8):
        active = active_rank_plan(plan, {"q": budget})
        assert sum(g["trainable_rank"] for g in active) == budget
        assert all(0 <= g["trainable_rank"] <= g["rank"] for g in active)
    assert [g["trainable_rank"] for g in active_rank_plan(plan, {"q": 4})] == [1, 0, 3]
    assert "trainable_rank" not in plan[0]
    with pytest.raises(ValueError):
        active_rank_plan(plan, {"q": 8})


def test_partial_rank_checkpoint_resume_restores_frozen_buffers_under_changed_init(tmp_path):
    config = common.read_json("experiments/chloe/configs/smoke-lora-r8.json")
    config, examples, make = tiny_run_setup(config, attention_dropout=.1)

    def partial(seed):
        model, original_plan = make()
        for group in original_plan:
            name = group["layers"][0]
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent), child, model.get_submodule(name).base)
        plan = [dict(family=p, layers=[f"model.layers.{i}.self_attn.{p}" for i in range(2)], rank=4)
                for p in ["q_proj", "k_proj", "v_proj", "o_proj"]]
        plan = active_rank_plan(plan, {p: 2 for p in ["q_proj", "k_proj", "v_proj", "o_proj"]})
        torch.manual_seed(seed)
        apply_partial_plan(model, plan, alpha=16, frozen_b_init_scale=.01)
        return model, plan

    full, plan = partial(5)
    fit(full, examples, plan, config, pad_token_id=2, checkpoint=tmp_path / "full.pt", fingerprint="partial")
    part, plan = partial(5)
    fit(part, examples, plan, config, pad_token_id=2, checkpoint=tmp_path / "part.pt", fingerprint="partial", stop_after_step=2)
    restored, plan = partial(999)
    fit(restored, examples, plan, config, pad_token_id=2, checkpoint=tmp_path / "part.pt", fingerprint="partial", resume=tmp_path / "part.pt")
    for key, value in adapter_state(full).items():
        torch.testing.assert_close(value, adapter_state(restored)[key], atol=0, rtol=0)


def test_matrix_covers_math_only_all_controls_and_has_no_invented_protocol_values():
    specs = specifications()
    assert len(specs) == 24 and sum(s["independent_runs"] for s in specs) == 72
    assert {s["task"] for s in specs} == {"math"}
    assert {s["method"] for s in specs if s["study"] == "table4"} == set(METHODS)
    for task in ["math"]:
        assert {s["base_rank"] for s in specs if s["task"] == task and s["study"] == "figure5_rank"} == {4, 8, 16, 32, 64}
        assert {s["sampling_steps"] for s in specs if s["task"] == task and s["study"] == "figure5_steps"} == {8, 16, 32, 64, 128, 256, 512}
    # A step sweep changes only the logical number of sampling steps.
    reference = next(s for s in specs if s["run_id"] == "f5-steps-math-n8")
    for spec in specs:
        assert spec["training_seeds"] == [42] and spec["eligible_for_formal_tables"] is False
        if spec["study"] == "figure5_steps" and spec["task"] == "math":
            assert not compare_configs(reference, spec, allowed=["run_id", "sampling_steps"])
    table4 = [s for s in specs if s["study"] == "table4"]
    full = next(s for s in table4 if s["method"] == "e2lora")
    for spec in table4:
        assert not compare_configs(full, spec, allowed=["run_id", "method", "lss", "hra", "similarity"])


def test_pending_protocol_cannot_be_enabled_by_changing_status_only():
    protocol = common.read_json("experiments/chloe/configs/formal-protocol.pending.json")
    protocol["status"] = "confirmed"
    for task in ["math"]:
        issues = protocol_issues(protocol, task, verify_files=False)
        assert "unconfirmed:confirmation.member4" in issues
        assert "paper parameter-count/partial-rank ambiguity is unresolved" in issues
    with pytest.raises(ValueError, match="Protocol blocked"):
        materialize(specifications()[0], protocol, 0)


def test_alignment_detects_prompt_manifest_scorer_and_extra_field_drift():
    protocol = common.read_json("experiments/chloe/configs/formal-protocol.pending.json")
    changed = deepcopy(protocol)
    changed["tasks"]["math"]["prompt"]["train_template"] = "changed"
    changed["tasks"]["math"]["manifest"]["sha256"] = "changed"
    changed["tasks"]["math"]["evaluator"]["sha256"] = "changed"
    changed["extra"] = None
    differences = compare_configs(protocol, changed)
    assert {d["field"] for d in differences} == {"tasks.math.prompt.train_template", "tasks.math.manifest.sha256",
                                               "tasks.math.evaluator.sha256", "extra"}


def test_smoke_records_and_local_evaluations_are_explicitly_excluded():
    for filename, correct in [("lora-smoke-54429.json", 11), ("e2lora-smoke-54466.json", 12)]:
        record = common.read_json("experiments/chloe/report/" + filename)
        assert record["eligible_for_formal_tables"] is False and record["run_type"] == "smoke"
        assert record["correct"] == correct and record["evaluation_examples"] == 32


def confirmed_fixture(tmp_path):
    """Explicit TEST-ONLY protocol. It never changes the repository pending file."""
    protocol = common.read_json("experiments/chloe/configs/formal-protocol.pending.json")

    def supplied(value):
        if isinstance(value, dict):
            return {key: supplied(item) for key, item in value.items()}
        return "TEST_ONLY" if value is None else value

    protocol = supplied(protocol)
    protocol["status"] = "confirmed"
    protocol["model"]["path"] = str(tmp_path)
    protocol["training"].update(seeds=[101, 202, 303], max_grad_norm=1.0,
                                gradient_checkpointing=True, checkpoint_every=2, log_every=10)
    protocol["adapter"].update(alpha=16, dropout=0, adapter_dtype="float32", scaling="alpha_over_group_rank",
        initialization="kaiming_A_zero_trainable_B", loss="response_only_token_weighted", packing=False,
        truncation="keep_BOS_and_prompt_tail_then_truncate_response_right")
    protocol["sampling"].update(batch_size=2, micro_batch_size=1, seed_policy="training_seed", fixed_seed=None,
                                order="manifest_shuffle_prefix", loss_weighting="supervised_response_tokens")
    protocol["mechanisms"].update(table4_base_rank=8, main_sampling_steps=16,
        figure5_rank_fixed_sampling_steps=32, figure5_steps_fixed_rank=16,
        metric_conventions=CONVENTIONS.copy(), adapter_policies={method: {"mode": "all_trainable"} for method in METHODS},
        paper_parameter_resolution_confirmed=True, table5_max_relative_budget_gap=.1)
    artifact = tmp_path / "fixture.json"
    artifact.write_text("{}")
    checksum = common.file_sha256(artifact)
    for task in ["math", "code"]:
        tp = protocol["tasks"][task]
        for label in ["data", "manifest", "evaluator"]:
            tp[label].update(path=str(artifact), sha256=checksum)
        tp["data"]["format"] = "normalized_query_response_json"
        tp["manifest"]["count"] = 100000
        tp["training"].update(epochs=1, max_steps=None, max_length=128, max_prompt_length=64,
                               micro_batch_size=1, gradient_accumulation=64)
        tp["prompt"]["eos_policy"] = "append_eos_before_right_truncation"
        tp["generation"].update(do_sample=False, temperature=None, top_p=None, top_k=None, stop_sequences=[])
        tp["evaluation"].update(task_ids_path=str(artifact), task_ids_sha256=checksum,
                                 environment_lock_path=str(artifact), environment_lock_sha256=checksum)
        tp["smoke_validation"].update(passed=True, report_path=str(artifact), report_sha256=checksum)
        if task == "math":
            tp["formal_preflight"].update(passed=True, report_path=str(artifact), report_sha256=checksum)
        tp["evaluator"]["argv"] = ["{python}", "{script}", "--config", "{config}", "--checkpoint", "{checkpoint}", "--output", "{output}"]
    return protocol, artifact


@pytest.mark.parametrize("task", ["math"])
def test_confirmed_protocol_compiles_and_detects_artifact_or_config_drift(tmp_path, task):
    protocol, artifact = confirmed_fixture(tmp_path)
    assert protocol_issues(protocol, task) == []
    source = next(s for s in specifications() if s["task"] == task and s["study"] == "figure5_rank")
    run = materialize(source, protocol, 1)
    assert run["seed"] == 202 and run["sampling_steps"] == 32
    assert not validate_resolved(run)
    config = runtime_config(run)
    assert config["training"]["max_steps"] is None and config["task"] == task
    changed = deepcopy(run)
    changed["lss"] = False
    assert "compiled experiment differs from canonical specification" in validate_resolved(changed)
    revised_protocol = deepcopy(protocol)
    revised_protocol["adapter"]["alpha"] = 32
    assert materialize(source, revised_protocol, 1)["output_dir"] != run["output_dir"]
    artifact.write_text("changed")
    assert "artifact digest mismatch:manifest" in protocol_issues(protocol, task)


def test_freezing_drift_between_table4_mechanisms_is_blocked(tmp_path):
    protocol, _ = confirmed_fixture(tmp_path)
    protocol["mechanisms"]["adapter_policies"]["sharing_only"] = {
        "mode": "partial_group_ab", "active_rank_per_projection_by_base_rank": {"8": 139},
        "frozen_b_init_scale": .01}
    assert "Table 4 mechanisms must use the same freezing/initialization policy" in protocol_issues(protocol, "math")


def test_formal_installer_uses_explicit_active_budget_for_actual_model(tmp_path):
    protocol, _ = confirmed_fixture(tmp_path)
    spec = materialize(next(s for s in specifications() if s["run_id"] == "t4-e2lora"), protocol, 0)
    protocol = spec["resolved_protocol"]
    protocol["mechanisms"]["adapter_policies"]["e2lora"] = {
        "mode": "partial_group_ab", "active_rank_per_projection_by_base_rank": {"8": 3},
        "frozen_b_init_scale": .01}
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    plan = [dict(family="q", layers=["0", "1"], rank=8)]
    actual, counts = install(model, plan, spec)
    assert actual[0]["trainable_rank"] == 3
    assert counts["trainable_adapter_parameters"] == 24
    assert counts["frozen_adapter_elements"] == 40
    del protocol["mechanisms"]["adapter_policies"]["e2lora"]["active_rank_per_projection_by_base_rank"]["8"]
    with pytest.raises(ValueError, match="No confirmed active budget"):
        install(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)), plan, spec)


def test_compiled_batch_expands_exactly_three_seeds_and_rejects_different_protocol_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path)
    protocol, _ = confirmed_fixture(tmp_path)
    directory = tmp_path / "batch"
    assert compile_batch(protocol, task="all", directory=directory)["runs"] == 72
    manifest = common.read_json(directory / "batch.json")
    assert len({entry["config"] for entry in manifest["entries"]}) == 72
    assert [entry["index"] for entry in manifest["entries"]] == list(range(72))
    protocol["adapter"]["alpha"] = 32
    with pytest.raises(ValueError, match="Existing compiled config differs"):
        compile_batch(protocol, task="all", directory=directory)


def test_code_exclusion_blocks_old_compiled_runs_even_with_complete_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path)
    protocol, _ = confirmed_fixture(tmp_path)
    assert protocol_issues(protocol, "code") == ["task code excluded by group-approved Math-only scope"]
    stale = dict(specifications()[0], task="code", run_id="f5-rank-code-r8")
    with pytest.raises(ValueError, match="excluded"):
        materialize(stale, protocol, 0)
    with pytest.raises(ValueError, match="excluded"):
        compile_batch(protocol, task="code", directory=tmp_path / "code")
    assert not (tmp_path / "code").exists()
    stale.update(resolved_protocol=protocol, protocol_sha256=common.digest(protocol), repeat=0)
    assert any("excluded" in issue for issue in validate_resolved(stale))
    with pytest.raises(ValueError, match="excluded"):
        runtime_config(stale)


def test_regeneration_retires_code_without_deleting_framework_and_writes_only_math_rows(tmp_path, monkeypatch):
    import csv
    import ablations
    root = tmp_path / "experiments/chloe"
    configs = root / "configs/formal"
    configs.mkdir(parents=True)
    retained = configs / "f5-rank-code-r8.json"
    common.write_json(retained, {"task": "code", "run_id": "f5-rank-code-r8"})
    monkeypatch.setattr(ablations, "ROOT", tmp_path)
    monkeypatch.setattr(ablations, "CONFIG_ROOT", configs)
    report = generate()
    assert report["configurations"] == 24 and report["planned_runs"] == 72
    assert report["code_gpu_runs"] == 0 and report["known_seed_rows"] == 24
    assert retained.exists() and common.read_json(retained)["independent_runs"] == 0
    with (root / "formal-matrix.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 72 and {r["task"] for r in rows} == {"math"}
    assert sum(r["seed"] == "PENDING_MEMBER4" for r in rows) == 48


def test_first_seed_batch_does_not_invent_other_seeds_or_accept_old_smoke_as_formal(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path)
    protocol, _ = confirmed_fixture(tmp_path)
    protocol["training"]["seeds"] = [42]
    directory = tmp_path / "first-seed"
    assert compile_batch(protocol, task="math", directory=directory)["runs"] == 24
    manifest = common.read_json(directory / "batch.json")
    assert {row["seed"] for row in manifest["entries"]} == {42}
    with pytest.raises(ValueError, match="has not been confirmed"):
        materialize(specifications()[0], protocol, 1)
    protocol["tasks"]["math"]["formal_preflight"]["passed"] = False
    assert any("short Smoke alone is insufficient" in x for x in protocol_issues(protocol, "math"))


def test_table5_budget_audit_uses_measured_counts_not_rank_labels():
    rows = [dict(run_id="t5-lora-r4", seed=1, study="table5", budget_pair="pair", method="lora",
                 trainable_adapter_parameters=3407872, eligible=True, score=.5, issues=[]),
            dict(run_id="t5-e2lora-r8", seed=1, study="table5", budget_pair="pair", method="e2lora",
                 trainable_adapter_parameters=6810624, eligible=True, score=.6, issues=[])]
    checks = audit_budget_pairs(rows, .1)
    assert checks[0]["matched"] is False
    assert all(row["eligible"] is False and row["score"] is None for row in rows)
    rows[1]["trainable_adapter_parameters"] = 3700736
    assert audit_budget_pairs(rows, .1)[0]["matched"] is True


def test_formal_summary_rejects_smoke_even_if_score_and_count_look_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path)
    protocol, _ = confirmed_fixture(tmp_path)
    run = materialize(specifications()[0], protocol, 0)
    run["output_dir"] = str(tmp_path / "run")
    directory = common.output_path(run["output_dir"])
    directory.mkdir()
    checkpoint = directory / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    record = dict(run_type="smoke", task="math", protocol_sha256=run["protocol_sha256"],
                  checkpoint_sha256=common.file_sha256(checkpoint), run_id=run["run_id"], seed=run["seed"],
                  training={"status": "completed", "progress": {"global_step": 1}, "planned_optimizer_steps": 1},
                  metric="accuracy", evaluation_examples=1319, score=.375,
                  task_ids_sha256=protocol["tasks"]["math"]["evaluation"]["task_ids_sha256"])
    common.write_json(directory / "result.json", record)
    common.write_json(directory / "evaluation.json", record)
    audit = audit_run(run)
    assert not audit["eligible"] and audit["score"] is None
    assert "evaluation: Smoke or wrong task" in audit["issues"]
