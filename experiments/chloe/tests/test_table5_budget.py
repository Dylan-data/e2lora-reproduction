"""Table 5 budget invariants, actual meta installation, and counting semantics."""
import sys

import pytest
import torch
from torch import nn

import audit_table5_budget as audit
from common import attach_trainable_plan, parameter_report
from e2lora.adapters.adaptive import allocate_ranks


@pytest.fixture(scope="module")
def static_audit():
    return audit.build_audit()


def test_actual_meta_counts_match_stage_a_b_and_explain_rounding(static_audit):
    cases = static_audit["cases"]
    assert {name: row["physical_unique_trainable_parameters"] for name, row in cases.items()} == {
        "lora_r4": 3407872, "lora_r8": 6815744, "e2lora_r8": 6810624, "e2lora_r16": 13631488}
    assert cases["e2lora_r8"]["parameter_rounding_delta"] == -5120
    assert cases["e2lora_r16"]["parameter_rounding_delta"] == 0
    assert cases["e2lora_r8"]["actual_minus_displayed_center"] == 3200624
    assert cases["e2lora_r16"]["actual_minus_displayed_center"] == 6411488
    assert static_audit["count_of_unique_trainable_tensors_matches_stored_stage_A_B"]


def test_every_cached_interval_covers_layers_once_and_counts_actual_shapes(static_audit):
    for label in ("e2lora_r8", "e2lora_r16"):
        case = static_audit["cases"][label]
        assert len(case["groups"]) == 24 and case["unique_trainable_tensors"] == 48
        for family in audit.FAMILIES:
            rows = [row for row in case["groups"] if row["family"] == family]
            assert [index for row in rows for index in row["layer_indices"]] == list(range(32))
            output = 4096 if family in ("q_proj", "o_proj") else 1024
            assert all(row["A_shape"] == [row["rank"], 4096] and
                       row["B_shape"] == [output, row["rank"]] for row in rows)
            assert all(row["actual_shared_parameter_identity_verified"] and
                       row["cross_group_parameters_distinct_verified"] for row in rows)
            assert sum(row["rank"] * (4096 + output) for row in rows) == case["families"][family]["physical_unique_trainable_parameters"]
            assert case["families"][family]["paper_family_parameter_target"] is None
        assert case["active_trainable_coordinate_parameters"] == case["physical_unique_trainable_parameters"]
        assert case["logical_layer_uses_parameter_elements_not_unique"] > case["physical_unique_trainable_parameters"]


@pytest.mark.parametrize("sign", [1, -1])
def test_all_partitions_preserve_rank_budget_up_to_nearest_integer_error(sign):
    entropy = [sign * (i + 1) for i in range(8)]
    for mask in range(2 ** 7):
        cuts = [0] + [i for i in range(1, 8) if mask & (1 << (i - 1))] + [8]
        intervals = list(zip(cuts, cuts[1:]))
        _, quotas, independently_rounded = audit.literal_allocation(entropy, intervals, 16)
        actual = allocate_ranks(entropy, intervals, 16, rank_policy="paper")
        assert actual == independently_rounded
        assert sum(quotas) == pytest.approx(128)
        assert abs(sum(actual) - 128) <= len(intervals) / 2


def test_even_maximal_rounding_cannot_reach_table5_parameter_scale(static_audit):
    bound = static_audit["maximum_rounding_delta_for_any_32_layer_partition"]
    assert bound == 425984
    for label in ("e2lora_r8", "e2lora_r16"):
        row = static_audit["cases"][label]
        lowest = row["same_nominal_rank_independent_lora_parameters"] - bound
        assert lowest > row["conventional_two_decimal_M_display_bin"][1]


def test_zero_a_gradient_at_zero_b_initialization_does_not_halve_parameters():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    plan = [dict(family="q_proj", layers=["0", "1"], rank=2)]
    attach_trainable_plan(model, plan, alpha=4)
    before = parameter_report(model, plan)
    model(torch.randn(3, 4)).square().sum().backward()
    assert model[0].down.weight is model[1].down.weight
    assert model[0].up.weight is model[1].up.weight
    assert model[0].down.weight.grad is not None
    assert torch.count_nonzero(model[0].down.weight.grad) == 0
    assert torch.count_nonzero(model[0].up.weight.grad) > 0
    assert parameter_report(model, plan)["total_trainable_parameters"] == before["total_trainable_parameters"] == 16


def test_forward_slice_changes_active_coordinates_without_shrinking_parameter():
    parameter = nn.Parameter(torch.ones(8, 4))
    parameter[:2].sum().backward()
    assert parameter.requires_grad and parameter.numel() == 32
    assert torch.count_nonzero(parameter.grad) == 8
    assert torch.count_nonzero(parameter.grad[2:]) == 0


def test_default_entropy_shift_is_not_the_explicit_paper_policy():
    entropy, intervals = [-1.0, -4.0], [(0, 1), (1, 2)]
    assert allocate_ranks(entropy, intervals, 8, rank_policy="paper") == [3, 13]
    assert allocate_ranks(entropy, intervals, 8) == [13, 3]


def test_audit_refuses_to_overwrite_existing_result(tmp_path, monkeypatch):
    destination = tmp_path / "audit.json"
    destination.write_text("existing audit\n")
    monkeypatch.setattr(sys, "argv", ["audit_table5_budget.py", "--output", str(destination)])
    with pytest.raises(FileExistsError, match="protected"):
        audit.main()
    assert destination.read_text() == "existing audit\n"
