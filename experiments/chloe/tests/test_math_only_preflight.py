import pytest

from gpu_preflight import probe_plan


def test_memory_probes_remain_bounded_and_cannot_claim_formal_readiness():
    train = probe_plan(stage="train", checkpointing="on", rank=8, updates=2, accumulation=2)
    assert train["sequence_length"] == 2048 and train["optimizer_updates"] == 2
    assert train["formal_global_batch_target"] == 64 and train["formal_batch_validated"] is False
    assert train["eligible_for_formal_tables"] is False and train["gpu_jobs_submitted"] == 0
    gradient = probe_plan(stage="gradient", checkpointing="off", rank=8, updates=2, accumulation=2)
    assert gradient["optimizer_updates"] == 0 and len(gradient["gradient_targets"]) == 1
    with pytest.raises(ValueError, match="eval"):
        probe_plan(stage="gradient", checkpointing="on", rank=8, updates=1, accumulation=1)
    with pytest.raises(ValueError, match="bounded"):
        probe_plan(stage="train", checkpointing="on", rank=8, updates=1563, accumulation=64)
