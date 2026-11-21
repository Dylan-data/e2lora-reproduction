from pathlib import Path
import sys

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
torch.set_num_threads(1)


def pytest_addoption(parser):
    parser.addoption('--cluster-assets',action='store_true',
                     help='Also run historical audits that require school-local model headers/smoke artifacts')


def pytest_configure(config):
    config.addinivalue_line('markers','cluster_assets: requires existing school-local model or historical smoke files')


def pytest_collection_modifyitems(config,items):
    asset_tests={
        'test_instantiated_llama_budget_is_not_halved_by_grouping_alone',
        'test_real_weight_headers_imply_five_gib_total_and_two_gib_family_cache',
        'test_actual_meta_counts_match_stage_a_b_and_explain_rounding',
        'test_every_cached_interval_covers_layers_once_and_counts_actual_shapes',
        'test_even_maximal_rounding_cannot_reach_table5_parameter_scale',
    }
    for item in items:
        if item.originalname in asset_tests:
            item.add_marker(pytest.mark.cluster_assets)
            if not config.getoption('--cluster-assets'):
                item.add_marker(pytest.mark.skip(reason='School-local assets audit; use --cluster-assets on the school checkout'))
