"""CPU numerical verification of the installed shared partial-rank primitives."""
import unittest
import torch
from torch import nn
from team_rank_check import exact_ranks, sampling_positions, snapshot, fixed_snapshot, assert_fixed, audit
from e2lora.adapters.partial_rank import active_rank_plan, apply_partial_plan, count_adapters


class Check(unittest.TestCase):
    def test_exact_budget(self):
        self.assertEqual(exact_ranks([-1., -2., -3.], 256), [43, 85, 128])
        with self.assertRaises(ValueError):
            exact_ranks([-1., 2.], 256)

    def test_sampling(self):
        positions = sampling_positions(100, 20, 42)
        self.assertEqual(positions, sampling_positions(100, 20, 42))
        self.assertEqual(len(set(positions)), 20)
        self.assertNotEqual(positions, sampling_positions(100, 20, 43))

    def test_frozen_update_and_shared_identity(self):
        torch.manual_seed(42)
        model = nn.Module()
        model.a, model.b = nn.Linear(8, 4), nn.Linear(8, 4)
        plan = active_rank_plan([dict(family='q_proj', layers=['a', 'b'], rank=8)], {'q_proj': 4})
        apply_partial_plan(model, plan, alpha=16, frozen_b_init_scale=.01)
        initial, frozen = snapshot(model), fixed_snapshot(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.1, weight_decay=.1)
        for _ in range(3):
            optimizer.zero_grad()
            inputs = torch.randn(2, 8)
            (model.a(inputs) + model.b(inputs)).square().mean().backward()
            optimizer.step()
            assert_fixed(model, frozen)
        self.assertIs(model.a.down, model.b.down)
        self.assertIs(model.a.up, model.b.up)
        changed = snapshot(model)
        self.assertFalse(torch.equal(initial['a.down.active'], changed['a.down.active']))
        self.assertFalse(torch.equal(initial['a.up.active'], changed['a.up.active']))
        self.assertEqual(count_adapters(model, plan)['trainable_adapter_parameters'], 48)

    def test_llama_shape_meta_counts(self):
        # Shape-only fixture; the GPU run verifies the actual loaded model separately.
        for share_a, expected in [(False, 3700736), (True, 1423360)]:
            with torch.device('meta'):
                model = nn.Module()
                plan = []
                for family, out_width in [('q_proj',4096),('k_proj',1024),('v_proj',1024),('o_proj',4096)]:
                    for i, rank in enumerate([80, 176]):
                        name = f'{family}_{i}'
                        setattr(model, name, nn.Linear(4096, out_width, bias=False))
                        plan.append(dict(family=family, layers=[name], rank=rank))
                plan = active_rank_plan(plan, {p:139 for p in ['q_proj','k_proj','v_proj','o_proj']})
                apply_partial_plan(model, plan, alpha=16, frozen_b_init_scale=.01, global_down=share_a)
            self.assertEqual(count_adapters(model, plan)['trainable_adapter_parameters'], expected)


if __name__ == '__main__':
    unittest.main()
