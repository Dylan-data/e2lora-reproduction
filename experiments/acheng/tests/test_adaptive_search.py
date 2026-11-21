import unittest

import torch

from experiments.acheng.scripts.adaptive_search import (
    dynamic_programming_intervals,
    greedy_intervals,
    perturb_rmi,
)


class AdaptiveSearchTests(unittest.TestCase):
    def setUp(self):
        self.rmi = torch.tensor(
            [
                [1.0, 0.95, 0.10, 0.10],
                [0.95, 1.0, 0.20, 0.10],
                [0.10, 0.20, 1.0, 0.90],
                [0.10, 0.10, 0.90, 1.0],
            ],
            dtype=torch.float64,
        )

    def test_greedy_uses_contiguous_boundaries_and_row_thresholds(self):
        result = greedy_intervals(self.rmi)
        self.assertEqual(result["intervals"], [(0, 2), (2, 4)])
        self.assertEqual(len(result["thresholds"]), 4)

    def test_dp_cost_uses_same_matrix_and_penalty_controls_partition(self):
        unpenalized = dynamic_programming_intervals(self.rmi, penalty_lambda=0.0)
        penalized = dynamic_programming_intervals(self.rmi, penalty_lambda=2.0)
        self.assertEqual(unpenalized["intervals"], [(0, 2), (2, 4)])
        self.assertEqual(penalized["intervals"], [(0, 4)])
        self.assertGreaterEqual(unpenalized["search_ns"], 0)
        self.assertIn("singleton cost=0", unpenalized["cost_definition"])

    def test_noise_is_reproducible_symmetric_and_pre_threshold(self):
        first = perturb_rmi(self.rmi, sigma=0.01, seed=18)
        second = perturb_rmi(self.rmi, sigma=0.01, seed=18)
        self.assertTrue(torch.equal(first["noise"], second["noise"]))
        self.assertTrue(torch.equal(first["noisy_rmi"], self.rmi + first["noise"]))
        self.assertTrue(torch.equal(first["noise"], first["noise"].T))
        self.assertTrue(torch.equal(torch.diag(first["noise"]), torch.zeros(4)))
        self.assertFalse(torch.equal(first["noise"], torch.zeros_like(first["noise"])))

    def test_invalid_matrix_and_lambda_fail_explicitly(self):
        with self.assertRaises(ValueError):
            greedy_intervals([[1.0, 0.2], [0.3, 1.0]])
        with self.assertRaises(ValueError):
            dynamic_programming_intervals(self.rmi, penalty_lambda=-0.1)


if __name__ == "__main__":
    unittest.main()
