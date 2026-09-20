from __future__ import annotations

import unittest

import torch

from research.reframe_v3.existing_data_reanalysis import (
    _action_metrics,
    _direction_counts,
)


class ExistingDataReanalysisTests(unittest.TestCase):
    def test_direction_counts_inherit_fixed_epsilon(self) -> None:
        result = _direction_counts([-2e-6, -1e-7, 0.0, 1e-7, 2e-6])
        self.assertEqual(result, {"improved": 1, "unchanged": 3, "worsened": 1})

    def test_action_metrics_separate_full_plan_and_first_chunk(self) -> None:
        frozen = torch.zeros(1, 5, 2)
        candidate = frozen.clone()
        candidate[:, 3, 0] = 2.0
        metrics = _action_metrics(candidate, frozen)
        self.assertEqual(metrics["first_chunk_delta_from_frozen_l2"], 0.0)
        self.assertEqual(metrics["full_action_delta_from_frozen_l2"], 2.0)
        self.assertTrue(metrics["first_chunk_exactly_frozen"])
        self.assertFalse(metrics["full_action_exactly_frozen"])


if __name__ == "__main__":
    unittest.main()
