from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from research.reframe_v3.propagation_audit import (
    _aggregate_ranges,
    _encode_each_frame,
    _range,
    _real_state_one_step_rollout,
    _sequential_rollout,
)


class PropagationAuditTests(unittest.TestCase):
    def test_range(self) -> None:
        self.assertEqual(_range([3.0, -1.0, 2.0]), 4.0)

    def test_aggregate_ranges_keeps_depths_separate(self) -> None:
        rows = [
            {
                "anchor_id": "a",
                "depth": depth,
                "real_goal_cost": real,
                "free_model_goal_cost": free,
                "teacher_forced_model_goal_cost": teacher,
                "real_state_one_step_model_goal_cost": teacher,
            }
            for depth, real, free, teacher in (
                (1, 0.0, 0.0, 0.0),
                (1, 2.0, 1.0, 1.5),
                (2, 1.0, 4.0, 3.0),
                (2, 4.0, 5.0, 3.5),
            )
        ]
        result = _aggregate_ranges(rows)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["real_goal_cost_range"], 2.0)
        self.assertEqual(result[0]["free_to_real_range_ratio"], 0.5)
        self.assertEqual(result[1]["teacher_forced_model_goal_cost_range"], 0.5)
        self.assertEqual(result[1]["real_state_one_step_model_goal_cost_range"], 0.5)

    def test_sequential_rollout_restarts_from_each_latest_frame(self) -> None:
        initial = {
            "visual": torch.tensor([[[1.0]]]),
            "proprio": torch.tensor([[[10.0]]]),
        }
        actions = torch.tensor([[[2.0], [3.0]]])

        def fake_rollout(_model, current, action):
            increment = action[:, :1]
            predicted = {
                "visual": current["visual"] + increment,
                "proprio": current["proprio"] + 10.0 * increment,
            }
            return predicted, predicted

        with patch(
            "research.reframe_v3.propagation_audit.rollout_from_zobs",
            side_effect=fake_rollout,
        ) as mocked:
            result = _sequential_rollout(object(), initial, actions)

        self.assertEqual(mocked.call_count, 2)
        torch.testing.assert_close(
            result["visual"], torch.tensor([[[1.0], [3.0], [6.0]]])
        )
        torch.testing.assert_close(
            result["proprio"], torch.tensor([[[10.0], [30.0], [60.0]]])
        )

    def test_encode_each_frame_preserves_original_time_order(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.batch_shapes = []

            def encode_obs(self, obs):
                self.batch_shapes.append(tuple(obs["visual"].shape))
                return {
                    "visual": obs["visual"] + 1.0,
                    "proprio": obs["proprio"] + 2.0,
                }

        model = FakeModel()
        obs = {
            "visual": torch.tensor([[[1.0], [2.0], [3.0]]]),
            "proprio": torch.tensor([[[10.0], [20.0], [30.0]]]),
        }
        result = _encode_each_frame(model, obs)

        self.assertEqual(model.batch_shapes, [(1, 1, 1)] * 3)
        torch.testing.assert_close(
            result["visual"], torch.tensor([[[2.0], [3.0], [4.0]]])
        )
        torch.testing.assert_close(
            result["proprio"], torch.tensor([[[12.0], [22.0], [32.0]]])
        )

    def test_real_state_one_step_rollout_resets_before_every_action(self) -> None:
        real = {
            "visual": torch.tensor([[[1.0], [100.0], [1000.0]]]),
            "proprio": torch.tensor([[[10.0], [200.0], [2000.0]]]),
        }
        actions = torch.tensor([[[2.0], [3.0]]])

        def fake_rollout(_model, current, action):
            increment = action[:, :1]
            predicted = {
                "visual": current["visual"] + increment,
                "proprio": current["proprio"] + 10.0 * increment,
            }
            return predicted, predicted

        with patch(
            "research.reframe_v3.propagation_audit.rollout_from_zobs",
            side_effect=fake_rollout,
        ) as mocked:
            result = _real_state_one_step_rollout(object(), real, actions)

        self.assertEqual(mocked.call_count, 2)
        torch.testing.assert_close(
            result["visual"], torch.tensor([[[1.0], [3.0], [103.0]]])
        )
        torch.testing.assert_close(
            result["proprio"], torch.tensor([[[10.0], [30.0], [230.0]]])
        )


if __name__ == "__main__":
    unittest.main()
