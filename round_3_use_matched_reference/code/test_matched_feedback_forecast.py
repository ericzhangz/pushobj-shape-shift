from __future__ import annotations

import unittest

import torch

from research.reframe_v3.matched_feedback_forecast import (
    _selection_diagnostics,
    rollout_from_zobs,
)


class _TinyModel:
    concat_dim = 0
    num_hist = 1

    def encode_act(self, action):
        return action

    def predict(self, z):
        return z + 0.1 * z[:, :, -1:, :].expand_as(z) + 0.25

    def replace_actions_from_z(self, z, action):
        z = z.clone()
        z[:, :, -1, :] = action
        return z

    def separate_emb(self, z):
        return {"visual": z[:, :, :-2], "proprio": z[:, :, -2]}, z[:, :, -1]


class MatchedFeedbackForecastTests(unittest.TestCase):
    def test_rollout_from_zobs_preserves_input_and_action_gradient(self) -> None:
        model = _TinyModel()
        z_obs = {
            "visual": torch.zeros(1, 1, 2, 3),
            "proprio": torch.zeros(1, 1, 3),
        }
        before = {key: value.clone() for key, value in z_obs.items()}
        actions = torch.arange(9.0).reshape(1, 3, 3).requires_grad_(True)
        predicted, full = rollout_from_zobs(model, z_obs, actions)
        self.assertEqual(tuple(predicted["visual"].shape), (1, 4, 2, 3))
        self.assertEqual(tuple(full.shape), (1, 4, 4, 3))
        self.assertTrue(all(torch.equal(z_obs[key], before[key]) for key in z_obs))
        predicted["proprio"].sum().backward()
        self.assertIsNotNone(actions.grad)

    def test_selection_diagnostics_reports_tie_aware_range(self) -> None:
        rows = [
            {"candidate_key": "a", "pred": 0.0, "truth": 3.0},
            {"candidate_key": "b", "pred": 5e-7, "truth": 1.0},
            {"candidate_key": "c", "pred": 2e-6, "truth": 2.0},
        ]
        result = _selection_diagnostics(rows, "pred", "truth", epsilon=1e-6)
        self.assertEqual(result["selected_key"], "a")
        self.assertEqual(result["epsilon_tie_keys"], "a;b")
        self.assertEqual(result["selected_truth_regret"], 2.0)
        self.assertEqual(result["tie_aware_truth_regret_min"], 0.0)
        self.assertEqual(result["tie_aware_truth_regret_max"], 2.0)


if __name__ == "__main__":
    unittest.main()
