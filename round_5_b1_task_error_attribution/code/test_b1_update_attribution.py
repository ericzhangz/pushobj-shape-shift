import unittest

import torch

from research.reframe_v3.b1_update_attribution import compare_pair, telescope_candidate
from research.reframe_v3.matched_feedback_forecast import rollout_from_zobs
from research.reframe_v3.test_processing_composition import HistoryModel


class ShiftHistoryModel(HistoryModel):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def predict(self, z):
        result = super().predict(z)
        result[..., :6] += self.shift
        return result


class B1AttributionTests(unittest.TestCase):
    def setUp(self):
        self.anchor = {"visual": torch.tensor([[[[.1, -.2, .3, .4]]]], dtype=torch.float64),
                       "proprio": torch.tensor([[[.2, -.1]]], dtype=torch.float64)}
        self.goal = {key: value + .25 for key, value in self.anchor.items()}
        self.actions = torch.tensor([[[.1], [.2], [-.1], [.3], [0.]]], dtype=torch.float64)

    def truth(self, actions):
        predicted, _ = rollout_from_zobs(HistoryModel(), self.anchor, actions)
        truth = {key: value.clone() for key, value in predicted.items()}
        for value in truth.values():
            value[:, 0:1] += .03
            value[:, 1:] += .05
        return truth

    def test_update_pair_telescope_with_changed_predictor_and_anchor(self):
        a, b = self.actions, self.actions + .2
        ta, tb = self.truth(a), self.truth(b)
        before_left = telescope_candidate(ShiftHistoryModel(0.), self.anchor, ta, a, self.goal)
        before_right = telescope_candidate(ShiftHistoryModel(0.), self.anchor, tb, b, self.goal)
        after_left = telescope_candidate(ShiftHistoryModel(.07), self.anchor, ta, a, self.goal)
        after_right = telescope_candidate(ShiftHistoryModel(.07), self.anchor, tb, b, self.goal)
        for row in (before_left, before_right, after_left, after_right):
            self.assertLess(abs(row["closure"]), 1e-12)
            self.assertGreater(abs(row["anchor_shift"]), 1e-5)
        pair = compare_pair(before_left, before_right, after_left, after_right)
        self.assertLess(abs(pair["closure"]), 1e-12)
        self.assertAlmostEqual(pair["pair_error_change"],
            (before_left["model_cost"] - before_right["model_cost"])
            - (after_left["model_cost"] - after_right["model_cost"]), places=12)
        self.assertNotEqual(pair["pair_error_change"], 0.)


if __name__ == "__main__":
    unittest.main()
