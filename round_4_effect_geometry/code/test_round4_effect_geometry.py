"""Independent arithmetic controls for the Round 4 evaluation geometry."""

import math
import unittest

import numpy as np

from research.reframe_v3.round4_effect_geometry import _finite_gain, pair_metrics


class Round4GeometryTests(unittest.TestCase):
    def test_signed_and_radial_direction_closure(self):
        zi, zj = np.array([2., 0.]), np.array([0., 0.])
        pi, pj = np.array([1.5, 0.6]), np.array([0.1, -0.2])
        result = pair_metrics(zi, zj, pi, pj, np.array([0.3, -0.5]))
        self.assertLess(abs(result["signed_closure"]), 1e-12)
        self.assertLess(abs(result["vector_closure"]), 1e-12)

    def test_common_translation_changes_cost_without_effect_norm_change(self):
        zi, zj, goal = np.array([2., 0.]), np.array([0., 0.]), np.zeros(2)
        shift = np.array([1., 0.])
        result = pair_metrics(zi, zj, zi - shift, zj - shift, goal)
        self.assertAlmostEqual(result["real_effect_norm"], result["predicted_effect_norm"])
        self.assertAlmostEqual(result["effect_contribution"], 0.)
        self.assertNotAlmostEqual(result["midpoint_contribution"], 0.)
        self.assertLess((np.linalg.norm(zi - shift) ** 2 + np.linalg.norm(zj - shift) ** 2),
                        np.linalg.norm(zi) ** 2 + np.linalg.norm(zj) ** 2)

    def test_rotation_preserves_effect_norm_but_flips_task_ranking(self):
        zi, zj = np.array([1., 0.]), np.array([-1., 0.])
        pi, pj = np.array([0., 1.]), np.array([0., -1.])
        result = pair_metrics(zi, zj, pi, pj, np.array([0.8, -0.8]))
        self.assertAlmostEqual(result["real_effect_norm"], result["predicted_effect_norm"])
        self.assertLess(result["real_contrast"], 0.)
        self.assertGreater(result["predicted_contrast"], 0.)

    def test_correct_contraction_has_zero_finite_pair_gain(self):
        true_d1, true_d5 = 2., 1.
        predicted_d1, predicted_d5 = true_d1, true_d5
        value = math.log(predicted_d5 / predicted_d1) - math.log(true_d5 / true_d1)
        self.assertAlmostEqual(value, 0.)

    def test_masked_intermediate_breaks_telescope_but_not_endpoint(self):
        rows = [{"pair_id": "a__b", "anchor_id": "a", "mode": "free", "depth": depth,
                 "frozen_nonfrozen": True, "frozen_adapted": False,
                 "real_effect_norm": 1.0, "predicted_effect_norm": 0.0 if depth == 3 else 1.0}
                for depth in range(1, 6)]
        gains = _finite_gain(rows)
        self.assertFalse(gains[2]["endpoint_gain_defined"])
        self.assertTrue(gains[3]["endpoint_gain_defined"])
        self.assertFalse(gains[3]["adjacent_gain_defined"])
        self.assertFalse(gains[3]["telescoping_defined"])
        self.assertFalse(gains[4]["telescoping_defined"])


if __name__ == "__main__":
    unittest.main()
