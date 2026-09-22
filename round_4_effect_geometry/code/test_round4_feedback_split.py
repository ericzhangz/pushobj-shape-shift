"""Algebraic control for the optional ordered feedback split."""

import unittest

import numpy as np

from research.reframe_v3.round4_feedback_split import record_terms


class FeedbackSplitTests(unittest.TestCase):
    def test_three_term_closure_and_signed_cross_terms(self):
        input_term = np.array([2., -1.])
        action_term = np.array([-0.5, 3.])
        residual = np.array([1., 0.25])
        observed = input_term + action_term + residual
        row = record_terms({}, input_term, action_term, residual, observed)
        reconstructed_sq = (row["input_term_norm_sq"] + row["action_term_norm_sq"]
                            + row["residual_term_norm_sq"]
                            + 2 * (row["input_action_inner"] + row["input_residual_inner"]
                                   + row["action_residual_inner"]))
        self.assertAlmostEqual(row["observed_error_norm_sq"], reconstructed_sq)
        self.assertAlmostEqual(row["closure_max_abs"], 0.)


if __name__ == "__main__":
    unittest.main()
