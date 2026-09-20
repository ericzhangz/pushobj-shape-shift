"""Unit tests for the common-anchor shadow-selection audit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.reframe_v3.shadow_selection_audit import (
    experience_reuse_schedules,
    official_prefix_schedule,
    processed_transition_counts,
    selection_summary,
)


class ShadowSelectionAuditTests(unittest.TestCase):
    def test_official_schedule_replays_growing_factual_prefix(self) -> None:
        self.assertEqual(official_prefix_schedule(2), [(0,), (0, 1)])
        self.assertEqual(
            official_prefix_schedule(4),
            [(0,), (0, 1), (0, 1, 2), (0, 1, 2, 3)],
        )

    def test_time_separated_schedules_keep_update_budget_controls(self) -> None:
        schedules = experience_reuse_schedules()
        self.assertEqual(schedules["frozen"], [])
        self.assertEqual(schedules["e1_once"], [(0, 1)])
        self.assertEqual(schedules["e2_once"], [(2, 3)])
        self.assertEqual(schedules["e1_then_e2"], [(0, 1), (2, 3)])
        self.assertEqual(schedules["e1_then_e1"], [(0, 1), (0, 1)])
        self.assertEqual(schedules["e2_then_e2"], [(2, 3), (2, 3)])
        self.assertEqual(processed_transition_counts(schedules["e1_then_e2"]), (4, 4))
        self.assertEqual(processed_transition_counts(schedules["e1_then_e1"]), (4, 2))

    def test_selection_summary_separates_exact_choice_from_numeric_ties(self) -> None:
        rows = [
            {"candidate_id": "a", "c_model": 0.0, "c_env": 5.0},
            {"candidate_id": "b", "c_model": 0.5e-6, "c_env": 1.0},
            {"candidate_id": "c", "c_model": 2.0, "c_env": 0.0},
        ]

        result = selection_summary(rows, epsilon=1e-6)

        self.assertEqual(result["model_best_candidate_id"], "a")
        self.assertEqual(result["env_best_candidate_id"], "c")
        self.assertEqual(result["model_tie_count_epsilon"], 2)
        self.assertEqual(result["r_selected"], 5.0)
        self.assertEqual(result["r_selected_tie_min"], 1.0)
        self.assertEqual(result["r_selected_tie_max"], 5.0)

    def test_selection_summary_rejects_nonfinite_or_duplicate_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            selection_summary(
                [
                    {"candidate_id": "a", "c_model": 0.0, "c_env": 1.0},
                    {"candidate_id": "a", "c_model": 1.0, "c_env": 0.0},
                ],
                epsilon=1e-6,
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            selection_summary(
                [{"candidate_id": "a", "c_model": float("nan"), "c_env": 0.0}],
                epsilon=1e-6,
            )


if __name__ == "__main__":
    unittest.main()

