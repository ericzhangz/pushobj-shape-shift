"""Unit tests for the pre-registered common-continuation experiment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.reframe_v3.common_continuation import (
    continuation_seed,
    deduplicate_first_chunks,
    paired_outcome,
)


class CommonContinuationTests(unittest.TestCase):
    def test_seed_depends_on_anchor_and_tail_step_not_branch(self) -> None:
        self.assertEqual(continuation_seed(260920, 3, 2), 260952)
        self.assertEqual(continuation_seed(260920, 3, 2), 260952)
        self.assertNotEqual(
            continuation_seed(260920, 3, 2),
            continuation_seed(260920, 3, 3),
        )

    def test_exact_first_chunk_dedup_keeps_all_logical_labels(self) -> None:
        shared = torch.zeros(1, 1, 10)
        different = torch.ones(1, 1, 10)
        branches = deduplicate_first_chunks(
            [
                {"label": "frozen", "candidate_id": "a", "first_chunk": shared},
                {"label": "adapted", "candidate_id": "b", "first_chunk": different},
                {"label": "oracle", "candidate_id": "a", "first_chunk": shared.clone()},
            ]
        )

        self.assertEqual(len(branches), 2)
        self.assertEqual(branches[0]["labels"], ["frozen", "oracle"])
        self.assertEqual(branches[1]["labels"], ["adapted"])

    def test_paired_outcome_uses_after_minus_before(self) -> None:
        before = {
            "label": "frozen",
            "final_state_dist": 5.0,
            "final_reference_terminal_cost": 2.0,
            "first_chunk_state_dist": 6.0,
            "first_chunk_reference_terminal_cost": 3.0,
            "final_success": False,
        }
        after = {
            "label": "adapted",
            "final_state_dist": 4.0,
            "final_reference_terminal_cost": 2.5,
            "first_chunk_state_dist": 5.5,
            "first_chunk_reference_terminal_cost": 2.8,
            "final_success": True,
        }

        result = paired_outcome(before, after, "adapted_minus_frozen")

        self.assertEqual(result["delta_final_state_dist"], -1.0)
        self.assertEqual(result["delta_final_reference_terminal_cost"], 0.5)
        self.assertEqual(result["delta_final_success"], 1)


if __name__ == "__main__":
    unittest.main()

