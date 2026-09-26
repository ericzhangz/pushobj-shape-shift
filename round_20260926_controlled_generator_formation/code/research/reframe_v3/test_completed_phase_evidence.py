"""CPU-only, time-numbered checks for completed native phase repacking."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from research.reframe_v3.completed_phase_evidence import (
    merge_completed_lowlevel, phase_chains,
)
from research.reframe_v3.shadow_selection_audit import _load_segments


def _numbered(length):
    time = torch.arange(length + 1, dtype=torch.float32)
    observed = {"visual": time.reshape(1, -1, 1, 1),
                "proprio": (100 + time).reshape(1, -1, 1)}
    low = torch.stack((10 * torch.arange(length), 10 * torch.arange(length) + 1), dim=-1)
    return observed, low.float().unsqueeze(0)


def _segments(count):
    observed, low = _numbered(count * 5)
    return [({key: value[:, start:start + 6] for key, value in observed.items()},
             low[:, start:start + 5].reshape(1, 1, 10))
            for start in range(0, count * 5, 5)]


class CompletedPhaseEvidenceTests(unittest.TestCase):
    def test_loader_default_preserves_original_endpoint_only_contract(self):
        class Archive(dict):
            @property
            def files(self):
                return list(self)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        class TensorPreprocessor:
            def transform_obs(self, observed):
                return {key: torch.as_tensor(value) for key, value in observed.items()}

        archive = Archive(visual=np.arange(6).reshape(6, 1, 1, 1),
                          proprio=np.arange(6).reshape(6, 1),
                          normalized_model_actions=np.arange(10, dtype=np.float32).reshape(1, 1, 10))
        with patch.object(Path, "exists", return_value=True), patch(
                "research.reframe_v3.shadow_selection_audit.np.load", return_value=archive):
            old = _load_segments(Path("unused"), 0, 1, 5, TensorPreprocessor())[0]
            fine = _load_segments(Path("unused"), 0, 1, 5, TensorPreprocessor(),
                                  full_resolution=True)[0]
        for key in old[0]:
            self.assertEqual(old[0][key].shape[1], 2)
            self.assertEqual(fine[0][key].shape[1], 6)
            self.assertTrue(torch.equal(old[0][key], fine[0][key][:, ::5]))
        self.assertTrue(torch.equal(old[1], fine[1]))

    def test_merge_preserves_fine_frames_action_order_and_phase_zero(self):
        segments = _segments(4)
        fine, low = merge_completed_lowlevel(segments)
        expected, expected_low = _numbered(20)
        for key in fine:
            self.assertTrue(torch.equal(fine[key], expected[key]))
        self.assertTrue(torch.equal(low, expected_low))
        phase_zero = phase_chains(fine, low)[0]
        self.assertTrue(torch.equal(phase_zero["actions"],
                                    torch.cat([actions for _, actions in segments], dim=1)))
        for key in fine:
            self.assertTrue(torch.equal(phase_zero["observed"][key], fine[key][:, ::5]))

    def test_counts_context_lengths_and_completed_endpoints(self):
        for count, expected_counts, expected_hist in (
                (2, [2, 1, 1, 1, 1], [5, 1, 0]),
                (4, [4, 3, 3, 3, 3], [5, 5, 6])):
            with self.subTest(chunks=count):
                fine, low = merge_completed_lowlevel(_segments(count))
                chains = phase_chains(fine, low)
                self.assertEqual([len(chain["lowlevel_starts"]) for chain in chains], expected_counts)
                histories = [min(index + 1, 3) for chain in chains
                             for index in range(len(chain["lowlevel_starts"]))]
                self.assertEqual([histories.count(n) for n in (1, 2, 3)], expected_hist)
                starts = []
                for chain in chains:
                    starts.extend(chain["lowlevel_starts"])
                    self.assertEqual(chain["observation_times"],
                                     [chain["phase"]] + chain["lowlevel_ends"])
                    for index, (start, end) in enumerate(zip(
                            chain["lowlevel_starts"], chain["lowlevel_ends"])):
                        self.assertLessEqual(end, count * 5)
                        self.assertEqual(end - start, 5)
                        self.assertTrue(torch.equal(chain["actions"][:, index],
                                                    low[:, start:end].reshape(1, 10)))
                        self.assertEqual(chain["observed"]["visual"][0, index, 0, 0], start)
                        self.assertEqual(chain["observed"]["visual"][0, index + 1, 0, 0], end)
                self.assertEqual(sorted(starts), list(range(count * 5 - 4)))

    def test_boundary_mismatch_fails_without_tolerance(self):
        for key in ("visual", "proprio"):
            with self.subTest(key=key):
                segments = _segments(2)
                segments[1][0][key] = segments[1][0][key].clone()
                segments[1][0][key][:, 0] += .001
                with self.assertRaisesRegex(ValueError, "noncontinuous"):
                    merge_completed_lowlevel(segments)

    def test_incomplete_tail_never_creates_future_endpoint(self):
        fine, low = _numbered(12)
        chains = phase_chains(fine, low)
        self.assertEqual([len(chain["lowlevel_starts"]) for chain in chains], [2, 2, 2, 1, 1])
        self.assertEqual(sorted(start for chain in chains for start in chain["lowlevel_starts"]),
                         list(range(8)))
        self.assertTrue(all(end <= 12 for chain in chains for end in chain["lowlevel_ends"]))
        # A held cut at low-level time 5 admits only the original first chunk.
        held = phase_chains({key: value[:, :6] for key, value in fine.items()}, low[:, :5])
        self.assertEqual(len(held), 1)
        self.assertEqual(held[0]["lowlevel_starts"], [0])

    def test_bad_frame_count_and_action_shape_fail(self):
        segments = _segments(1)
        short = {key: value[:, :-1] for key, value in segments[0][0].items()}
        with self.assertRaises(ValueError):
            merge_completed_lowlevel([(short, segments[0][1])])
        with self.assertRaises(ValueError):
            merge_completed_lowlevel([(segments[0][0], torch.zeros(1, 5, 2))])
        with self.assertRaises(ValueError):
            merge_completed_lowlevel([])


if __name__ == "__main__":
    unittest.main()
