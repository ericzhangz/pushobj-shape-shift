"""Integration test for ON/OFF qualification comparison."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


COMPARE = Path(__file__).with_name("compare_qualification_runs.py")


def write_run(path: Path) -> None:
    snapshots = path / "qualification_snapshots"
    snapshots.mkdir(parents=True)
    torch.save(
        {"actions": torch.ones(1, 2, 10), "action_len": np.asarray([2.0])},
        path / "baseline_actions.pt",
    )
    torch.save(
        {
            "sample_idx": 0,
            "mpc_iter": 0,
            "adapted_predictor_params": [torch.arange(3)],
            "adapted_encoder_params": [torch.arange(2)],
            "predictor_buffers": {"x": torch.ones(1)},
            "encoder_buffers": {},
            "obs_buffer": [{"visual": torch.zeros(1, 2)}],
            "act_buffer": [torch.ones(1, 1, 2)],
            "segment_scores": [0.5],
            "rng": {
                "python": (1, (2, 3), None),
                "numpy": ("MT19937", np.arange(4, dtype=np.uint32), 1, 0, 0.0),
                "torch_cpu": torch.arange(4, dtype=torch.uint8),
                "torch_cuda": [],
            },
        },
        snapshots / "s0_after_mpc0.pt",
    )


class CompareQualificationRunsCliTests(unittest.TestCase):
    def test_identical_runs_pass_all_state_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "off"
            candidate = root / "on"
            write_run(baseline)
            write_run(candidate)
            output = root / "comparison.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(COMPARE),
                    str(baseline),
                    str(candidate),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertTrue(report["actions_exact"])
            self.assertTrue(report["parameters_and_buffers_exact"])
            self.assertTrue(report["experience_buffers_exact"])
            self.assertTrue(report["rng_exact"])


if __name__ == "__main__":
    unittest.main()
