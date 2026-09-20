"""Tests for the non-consuming qualification snapshot boundary."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.instrumentation import qualification_snapshot_payload
from research.replay_oracle import preserve_global_rng_state


class _WorldModel:
    def __init__(self) -> None:
        self.predictor = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        self.encoder = nn.Linear(2, 2)


class _Trainer:
    def __init__(self, wm: _WorldModel) -> None:
        self.wm = wm
        self._ada_predictor_params = list(wm.predictor[0].parameters())
        self._ada_encoder_params = list(wm.encoder.parameters())
        self.finetune_encoder = True


class _Planner:
    def __init__(self) -> None:
        wm = _WorldModel()
        self.adajepa_trainer = _Trainer(wm)
        self._obs_buffer = [{"visual": torch.arange(4).reshape(1, 2, 2)}]
        self._act_buffer = [torch.ones(1, 1, 2)]
        self._segment_scores = [0.25]
        self._sample_idx = 0
        self.iter = 0


class QualificationSnapshotTests(unittest.TestCase):
    def test_oracle_rng_guard_restores_all_global_generators(self) -> None:
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        cuda_before = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        )

        with preserve_global_rng_state():
            random.random()
            np.random.rand()
            torch.rand(1)
            if torch.cuda.is_available():
                torch.rand(1, device="cuda")

        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        self.assertTrue(np.array_equal(numpy_after[1], numpy_before[1]))
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        if torch.cuda.is_available():
            self.assertTrue(
                all(
                    torch.equal(after, before)
                    for after, before in zip(torch.cuda.get_rng_state_all(), cuda_before)
                )
            )

    def test_snapshot_captures_state_without_advancing_rng(self) -> None:
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        planner = _Planner()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()

        payload = qualification_snapshot_payload(planner)

        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_before[0])
        self.assertTrue(np.array_equal(numpy_after[1], numpy_before[1]))
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        self.assertEqual(len(payload["adapted_predictor_params"]), 2)
        self.assertEqual(len(payload["adapted_encoder_params"]), 2)
        self.assertEqual(len(payload["obs_buffer"]), 1)
        self.assertEqual(len(payload["act_buffer"]), 1)
        self.assertEqual(payload["segment_scores"], [0.25])


if __name__ == "__main__":
    unittest.main()
