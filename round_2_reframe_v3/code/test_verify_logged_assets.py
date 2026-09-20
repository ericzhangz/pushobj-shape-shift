"""Integration test for raw case, returned-prefix, and tensor-pool validation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


VERIFY = Path(__file__).with_name("verify_logged_assets.py")
GD_STEPS = (0, 24, 49, 74, 99)
ARMS = ("FROZEN", "OFFICIAL_ADAJEPA", "PREDICTOR_ONLY")


def write_fixture(root: Path) -> Path:
    rows = []
    for arm_index, arm in enumerate(ARMS):
        run_path = root / f"run_{arm_index}"
        (run_path / "real_evidence").mkdir(parents=True)
        (run_path / "tensors").mkdir()
        (run_path / "oracle" / "sample_000").mkdir(parents=True)
        np.savez_compressed(
            run_path / "real_evidence" / "sample_000_initial.npz",
            visual=np.arange(6, dtype=np.uint8),
            proprio=np.arange(4, dtype=np.float32),
            initial_state=np.arange(7, dtype=np.float32),
            goal_visual=np.arange(6, dtype=np.uint8) + 1,
            goal_proprio=np.arange(4, dtype=np.float32) + 1,
            goal_state=np.arange(7, dtype=np.float32) + 1,
        )
        torch.save(
            {
                "actions": torch.full((1, 1, 10), 100.0),
                "action_len": np.asarray([1.0]),
            },
            run_path / "baseline_actions.pt",
        )
        real_visual = np.arange(18, dtype=np.uint8).reshape(6, 3)
        real_proprio = np.arange(24, dtype=np.float32).reshape(6, 4)
        real_states = np.arange(42, dtype=np.float32).reshape(6, 7)
        np.savez_compressed(
            run_path / "real_evidence" / "s0_executed_mpc0.npz",
            visual=real_visual,
            proprio=real_proprio,
            states=real_states,
            normalized_model_actions=np.full((1, 1, 10), 100.0, dtype=np.float32),
        )
        oracle_visual = np.repeat(real_visual[-1:], 6, axis=0)
        oracle_proprio = np.repeat(real_proprio[-1:], 6, axis=0)
        oracle_states = np.repeat(real_states[-1:], 6, axis=0)
        oracle_visual[0], oracle_visual[1] = real_visual[0], real_visual[-1]
        oracle_proprio[0], oracle_proprio[1] = real_proprio[0], real_proprio[-1]
        oracle_states[0], oracle_states[1] = real_states[0], real_states[-1]
        np.savez_compressed(
            run_path
            / "oracle"
            / "sample_000"
            / "s0_m0_g99_after_outcome.npz",
            visual=oracle_visual,
            proprio=oracle_proprio,
            states=oracle_states,
            final_state=oracle_states[-1],
        )
        for index, gd_iter in enumerate(GD_STEPS):
            before = torch.full((1, 5, 10), float(index * 2 + 1))
            after_value = 100.0 if gd_iter == 99 else float(index * 2 + 2)
            after = torch.full((1, 5, 10), after_value)
            torch.save(
                {"u_before": before, "u_after": after},
                run_path / "tensors" / f"s0_m0_g{gd_iter}.pt",
            )
            cm_before = 10.0 - index
            cm_after = 9.5 - index
            ce_before = 11.0 - index
            ce_after = 10.75 - index
            rows.append(
                {
                    "phase": "development",
                    "split": "val_T",
                    "run_path": str(run_path),
                    "action_unit_version": "pushobj_norm_envapi_v1",
                    "adaptation_buffer_ids": "[]",
                    "arm": arm,
                    "c_env_ref_after": ce_after,
                    "c_env_ref_before": ce_before,
                    "c_env_samever_after": ce_after,
                    "c_env_samever_before": ce_before,
                    "c_hat_native_after": cm_after,
                    "c_hat_native_before": cm_before,
                    "delta_env_ref": ce_after - ce_before,
                    "delta_env_samever": ce_after - ce_before,
                    "delta_pred": cm_after - cm_before,
                    "encoder_version": 0,
                    "gd_iter": gd_iter,
                    "model_version": 0,
                    "mpc_iter": 0,
                    "objective_version": 0,
                    "predictor_version": 0,
                    "real_history_id": "s0_history_mpc0",
                    "run_id": f"run_{arm_index}",
                    "sample_id": 0,
                    "seed": 1,
                    "shape": "T",
                }
            )
    source = root / "contrast_pairs.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return source


class VerifyLoggedAssetsCliTests(unittest.TestCase):
    def test_matching_cases_and_returned_prefixes_pass_with_ten_unique_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_fixture(root)
            output = root / "output"

            result = subprocess.run(
                [sys.executable, str(VERIFY), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            self.assertTrue(validation["passed"])
            self.assertEqual(validation["case_mappings_checked"], 1)
            self.assertEqual(validation["returned_prefixes_checked"], 3)
            self.assertEqual(validation["continuations_checked"], 3)
            self.assertEqual(validation["continuation_failures"], 0)
            with (output / "deduplicated_selection_metrics.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                dedup = list(csv.DictReader(handle))
            self.assertEqual(len(dedup), 3)
            self.assertTrue(all(int(row["unique_pool_size"]) == 10 for row in dedup))

    def test_first_chunk_outcome_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_fixture(root)
            outcome = (
                root
                / "run_0"
                / "oracle"
                / "sample_000"
                / "s0_m0_g99_after_outcome.npz"
            )
            with np.load(outcome, allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
            arrays["states"][1, 0] += 1.0
            np.savez_compressed(outcome, **arrays)
            output = root / "output"

            result = subprocess.run(
                [sys.executable, str(VERIFY), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            self.assertFalse(validation["passed"])
            self.assertEqual(validation["continuation_failures"], 1)


if __name__ == "__main__":
    unittest.main()
