"""Behavioral tests for the V3 offline selection audit CLI."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


AUDIT = Path(__file__).with_name("offline_selection_audit.py")
GD_STEPS = (0, 24, 49, 74, 99)


def make_row(gd_iter: int, cm_before: float, cm_after: float, ce_before: float, ce_after: float) -> dict:
    return {
        "phase": "development",
        "split": "val_T",
        "run_path": "fixture/run",
        "action_unit_version": "pushobj_norm_envapi_v1",
        "adaptation_buffer_ids": "[]",
        "arm": "FROZEN",
        "c_env_ref_after": ce_after,
        "c_env_ref_before": ce_before,
        "c_env_samever_after": ce_after,
        "c_env_samever_before": ce_before,
        "c_hat_native_after": cm_after,
        "c_hat_native_before": cm_before,
        "contrast_error": (cm_after - cm_before) - (ce_after - ce_before),
        "delta_env_ref": ce_after - ce_before,
        "delta_env_samever": ce_after - ce_before,
        "delta_pred": cm_after - cm_before,
        "encoder_version": 0,
        "false_improvement": False,
        "gd_iter": gd_iter,
        "model_version": 0,
        "mpc_iter": 0,
        "objective_version": 0,
        "predictor_version": 0,
        "real_history_id": "s0_history_mpc0",
        "run_id": "fixture_run",
        "sample_id": 0,
        "seed": 1,
        "shape": "T",
        "sign_correct": True,
        "sign_env": -1,
        "sign_pred": -1,
    }


def valid_rows() -> list[dict]:
    costs = [
        (10.0, 8.0, 10.0, 9.0),
        (7.5, 7.0, 8.5, 8.0),
        (6.5, 6.0, 7.5, 7.0),
        (5.5, 5.0, 6.5, 6.0),
        (4.5, 4.0, 5.5, 5.0),
    ]
    return [make_row(gd, *values) for gd, values in zip(GD_STEPS, costs)]


def regret_rows() -> list[dict]:
    costs = [
        (10.0, 1.0, 10.0, 8.0),
        (3.0, 2.0, 3.0, 2.0),
        (5.0, 4.5, 6.0, 5.5),
        (6.0, 5.5, 7.0, 6.5),
        (4.2, 4.0, 5.2, 5.0),
    ]
    return [make_row(gd, *values) for gd, values in zip(GD_STEPS, costs)]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class OfflineSelectionAuditCliTests(unittest.TestCase):
    def test_valid_replan_writes_endpoint_gains_and_selection_regret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            output = root / "output"
            write_rows(source, valid_rows())

            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "selection_metrics.csv").open(encoding="utf-8", newline="") as handle:
                selection = list(csv.DictReader(handle))
            with (output / "path_metrics.csv").open(encoding="utf-8", newline="") as handle:
                path = list(csv.DictReader(handle))
            self.assertEqual(len(selection), 1)
            self.assertEqual(len(path), 5)
            self.assertEqual(float(selection[0]["r_selected"]), 0.0)
            self.assertEqual(float(selection[0]["r_final_record"]), 0.0)
            self.assertEqual(float(selection[0]["eta_model_final_record"]), 0.0)
            self.assertEqual(float(path[-1]["g_model"]), 6.0)
            self.assertEqual(float(path[-1]["g_env"]), 5.0)
            self.assertEqual(float(path[-1]["omega"]), 1.0)

    def test_missing_gd_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            output = root / "output"
            write_rows(source, valid_rows()[:-1])

            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected GD snapshots", result.stderr)
            self.assertFalse(output.exists())

    def test_mixed_model_versions_within_replan_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            output = root / "output"
            rows = valid_rows()
            rows[-1]["model_version"] = 1
            write_rows(source, rows)

            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mixed model_version", result.stderr)
            self.assertFalse(output.exists())

    def test_non_finite_cost_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            output = root / "output"
            rows = valid_rows()
            rows[2]["c_env_samever_after"] = "nan"
            write_rows(source, rows)

            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-finite", result.stderr)
            self.assertFalse(output.exists())

    def test_delta_that_does_not_match_endpoints_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            output = root / "output"
            rows = valid_rows()
            rows[1]["delta_pred"] = 123.0
            write_rows(source, rows)

            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("delta_pred does not match endpoints", result.stderr)
            self.assertFalse(output.exists())

    def test_algebra_validation_is_saved_for_nonzero_selection_regret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            output = root / "output"
            write_rows(source, regret_rows())

            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "selection_metrics.csv").open(encoding="utf-8", newline="") as handle:
                selection = list(csv.DictReader(handle))[0]
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            self.assertEqual(float(selection["r_selected"]), 6.0)
            self.assertEqual(float(selection["r_final_record"]), 3.0)
            self.assertEqual(float(selection["eta_model_final_record"]), 3.0)
            self.assertTrue(validation["algebra"]["passed"])
            self.assertEqual(validation["algebra"]["gain_identity_max_abs_residual"], 0.0)
            self.assertGreaterEqual(validation["algebra"]["selected_bound_min_slack"], 0.0)
            self.assertGreaterEqual(validation["algebra"]["final_bound_min_slack"], 0.0)

    def test_missing_required_column_reports_schema_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            output = root / "output"
            rows = valid_rows()
            for row in rows:
                del row["c_env_ref_after"]
            write_rows(source, rows)

            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing required columns", result.stderr)
            self.assertIn("c_env_ref_after", result.stderr)

    def test_case_and_strata_outputs_use_case_as_independent_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            output = root / "output"
            write_rows(source, regret_rows())

            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "case_metrics.csv").open(encoding="utf-8", newline="") as handle:
                case_rows = list(csv.DictReader(handle))
            with (output / "strata_summary.csv").open(encoding="utf-8", newline="") as handle:
                strata_rows = list(csv.DictReader(handle))
            self.assertEqual(len(case_rows), 1)
            self.assertEqual(case_rows[0]["case_id"], "development|val_T|T|0")
            self.assertEqual(case_rows[0]["independent_unit"], "matched_initial_goal_case")
            self.assertEqual(int(case_rows[0]["n_replans"]), 1)
            self.assertEqual(float(case_rows[0]["mean_r_selected"]), 6.0)
            self.assertEqual(len(strata_rows), 1)
            self.assertEqual(int(strata_rows[0]["n_cases"]), 1)
            self.assertEqual(float(strata_rows[0]["mean_case_r_selected"]), 6.0)

    def test_near_tied_model_minima_report_regret_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            output = root / "output"
            rows = regret_rows()
            rows[1] = make_row(24, 3.0, 1.0000005, 3.0, 2.0)
            write_rows(source, rows)

            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "selection_metrics.csv").open(encoding="utf-8", newline="") as handle:
                selection = list(csv.DictReader(handle))[0]
            self.assertEqual(int(selection["model_tie_count_epsilon"]), 2)
            self.assertEqual(float(selection["r_selected_tie_min"]), 0.0)
            self.assertEqual(float(selection["r_selected_tie_max"]), 6.0)
            self.assertAlmostEqual(float(selection["model_best_margin"]), 0.0000005)

    def test_nonempty_output_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            output = root / "output"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("preserve", encoding="utf-8")
            write_rows(source, valid_rows())

            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite nonempty output directory", result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_duplicate_gd_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            output = root / "output"
            rows = valid_rows()
            rows[-1]["gd_iter"] = 74
            write_rows(source, rows)

            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected GD snapshots", result.stderr)

    def test_common_model_cost_bias_does_not_change_selection_regret(self) -> None:
        def run_fixture(root: Path, name: str, rows: list[dict]) -> dict:
            source = root / f"{name}.csv"
            output = root / name
            write_rows(source, rows)
            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "selection_metrics.csv").open(encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))[0]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_rows = regret_rows()
            biased_rows = regret_rows()
            for row in biased_rows:
                row["c_hat_native_before"] += 100.0
                row["c_hat_native_after"] += 100.0
            original = run_fixture(root, "original", original_rows)
            biased = run_fixture(root, "biased", biased_rows)
            self.assertEqual(float(original["r_selected"]), float(biased["r_selected"]))
            self.assertEqual(
                float(original["error_oscillation"]),
                float(biased["error_oscillation"]),
            )

    def test_local_wrong_step_can_end_with_zero_final_pool_regret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            output = root / "output"
            costs = [
                (10.0, 9.0, 10.0, 11.0),
                (8.0, 7.0, 9.0, 8.0),
                (6.0, 5.0, 7.0, 6.0),
                (4.0, 3.0, 5.0, 4.0),
                (2.0, 1.0, 3.0, 2.0),
            ]
            rows = [make_row(gd, *values) for gd, values in zip(GD_STEPS, costs)]
            write_rows(source, rows)

            result = subprocess.run(
                [sys.executable, str(AUDIT), "--input", str(source), "--out", str(output)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "selection_metrics.csv").open(encoding="utf-8", newline="") as handle:
                selection = list(csv.DictReader(handle))[0]
            self.assertEqual(float(selection["r_selected"]), 0.0)
            self.assertEqual(float(selection["r_final_record"]), 0.0)


if __name__ == "__main__":
    unittest.main()
