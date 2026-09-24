"""Retrospective B1 predictor-update attribution on the already recorded fixed-plan pool.

The true B trajectories are evaluation inputs only. This module reproduces the
original AdaJEPA update before interpreting any term and never changes the
online operator, planner, candidate pool, or environment.
"""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import torch

from planning.adajepa import AdaJEPATrainer
from planning.objectives import create_objective_fn
from research.reframe_v3.history_pairwise_predict import cut
from research.reframe_v3.matched_feedback_forecast import rollout_from_zobs
from research.reframe_v3.processing_evaluation import TRUTH, npz_latent
from research.reframe_v3.processing_feasibility import Runner, read_csv, sync, write_csv, write_json
from research.reframe_v3.shadow_selection_audit import (
    _apply_schedule, _encoder_versions, _load_anchor_observations, _load_segments,
    _reset_to_base, _score_pool, _seed_all, _support_loss, _transform_anchor_obs,
    official_prefix_schedule, selection_summary,
)
from research.reframe_v3.task_defect_probe import complete_prefix


TOL_COST = 1e-6
TOL_CLOSURE = 1e-5
VARIANTS = ("frozen", "predictor_official_shadow")


def telescope_candidate(model, anchor, truth, actions, goal):
    """Exact ordered task-cost telescope for one fixed five-action candidate."""
    if actions.shape[1] != 5 or any(value.shape[1] != 6 for value in truth.values()):
        raise ValueError("expected one full native H=5 action and H+1 truth path")
    objective = create_objective_fn(1., 2., "last")
    with torch.no_grad():
        prediction, _ = rollout_from_zobs(model, anchor, actions)
        model_cost = float(objective(prediction, goal).sum())
        costs = [float(objective(complete_prefix(model, cut(truth, 0, s + 1), actions),
                                 goal).sum()) for s in range(6)]
    terms = [costs[s + 1] - costs[s] for s in range(5)]
    anchor_shift = costs[0] - model_cost
    true_error = costs[-1] - model_cost
    return {"model_cost": model_cost, "true_cost": costs[-1],
            "anchor_shift": anchor_shift, "true_error": true_error,
            "terms": terms, "closure": anchor_shift + sum(terms) - true_error}


def compare_pair(before_left, before_right, after_left, after_right):
    """Same true candidate pair, with signed error truth minus prediction."""
    def diff(left, right, key):
        return left[key] - right[key]

    before_error = diff(before_left, before_right, "true_error")
    after_error = diff(after_left, after_right, "true_error")
    anchor_change = (diff(after_left, after_right, "anchor_shift")
                     - diff(before_left, before_right, "anchor_shift"))
    term_changes = [
        (after_left["terms"][s] - after_right["terms"][s])
        - (before_left["terms"][s] - before_right["terms"][s])
        for s in range(5)
    ]
    change = after_error - before_error
    return {"before_pair_error": before_error, "after_pair_error": after_error,
            "pair_error_change": change, "anchor_change": anchor_change,
            "term_changes": term_changes,
            "closure": change - anchor_change - sum(term_changes)}


def _check(name, value, limit):
    if value > limit:
        raise AssertionError(f"{name}={value:.9g} exceeds predeclared {limit:.9g}")


def _cached_load():
    cache = Path("D:/EV-TTT/adajepa_runtime/torch/hub/facebookresearch_dinov2_main")
    if not (cache / "hubconf.py").is_file() or not (
            cache.parent / "checkpoints/dinov2_vits14_pretrain.pth").is_file():
        raise FileNotFoundError("local DINO cache and weights required; no download")
    original = torch.hub.load

    def load(repo, name, *args, **kwargs):
        if repo != "facebookresearch/dinov2" or name != "dinov2_vits14":
            raise ValueError("unexpected hub dependency")
        return original(str(cache), name, *args, source="local", **kwargs)

    return load


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "RUN_CONTRACT.json", {
        "status": "B1_RETROSPECTIVE", "protocol":
        "artifacts/processing_math_bridge/B1_UPDATE_ATTRIBUTION_PROTOCOL.md",
        "original_variant": "predictor_official_shadow", "cases": 6, "anchors": 12,
        "candidates_per_anchor": 10, "fixed_full_plan_only": True,
        "future_latent_sidecars_loaded_after_score_reproduction": True,
        "new_environment_branches": 0, "new_operator": False,
        "cost_parity_limit": TOL_COST, "closure_limit": TOL_CLOSURE})
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    with patch.object(torch.hub, "load", _cached_load()):
        runner = Runner(out)
    loaded = time.perf_counter()
    model, device = runner.model, runner.device
    if model.concat_dim != 1 or model.num_proprio_repeat != 1 or model.num_action_repeat != 1:
        raise ValueError("unverified native history/action layout")
    predictor_trainer = AdaJEPATrainer(
        wm=model, lr=runner.lr, steps=1, optimizer_name="adam",
        finetune_encoder=False, last_layer_only=True)
    full_trainer = AdaJEPATrainer(
        wm=model, lr=runner.lr, steps=1, optimizer_name="adam",
        finetune_encoder=True, last_layer_only=True, encoder_lr=1e-5,
        encoder_last_layer_only=True)
    trainers = (predictor_trainer, full_trainer)
    if not _reset_to_base(model, trainers):
        raise AssertionError("initial checkpoint reset failed")
    original = Path("artifacts/reframe_v3/common_anchors_v2")
    saved_cost = {(row["anchor_id"], row["candidate_id"], row["variant"]): row
                  for row in read_csv(original / "candidate_scores.csv")
                  if row["analysis"] == "b1_common_anchor" and row["variant"] in VARIANTS}
    saved_selection = {(row["anchor_id"], row["variant"]): row
                       for row in read_csv(original / "selection_metrics.csv")
                       if row["analysis"] == "b1_common_anchor" and row["variant"] in VARIANTS}
    cases = read_csv(Path("artifacts/processing_math_bridge/history_pairwise_20260923/CASES.csv"))
    truth = {(row["anchor_id"], row["candidate_id"]): row
             for row in read_csv(TRUTH / "REENCODED_COSTS.csv")}
    if len(saved_cost) != 240 or len(saved_selection) != 24 or len(cases) != 12 or len(truth) != 120:
        raise AssertionError("original B1 records or B truth are incomplete")
    checks = {key: 0. for key in (
        "saved_model_cost", "latent_model_cost", "support_loss", "truth_cost",
        "truth_same_variants", "candidate_telescope", "pair_update_telescope",
        "selection_regret")}
    candidates, steps, pairs, anchors = [], [], [], []
    adaptation_optimizer_steps = factual_transition_uses = 0
    torch.cuda.reset_peak_memory_stats()
    for ordinal, case in enumerate(cases):
        case_id, anchor_id, mpc = case["case_id"], case["anchor_id"], int(case["mpc"])
        query = runner.query_map[(case_id, mpc)]
        donor, sample = Path(query["donor_dir"]), int(case_id.split("|")[2][1:])
        segments = _load_segments(donor, sample, mpc, 5, runner.preprocessor)
        current, goal_observation = _load_anchor_observations(donor, sample, mpc)
        transformed_current, transformed_goal = _transform_anchor_obs(
            runner.preprocessor, current, goal_observation, device)
        anchor, pool = runner.anchor(case_id, mpc)
        goal = torch.load(case["goal_file"], map_location=device)
        if {cid for cid, _ in pool} != {
                cid for aid, cid, variant in saved_cost if aid == anchor_id and variant == VARIANTS[0]}:
            raise AssertionError("candidate IDs diverge from original B1")
        if not _reset_to_base(model, trainers):
            raise AssertionError(f"checkpoint reset failed at {anchor_id}")
        base_scores, base_stage = _score_pool(model, transformed_current,
                                              transformed_goal, [
                                                  {"candidate_id": cid, "gd_iter": int(cid.split("_")[0][1:]),
                                                   "side": cid.split("_")[1], "action": action,
                                                   "c_env": float(saved_cost[(anchor_id, cid, VARIANTS[0])]["c_env_ref"])}
                                                  for cid, action in pool], mpc)
        if base_stage != "terminal":
            raise AssertionError("original B objective is not terminal")
        for row in base_scores:
            cid = row["candidate_id"]
            checks["saved_model_cost"] = max(checks["saved_model_cost"],
                abs(row["c_model"] - float(saved_cost[(anchor_id, cid, VARIANTS[0])]["c_model"])))
        _check("frozen B1 cost reproduction", checks["saved_model_cost"], TOL_COST)

        def evaluate(variant, scores):
            score_map = {row["candidate_id"]: row for row in scores}
            local = {}
            for candidate_id, action in pool:
                old = saved_cost[(anchor_id, candidate_id, variant)]
                checks["saved_model_cost"] = max(checks["saved_model_cost"],
                    abs(score_map[candidate_id]["c_model"] - float(old["c_model"])))
                oracle = truth[(anchor_id, candidate_id)]
                true_path = {key: value.to(device) for key, value in
                             npz_latent(oracle["latent_sidecar"]).items()}
                result = telescope_candidate(model, anchor, true_path, action, goal)
                checks["latent_model_cost"] = max(checks["latent_model_cost"],
                    abs(result["model_cost"] - score_map[candidate_id]["c_model"]))
                checks["truth_cost"] = max(checks["truth_cost"],
                    abs(result["true_cost"] - float(oracle["reencoded_cost"])))
                checks["candidate_telescope"] = max(checks["candidate_telescope"],
                    abs(result["closure"]))
                row = {"case_id": case_id, "anchor_id": anchor_id, "mpc": mpc,
                       "candidate_id": candidate_id, "variant": variant,
                       **{key: result[key] for key in (
                           "model_cost", "true_cost", "anchor_shift", "true_error")}}
                candidates.append(row)
                local[candidate_id] = result
                steps.extend({"case_id": case_id, "anchor_id": anchor_id, "mpc": mpc,
                              "candidate_id": candidate_id, "variant": variant,
                              "step": s, "task_error_term": term}
                             for s, term in enumerate(result["terms"]))
            chosen = selection_summary([{"candidate_id": cid,
                                         "c_model": value["model_cost"],
                                         "c_env": value["true_cost"]}
                                        for cid, value in local.items()], 1e-6)
            old_selection = saved_selection[(anchor_id, variant)]
            if chosen["model_best_candidate_id"] != old_selection["model_best_candidate_id"]:
                raise AssertionError(f"B1 selected candidate parity failed at {anchor_id}/{variant}")
            checks["selection_regret"] = max(checks["selection_regret"],
                abs(chosen["r_selected_tie_min"] - float(old_selection["r_selected_tie_min"])))
            return local, chosen

        if not _reset_to_base(model, trainers):
            raise AssertionError(f"B1 reset failed at {anchor_id}")
        _seed_all(260920 + ordinal)
        all_indices = tuple(range(mpc))
        support_before = _support_loss(predictor_trainer, segments, all_indices)
        _support_loss(predictor_trainer, segments, all_indices)  # original training support call
        encoder_before = _encoder_versions(model)
        update_losses = _apply_schedule(predictor_trainer, segments,
                                       official_prefix_schedule(mpc))
        adaptation_optimizer_steps += len(update_losses)
        factual_transition_uses += sum(len(batch) for batch in official_prefix_schedule(mpc))
        encoder_after = _encoder_versions(model)
        if encoder_before != encoder_after or len(update_losses) != mpc:
            raise AssertionError("fixed-encoder or B1 update-count contract failed")
        support_after = _support_loss(predictor_trainer, segments, all_indices)
        _support_loss(predictor_trainer, segments, all_indices)
        old_support = saved_selection[(anchor_id, VARIANTS[1])]
        checks["support_loss"] = max(checks["support_loss"],
            abs(support_before - float(old_support["common_support_loss_before"])),
            abs(support_after - float(old_support["common_support_loss_after"])))
        updated_scores, updated_stage = _score_pool(model, transformed_current,
                                                     transformed_goal, [
                                                         {"candidate_id": cid, "gd_iter": int(cid.split("_")[0][1:]),
                                                          "side": cid.split("_")[1], "action": action,
                                                          "c_env": float(saved_cost[(anchor_id, cid, VARIANTS[1])]["c_env_ref"])}
                                                         for cid, action in pool], mpc)
        if updated_stage != base_stage:
            raise AssertionError("objective stage changed after B1")
        # Gate reproduced B1 scores before attributing the updated version.
        _check("saved_model_cost", checks["saved_model_cost"], TOL_COST)
        _check("support_loss", checks["support_loss"], TOL_COST)
        for row in updated_scores:
            cid = row["candidate_id"]
            checks["saved_model_cost"] = max(checks["saved_model_cost"],
                abs(row["c_model"] - float(saved_cost[(anchor_id, cid, VARIANTS[1])]["c_model"])))
        _check("updated B1 cost reproduction", checks["saved_model_cost"], TOL_COST)
        adapted_params = [parameter.detach().clone()
                          for parameter in predictor_trainer._ada_predictor_params]
        if not _reset_to_base(model, trainers):
            raise AssertionError(f"B1 attribution frozen-state reset failed at {anchor_id}")
        frozen, before_choice = evaluate(VARIANTS[0], base_scores)
        with torch.no_grad():
            for parameter, saved in zip(predictor_trainer._ada_predictor_params,
                                        adapted_params):
                parameter.copy_(saved)
        model.eval()
        adapted, after_choice = evaluate(VARIANTS[1], updated_scores)
        for cid in frozen:
            checks["truth_same_variants"] = max(checks["truth_same_variants"],
                abs(frozen[cid]["true_cost"] - adapted[cid]["true_cost"]))
        anchor_pairs = []
        for left, right in itertools.combinations(frozen, 2):
            comparison = compare_pair(frozen[left], frozen[right],
                                      adapted[left], adapted[right])
            checks["pair_update_telescope"] = max(checks["pair_update_telescope"],
                abs(comparison["closure"]))
            row = {"case_id": case_id, "anchor_id": anchor_id, "mpc": mpc,
                   "candidate_left": left, "candidate_right": right,
                   **{key: comparison[key] for key in (
                       "before_pair_error", "after_pair_error", "pair_error_change",
                       "anchor_change")},
                   **{f"term{s}_change": comparison["term_changes"][s]
                      for s in range(5)}, "signed_closure": comparison["closure"]}
            pairs.append(row)
            anchor_pairs.append(row)
        selected = after_choice["model_best_candidate_id"]
        selected_before = before_choice["model_best_candidate_id"]
        oracle_id = after_choice["env_best_candidate_id"]
        selected_oracle = compare_pair(frozen[selected], frozen[oracle_id],
                                       adapted[selected], adapted[oracle_id])
        switch = compare_pair(frozen[selected], frozen[selected_before],
                              adapted[selected], adapted[selected_before])
        anchors.append({"case_id": case_id, "anchor_id": anchor_id, "mpc": mpc,
                        "support_before": support_before, "support_after": support_after,
                        "support_change": support_after - support_before,
                        "regret_before": before_choice["r_selected_tie_min"],
                        "regret_after": after_choice["r_selected_tie_min"],
                        "regret_change": after_choice["r_selected_tie_min"]
                        - before_choice["r_selected_tie_min"],
                        "selected_before": selected_before,
                        "selected_after": selected, "oracle_id": oracle_id,
                        "true_cost_span": after_choice["environment_pool_span"],
                        "margin_before": before_choice["model_best_margin"],
                        "margin_after": after_choice["model_best_margin"],
                        "pair_mae_before": statistics.mean(abs(r["before_pair_error"]) for r in anchor_pairs),
                        "pair_mae_after": statistics.mean(abs(r["after_pair_error"]) for r in anchor_pairs),
                        "selected_oracle_true_gap": adapted[selected]["true_cost"] - adapted[oracle_id]["true_cost"],
                        "selected_oracle_predicted_gap_after": adapted[selected]["model_cost"] - adapted[oracle_id]["model_cost"],
                        "selected_oracle_pair_error_before": selected_oracle["before_pair_error"],
                        "selected_oracle_pair_error_after": selected_oracle["after_pair_error"],
                        "selected_oracle_pair_error_change": selected_oracle["pair_error_change"],
                        "selected_oracle_anchor_change": selected_oracle["anchor_change"],
                        **{f"selected_oracle_term{s}_change": selected_oracle["term_changes"][s]
                           for s in range(5)},
                        "switch_true_gap": adapted[selected]["true_cost"]
                        - adapted[selected_before]["true_cost"],
                        "switch_predicted_gap_before": frozen[selected]["model_cost"]
                        - frozen[selected_before]["model_cost"],
                        "switch_predicted_gap_after": adapted[selected]["model_cost"]
                        - adapted[selected_before]["model_cost"],
                        "switch_pair_error_change": switch["pair_error_change"],
                        "switch_anchor_change": switch["anchor_change"],
                        **{f"switch_term{s}_change": switch["term_changes"][s]
                           for s in range(5)}})
        _check("saved_model_cost", checks["saved_model_cost"], TOL_COST)
        _check("latent_model_cost", checks["latent_model_cost"], TOL_COST)
        _check("truth_cost", checks["truth_cost"], TOL_COST)
        _check("candidate_telescope", checks["candidate_telescope"], TOL_CLOSURE)
        _check("pair_update_telescope", checks["pair_update_telescope"], TOL_CLOSURE)
        if not _reset_to_base(model, trainers):
            raise AssertionError(f"B1 post-attribution reset failed at {anchor_id}")
        print(anchor_id, "B1 reproduced and attributed", flush=True)
    for key, limit in (("support_loss", TOL_COST), ("truth_same_variants", TOL_COST),
                       ("selection_regret", TOL_COST)):
        _check(key, checks[key], limit)
    if len(candidates) != 240 or len(steps) != 1200 or len(pairs) != 540 or len(anchors) != 12:
        raise AssertionError("incomplete B1 attribution")
    write_csv(out / "CANDIDATE_ATTRIBUTION.csv", candidates)
    write_csv(out / "STEP_ATTRIBUTION.csv", steps)
    write_csv(out / "PAIR_UPDATE_ATTRIBUTION.csv", pairs)
    write_csv(out / "ANCHOR_SUMMARY.csv", anchors)
    sync()
    groups = []
    for mpc in (2, 4):
        subset = [row for row in anchors if row["mpc"] == mpc]
        groups.append({"mpc": mpc, "cases": len(subset),
                       "support_down": sum(row["support_change"] < 0 for row in subset),
                       "regret_improved": sum(row["regret_change"] < -1e-6 for row in subset),
                       "regret_worsened": sum(row["regret_change"] > 1e-6 for row in subset),
                       "regret_unchanged": sum(abs(row["regret_change"]) <= 1e-6 for row in subset),
                       "mean_pair_mae_before": statistics.mean(row["pair_mae_before"] for row in subset),
                       "mean_pair_mae_after": statistics.mean(row["pair_mae_after"] for row in subset),
                       "mean_regret_before": statistics.mean(row["regret_before"] for row in subset),
                       "mean_regret_after": statistics.mean(row["regret_after"] for row in subset),
                       "median_true_cost_span": statistics.median(row["true_cost_span"] for row in subset),
                       "median_margin_after": statistics.median(row["margin_after"] for row in subset)})
    write_csv(out / "GROUP_SUMMARY.csv", groups)
    write_json(out / "RUN_SUMMARY.json", {
        "status": "COMPLETE_B1_RETROSPECTIVE", "checks": checks,
        "groups": groups, "candidate_rows": len(candidates), "step_rows": len(steps),
        "pair_rows": len(pairs), "anchor_rows": len(anchors),
        "load_seconds": loaded - started,
        "compute_seconds": time.perf_counter() - loaded,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "ledger": runner.ledger.snapshot(), "new_environment_branches": 0,
        "adaptation_optimizer_steps": adaptation_optimizer_steps,
        "factual_transition_uses": factual_transition_uses,
        "ledger_backward_and_updates_exclude_AdaJEPA_trainer": True,
        "ledger_encoder_calls_cover_runner_only": True,
        "fixed_full_plan_not_native_closed_loop": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
