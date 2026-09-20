"""Non-decision-changing instrumentation for the PushObj contrast pilot.

The official classes are patched only inside ``run_pushobj_pilot.py``.  With
instrumentation disabled, the only wrapper captures the returned action tensor.
With instrumentation enabled, the released MPC/GD loops are reproduced line for
line and augmented with measurements after the official operations they observe.
"""

from __future__ import annotations

import copy
import json
import pickle
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from research.contrast_probe import (
    native_objective_breakdown,
    optimizer_state_summary,
    scheduler_state_summary,
    signed,
)
from research.replay_oracle import PushObjReplayOracle, encode_obs_with_modules
from research.schemas import JsonlWriter, jsonable, tensor_stats
from utils import move_to_device, slice_trajdict_with_t


_ACTIVE_RECORDER = None


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PilotRecorder:
    def __init__(
        self,
        output_dir: Path,
        run_id: str,
        arm: str,
        enabled: bool,
        oracle_steps=(0, 24, 49, 74, 99),
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.arm = str(arm)
        self.enabled = bool(enabled)
        self.oracle_steps = tuple(int(step) for step in oracle_steps)

        self.traces_dir = self.output_dir / "traces"
        self.oracle_dir = self.output_dir / "oracle"
        self.metrics_dir = self.output_dir / "metrics"
        self.tensor_dir = self.output_dir / "tensors"
        self.evidence_dir = self.output_dir / "real_evidence"
        for directory in (
            self.traces_dir,
            self.oracle_dir,
            self.metrics_dir,
            self.tensor_dir,
            self.evidence_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.planner_steps = JsonlWriter(self.traces_dir / "planner_steps.jsonl")
        self.adaptation_steps = JsonlWriter(
            self.traces_dir / "adaptation_steps.jsonl"
        )
        self.version_table = JsonlWriter(self.traces_dir / "version_table.jsonl")
        self.branch_records = JsonlWriter(self.oracle_dir / "branch_records.jsonl")
        self.contrast_pairs = JsonlWriter(self.oracle_dir / "contrast_pairs.jsonl")
        self.replay_errors = JsonlWriter(self.metrics_dir / "replay_errors.jsonl")
        self.query_budget = JsonlWriter(self.oracle_dir / "query_budget.jsonl")

        self.workspace = None
        self.cfg = None
        self.shapes = []
        self.env_infos = []
        self.env_kwargs = {"with_velocity": True, "with_target": True}
        self.alpha = 1.0
        self.base = 2.0

        self.sample_idx = None
        self.shape = None
        self.seed = None
        self.initial_state = None
        self.goal_state = None
        self.reference_encoder = None
        self.reference_proprio_encoder = None
        self.reference_encoder_transform = None
        self.reference_goal_z = None
        self.oracle = None
        self.current_planner = None
        self.current_prefix = None
        self.current_obs = None
        self.current_state = None
        self.current_history_id = None
        self.active_buffer_ids = []
        self.latest_support = None
        self.model_version = 0
        self.predictor_version = 0
        self.encoder_version = 0
        self.objective_version = 0
        self.action_unit_version = "pushobj_norm_envapi_v1"

        self.sample_wall_started = None
        self.run_wall_started = time.perf_counter()
        self.sample_counter_start = {}
        self.wm_rollout_count = 0
        self.predict_call_count = 0
        self.encode_obs_call_count = 0
        self.backward_count = 0
        self.deployed_real_steps = 0
        self.deployed_trajectory_count = 0
        self.baseline_replay_environment_steps = 0
        self.total_deployed_real_steps = 0
        self.total_executed_trajectories = 0
        self.total_baseline_replay_environment_steps = 0
        self.total_oracle_rollout_count = 0
        self.total_oracle_environment_steps = 0
        self.final_actions = None
        self.final_action_len = None

    def attach_workspace(self, workspace) -> None:
        self.run_wall_started = time.perf_counter()
        self.workspace = workspace
        self.cfg = workspace.cfg_dict
        objective_cfg = self.cfg.get("objective", {})
        self.alpha = float(objective_cfg.get("alpha", 1.0))
        self.base = float(objective_cfg.get("base", 2.0))
        override = self.cfg.get("env_kwargs_override") or {}
        self.env_kwargs.update(dict(override))

        eval_path = Path(self.cfg["eval_data_path"])
        with eval_path.open("rb") as handle:
            payload = pickle.load(handle)
        chosen = random.Random(int(self.cfg["seed"])).sample(
            payload["segments"], min(int(self.cfg["n_evals"]), len(payload["segments"]))
        )
        self.shapes = [segment.get("shape") for segment in chosen]
        self.env_infos = []
        for segment in chosen:
            info = {"shape": segment["shape"]} if "shape" in segment else {}
            for key in ("color", "agent_color", "goal_color"):
                if key in override:
                    info[key] = override[key]
            self.env_infos.append(info)

        metadata = {
            "run_id": self.run_id,
            "arm": self.arm,
            "instrumentation_enabled": self.enabled,
            "oracle_gd_steps": list(self.oracle_steps),
            "eval_data_path": str(eval_path.resolve()),
            "n_evals": int(self.cfg["n_evals"]),
            "seed": int(self.cfg["seed"]),
            "shapes": self.shapes,
            "frameskip": int(workspace.frameskip),
            "model_action_dim": int(workspace.action_dim),
            "objective": {
                "mode": objective_cfg.get("mode"),
                "alpha": self.alpha,
                "base": self.base,
            },
            "action_units": {
                "model": "normalized environment API action packed by frameskip",
                "environment_api": "relative displacement before internal x100 scale",
            },
        }
        with (self.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def begin_sample(
        self,
        planner,
        obs_0_i: dict,
        obs_g_i: dict,
        seed_i,
        state_0_i: np.ndarray,
        state_g_i: np.ndarray,
    ) -> None:
        if not self.enabled:
            return
        self.current_planner = planner
        self.sample_idx = int(planner._sample_idx)
        self.shape = self.shapes[self.sample_idx]
        self.seed = int(seed_i[0])
        self.initial_state = np.asarray(state_0_i[0]).copy()
        self.goal_state = np.asarray(state_g_i[0]).copy()
        self.active_buffer_ids = []
        self.latest_support = None
        self.model_version = 0
        self.predictor_version = 0
        self.encoder_version = 0
        self.objective_version = 0
        self.deployed_real_steps = 0
        self.deployed_trajectory_count = 1
        self.baseline_replay_environment_steps = 0
        self.sample_wall_started = time.perf_counter()
        self.sample_counter_start = {
            "wm_rollout_count": self.wm_rollout_count,
            "predict_call_count": self.predict_call_count,
            "encode_obs_call_count": self.encode_obs_call_count,
            "backward_count": self.backward_count,
        }
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.reference_encoder = copy.deepcopy(planner.wm.encoder).eval()
        self.reference_proprio_encoder = copy.deepcopy(
            planner.wm.proprio_encoder
        ).eval()
        for module in (self.reference_encoder, self.reference_proprio_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.reference_encoder_transform = planner.wm.encoder_transform
        transformed_goal = move_to_device(
            planner.preprocessor.transform_obs(obs_g_i), planner.device
        )
        with torch.no_grad():
            self.reference_goal_z = encode_obs_with_modules(
                transformed_goal,
                self.reference_encoder,
                self.reference_proprio_encoder,
                self.reference_encoder_transform,
            )

        sample_oracle_dir = self.oracle_dir / f"sample_{self.sample_idx:03d}"
        self.oracle = PushObjReplayOracle(
            initial_state=self.initial_state,
            goal_state=self.goal_state,
            seed=self.seed,
            env_info=self.env_infos[self.sample_idx],
            env_kwargs=self.env_kwargs,
            preprocessor=planner.preprocessor,
            frameskip=planner.evaluator.frameskip,
            alpha=self.alpha,
            base=self.base,
            output_dir=sample_oracle_dir,
        )
        initial_sidecar = self.evidence_dir / f"sample_{self.sample_idx:03d}_initial.npz"
        np.savez_compressed(
            initial_sidecar,
            visual=np.asarray(obs_0_i["visual"]),
            proprio=np.asarray(obs_0_i["proprio"]),
            initial_state=self.initial_state,
            goal_visual=np.asarray(obs_g_i["visual"]),
            goal_proprio=np.asarray(obs_g_i["proprio"]),
            goal_state=self.goal_state,
        )
        self._record_version(event="sample_start", mpc_iter=-1)

    def end_sample(self, actions: torch.Tensor, action_len: np.ndarray) -> None:
        if not self.enabled:
            return
        wall_s = time.perf_counter() - self.sample_wall_started
        peak_vram = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        )
        sample_wm_rollouts = (
            self.wm_rollout_count - self.sample_counter_start["wm_rollout_count"]
        )
        sample_predict_calls = (
            self.predict_call_count - self.sample_counter_start["predict_call_count"]
        )
        sample_encode_calls = (
            self.encode_obs_call_count
            - self.sample_counter_start["encode_obs_call_count"]
        )
        sample_backward_count = (
            self.backward_count - self.sample_counter_start["backward_count"]
        )
        record = {
            "run_id": self.run_id,
            "arm": self.arm,
            "sample_id": self.sample_idx,
            "shape": self.shape,
            "seed": self.seed,
            "deployed_real_environment_steps": self.deployed_real_steps,
            "executed_trajectories": self.deployed_trajectory_count,
            "official_cumulative_replay_environment_steps": self.baseline_replay_environment_steps,
            "wm_rollout_count": sample_wm_rollouts,
            "model_predict_call_count": sample_predict_calls,
            "model_encode_obs_call_count": sample_encode_calls,
            "model_forward_count": sample_predict_calls + sample_encode_calls,
            "model_backward_count": sample_backward_count,
            "oracle_environment_rollout_count": self.oracle.rollout_count,
            "oracle_environment_steps": self.oracle.environment_steps,
            "wall_clock_s": wall_s,
            "peak_vram_bytes": peak_vram,
            "returned_model_actions": int(actions.shape[1]),
            "action_len": np.asarray(action_len).tolist(),
        }
        self.query_budget.write(record)
        self.total_deployed_real_steps += self.deployed_real_steps
        self.total_executed_trajectories += self.deployed_trajectory_count
        self.total_baseline_replay_environment_steps += (
            self.baseline_replay_environment_steps
        )
        self.total_oracle_rollout_count += self.oracle.rollout_count
        self.total_oracle_environment_steps += self.oracle.environment_steps
        self.reference_encoder = None
        self.reference_proprio_encoder = None
        self.reference_goal_z = None
        self.oracle = None
        self.current_planner = None

    def capture_final_actions(self, actions: torch.Tensor, action_len: np.ndarray) -> None:
        self.final_actions = actions.detach().cpu()
        self.final_action_len = np.asarray(action_len).copy()
        torch.save(
            {
                "run_id": self.run_id,
                "arm": self.arm,
                "instrumentation_enabled": self.enabled,
                "actions": actions.detach().cpu(),
                "action_len": np.asarray(action_len),
            },
            self.output_dir / "baseline_actions.pt",
        )

    def finalize_run(self) -> None:
        """Write counters after the official final evaluator has also run."""
        if not self.enabled:
            return
        finite_lengths = np.where(
            np.isfinite(self.final_action_len),
            self.final_action_len,
            self.final_actions.shape[1],
        )
        final_eval_environment_steps = int(
            np.sum(finite_lengths) * self.workspace.frameskip
        )
        record = {
            "run_id": self.run_id,
            "arm": self.arm,
            "n_samples": int(self.cfg["n_evals"]),
            "deployed_real_environment_steps": self.total_deployed_real_steps,
            "executed_trajectories": self.total_executed_trajectories,
            "official_mpc_cumulative_replay_environment_steps": self.total_baseline_replay_environment_steps,
            "official_final_eval_environment_steps": final_eval_environment_steps,
            "official_total_evaluation_environment_steps": self.total_baseline_replay_environment_steps
            + final_eval_environment_steps,
            "wm_rollout_count": self.wm_rollout_count,
            "model_predict_call_count": self.predict_call_count,
            "model_encode_obs_call_count": self.encode_obs_call_count,
            "model_forward_count": self.predict_call_count
            + self.encode_obs_call_count,
            "model_backward_count": self.backward_count,
            "oracle_environment_rollout_count": self.total_oracle_rollout_count,
            "oracle_environment_steps": self.total_oracle_environment_steps,
            "wall_clock_s": time.perf_counter() - self.run_wall_started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0,
        }
        with (self.oracle_dir / "run_budget_final.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(jsonable(record), handle, indent=2)

    def before_mpc(self, planner, cur_obs_0: dict, obs_g: dict) -> None:
        self.current_planner = planner
        self.current_prefix = (
            torch.cat(planner.planned_actions, dim=1).detach().clone()
            if planner.planned_actions
            else None
        )
        self.current_obs = {key: np.asarray(value).copy() for key, value in cur_obs_0.items()}
        self.current_state = np.asarray(planner.evaluator.state_0).copy()
        self.current_history_id = f"s{self.sample_idx}_history_mpc{planner.iter}"

        transformed_goal = move_to_device(
            planner.preprocessor.transform_obs(obs_g), planner.device
        )
        with torch.no_grad():
            current_goal_z = planner.wm.encode_obs(transformed_goal)
        replay = self.oracle.validate_anchor(
            prefix=self.current_prefix,
            online_obs=self.current_obs,
            online_state=self.current_state,
            model=planner.wm,
            current_goal_z=current_goal_z,
            mpc_iter=int(planner.iter),
            record_name=self.current_history_id,
        )
        replay.update(
            {
                "run_id": self.run_id,
                "arm": self.arm,
                "sample_id": self.sample_idx,
                "shape": self.shape,
                "seed": self.seed,
                "mpc_iter": int(planner.iter),
                "real_history_id": self.current_history_id,
                **self._versions(),
            }
        )
        self.replay_errors.write(replay)

        history_sidecar = self.evidence_dir / f"{self.current_history_id}.pt"
        torch.save(
            {
                "executed_prefix": None
                if self.current_prefix is None
                else self.current_prefix.detach().cpu(),
                "current_real_observation": {
                    key: np.asarray(value) for key, value in self.current_obs.items()
                },
                "current_public_state": self.current_state,
            },
            history_sidecar,
        )
        if not replay["passed"]:
            raise RuntimeError(
                "Zero-perturbation replay hard gate failed; counterfactual oracle "
                f"is disabled. See {self.metrics_dir / 'replay_errors.jsonl'}"
            )

    def record_real_experience(
        self,
        planner,
        taken_actions: torch.Tensor,
        e_obses: dict,
        e_states: np.ndarray,
    ) -> None:
        frameskip = int(planner.evaluator.frameskip)
        model_steps = int(taken_actions.shape[1])
        start = int(planner.iter * model_steps * frameskip)
        stop = start + model_steps * frameskip + 1
        evidence_id = f"s{self.sample_idx}_executed_mpc{planner.iter}"
        sidecar = self.evidence_dir / f"{evidence_id}.npz"
        np.savez_compressed(
            sidecar,
            visual=np.asarray(e_obses["visual"])[0, start:stop],
            proprio=np.asarray(e_obses["proprio"])[0, start:stop],
            states=np.asarray(e_states)[0, start:stop],
            normalized_model_actions=taken_actions.detach().cpu().numpy(),
        )
        self.active_buffer_ids.append(evidence_id)
        replay_mode = getattr(planner, "replay_mode", None)
        replay_size = int(getattr(planner, "replay_size", 0))
        if replay_mode == "recent" and replay_size > 0:
            self.active_buffer_ids = self.active_buffer_ids[-replay_size:]
        self.deployed_real_steps += model_steps * frameskip
        self.baseline_replay_environment_steps += (planner.iter + 1) * model_steps * frameskip

    def record_adaptation(
        self,
        trainer,
        before_loss: Optional[float],
        after_loss: Optional[float],
        step_losses: list,
        merge: bool,
    ) -> None:
        n_updates = len(step_losses)
        before_versions = self._versions()
        if n_updates:
            self.model_version += n_updates
            self.predictor_version += n_updates
            if trainer.finetune_encoder:
                self.encoder_version += n_updates
        after_versions = self._versions()
        record = {
            "run_id": self.run_id,
            "arm": self.arm,
            "sample_id": self.sample_idx,
            "shape": self.shape,
            "seed": self.seed,
            "mpc_iter": int(self.current_planner.iter),
            "adaptation_buffer_ids": list(self.active_buffer_ids),
            "merge_contiguous_segments": bool(merge),
            "support_loss_eval_before": before_loss,
            "support_loss_eval_after": after_loss,
            "support_loss_eval_mode": "deterministic_module_eval",
            "official_pre_update_step_losses": list(step_losses),
            "optimizer": trainer.optimizer_name,
            "predictor_lr": float(trainer.lr),
            "encoder_lr": float(trainer.encoder_lr),
            "finetune_encoder": bool(trainer.finetune_encoder),
            "versions_before": before_versions,
            "versions_after": after_versions,
        }
        self.adaptation_steps.write(record)
        self.latest_support = {
            "before": before_loss,
            "after": after_loss,
            "buffer_ids": list(self.active_buffer_ids),
            "mpc_iter": int(self.current_planner.iter),
        }
        self._record_version(event="adaptation", mpc_iter=int(self.current_planner.iter))

    def record_gd_step(
        self,
        gd_planner,
        gd_iter: int,
        actions_before: torch.Tensor,
        actions_after: torch.Tensor,
        raw_gradient: torch.Tensor,
        before_breakdown: dict,
        after_breakdown: dict,
        optimizer: torch.optim.Optimizer,
        lr_before: list,
        lr_after: list,
        scheduler,
        current_goal_z: dict,
    ) -> None:
        update = actions_after - actions_before
        c_before = float(before_breakdown["total"].item())
        c_after = float(after_breakdown["total"].item())
        delta_pred = c_after - c_before
        selected = int(gd_iter) in self.oracle_steps
        record_name = f"s{self.sample_idx}_m{self.current_planner.iter}_g{gd_iter}"
        tensor_sidecar = None
        if selected:
            tensor_sidecar = self.tensor_dir / f"{record_name}.pt"
            torch.save(
                {
                    "u_before": actions_before.detach().cpu(),
                    "u_after": actions_after.detach().cpu(),
                    "actual_update": update.detach().cpu(),
                    "raw_gradient_before": raw_gradient.detach().cpu(),
                },
                tensor_sidecar,
            )

        record = {
            "run_id": self.run_id,
            "arm": self.arm,
            "sample_id": self.sample_idx,
            "shape": self.shape,
            "seed": self.seed,
            "mpc_iter": int(self.current_planner.iter),
            "gd_iter": int(gd_iter),
            **self._versions(),
            "goal_id": f"s{self.sample_idx}_goal",
            "goal_latent_encoder_version": self.encoder_version,
            "real_history_id": self.current_history_id,
            "objective_stage": before_breakdown["stage"],
            "objective_coefficients": before_breakdown["coefficients"],
            "objective_horizon": before_breakdown["horizon"],
            "native_cost_before": c_before,
            "native_cost_before_visual": float(before_breakdown["visual"].item()),
            "native_cost_before_proprio": float(before_breakdown["proprio"].item()),
            "native_cost_after": c_after,
            "native_cost_after_visual": float(after_breakdown["visual"].item()),
            "native_cost_after_proprio": float(after_breakdown["proprio"].item()),
            "predicted_delta_cost": delta_pred,
            "u_before_summary": tensor_stats(actions_before),
            "u_after_summary": tensor_stats(actions_after),
            "actual_update_summary": tensor_stats(update),
            "raw_gradient_summary": tensor_stats(raw_gradient),
            "optimizer_name": gd_planner.optimizer_name,
            "optimizer_state_summary": optimizer_state_summary(
                optimizer, next(iter(optimizer.param_groups[0]["params"]))
            ),
            "learning_rate_before": lr_before,
            "learning_rate_after": lr_after,
            "scheduler_state": scheduler_state_summary(scheduler),
            "adaptation_buffer_ids": list(self.active_buffer_ids),
            "latest_support_loss": self.latest_support,
            "oracle_selected": selected,
            "tensor_sidecar": None if tensor_sidecar is None else str(tensor_sidecar),
        }
        self.planner_steps.write(record)

        if not selected:
            return
        common = {
            "run_id": self.run_id,
            "arm": self.arm,
            "sample_id": self.sample_idx,
            "shape": self.shape,
            "seed": self.seed,
            "mpc_iter": int(self.current_planner.iter),
            "gd_iter": int(gd_iter),
            **self._versions(),
            "real_history_id": self.current_history_id,
            "adaptation_buffer_ids": list(self.active_buffer_ids),
        }
        before_oracle = self.oracle.evaluate_candidate(
            prefix=self.current_prefix,
            candidate=actions_before,
            model=gd_planner.wm,
            current_goal_z=current_goal_z,
            reference_encoder=self.reference_encoder,
            reference_proprio_encoder=self.reference_proprio_encoder,
            reference_encoder_transform=self.reference_encoder_transform,
            reference_goal_z=self.reference_goal_z,
            mpc_iter=int(self.current_planner.iter),
            record_name=f"{record_name}_before",
        )
        after_oracle = self.oracle.evaluate_candidate(
            prefix=self.current_prefix,
            candidate=actions_after,
            model=gd_planner.wm,
            current_goal_z=current_goal_z,
            reference_encoder=self.reference_encoder,
            reference_proprio_encoder=self.reference_proprio_encoder,
            reference_encoder_transform=self.reference_encoder_transform,
            reference_goal_z=self.reference_goal_z,
            mpc_iter=int(self.current_planner.iter),
            record_name=f"{record_name}_after",
        )
        self.branch_records.write(
            {**common, "candidate": "u_before", "c_hat_native": c_before, **before_oracle}
        )
        self.branch_records.write(
            {**common, "candidate": "u_after", "c_hat_native": c_after, **after_oracle}
        )
        delta_samever = after_oracle["c_env_samever"] - before_oracle["c_env_samever"]
        delta_ref = after_oracle["c_env_ref"] - before_oracle["c_env_ref"]
        contrast = {
            **common,
            "c_hat_native_before": c_before,
            "c_hat_native_after": c_after,
            "c_env_samever_before": before_oracle["c_env_samever"],
            "c_env_samever_after": after_oracle["c_env_samever"],
            "c_env_ref_before": before_oracle["c_env_ref"],
            "c_env_ref_after": after_oracle["c_env_ref"],
            "delta_pred": delta_pred,
            "delta_env_samever": delta_samever,
            "delta_env_ref": delta_ref,
            "contrast_error": delta_pred - delta_samever,
            "sign_pred": signed(delta_pred),
            "sign_env": signed(delta_samever),
            "sign_correct": signed(delta_pred) == signed(delta_samever),
            "false_improvement": delta_pred < 0.0 and delta_samever > 0.0,
        }
        self.contrast_pairs.write(contrast)

    def _versions(self) -> dict:
        return {
            "model_version": self.model_version,
            "predictor_version": self.predictor_version,
            "encoder_version": self.encoder_version,
            "objective_version": self.objective_version,
            "action_unit_version": self.action_unit_version,
        }

    def _record_version(self, event: str, mpc_iter: int) -> None:
        self.version_table.write(
            {
                "run_id": self.run_id,
                "arm": self.arm,
                "sample_id": self.sample_idx,
                "shape": self.shape,
                "seed": self.seed,
                "event": event,
                "mpc_iter": mpc_iter,
                **self._versions(),
            }
        )


def _support_eval_loss(trainer, obs_seqs: list, act_seqs: list, merge: bool):
    if not obs_seqs:
        return None
    if merge and len(obs_seqs) > 1:
        obs_seqs, act_seqs = trainer._merge_segments(obs_seqs, act_seqs)
    segments = [trainer._prepare_segment(o, a) for o, a in zip(obs_seqs, act_seqs)]
    predictor_training = trainer.wm.predictor.training
    encoder_training = trainer.wm.encoder.training
    trainer.wm.predictor.eval()
    trainer.wm.encoder.eval()
    try:
        with torch.no_grad():
            losses = [
                trainer._prediction_loss(trainer.wm.encode(o, a))
                for o, a in segments
            ]
            return float(torch.stack(losses).mean().item())
    finally:
        trainer.wm.predictor.train(predictor_training)
        trainer.wm.encoder.train(encoder_training)


def _instrumented_gd_plan(self, obs_0, obs_g, actions=None, step=None):
    recorder = _ACTIVE_RECORDER
    trans_obs_0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
    trans_obs_g = move_to_device(self.preprocessor.transform_obs(obs_g), self.device)
    with torch.no_grad():
        z_obs_g = self.wm.encode_obs(trans_obs_g)

    actions = self.init_actions(obs_0, actions).to(self.device)
    actions.requires_grad = True
    optimizer = self.get_action_optimizer(actions)
    scheduler = self.get_scheduler(optimizer)
    n_evals = actions.shape[0]

    for i in tqdm(range(self.opt_steps)):
        actions_before = actions.detach().clone()
        lr_before = [float(group["lr"]) for group in optimizer.param_groups]
        optimizer.zero_grad()
        i_z_obses, _ = self.wm.rollout(obs_0=trans_obs_0, act=actions)
        loss = self.objective_fn(i_z_obses, z_obs_g, step=step)
        before_breakdown = native_objective_breakdown(
            i_z_obses, z_obs_g, step, recorder.alpha, recorder.base
        )
        closure_error = torch.max(torch.abs(loss - before_breakdown["total"]))
        if float(closure_error.detach().cpu()) > 1e-6:
            raise AssertionError(
                f"Native objective instrumentation mismatch: {float(closure_error)}"
            )
        total_loss = loss.mean() * n_evals
        total_loss.backward()
        recorder.backward_count += 1
        raw_gradient = actions.grad.detach().clone()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            actions += torch.randn_like(actions) * self.action_noise
        actions_after = actions.detach().clone()
        lr_after = [float(group["lr"]) for group in optimizer.param_groups]
        with torch.no_grad():
            after_z_obses, _ = self.wm.rollout(obs_0=trans_obs_0, act=actions)
            after_breakdown = native_objective_breakdown(
                after_z_obses, z_obs_g, step, recorder.alpha, recorder.base
            )

        recorder.record_gd_step(
            gd_planner=self,
            gd_iter=i,
            actions_before=actions_before,
            actions_after=actions_after,
            raw_gradient=raw_gradient,
            before_breakdown=before_breakdown,
            after_breakdown=after_breakdown,
            optimizer=optimizer,
            lr_before=lr_before,
            lr_after=lr_after,
            scheduler=scheduler,
            current_goal_z=z_obs_g,
        )

        self.wandb_run.log(
            {f"{self.logging_prefix}/loss": total_loss.item(), "step": i + 1}
        )
        if self.evaluator is not None and self.eval_every != -1 and i % self.eval_every == 0:
            logs, successes, _, _ = self.evaluator.eval_actions(
                actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
            )
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": i + 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)
            if np.all(successes):
                break
    return actions, np.full(n_evals, np.inf)


def _instrumented_mpc_plan(self, obs_0, obs_g, actions=None):
    recorder = _ACTIVE_RECORDER
    n_evals = obs_0["visual"].shape[0]
    self.is_success = np.zeros(n_evals, dtype=bool)
    self.action_len = np.full(n_evals, np.inf)
    self.iter = 0
    self.planned_actions = []
    init_obs_0, init_state_0 = self.evaluator.get_init_cond()

    cur_obs_0 = obs_0
    memo_actions = None
    while not np.all(self.is_success) and self.iter < self.max_iter:
        recorder.before_mpc(self, cur_obs_0, obs_g)
        self.sub_planner.logging_prefix = f"plan_{self.iter}{self.plan_suffix}"
        _cuda_sync()
        t0 = time.perf_counter()
        actions, _ = self.sub_planner.plan(
            obs_0=cur_obs_0,
            obs_g=obs_g,
            actions=memo_actions,
            step=self.iter,
        )
        _cuda_sync()
        self.t_plan_s = time.perf_counter() - t0
        taken_actions = actions.detach()[:, : self.n_taken_actions]
        self._apply_success_mask(taken_actions)
        memo_actions = (
            actions.detach()[:, self.n_taken_actions :] if self.reuse_actions else None
        )
        self.planned_actions.append(taken_actions)

        print(f"MPC iter {self.iter} Eval ------- ")
        action_so_far = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(obs_0=init_obs_0, state_0=init_state_0)
        logs, successes, e_obses, e_states = self.evaluator.eval_actions(
            action_so_far,
            self.action_len,
            filename=f"plan{self.iter}{self.plan_suffix}",
            save_video=True,
        )
        new_successes = successes & ~self.is_success
        self.is_success = self.is_success | successes
        self.action_len[new_successes] = (self.iter + 1) * self.n_taken_actions

        print("self.is_success: ", self.is_success)
        recorder.record_real_experience(self, taken_actions, e_obses, e_states)
        extra_logs = self._post_env_feedback(taken_actions, e_obses)
        logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
        logs.update({"step": self.iter + 1, **(extra_logs or {})})
        self.wandb_run.log(logs)
        self.dump_logs(logs)

        e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
        cur_obs_0 = e_final_obs
        e_final_state = e_states[:, -1]
        self.evaluator.assign_init_cond(obs_0=e_final_obs, state_0=e_final_state)
        self.iter += 1

    planned_actions = torch.cat(self.planned_actions, dim=1)
    self.evaluator.assign_init_cond(obs_0=init_obs_0, state_0=init_state_0)
    return planned_actions, self.action_len


def install_instrumentation(recorder: PilotRecorder) -> None:
    """Install process-local wrappers before Hydra constructs the workspace."""

    global _ACTIVE_RECORDER
    _ACTIVE_RECORDER = recorder

    import plan as plan_module
    from models.visual_world_model import VWorldModel
    from planning.adajepa import AdaJEPATrainer
    from planning.adajepa_mpc import AdaJEPAMPCPlanner
    from planning.gd import GDPlanner
    from planning.mpc import MPCPlanner

    patch_log = [
        {
            "target": "PlanWorkspace.__init__",
            "purpose": "attach run metadata after official workspace construction",
            "enabled_modes": ["capture", "off", "on"],
        },
        {
            "target": "AdaJEPAMPCPlanner.plan",
            "purpose": "capture returned action tensor without changing it",
            "enabled_modes": ["capture", "off", "on"],
        },
    ]

    original_workspace_init = plan_module.PlanWorkspace.__init__
    original_perform_planning = plan_module.PlanWorkspace.perform_planning

    def workspace_init_wrapper(self, *args, **kwargs):
        original_workspace_init(self, *args, **kwargs)
        recorder.attach_workspace(self)

    plan_module.PlanWorkspace.__init__ = workspace_init_wrapper

    original_outer_plan = AdaJEPAMPCPlanner.plan

    def outer_plan_wrapper(self, obs_0, obs_g, actions=None):
        result_actions, result_lens = original_outer_plan(self, obs_0, obs_g, actions)
        recorder.capture_final_actions(result_actions, result_lens)
        return result_actions, result_lens

    AdaJEPAMPCPlanner.plan = outer_plan_wrapper

    if recorder.enabled:
        patch_log.extend(
            [
                {
                    "target": "PlanWorkspace.perform_planning",
                    "purpose": "finalize run budget after the official final evaluator",
                    "enabled_modes": ["on"],
                },
                {
                    "target": "AdaJEPAMPCPlanner._plan_single",
                    "purpose": "sample-scoped reference snapshot and budget accounting",
                    "enabled_modes": ["on"],
                },
                {
                    "target": "MPCPlanner.plan",
                    "purpose": "zero-prefix replay gate and executed-evidence capture",
                    "enabled_modes": ["on"],
                },
                {
                    "target": "GDPlanner.plan",
                    "purpose": "actual optimizer-step and native contrast trace",
                    "enabled_modes": ["on"],
                },
                {
                    "target": "AdaJEPATrainer.finetune",
                    "purpose": "deterministic support loss before/after and versions",
                    "enabled_modes": ["on"],
                },
                {
                    "target": "VWorldModel rollout/predict/encode_obs",
                    "purpose": "model-query accounting",
                    "enabled_modes": ["on"],
                },
            ]
        )

        def perform_planning_wrapper(self):
            result = original_perform_planning(self)
            recorder.finalize_run()
            return result

        plan_module.PlanWorkspace.perform_planning = perform_planning_wrapper

        original_plan_single = AdaJEPAMPCPlanner._plan_single

        def plan_single_wrapper(
            self, obs_0_i, obs_g_i, env_i, seed_i, state_0_i, state_g_i
        ):
            recorder.begin_sample(
                self, obs_0_i, obs_g_i, seed_i, state_0_i, state_g_i
            )
            actions_i, action_len_i = original_plan_single(
                self, obs_0_i, obs_g_i, env_i, seed_i, state_0_i, state_g_i
            )
            recorder.end_sample(actions_i, action_len_i)
            return actions_i, action_len_i

        AdaJEPAMPCPlanner._plan_single = plan_single_wrapper
        MPCPlanner.plan = _instrumented_mpc_plan
        GDPlanner.plan = _instrumented_gd_plan

        original_finetune = AdaJEPATrainer.finetune

        def finetune_wrapper(self, obs_seqs, act_seqs, merge=True):
            before = _support_eval_loss(self, obs_seqs, act_seqs, merge)
            step_losses = original_finetune(self, obs_seqs, act_seqs, merge=merge)
            after = _support_eval_loss(self, obs_seqs, act_seqs, merge)
            recorder.backward_count += len(step_losses)
            recorder.record_adaptation(self, before, after, step_losses, merge)
            return step_losses

        AdaJEPATrainer.finetune = finetune_wrapper

        original_rollout = VWorldModel.rollout
        original_predict = VWorldModel.predict
        original_encode_obs = VWorldModel.encode_obs

        def rollout_wrapper(self, *args, **kwargs):
            recorder.wm_rollout_count += 1
            return original_rollout(self, *args, **kwargs)

        def predict_wrapper(self, *args, **kwargs):
            recorder.predict_call_count += 1
            return original_predict(self, *args, **kwargs)

        def encode_obs_wrapper(self, *args, **kwargs):
            recorder.encode_obs_call_count += 1
            return original_encode_obs(self, *args, **kwargs)

        VWorldModel.rollout = rollout_wrapper
        VWorldModel.predict = predict_wrapper
        VWorldModel.encode_obs = encode_obs_wrapper

    with (recorder.output_dir / "instrumentation_patch_log.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(jsonable(patch_log), handle, indent=2)
