"""Fresh-environment replay oracle for PushObj counterfactual branches."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from einops import rearrange

from env.pusht.pusht_wrapper import PushTWrapper
from research.contrast_probe import native_objective_breakdown
from utils import move_to_device


def encode_obs_with_modules(
    transformed_obs: dict,
    encoder: torch.nn.Module,
    proprio_encoder: torch.nn.Module,
    encoder_transform,
) -> dict:
    visual = transformed_obs["visual"]
    batch = visual.shape[0]
    visual = rearrange(visual, "b t ... -> (b t) ...")
    visual = encoder_transform(visual)
    visual_latent = encoder(visual)
    visual_latent = rearrange(visual_latent, "(b t) p d -> b t p d", b=batch)
    proprio_latent = proprio_encoder(transformed_obs["proprio"])
    return {"visual": visual_latent, "proprio": proprio_latent}


class PushObjReplayOracle:
    """Recreate a fresh simulator and replay the full executed prefix.

    No mid-episode public-state reset is used.  Every anchor and candidate
    branch starts from the episode's original initial condition.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        goal_state: np.ndarray,
        seed: int,
        env_info: dict,
        env_kwargs: dict,
        preprocessor,
        frameskip: int,
        alpha: float,
        base: float,
        output_dir: Path,
    ):
        self.initial_state = np.asarray(initial_state).copy()
        self.goal_state = np.asarray(goal_state).copy()
        self.seed = int(seed)
        self.env_info = dict(env_info)
        self.env_kwargs = dict(env_kwargs)
        self.preprocessor = preprocessor
        self.frameskip = int(frameskip)
        self.alpha = float(alpha)
        self.base = float(base)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_count = 0
        self.environment_steps = 0

    def _fresh_env(self) -> PushTWrapper:
        env = PushTWrapper(**self.env_kwargs)
        env.update_env(self.env_info)
        return env

    def _to_env_actions(self, model_actions: Optional[torch.Tensor]) -> np.ndarray:
        if model_actions is None or model_actions.shape[1] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        normalized = rearrange(
            model_actions.detach().cpu(),
            "b t (f d) -> b (t f) d",
            f=self.frameskip,
        )
        return self.preprocessor.denormalize_actions(normalized)[0].numpy()

    def _replay(self, model_actions: Optional[torch.Tensor]):
        env_actions = self._to_env_actions(model_actions)
        env = self._fresh_env()
        started = time.perf_counter()
        try:
            if env_actions.shape[0] == 0:
                obs, state = env.prepare(self.seed, self.initial_state)
                obs = {key: np.expand_dims(value, 0) for key, value in obs.items()}
                states = np.expand_dims(state, 0)
            else:
                obs, states = env.rollout(
                    self.seed, self.initial_state, env_actions
                )
            self.rollout_count += 1
            self.environment_steps += int(env_actions.shape[0])
            return env, obs, states, time.perf_counter() - started
        except Exception:
            env.close()
            raise

    @staticmethod
    def _as_batch_time(obs: dict) -> dict:
        return {
            key: np.expand_dims(np.expand_dims(value, 0), 0)
            for key, value in obs.items()
        }

    @staticmethod
    def _final_obs(obses: dict) -> dict:
        return {key: value[-1] for key, value in obses.items()}

    def validate_anchor(
        self,
        prefix: Optional[torch.Tensor],
        online_obs: dict,
        online_state: np.ndarray,
        model,
        current_goal_z: dict,
        mpc_iter: int,
        record_name: str,
    ) -> dict:
        env, replay_obses, replay_states, elapsed = self._replay(prefix)
        try:
            replay_final = self._final_obs(replay_obses)
            online_final = {
                key: np.asarray(value)[0, -1] for key, value in online_obs.items()
            }
            replay_state = np.asarray(replay_states)[-1]
            online_state = np.asarray(online_state).reshape(-1, replay_state.shape[-1])[0]

            rgb_delta = np.abs(
                replay_final["visual"].astype(np.int16)
                - online_final["visual"].astype(np.int16)
            )
            proprio_delta = np.abs(
                replay_final["proprio"].astype(np.float64)
                - online_final["proprio"].astype(np.float64)
            )
            state_delta = np.abs(
                replay_state.astype(np.float64) - online_state.astype(np.float64)
            )

            device = next(model.parameters()).device
            replay_transformed = move_to_device(
                self.preprocessor.transform_obs(self._as_batch_time(replay_final)),
                device,
            )
            online_transformed = move_to_device(
                self.preprocessor.transform_obs(self._as_batch_time(online_final)),
                device,
            )
            with torch.no_grad():
                replay_z = model.encode_obs(replay_transformed)
                online_z = model.encode_obs(online_transformed)
                replay_cost = native_objective_breakdown(
                    replay_z, current_goal_z, mpc_iter, self.alpha, self.base
                )["total"]
                online_cost = native_objective_breakdown(
                    online_z, current_goal_z, mpc_iter, self.alpha, self.base
                )["total"]

            replay_metrics = env.eval_state(self.goal_state, replay_state)
            online_metrics = env.eval_state(self.goal_state, online_state)
            metric_error = {
                key: float(
                    np.max(
                        np.abs(
                            np.asarray(replay_metrics[key], dtype=np.float64)
                            - np.asarray(online_metrics[key], dtype=np.float64)
                        )
                    )
                )
                for key in replay_metrics
            }
            cost_error = float(torch.max(torch.abs(replay_cost - online_cost)).cpu())
            passed = bool(
                np.array_equal(replay_final["visual"], online_final["visual"])
                and np.array_equal(replay_final["proprio"], online_final["proprio"])
                and np.array_equal(replay_state, online_state)
                and cost_error == 0.0
                and all(value == 0.0 for value in metric_error.values())
            )
            sidecar = self.output_dir / f"{record_name}_anchor.npz"
            np.savez_compressed(
                sidecar,
                replay_visual=replay_final["visual"],
                online_visual=online_final["visual"],
                replay_proprio=replay_final["proprio"],
                online_proprio=online_final["proprio"],
                replay_state=replay_state,
                online_state=online_state,
            )
            return {
                "passed": passed,
                "rgb_max_abs_error": int(rgb_delta.max()),
                "rgb_mean_abs_error": float(rgb_delta.mean()),
                "rgb_exact_pixel_match_rate": float(
                    np.mean(replay_final["visual"] == online_final["visual"])
                ),
                "proprio_max_abs_error": float(proprio_delta.max()),
                "state_max_abs_error": float(state_delta.max()),
                "native_same_version_cost_replay": float(replay_cost.item()),
                "native_same_version_cost_online": float(online_cost.item()),
                "native_same_version_cost_abs_error": cost_error,
                "environment_metric_abs_error": metric_error,
                "prefix_model_actions": 0 if prefix is None else int(prefix.shape[1]),
                "prefix_environment_actions": int(self._to_env_actions(prefix).shape[0]),
                "oracle_wall_s": elapsed,
                "sidecar": str(sidecar),
            }
        finally:
            env.close()

    def evaluate_candidate(
        self,
        prefix: Optional[torch.Tensor],
        candidate: torch.Tensor,
        model,
        current_goal_z: dict,
        reference_encoder: torch.nn.Module,
        reference_proprio_encoder: torch.nn.Module,
        reference_encoder_transform,
        reference_goal_z: dict,
        mpc_iter: int,
        record_name: str,
    ) -> dict:
        if prefix is None or prefix.shape[1] == 0:
            combined = candidate
            prefix_env_len = 0
        else:
            combined = torch.cat([prefix, candidate], dim=1)
            prefix_env_len = int(prefix.shape[1] * self.frameskip)

        env, all_obses, all_states, elapsed = self._replay(combined)
        try:
            branch_obses = {
                key: value[prefix_env_len :: self.frameskip]
                for key, value in all_obses.items()
            }
            branch_states = all_states[prefix_env_len :: self.frameskip]
            expected = int(candidate.shape[1]) + 1
            if branch_obses["visual"].shape[0] != expected:
                raise AssertionError(
                    f"Expected {expected} oracle observations, got "
                    f"{branch_obses['visual'].shape[0]}"
                )
            batched_obs = {
                key: np.expand_dims(value, 0) for key, value in branch_obses.items()
            }
            device = next(model.parameters()).device
            transformed = move_to_device(
                self.preprocessor.transform_obs(batched_obs), device
            )
            with torch.no_grad():
                current_z = model.encode_obs(transformed)
                reference_z = encode_obs_with_modules(
                    transformed,
                    reference_encoder,
                    reference_proprio_encoder,
                    reference_encoder_transform,
                )
                current_breakdown = native_objective_breakdown(
                    current_z, current_goal_z, mpc_iter, self.alpha, self.base
                )
                reference_breakdown = native_objective_breakdown(
                    reference_z, reference_goal_z, mpc_iter, self.alpha, self.base
                )

            final_state = np.asarray(all_states)[-1]
            metrics = env.eval_state(self.goal_state, final_state)
            sidecar = self.output_dir / f"{record_name}_outcome.npz"
            np.savez_compressed(
                sidecar,
                visual=branch_obses["visual"],
                proprio=branch_obses["proprio"],
                states=branch_states,
                final_state=final_state,
            )
            return {
                "c_env_samever": float(current_breakdown["total"].item()),
                "c_env_samever_visual": float(current_breakdown["visual"].item()),
                "c_env_samever_proprio": float(current_breakdown["proprio"].item()),
                "c_env_ref": float(reference_breakdown["total"].item()),
                "c_env_ref_visual": float(reference_breakdown["visual"].item()),
                "c_env_ref_proprio": float(reference_breakdown["proprio"].item()),
                "objective_stage": current_breakdown["stage"],
                "objective_coefficients": current_breakdown["coefficients"],
                "official_environment_metrics": metrics,
                "candidate_model_actions": int(candidate.shape[1]),
                "candidate_environment_actions": int(
                    candidate.shape[1] * self.frameskip
                ),
                "oracle_wall_s": elapsed,
                "sidecar": str(sidecar),
            }
        finally:
            env.close()

