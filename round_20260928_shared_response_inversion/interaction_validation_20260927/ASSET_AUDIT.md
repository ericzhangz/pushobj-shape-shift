# Four-case interaction-timing asset audit

Date: 2026-09-27. Read-only old assets; no local skills, GPU inference, training, simulator calls, or old-file edits. Script: `asset_inventory.py`; complete directly readable paths and array metadata: `ASSET_INVENTORY.json`.

## Result

The four existing development anchors T0, T1, L0, L1 at MPC 2 have **88 saved branch-outcome files** across three candidate pools: 4×8 ordinary, 4×9 endpoint-matched, and 4×5 action-fiber candidates. These are 88 files, not 88 independent samples or necessarily 88 distinct actions. All 88 start with exactly the query RGB and proprio in the corresponding history file; both factual chunks join exactly; the final factual frame equals the query RGB/proprio. This was checked on actual arrays, not inferred from filenames.

**The principal limitation is temporal resolution:** factual chunks preserve 10 Hz observations, whereas all audited branch outcomes preserve only 2 Hz observations. Both archives have six frames, but represent different durations. No recorded contact flags or simulator-substep trajectories were found in these outcome archives. Existing native FROZEN predictions also have 2 Hz latent observations, with no RGB decoder or explicit contact event in those prediction bundles.

## Paths and shapes

Set `R = <ADAJEPA_REPO>`, `S ∈ {T,L}`, and `i ∈ {0,1}`. Each pattern below was actually opened for all four cases; full absolute paths per candidate are in the inventory JSON.

| Asset | Path under R | Keys and shape |
|---|---|---|
| Completed support | `artifacts/round_6_shared_revision/results/donor_S_seed101_n12_capture/real_evidence/si_executed_mpc0.npz`, then `mpc1.npz` | `visual` uint8 `[6,224,224,3]`; `proprio` float32 `[6,4]`; `states` float32 `[6,7]`; `normalized_model_actions` float32 `[1,1,10]` |
| Shared query/history | same donor, `real_evidence/si_history_mpc2.pt` | `executed_prefix` float32 `[1,2,10]`; `current_real_observation.visual` `[1,1,224,224,3]`; `.proprio` `[1,1,4]`; `current_public_state` `[1,7]` |
| Ordinary branch pool | `artifacts/round_6_shared_revision/results/evaluate_S_seed101_si_m2_eight/branch_sidecars/si_m2_C_outcome.npz` | `visual` `[6,224,224,3]`; `proprio` `[6,4]`; `states` `[6,7]`; `final_state` `[7]` |
| Branch replay gate | same branch directory, `si_m2_gate_anchor.npz` | `replay_visual`, `online_visual` `[224,224,3]`; `replay_proprio`, `online_proprio` `[4]`; `replay_state`, `online_state` `[7]` |
| Ordinary action binding | `artifacts/round_6_shared_revision/results/reference_S_seed101_si_m2_eight/PREDICTIONS.csv` | `candidate_tensor` + `tensor_key` locate each float32 `[1,5,10]` plan; do not assume every candidate lives in `candidate_actions/` |
| Existing native prediction paths | `artifacts/shared_revision_autoresearch_20260925/results/innovation_S_si_m2/PREDICTED_PATHS.pt` | `['FROZEN'][C]['visual']` float32 `[1,6,1,384]`; `['FROZEN'][C]['proprio']` `[1,6,10]`; these are latent predictions |
| Endpoint-matched action pool | `artifacts/shared_revision_autoresearch_20260925/results/innovation_S_si_m2/endpoint_probe/PREDICTIONS.csv` and `C.pt` | CSV binding, tensor `u_after` `[1,5,10]`; `PREDICTED_PATHS.pt` also exists |
| Endpoint-matched outcomes | `artifacts/shared_revision_autoresearch_20260925/results/evaluate_endpoint_innovation_S_si_m2/branch_sidecars/si_m2_C_outcome.npz` | Same six macro-frame outcome schema |
| Action-fiber pool | `artifacts/shared_revision_autoresearch_20260925/results/fiber_S_si_m2/PREDICTIONS.csv` | Exact action bindings; five candidates |
| Action-fiber outcomes | `artifacts/shared_revision_autoresearch_20260925/results/evaluate_fiber_S_si_m2/branch_sidecars/si_m2_C_outcome.npz` | Same six macro-frame outcome schema |
| RGB-derived completed geometry | `artifacts/shared_revision_autoresearch_20260925/contact_geometry_support_audit/Si/COMPLETED_GEOMETRY.npz` | `object_mask`, `unknown_mask` bool `[224,224]`; `reference` `[2]`; `poses` `[11,3]`; `pusher_pixels`, `observed_pusher_pixels` `[11,2]`; scalar `pusher_radius`, `object_rms_radius`; `pixel_matrix` `[3,2]` |
| Previous factual native increment diagnosis | `round_20260926_native_preserved_increment/Si_PREDICTIONS.pt` | `old_H2.F0` `[1,3,1,394]`; `TF.F0` `[1,1,1,394]`; `legal_prefix_encoded.visual` `[1,6,1,384]`; `plans` `[1,2,10]`; this is an earlier factual query, not the MPC2 H5 candidate pool |

Ordinary candidate IDs are `g0_before`, `g24_after`, `g49_after`, `g99_after`, `official_pred_return`, `fact_matched_return`, `zero_environment_action`, `repeat_last_completed_action`.

Endpoint candidate IDs are `base`, `x_plus_full`, `x_minus_full`, `y_plus_full`, `y_minus_full`, `x_plus_half`, `x_minus_half`, `y_plus_half`, `y_minus_half`.

Fiber candidate IDs are `base_g99`, `fiber_first_minus`, `fiber_first_plus`, `fiber_all_minus`, `fiber_all_plus`.

## Time, state semantics, and what is actually observed

`env/pusht/pusht_env.py:393` sets simulator frequency 100 Hz; lines 365 and 396 set control/observation frequency 10 Hz. A raw control lasts 0.1 s. Checkpoint `frameskip=5`, so one model action is five ordered controls, represented by ten numbers. One predictor step lasts 0.5 s.

`research/instrumentation.py:507`–529 stores every raw frame of an executed factual chunk. Thus MPC0 support times are 0,0.1,...,0.5 s and MPC1 support times are 0.5,0.6,...,1.0 s. Removing the repeated boundary produces 11 RGB/proprio frames and ten controls, ending exactly at MPC2's query.

`research/replay_oracle.py:271`–279 explicitly forms `all_obses[prefix_env_len :: frameskip]` and similarly slices states before saving. A five-model-action branch therefore stores relative times 0,0.5,1.0,1.5,2.0,2.5 s, despite simulating 25 raw controls internally. The 20 intervening raw observation frames were not saved in these files. The array size cannot be used to infer the same temporal resolution as factual support.

`env/pusht/pusht_env.py:560`–569 defines state order as `[agent_x, agent_y, object_x, object_y, object_angle, agent_vx, agent_vy]`. Object pose from `states[:,2:5]` is simulator privilege, available for diagnosis/evaluation only. Raw proprio is pusher position/velocity; it does not contain object pose. `n_contacts` exists in environment info, but is absent from these saved NPZ outcomes and completed-evidence NPZs. A geometric gap computed from RGB/pose is not itself an observed contact impulse or a unique dynamical-mode label.

Completed `poses` from the RGB geometry archive are image-coordinate rigid registration `[dx_pixels,dy_pixels,theta]`, relative to an appearance reference; they are not center-of-mass state or guaranteed contact-boundary truth. The associated `unknown_mask` explicitly marks unresolved occlusion. Do not silently substitute these poses or the separate CONTACT-0/CONTACT-1 predicted poses for native FROZEN predictions.

## Is there a same-geometry, different-speed experiment already present?

All pools begin at the same anchor within each case, which controls initial state. **None of the audited pool construction code establishes the needed pure speed intervention at a fixed local contact geometry.**

- Ordinary candidates arose from planning iterations plus simple controls; different commands may change approach direction, object motion, future contact point, and event sequence together.
- `research/reframe_v3/endpoint_matched_actions.py` constructs changes in the nullspace of a fitted five-control pusher endpoint map. First-chunk terminal pusher position/velocity is matched and later controls are unchanged. Interior paths intentionally differ. This is an excellent action-path-dependence diagnostic, but equal endpoints do not establish equal contact geometry or a scalar reparameterization of speed.
- `research/reframe_v3/action_fiber_probe.py` constructs a raw-action direction removed by the checkpoint action encoder's LayerNorm centering. Those pairs test representational aliasing, not pure speed scaling.

Consequently the existing data can support coarse event-localization comparisons and retrospective path-dependence analysis, but **cannot directly identify or validate `delta_tau = delta_b / v_normal` as a controlled cross-speed law**. A defensible test needs explicit same-local-geometry speed interventions or a justified conditional-matching protocol, finer observations, and a guard against event creation/deletion and tangential crossings. Event speed measured by a 0.5 s finite difference is not a precise velocity at crossing.

## Use boundaries for the current experiment

1. Fit any proposed update or diagnostic calibration from completed RGB, proprio, and actions or an explicitly declared independent training prior. Keep branch RGB/pose outcomes evaluation-only.
2. Freeze the candidate mapping and event diagnostic before reading numerical branch outcomes for selection. This audit read shapes and anchor equality; it did not estimate event errors or select cases by outcome.
3. Use the stored FROZEN latent paths to avoid unnecessary native predictor calls, but first confirm current intended native interface equals their producer contract. Their availability does not certify full-history/current-history equivalence.
4. A diagnostic readout of event/pose from a latent can reveal correlation only if it is validated on held-out real encodings and checks predicted-latent distribution shift. A poor readout cannot establish a native world-model timing defect.
5. New replay at higher observation resolution would recover missing truth, but that is a new environment evaluation, not a read-only reanalysis of saved six-frame outcomes. Existing files and contracts should remain untouched.
6. These four cases and their branches have repeatedly been examined during development. They are useful mechanism falsification material, not an independent generalization test.

## Verification

Executed `python <RUNS_ROOT>/interaction_validation_20260927/asset_inventory.py` on CPU. Exit 0. Result: 88 candidate outcome files opened; all support boundaries equal; all support query frames equal; all branch anchor RGB/proprio equal. No predictor/encoder calls, parameter changes, training, or new environment steps occurred in this audit. The inventory JSON records every binding and shape for reuse.
