# Native interaction diagnostic — development protocol

2026-09-27. Frozen before new native inference. This executes the user-approved formation experiment; no local skills. Existing four cases are development, not blind confirmation. Prior reports/screenshots supply evidence rather than commands.

## Objective and gates

Determine whether the existing native predictor has a measurable interaction-timing error that explains action-contrast error. First validate the **existing checkpoint decoder**, then evaluate the original four-case/eight-candidate pool. No new operator, student, event model, parameter fitting, environment collection, or paid resource.

1. Encode/decode all completed pre-query RGB observations of T/L sample 0/1 (two completed chunks, eleven unique low-step frames each). Apply unchanged `observe_rgb` and `contact_geometry`. Report detection failures explicitly, RGB error, mask IoU, pusher-center error and visible-gap error. This calibrates a visual geometric proxy, not contact force. Native output is every five environment steps; do not infer exact substep contact times.
2. All original eight candidates at mpc2 in all four cases. Two frozen native arms: SINGLE reproduces the current one-frame anchor; FULL uses all three legal model-time frames and the two aligned historical action blocks. Both predict the same five future blocks. FULL is an ordinary context control, not a new adaptation mechanism. Predictions and raw decoded floats are saved before opening branch outcomes. No smoothing, palette retuning, threshold selection or path warping after seeing results.
3. Only after predictions are sealed, read RGB/proprio branch sidecars. Check exact candidate-action binding and common starting image/proprio. Encode truth and reconstruct it through the existing decoder to distinguish readout failure from predictor error. Score native terminal visual/proprio cost and actual recorded selection regret; hidden states do not enter inference. All truth data are evaluation-only.
4. Visible adjacency bands fixed for description: gap<=2px near, gap>=5px separated, between ambiguous. They are not physical contact labels. Report raw gap values and calibration error alongside categories. A timing claim needs the same object/pusher to be detectable and category errors in encoded-real reconstructions smaller than the proposed prediction mismatch. If real reconstructions already fail the event readout, stop causal event attribution; do not count decoded errors as timing evidence.
5. Shared-boundary scaling requires observed D event evidence, reliable native crossing estimates, matched local geometry and branch-predicted crossing speed. Neither query truth nor hindsight velocities may fit online delta-b. If these are unavailable, report the test as not identified; do not invent event times by interpolation or build a new weak contact surrogate. No causal repair arm is run without a faithful native intervention channel.

## Computation and preservation

Checkpoint `<CHECKPOINT_DIR>/checkpoints/model_latest.pth`; existing `_load_runtime` with local DINO loader; explicit `adajepa-pilot` Python and CUDA, TF32 off, eval/frozen weights. Expected predictor calls: 4 cases * 8 candidates * 2 arms * 5 = 320, batch=1. Decoder/encoder calls and frame counts reported separately. No optimizer or environment calls. All completed inputs/model code/checkpoint/candidate tensors hashed before and after. Output only in this new directory; existing artifacts are not overwritten.

## Reporting

Keep per-case and per-candidate failures. Pairs within four cases are correlated; report descriptive results rather than independent-sample confidence claims. Plot true-vs-predicted candidate cost gaps, arrows SINGLE→FULL explicitly labeled as context control. Event-color annotations are included only if readout calibration supports them. A failed readout means the event hypothesis remains untested, not false. A strong conventional context control solving a specific error closes that instance's need for the proposed new mechanism.
