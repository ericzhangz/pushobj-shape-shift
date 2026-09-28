# Native cycle intervention: formation probe

This probe was specified before computing its new outputs. All four anchors and all eight candidates are old development data. This is not a blind confirmation or a new-method benchmark.

Question: does the existing C = encoder(decoder(z)) behave sufficiently like a retraction to justify using it as the mathematical realization of manifold correction? No claim that C is an orthogonal projection, an identified physical manifold, or a causal-history retraction.

No fitting, threshold selection, early stopping, or query-outcome access in prediction. Fixed arms: native FULL replay; READOUT_CYCLE (cycle each already predicted visual code, no feedback); FEEDBACK_CYCLE (cycle each new visual code before inserting it into the same native history). Proprio and actions are preserved. Use the existing observation_transition callback, invoking the same model.predict. No alternative rollout implementation or backbone update.

Calibration uses all 44 past encoded images and their exact unrounded native reconstruction. Report C(z)-z, C(C(z))-C(z), and past temporal differences before/after cycling. No centering, learned compensator, damping, parameter sweep, or threshold relaxation.

Seal predictions before opening EVALUATION_TARGETS.pt. Score all 160 future steps, all 112 unordered candidate pairs, and four selections per arm. Report visual error, candidate gap error and native objective regret separately. READOUT_CYCLE is only a readout control; FEEDBACK_CYCLE is a deployable but uncalibrated baseline, not a D-conditioned adaptation mechanism. Large real-code changes or worse action contrast falsify this particular implementation, not all possible learned retractions.

Expected calls: 320 predictor calls (FULL and FEEDBACK, four times eight times five). Cycle call counts and frame counts recorded directly in the wrapper, since encoder.forward bypasses module hooks. Original checkpoint, source, native prediction bundle and evaluation targets remain unchanged.

Decoder emits the normalized image space used by its reconstruction training. Re-encode these floating point images directly with model.encoder_transform and encoder.forward; do not round to pixels, clip, or normalize a second time. Assert factual reconstruction agrees with the input normalization convention by reporting input/reconstruction ranges.

The original decoder may be unreliable off the encoded-image distribution. No cycle distance is to be reported as ground-truth manifold distance.
