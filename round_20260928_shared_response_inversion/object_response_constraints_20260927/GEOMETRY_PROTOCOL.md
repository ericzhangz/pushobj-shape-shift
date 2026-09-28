# Response-constraint power audit, fixed before measurements

2026-09-27. This is a CPU audit of an existing model's forecasts, not a new forecasting method. No model/environment calls or training. No local skills.

Question: does transported material support observed in completed D exclude the wrong object responses in the old development pool? Raw same-frame colored object/pusher masks trivially exclude one another and are not an adequate penetration check.

Use the existing `rgb_rigid_observer.register_pair`, `observe_rgb`, `SingleContactScene` and completed-only fused masks from `contact_geometry_support_audit`. The mask is in the final completed frame. For each target image, register the final completed RGB to that image; transport this D-only mask using that SE(2) pose and evaluate its signed gap to the image's visible pusher center with the D radius. A negative gap is penetration of this estimated known support. Unknown pixels are never filled; a nonnegative gap is NOT proof of complete nonpenetration. This is an endpoint condition only: 0.5 s endpoint samples cannot certify swept-path viability.

All 4 old cases, all 8 original candidates, all 5 future steps. Read FULL native decoded forecasts, REAL evaluation RGB and RECON_REAL decoded true observations, with the same measurement. Evaluation truth is only used to measure errors and readout limitations. No fit, branch selection, constraint tuning, or forecast modification uses it. Decode uses the existing [-1,1] convention with uint8 clipping; this is a readout operation only.

Compute all 11 D frame measurements per case with the same registration direction. Save every affine, signed gap, centroid, registration residual, convergence, and alternative-start objective gap. Report continuous penetration and D maximum-penetration envelope exceedance. The envelope is an empirical readout reference, not a confidence bound or proof. Do not discard failed registrations or unfavorable frames. Compare reconstructed-real and real to quantify decoder/readout limits. Registration sees the target image and thus is a measurement, not a dynamics predictor.

Protect input hashes, assert current RGB matches all evaluated branch anchors, save all rows and recomputable derived results. No physical state, force, contact label, or query feedback is supplied to a predictor. Lack of violations does not establish correctness; violations do not identify the true response law. An implementable constraint must pass this gate before being proposed as a correction mechanism.
