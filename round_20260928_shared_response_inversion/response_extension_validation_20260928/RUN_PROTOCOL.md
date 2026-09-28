# Bounded extension audit: frozen execution contract

2026-09-28. Executes stage A of ../response_extension_plan_20260928/PLAN.md.
No local skills, training, new environment interaction, posterior fitting, new
deployment operator, or query-label-directed construction.

Purpose: decide whether the existing extension rule exposes a substantial
cross-action prediction problem. Projection, nullspaces, arbitrary perturbations,
or a better old-case score are not proposed contributions. A positive sensitivity
result does not establish a physical cause, a new algorithm, or paper readiness.

1. Import the unchanged formation.py runtime, phi, support_view and cached_rollout.
   Use saved DIRECT/PRIOR, PREPARED encodings, and the old prediction bundle's
   actions, initial context and goal. Do not open query outcomes in the runner.
2. For all four cases, replay DIRECT support and recover its own feature matrix V.
   Use float64 thin SVD (relative threshold 1e-12) to form a basis, never a dense
   13343-square projector. Project the online delta about the same prior.
   Check support token preservation at the existing 2e-5 maximum absolute
   tolerance. Report projection residual and objective change without retuning.
3. Replay DIRECT and projected DIRECT on all 32 H5 queries. Verify DIRECT tokens
   against the saved predictions at 2e-5; keep old PRIOR/NATIVE/STEP_RIDGE outputs
   as explicit cached comparisons. Record exact input/code/model hashes.
4. The output direction u for each shape is the leading left singular vector of
   uncentered native one-step visual residuals from that shape's legal offline
   episodes 2..11, using the original 205 overlapping windows. Canonicalize sign
   by making the largest absolute element positive. No query labels are used.
   These calls count toward the 1600-sample-transition ceiling.
5. For T0/L1 only, construct two normalized input directions in the support-null
   subspace: mean of eight projected first-query features (COMMON); leading left
   singular vector of their centered matrix (DIFFERENTIAL). Fail explicitly on
   a zero direction; do not invent a replacement or inspect outcomes.
6. About the support-equivalent projected center, use eta={.01,.1,1}, rho equal
   to sqrt(eta*J_center/lambda), lambda=the existing 10. Eta=.1 is the primary
   comparison, .01 smaller-perturbation check, 1 stress only. Construct both
   signs rho*u*v^T; the same theta is used for every candidate and every step.
   These objective budgets are sensitivity scales, not calibrated probabilities.
7. Each variant replays two support steps and eight H5 paths. Verify support
   preservation at 2e-5 and record actual parameter penalty and data error;
   do not hide nonzero roundoff cross terms. Check first-step plus/minus outputs
   against the exact affine expression at 2e-5. No optimization of u/v/rho.
8. Expected predictor sample transitions: 205 offline directions + 336 base/
   projected paths + 1008 perturbation paths =1549, all batch size one. Cap1600.
   Count calls and sample transitions separately. Decoder/encoder cost is separate.
   Freeze backbone version counters; no gradients or backwards are needed.
9. Save audit traces/recipes, seal predictions and source hashes before independent
   evaluation. All old cases are development data already inspected in prior work;
   sealing construction does not turn them into blind evaluation.
10. The evaluator uses the old sealed TARGETS cache and recorded truth costs,
    reports step1/steps1-2/steps3-5 and all5 errors, per-step common/contrast error,
    candidate cost-gap MAE, regret, decoded object readout and failures. Retain
    the earlier 4/32 strict truth-cost replay failures as an inherited limitation.
11. No B-stage probe collection is automatic in this contract. A result may
    justify a separately specified, query-relevant information experiment; it
    cannot justify unlimited parameter tweaks, more directions or probability
    machinery just to make this particular residual adapter look novel.
