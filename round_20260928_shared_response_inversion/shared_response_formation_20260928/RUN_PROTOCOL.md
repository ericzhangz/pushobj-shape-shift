# Shared response formation: frozen protocol

2026-09-28. Written before loading MPC4 outcome payloads.

Objective: evaluate the concrete shared response model and the attachment's
two-step local CSI source solver on the existing native callback path.
No skills, new simulator calls, backbone changes, or candidate regeneration.

1. Offline prior: separate T and L laws; each uses episodes 2..11 only.
   Full three-frame histories at stride five; all valid phase offsets.
   Alpha selected once from {0.1,1,10} by equal-episode leave-one-episode-out
   one-step latent residual MSE. Refit on the ten episodes of that shape.
2. Online support: T0,T1,L0,L1 completed chunks 0..3. Use initial frames
   [0,5,10], targets [15,20] and four packed actions. Query anchor moves to MPC4:
   history [10,15,20], preceding two actions plus the eight existing H5 plans.
   MPC2 outcomes are not used as queries. MPC4 outcomes remain evaluator-only.
3. Features: full flattened 3x404 native cache s and normalized packed action a;
   phi=[1;s/sqrt(1212);a/sqrt(10);(a tensor s)/sqrt(12120)]. No compression,
   physical state, geometry labels, future observation, or candidate-dependent fit.
   Dense response matrix is 384x13343. Writes only visual in the new observation.
4. Prior regression uses raw-sum squared error + alpha||Theta||_F².
   All online arms use lambda=the offline selected alpha and center Theta_Pi.
   Hard objective is sum of both support visual squared errors plus
   lambda||Theta-Theta_Pi||_F². Division by 768 is numerical scaling only.
5. Arms: NATIVE; PRIOR; STEP_RIDGE (sequential observed-visual source recovery
   with native proprio propagation, then ridge); DIRECT (same model, nonlinear
   multistep L-BFGS, 100 iterations / 120 nominal evaluations, strong Wolfe,
   history 8); SOURCE_MU1 and a fixed stress test SOURCE_MU10.
   L-BFGS line-search evaluations are counted even if they exceed its nominal cap.
6. Source method: eight outer iterations. Initialize sources by the prior's
   actual rollout. In the H2 support problem, K=[I,0;J,I], where J is the native
   second visual's derivative with respect to first visual source. Compute exact
   derivative rows with batched autograd. Feature feedback is affine in w1 at
   fixed second action, so Q=Theta*dphi2/dw1 is analytic. Refresh J and affine
   offsets each iteration. Solve the 768-dimensional Eq5 in float64. Backtrack
   source step over {1,.5,.25,.125,.0625} using the true nonlinear joint penalty
   objective; reject the step if none passes. Fixed-source Theta update is
   exact dual ridge with alpha=lambda/mu. No detached-feedback approximation.
7. All solvers return the checkpoint with smallest actual hard support objective
   among their evaluated iterates, including the prior start. Never select by
   query outcomes. Also save the final source Theta and its auxiliary,
   consistency and hard deployment losses to expose relaxation gaps.
8. Report forward calls, batched transitions, backward calls, derivative seed
   vectors and wall time. The methods do not have equal computational cost.
   Do not claim solver efficiency from a raw forward-call comparison. If direct
   stops at its cap with unresolved descent and source uses substantially more
   time, allow a documented support-only budget extension before outcome access.
9. Validate cached encoding vs actual native rollout, exact zero-delta behavior,
   support-source gradient vs finite difference, source initialization consistency,
   native parameter/buffer immutability and common visual prefix.
10. Seal predictions before a separate evaluator reads MPC4 visual/proprio.
    Report full visual/proprio MSE, visible object centroid error with explicit
    failures, endpoint pairwise consequence difference MSE, cost-gap error and
    fixed-pool regret. These are old development assets, not a fresh blind test
    or newly executed closed-loop success rate.

Continue only if the actual deployed shared law improves held action outcomes,
and any claimed solver advantage survives same-model direct adaptation. Support
fit alone, auxiliary fit alone or regret alone does not establish the mechanism.
Failure only rejects or limits this concrete feature/model/solver construction.
