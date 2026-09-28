# Post-hoc evaluation-only contraction audit

Registered after the cycle probe was scored. This audit uses saved query truth solely to evaluate mathematical compatibility of a state-only correction class. It neither trains a map nor measures deployment-identifiable directions. All four development cases, five horizons and all 28 unordered pairs are retained (560 rows).

Let p_i be a native FULL predicted visual token and q_i its actual encoded target. For any map R applied to these same tokens with Lipschitz constant at most L in visual Euclidean norm, the pair-average squared error is at least max(0, ||q_i-q_j|| - L||p_i-p_j||)^2/4. Use RMS norms (Euclidean norm / sqrt(384)) consistently. Exact recovery of the pair requires L >= ||q_i-q_j|| / ||p_i-p_j|| when the denominator is nonzero. An infinite bound is explicitly recorded when only the denominator is zero.

This theorem concerns a state-only map applied to a frozen pair. It does not apply directly to feedback-corrected rollout (which changes later inputs), or an action/history-conditioned map (which has more arguments). Nonlinear retractions are not universally nonexpansive; a positive lower bound at L=1 does not disprove all manifold retractions. It does test whether simple contraction is a sufficient account.

No claim of physical distance: all norms use the frozen native 384-dimensional visual representation. The original 42px visible separation is a separate diagnostic and is not mixed with latent norms. No fitted directions from this audit enter a deployed method.
