# Round 3 — Existing-data reanalysis

This is a zero-model-call, zero-environment-call reconciliation from the recorded CSVs.

## A: finite recorded-pool selection

- 800 replans; tie-aware regret exceeds `1e-6` in 682.
- Recorded-pool optimizer gap exceeds `1e-6` in 0.
- Full optimization improves recorded environment outcome in 658 and worsens it in 142.
- These are nested replans, not 800 independent tasks.

## B: same experience, same query pool

- Support loss decreases in 12/12 anchors.
- Candidate ID changes in 11/12; 3 are only `g99_before`/`g99_after` side changes.
- Tie-aware regret: 2 improved, 6 worsened, 4 unchanged; mean delta `0.000154397062336`.
- `b_selection_existing_results.csv` records exact argmin sets, epsilon-tie sets, and full/first-chunk action magnitudes without adding a post-hoc behavioral threshold.

## C: open-plan target versus common-continuation target

- `matched_target_existing_results.csv` places `J_env_open_plan` and `Q_env_common_continuation` on the same logical row.
- The difference `b_pi = Q_env - J_env` is an intervention-target shift, not model error.
- Final successes are Frozen 4/6, adapted 4/6, oracle 4/6.
- The three labels within an anchor are interventions on one starting case, not independent episodes.

No new threshold, candidate selection, model query, or environment branch was introduced.
