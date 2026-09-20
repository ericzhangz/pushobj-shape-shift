# C effect-size report

All deltas are label minus Frozen at the same anchor; `epsilon=1e-6` is inherited.

| label | final cost improve / tie / worsen | mean delta | median delta | state improve / tie / worsen | mean state delta | median state delta |
|---|---:|---:|---:|---:|---:|---:|
| adapted | 0 / 3 / 3 | 0.000296614443262 | 0.000155052170157 | 2 / 0 / 4 | 0.312898000081 | 0.00274085998535 |
| oracle | 2 / 1 / 3 | -0.000565058086067 | 0.000307271722704 | 2 / 1 / 3 | -0.715592066447 | 0.155065536499 |

The oracle label was selected with full-plan environment outcomes and is diagnostic only.  It is not a deployable candidate generator.  In the legal Frozen/adapted subset, adapted has no anchor with a final fixed-reference cost improvement beyond epsilon.
