# SMDM GSM8K behavioral retention results

Full-test, three-seed behavioral endpoints from one released 1.14B GSM8K checkpoint; the reduced-data four-scale screen is not a matched cross-scale retention comparison.

## Released-checkpoint validation

The released 1.14B GSM8K checkpoint scores 784/1319 (59.44%) strict exact match on the full test set. Its Wilson 95% interval is 56.77%–62.06%; the published SMDM Table 2 value is 58.50%.

## Reduced-data scale feasibility screen

| Config | Actual parameters | Updates | Questions | Strict | Fallback numeric | Delimiter | Gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 170M | 219,050,496 | 5000 | 64 | 0.00 | 1.56 | 4.69 | fail |
| 336M | 401,123,328 | 5000 | 64 | 0.00 | 4.69 | 0.00 | fail |
| 472M | 553,827,840 | 5000 | 64 | 0.00 | 9.38 | 7.81 | fail |
| 1028M | 1,142,367,744 | 5000 | 64 | 0.00 | 0.00 | 9.38 | fail |

This screen used only 5,000 updates on the unaugmented training set; it is not matched to the released 40-epoch augmented recipe.

## Stiffness calibration

The held-out later-task-loss rule selected `c=1e-05`; the 100-step GD reference loss was 7.3890.

## Primary full-test endpoint

All endpoint accuracies below use the complete 1,319-question GSM8K test set. Conditional retention remains a paired diagnostic on the fixed 128-question audit subset.

| Seed | Method | Final strict accuracy | Change from released checkpoint | Conditional retained | Later loss |
|---:|:---|---:|---:|---:|---:|
| 3407 | seq | 20.47 | -38.97 pp | 30.77 | 7.2804 |
| 3407 | gd | 19.56 | -39.88 pp | 30.77 | 7.3692 |
| 3407 | rank1_gd | 50.49 | -8.95 pp | 80.77 | 7.4382 |
| 3407 | diag_gd | 52.99 | -6.44 pp | 83.33 | 7.3274 |
| 3408 | seq | 21.91 | -37.53 pp | 43.59 | 7.1745 |
| 3408 | gd | 40.64 | -18.80 pp | 69.23 | 7.1951 |
| 3408 | rank1_gd | 51.48 | -7.96 pp | 85.90 | 7.3988 |
| 3408 | diag_gd | 47.01 | -12.43 pp | 76.92 | 7.1878 |
| 3409 | seq | 23.65 | -35.78 pp | 37.18 | 7.1698 |
| 3409 | gd | 49.81 | -9.63 pp | 79.49 | 7.1819 |
| 3409 | rank1_gd | 54.97 | -4.47 pp | 92.31 | 7.5244 |
| 3409 | diag_gd | 47.69 | -11.75 pp | 75.64 | 7.2007 |

| Method | Final strict accuracy | Change from checkpoint | Conditional retained | Later loss | Mean EWC term |
|:---|---:|---:|---:|---:|---:|
| seq | 22.01 ± 1.59 | -37.43 ± 1.59 pp | 37.18 ± 6.41 | 7.2082 ± 0.0626 | 0.0000 |
| gd | 36.67 ± 15.51 | -22.77 ± 15.51 pp | 59.83 ± 25.68 | 7.2487 ± 0.1046 | 0.0000 |
| rank1_gd | 52.31 ± 2.35 | -7.13 ± 2.35 pp | 86.32 ± 5.78 | 7.4538 ± 0.0643 | 2.9567 |
| diag_gd | 49.23 ± 3.28 | -10.21 ± 3.28 pp | 78.63 ± 4.12 | 7.2386 ± 0.0771 | 0.0020 |

Rank-1 + GD minus GD was +15.64±13.54 percentage points across seeds (wins/ties/losses: [3, 0, 0]); Rank-1 + GD minus diagonal + GD was +3.08±5.04 percentage points (wins/ties/losses: [2, 0, 1]).

## Secondary 128-question trajectory audit

| Seed | Method | Task A | After B | After C | Conditional retained | Later loss |
|---:|:---|---:|---:|---:|---:|---:|
| 3407 | seq | 60.94 | 23.44 | 19.53 | 29.49 | 7.3066 |
| 3407 | gd | 60.94 | 36.72 | 8.59 | 11.54 | 7.3385 |
| 3407 | rank1_gd | 60.94 | 60.94 | 56.25 | 89.74 | 7.5689 |
| 3407 | diag_gd | 60.94 | 53.91 | 55.47 | 85.90 | 7.3238 |
| 3408 | seq | 60.94 | 27.34 | 17.19 | 25.64 | 7.2055 |
| 3408 | gd | 60.94 | 49.22 | 47.66 | 71.79 | 7.1882 |
| 3408 | rank1_gd | 60.94 | 61.72 | 56.25 | 88.46 | 7.4712 |
| 3408 | diag_gd | 60.94 | 47.66 | 48.44 | 75.64 | 7.1987 |
| 3409 | seq | 60.94 | 34.38 | 23.44 | 34.62 | 7.1691 |
| 3409 | gd | 60.94 | 49.22 | 51.56 | 78.21 | 7.1794 |
| 3409 | rank1_gd | 60.94 | 60.94 | 53.91 | 84.62 | 7.5209 |
| 3409 | diag_gd | 60.94 | 55.47 | 56.25 | 85.90 | 7.1880 |

Values except later loss are percentages.

## 128-question aggregate

| Method | Final strict accuracy | Conditional retained | Later loss | Mean EWC term | Clip rate |
|:---|---:|---:|---:|---:|---:|
| seq | 20.05 ± 3.16 | 29.91 ± 4.50 | 7.2271 ± 0.0712 | 0.0000 | 99.95 |
| gd | 35.94 ± 23.76 | 53.85 ± 36.78 | 7.2354 ± 0.0894 | 0.0000 | 99.98 |
| rank1_gd | 55.47 ± 1.35 | 87.61 ± 2.67 | 7.5203 ± 0.0489 | 1.6811 | 100.00 |
| diag_gd | 53.39 ± 4.30 | 82.48 ± 5.92 | 7.2368 ± 0.0755 | 0.0020 | 99.97 |

## High-quality replay sensitivity (seed 3407)

| Method | Final strict accuracy | Conditional retained | Later loss |
|:---|---:|---:|---:|
| gd | 53.12 | 82.05 | 7.4141 |
| rank1_gd | 54.69 | 87.18 | 7.4667 |
| diag_gd | 54.69 | 85.90 | 7.3379 |

## Rank-1 without replay

| Seed | After B | After C | Conditional retained | Later loss |
|---:|---:|---:|---:|---:|
| 3407 | 58.59 | 57.81 | 89.74 | 7.6807 |
| 3408 | 60.94 | 53.91 | 83.33 | 7.5359 |
| 3409 | 60.94 | 58.59 | 89.74 | 7.9220 |

The three low-cost replay caches had 0.00%–1.56% strict accuracy, 79.69%–85.94% fallback numeric accuracy, and 0.00%–1.56% delimiter rate; released two-pass replay had 82.81% and 100.00%, respectively.

## Interpretation

The full-test comparison is the behavioral endpoint; the 128-question trajectories are mechanism diagnostics rather than the headline estimate. A rank-1-specific claim requires a stable advantage over trace-matched diagonal EWC, not only an advantage over low-quality generated replay.

Rank-1 + GD improved strict exact match over low-cost GD by 15.64 points, but its fallback-numeric margin was only 5.66 points while its delimiter-rate margin was 14.61 points. Much of the strict-score gap therefore tracks preservation of the required output marker, especially in the unstable GD seed.

Against trace-matched diagonal + GD, the Rank-1 + GD difference was +3.08±5.04 points with per-seed signs [2, 0, 1]. This three-seed result does not establish a stable rank-1-specific advantage.

Rank-1 without replay differed from Rank-1 + GD by 1.30±3.52 percentage points; the per-seed signs were mixed, so replay supplied no stable increment under the strong rank-1 penalty.

Equal implemented trace did not equal equal realized constraint: the mean weighted EWC term was 2.9567 for rank-1 and 0.0020 for diagonal. This accompanies higher later-task loss for rank-1 and limits utility claims from trace matching alone.

The unadapted checkpoint had later-task loss 13.8640; all full-endpoint methods reduced it by 6.4102–6.6557. Retention therefore did not arise from completely refusing to learn the later tasks.

The same-seed GPU repeat was not bit-exact: outputs matched on 43.75% of 16 audited questions and correctness matched on 87.50%. Small percentage-point differences should therefore not be interpreted as stable method effects.

The four reduced-data scale pilots all failed the predeclared Task-A gate. They therefore do not identify cross-scale retention, and they are not evidence that the smaller checkpoints cannot learn GSM8K.
