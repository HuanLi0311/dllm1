# R18 scope-extension evidence

This directory is the public, identity-scrubbed counterpart of the private R18
run family.  It is separate from the primary 786-row release manifest and does
not change that manifest's scope or counts.

The bundle contains three complete SMDM-1.14B dense-slice envelopes (seeds
0--2) and the machine-readable R18 summary.  Each compressed envelope is
deterministic and records the SHA-256 of its private source artifact in
`release_provenance`.  The summary also authenticates and independently
recomputes the reused R12 cross-text and R16 continual-learning evidence from
their public raw envelopes.

Validation requires:

- exactly 18 unique dense seed--layer--mask cells: seeds 0--2, layers 0/19,
  and mask probabilities 0.1/0.5/0.9;
- 64 calibration and 64 disjoint test examples in every dense cell;
- finite rank-1, diagonal, fitted-top-1, and oracle errors, with the held-out
  oracle no worse than either fitted rank-1 estimator;
- exact probe, checkpoint, data, and source hashes;
- all six R12 cross-text envelopes present in their audited comparison
  contract; and
- all 33 R16 continual runs recomputed from stage-level metrics and equal to
  the locked aggregate and paired summaries.

The dense extension is mixed rather than selectively favorable: rank-1 wins
8/18 raw seed-cells and 2/6 layer--mask cell means.  Its seed-level paired log
score is `-0.1063 +/- 0.0705 SEM`, or a geometric-mean rank-1/diagonal error
ratio of 1.112.  The seed-0, layer-0, mask-0.1 rank-1 error of 4.456 is retained.

The R12 and R16 entries are independent recomputations of existing public
evidence, not new runs.  Cross-text results confound source and preprocessing;
the downstream matrix uses synthetic factual associations and does not
causally link reconstruction error to forgetting.  A short development smoke
run remains private and contributes no reported number.

From the `iclr_1` project root, verify the bundle with:

```bash
PYTHONNOUSERSITE=1 python experiments/build_review_bundle.py \
  --verify evidence/r18_completion/release_manifest.json
```
