# R19 MNIST UNet source-setting evidence

This directory is the public, identity-scrubbed counterpart of the private R19
run family. It is a separately scoped appendix check and does not change the
primary 786-row release manifest or the R18 scope-extension bundle.

The bundle contains three complete audit envelopes (training seeds 0--2) and
their machine-readable summary. Each seed trains the source paper's
152,497-parameter MNIST UNet for 200 epochs, then evaluates the full parameter
gradient at timesteps 100--900. Every timestep uses 1,024 calibration examples
and 1,024 disjoint test examples with independent diffusion noise. The source
implementation is pinned to commit
`c7577f22551941e4bf58e33405fc78e8fcb608aa`.

Validation requires:

- exactly three successful audits with seeds 0--2 and all 27
  seed--timestep cells;
- the fixed architecture, parameter count, training recipe, source commit,
  and 1,024/1,024 split sizes;
- disjoint calibration and test indices in every seed;
- finite rank-1, diagonal, and oracle errors, with the oracle no worse than
  the fitted rank-1 estimator; and
- exact hashes for the audits, summary, probe, summarizer, and this report.

Rank-1 has lower held-out relative Frobenius error than diagonal in 26/27
seed--timestep cells. Averaged equally over the grid, the errors are 0.644 for
rank-1, 0.994 for diagonal, and 0.179 for the test-Fisher rank-1 oracle. The
result supports favorable rank-1 structure in this small source setting, but
the cross-seed variation and oracle gap do not establish uniformly faithful
mean-gradient reconstruction or continual image-generation utility.

From the workspace root, verify the compressed hashes and recompute the public
summary with:

```bash
PYTHONNOUSERSITE=1 python iclr_1/experiments/build_review_bundle.py \
  --verify iclr_1/evidence/r19_mnist_unet/release_manifest.json

PYTHONNOUSERSITE=1 python iclr_1/experiments/summarize_mnist_unet_fisher.py \
  iclr_1/evidence/r19_mnist_unet/raw/runs/r19_mnist_unet_heldout/seed_*/audit.json.gz \
  --summary /tmp/r19_mnist_unet_summary.json \
  --figure-stem /tmp/r19_mnist_unet_fisher
```
