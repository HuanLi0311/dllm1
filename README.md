# When Is Rank-1 Fisher Geometry Faithful?

Code and public evidence for **When Is Rank-1 Fisher Geometry Faithful? An
Empirical Audit in Masked Diffusion Language Models**.

The project tests whether a mean-gradient rank-1 surrogate is a reliable
approximation to empirical Fisher geometry when fitting and evaluation use
disjoint gradient samples. It measures direct relative Frobenius error on
selected SMDM-219M, SMDM-1.14B, and LLaDA-8B parameter slices. GSM8K supplies
standardized text only. A separately scoped appendix reanalyzes synthetic-fact
continual-learning endpoints and a full-test GSM8K retention study for context;
neither establishes a causal link between reconstruction error and forgetting.

## Quick start

The small checks run without checkpoints or GPUs. The SMDM environment used
Python 3.9 and CUDA 12.1:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

for script in \
  simulate_fisher_null.py \
  dllm_rank1_probe.py \
  llada_geometry_probe.py \
  run_audited_geometry_probe.py \
  make_paper_figures.py \
  build_comparison_contract.py \
  build_submission_manifest.py \
  build_review_bundle.py
do
  PYTHONNOUSERSITE=1 python "experiments/$script" --self-check
done

PYTHONNOUSERSITE=1 python continual_mdm.py --self-check
PYTHONNOUSERSITE=1 python experiments/dllm_rank1_transfer.py --self-check
PYTHONNOUSERSITE=1 python experiments/build_dolly_stream.py --self-check
PYTHONNOUSERSITE=1 python experiments/smdm_gsm8k_rank1_benchmark.py --self-check
PYTHONNOUSERSITE=1 python experiments/summarize_gsm8k_rank1_behavior.py --self-check
```

LLaDA-8B needs a separate Python 3.10 environment because its model code uses a
newer Transformers release:

```bash
python -m pip install -r requirements-llada.txt
```

Checkpoints are intentionally excluded. Download instructions, complete probe
commands, and figure regeneration are in [REPRODUCING.md](REPRODUCING.md).

## GSM8K behavioral study

The completed benchmark validates the released 1.14B GSM8K checkpoint, caches
one shared learned Task-A state per seed, and compares four paired adaptation
methods on all 1,319 test questions. Mean-rank-1 and diagonal EWC have matched
implemented trace. The protocol and limitations are recorded in
[`report/gsm8k_rank1_behavior_protocol.md`](report/gsm8k_rank1_behavior_protocol.md);
the sanitized aggregates and interpretation are in
[`report/gsm8k_rank1_behavior_results.md`](report/gsm8k_rank1_behavior_results.md).

## Public evidence

`evidence/raw/` contains the 25 accepted primary experiment envelopes as
deterministic, anonymized JSON gzip files. No file exceeds GitHub's 100MB file
limit. Verify every compressed-file hash and scan the decompressed content with:

```bash
python experiments/build_review_bundle.py --verify evidence/release_manifest.json
```

The separately validated appendix extension is under `evidence/r18_completion/`
and has its own `release_manifest.json`.

Local `runs/` outputs record exact machines and paths and are therefore ignored
by Git except for the deterministic input data. Rebuild `evidence/` before a
release; never publish the local envelopes directly.

## Layout

- `experiments/` — probes, controls, plots, and evidence verifiers.
- `evidence/` — public compressed envelopes and their hash manifests.
- `runs/data/` — deterministic input subsets and token IDs used by the probes.
- `SMDM/` — the Apache-2.0 upstream model code needed to load SMDM checkpoints.
- `runs/data/gsm8k_rank1_behavior_results.json` — sanitized behavioral
  aggregates and input hashes.
- `report/gsm8k_rank1_behavior_protocol.md` — frozen behavioral protocol and
  scale-screen boundary.
- `report/gsm8k_rank1_behavior_results.md` — completed behavioral results.
- `report/INVALIDATED_RESULTS.md` — reasons superseded pilots are excluded.

## License

Original code in this repository is MIT licensed. Vendored code, data, and
downloaded checkpoints retain their upstream licenses; see
[NOTICE.md](NOTICE.md).
