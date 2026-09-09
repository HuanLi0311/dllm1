# Reproducing the audit

Run every command from the repository root. The published evidence can be
verified on CPU; rerunning model probes requires an NVIDIA GPU and downloaded
checkpoints.

## Environments

The SMDM runs used Python 3.9, PyTorch 2.4.1+cu121, and an A100 40GB:

```bash
python -m venv .venv-smdm
source .venv-smdm/bin/activate
python -m pip install -r requirements.txt
```

LLaDA used Python 3.10, PyTorch 2.5.1+cu121, and Transformers 4.46.0. Install it
in a separate environment:

```bash
python -m venv .venv-llada
source .venv-llada/bin/activate
python -m pip install -r requirements-llada.txt
```

## Checkpoints

Weights are not included. With `huggingface-hub` installed:

```bash
hf download nieshen/SMDM \
  mdm_safetensors/mdm-170M-100e18.safetensors \
  mdm_safetensors/mdm-336M-100e18.safetensors \
  mdm_safetensors/mdm-472M-100e18.safetensors \
  mdm_safetensors/mdm-1028M-100e18.safetensors \
  mdm_safetensors/mdm-1028M-1600e18.safetensors \
  gsm8k_safetensors/mdm-1028M-3300e18-rsl-gsm8k.safetensors \
  --local-dir checkpoints

hf download GSAI-ML/LLaDA-8B-Base \
  --local-dir checkpoints/llada-8b-base
```

Expected SHA-256 values are:

- SMDM-219M: `2d8c9b9a730715f2c772d5bc740e12951fc160e5e8511a16835f3537401ea9bb`
- SMDM-401M (`100e18`): `3cd6ec869fc29be1943d0ab2e74f47b59b28d40a2f22314021d60e8ccbb031b2`
- SMDM-554M (`100e18`): `aa672982e20eecb5fd850b3e22053846638cbc65b73a06e304da6588deb6799d`
- SMDM-1.14B (`100e18`): `ed7d52165307e231c3d1882566512d93a6fbf35f8487aa9593a2c6e795964ec6`
- SMDM-1.14B (`1600e18`): `ce96ce67a051613b6d7feb419c99c0b4db5bfcfaaa0833ed7f7ecbc6632841d6`
- SMDM-1.14B GSM8K SFT: `1e968c26419d5b041adf3b1825e6d2b10887c45cdab76e60ea2d8341df31618f`
- LLaDA-8B aggregate: `b84552bd96af3dc51fb9782085672269e95c1e4fe1eebd2a901863a8739a1b95`

The LLaDA aggregate hashes the model index, configuration, remote model code,
and all weight shards; individual hashes are stored in each public envelope.

## Probe examples

The checked-in SMDM input is `runs/data/gsm8k_tasks.jsonl`. A representative
SMDM-219M seed is:

```bash
python experiments/dllm_rank1_probe.py \
  --checkpoint checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors \
  --model-size 170 \
  --code-root SMDM \
  --data runs/data/gsm8k_tasks.jsonl \
  --split eval \
  --mask-probabilities 0.1,0.3,0.5,0.7,0.9 \
  --sample-sizes 32,64,128 \
  --test-samples 128 \
  --shuffle-records \
  --loss-mode native_conditional \
  --include-native-schedule \
  --sequence-length 64 \
  --parameter transformer.h.0.norm_1.weight,transformer.h.8.norm_1.weight,transformer.h.17.norm_1.weight \
  --seed 0 \
  --device cuda \
  --output runs/local/smdm-219m-seed0.json
```

The LLaDA probe tokenizes source documents independently:

```bash
python experiments/llada_geometry_probe.py \
  --checkpoint checkpoints/llada-8b-base \
  --data SMDM/data/gsm8k/test.jsonl \
  --mask-probabilities 0.1,0.3,0.5,0.7,0.9 \
  --sample-sizes 16,32,64 \
  --test-samples 64 \
  --sequence-length 64 \
  --parameter model.transformer.blocks.0.attn_norm.weight,model.transformer.blocks.15.attn_norm.weight,model.transformer.blocks.31.attn_norm.weight \
  --seed 0 \
  --device cuda \
  --output runs/local/llada-8b-seed0.json
```

Every successful envelope records its full command, software versions, source
hashes, checkpoint hashes, selected data, configuration, and direct metrics.

## GSM8K behavioral study

Read `report/gsm8k_rank1_behavior_protocol.md` before rerunning the study. The
runner first evaluates the released GSM8K checkpoint with the upstream
two-pass decoder, then prepares one shared state/Fisher/replay cache without
retraining Task A, and finally starts every paired method from that cache:

```bash
python experiments/smdm_gsm8k_rank1_benchmark.py --self-check

python experiments/smdm_gsm8k_rank1_benchmark.py \
  --mode evaluate --model 1028 \
  --checkpoint checkpoints/gsm8k_safetensors/mdm-1028M-3300e18-rsl-gsm8k.safetensors \
  --output runs/gsm8k_rank1_scale/released_sft_eval.json

python experiments/smdm_gsm8k_rank1_benchmark.py \
  --mode prepare --model 1028 --a-steps 0 \
  --checkpoint checkpoints/gsm8k_safetensors/mdm-1028M-3300e18-rsl-gsm8k.safetensors \
  --gsm-train SMDM/data/gsm8k/train_augmented.txt \
  --cache-prefix runs/gsm8k_rank1_scale/cache/m1028_gsm_s3407

python experiments/smdm_gsm8k_rank1_benchmark.py \
  --mode adapt --model 1028 --a-steps 0 \
  --checkpoint checkpoints/gsm8k_safetensors/mdm-1028M-3300e18-rsl-gsm8k.safetensors \
  --gsm-train SMDM/data/gsm8k/train_augmented.txt \
  --cache-prefix runs/gsm8k_rank1_scale/cache/m1028_gsm_s3407 \
  --method rank1_gd \
  --output runs/gsm8k_rank1_scale/pilot/m1028_rank1_gd_s3407.json
```

The released SMDM recipe uses the Git-LFS `data/gsm8k/train.txt` from
`da03/implicit_chain_of_thought`: 384,620 problems become 769,240 conditional
training instances and are trained for 40 epochs. Download it only when
auditing or reproducing upstream SFT; it is intentionally ignored rather than
redistributed by this repository:

```bash
curl -L --fail \
  -o SMDM/data/gsm8k/train_augmented.txt \
  https://media.githubusercontent.com/media/da03/implicit_chain_of_thought/main/data/gsm8k/train.txt
sha256sum SMDM/data/gsm8k/train_augmented.txt
```

The expected hash is
`6f9a20bc1476ca65eee9bc5117c2d0582b1f1733d5148ffc0ff29cad2a9e9c6b`.
The commands above illustrate the runner API, not every completed matrix cell. All
preparation-affecting flags must be identical between `prepare` and `adapt`;
the cache rejects drift.

After all 12 full-test endpoints plus the prespecified sensitivity, ablation,
and audit cells have finished, build the sanitized aggregate and the appendix
table without publishing local envelopes:

```bash
python experiments/summarize_gsm8k_rank1_behavior.py \
  --tex-table ../assets/iclr_1/inputs/gsm8k_behavior_table.tex
```

The summarizer rejects missing or non-1,319-question full endpoints. It writes
`runs/data/gsm8k_rank1_behavior_results.json` and
`report/gsm8k_rank1_behavior_results.md`; the former contains hashes of every
consumed private envelope but no machine paths.

## Null and figures

Regenerate the CPU null:

```bash
python experiments/simulate_fisher_null.py \
  --case 768:8:128 --case 768:16:128 --case 768:32:128 \
  --case 768:64:128 --case 768:128:128 \
  --case 1792:8:64 --case 1792:16:64 --case 1792:32:64 \
  --case 1792:64:64 \
  --repetitions 200 \
  --seed 20260825 \
  --device cpu \
  --output runs/local/isotropic_null.json
```

Then regenerate all plots directly from the compressed public evidence:

```bash
python experiments/make_paper_figures.py \
  --geometry \
    evidence/raw/runs/r08_split_primary/gsm_170_s*/benchmark.json.gz \
    evidence/raw/runs/r08_split_primary/gsm_1028_s*/benchmark.json.gz \
    evidence/raw/runs/r16_llada_geometry/*.json.gz \
  --comparison-control \
    evidence/raw/runs/r12_audited_controls/gsm_all_170_s*/benchmark.json.gz \
    evidence/raw/runs/r12_audited_controls/gsm_one_170_s*/benchmark.json.gz \
  --comparison-contract evidence/comparison_contract.json \
  --slice-control \
    evidence/raw/runs/r10_attention_split/*/benchmark.json.gz \
    evidence/raw/runs/r17_late_dense_matched/*.json.gz \
  --null runs/local/isotropic_null.json \
  --output-dir figures
```

`figures/figure_data.json` records the hash of every consumed input. Generated
figures are ignored by Git because they are reproducible products.

## Public evidence verification

```bash
python experiments/build_review_bundle.py --verify evidence/release_manifest.json
python experiments/build_review_bundle.py --verify evidence/r18_completion/release_manifest.json
python experiments/build_comparison_contract.py --self-check
```

The first command verifies all 25 primary compressed hashes; the second verifies
the separate appendix extension. Both scan decompressed JSON for identity-bearing
absolute paths. The checked-in comparison contract also recomputes rank-1 and
diagonal errors from stored sufficient statistics when the figure script loads
it.
