# GSM8K rank-1 Fisher behavior protocol

**Status: frozen after calibration; full-test matrix completed 2026-09-09.**
The released checkpoint passed the independent evaluator, the stiffness rule
selected `c=1e-5` without using a retention endpoint, and all 12 predeclared
full-test cells completed. Development and reduced-data scale-screen outputs
remain separate from the primary behavioral endpoint.

## Question and claim boundary

The behavioral experiment asks whether mean-gradient rank-1 EWC improves
retention of learned GSM8K behavior, beyond the same generated-distillation
replay, and how it compares with trace-matched diagonal EWC. It does not equate
held-out Fisher reconstruction error with continual-learning utility.

The confirmatory downstream comparison is initially restricted to the released
1.14B GSM8K checkpoint. The four released base-model sizes remain a geometry
panel and a Task-A feasibility panel; they become a cross-scale retention panel
only if each passes the same predeclared behavioral learnability gate. A missing
math-tuned checkpoint is not interpreted as zero retention.

## Checkpoints and staged eligibility

The scale-feasibility panel uses checkpoints carrying the same advertised
`100e18` pretraining-compute label:

| Config | Actual parameters | Checkpoint SHA-256 |
|---:|---:|---|
| 170M | 219,050,496 | `2d8c9b9a...a9bb` |
| 336M | 401,123,328 | `3cd6ec86...31b2` |
| 472M | 553,827,840 | `aa672982...799d` |
| 1028M | 1,142,367,744 | `ed7d5216...4ec6` |

The `mdm-1028M-1600e18.safetensors` checkpoint is a same-size,
different-pretraining-compute control. The released
`mdm-1028M-3300e18-rsl-gsm8k.safetensors` checkpoint (SHA-256
`1e968c26...1618f`) is the primary learned Task-A state. It first validates the
decoding/scoring chain, then supplies the shared start state for downstream
adaptation with `--a-steps 0`.

## Task-A data and released-recipe audit

The SMDM paper and code train on the augmented GSM8K file released with Deng et
al. (2023), not on the ordinary GSM8K training split. The upstream file contains
384,620 valid source problems (SHA-256
`6f9a20bc1476ca65eee9bc5117c2d0582b1f1733d5148ffc0ff29cad2a9e9c6b`).
The released preprocessor turns every source problem into two conditional
instances—question to chain of thought, then question-plus-thought to final
answer—giving the 769,240 instances hard-coded by the SMDM trainer. It trains
for 40 epochs with global batch 256, AdamW betas `(0.9, 0.95)`, weight decay
0.1, cosine decay from `2e-4` to `2e-5`, 1% warmup, and clip 1.

The unaugmented local source has 5,249 valid problems and one truncated final
line, which is rejected. The runner now applies the same two-conditional,
separately-tokenized representation, yielding 10,498 instances. Forty seeded
source problems are held out by source identity for Fisher estimation, giving
80 conditional Fisher rows; the remaining 5,209 problems give 10,418 possible
optimization instances. The untouched 1,319-row official test set is the
behavioral endpoint.

Two early 5,000-update pilots at 219M and 401M used the unaugmented file before
the released two-conditional encoding and two-pass decoder were recovered.
Both produced 0/128 strict exact match and clipped on more than 99.7% of steps.
They are protocol-discovery runs, not evidence that those scales cannot learn
GSM8K. Repeating a reduced-data recipe is optional only after the released
checkpoint path is validated; reproducing the 40-epoch augmented recipe at all
four scales is a separate, much larger study.

## Continual stream

- Task A is learned GSM8K behavior from the released checkpoint.
- Task B is a fixed, prompt-disjoint 120/40 Dolly summarization split.
- Task C is a fixed, prompt-disjoint 120/40 Dolly creative-writing split.
- Dolly is an adaptation stressor, not reported as a benchmark. Held-out loss
  and answer-token accuracy measure plasticity, so retaining GSM8K by refusing
  to learn later tasks cannot count as success.

Expected local input hashes are:

- GSM8K unaugmented train: `52ebf7c73927f7434abbb2f7b705a82fb3dbdd4695438b7654de78b701c23b36`
- GSM8K augmented train: `6f9a20bc1476ca65eee9bc5117c2d0582b1f1733d5148ffc0ff29cad2a9e9c6b`
- GSM8K test: `8530a3775b96385370842171f226d83de7c5b27d779be54ef0411d255a939818`
- Dolly stream: `a3847b527b517a6a778d85c17ad6a597e66c0f27f4137dde25da496092e219a0`

## Paired methods

Every method within a seed starts from the byte-identical learned Task-A state,
Fisher rows, generated replay rows, later-task minibatches, and masking streams.

- `seq`: no replay and no Fisher penalty.
- `gd`: generated-distillation replay, no Fisher penalty.
- `rank1_gd`: mean-gradient rank-1 EWC plus identical replay.
- `diag_gd`: diagonal EWC plus identical replay.
- `rank1`: rank-1 EWC without replay, a mechanism ablation.

Primary contrasts are paired `rank1_gd - gd` and `rank1_gd - diag_gd`.
Sequential training alone cannot identify the Fisher increment.

Generated replay is audited before adaptation for nonempty completions,
`Answer:` prefixes, and `####` delimiter rate. The frozen low-cost generator
(32 diffusion steps, 128 new-token slots) is not filtered or regenerated after
endpoint inspection. If its delimiter rate is below 90%, the primary matrix
still identifies the Fisher increment over that byte-identical replay stream,
but it is explicitly a low-cost-replay comparison rather than evidence against
a strong replay baseline. In that case, one seed is repeated with released
two-pass decoding for replay as a prespecified replay-quality sensitivity.

## Fisher and stiffness control

The Fisher uses the native answer-only masked-diffusion objective and 40 held-out
source problems represented as 80 released-recipe conditionals. Independent
Bernoulli masks use `t ~ Uniform(1e-3, 1)`. Mean and diagonal moments are
accumulated in one streaming pass; a second identical-mask pass estimates the
Fisher coefficient along the stored mean direction.

Rank-1 and diagonal penalties have equal implemented total trace. If the stored
float32 direction is `u`, its coefficient is `alpha`, diagonal trace is `tau`,
trainable parameter count is `P`, and selected per-parameter target is `c`, then

`lambda_R = c P / (alpha ||u||^2)` and `lambda_D = c P / tau`.

`||u||^2` and `tau` are reduced from stored tensors in fixed chunks using
float64. Before inspecting any GSM8K retention endpoint, development seed 3407
runs 100 updates per later task for `gd` and for both EWC variants on the fixed
grid `c in {1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}`. Select the largest `c` for which both EWC
variants have finite metrics and final mean later-task held-out loss no more
than 10% above `gd`; if no value qualifies, the confirmatory EWC comparison is
ineligible. The selected `c` is then frozen across formal seeds. Realized
traces, penalty/task-loss ratio, gradients, and clipping are reported. GSM8K
outputs produced during calibration are mechanically required by the runner
but are not inspected or used for selection.

## Decoding, gate, and endpoints

The evaluator mirrors released SMDM GSM8K inference: `Question: ...` prompts,
two consecutive confidence-transfer diffusion passes, total context length 256
and 256 steps per pass, CFG 0.1, and temperature 0.1. Special tokens are removed
and the first pass is retokenized before the second. Batched rows retain their
own prompt boundary instead of inheriting the longest padded prompt. Rare
retokenization growth beyond 256 tokens is right-truncated and counted in the
envelope. Primary exact match requires a final number after `####`;
delimiter-free numeric agreement and delimiter rate are diagnostics only.

No adaptation cell is eligible unless the shared start checkpoint exceeds the
predeclared 1% exact-match mechanical floor. Released-checkpoint accuracy is
also compared with the upstream evaluator before freezing the protocol.

Primary endpoint is the exact-match change from the learned Task-A state to
after Task C. Secondary endpoints are exact match after Task B, retained fraction
conditional on initially correct examples, delimiter rate, and paired seed
differences. Plasticity endpoints are Task-B/Task-C held-out loss and answer-token
accuracy.

The first eligible confirmatory matrix is one released checkpoint by four
primary methods (`seq`, `gd`, `rank1_gd`, `diag_gd`) by three optimization seeds:
12 cells. The `rank1`-only ablation follows if the mechanical subset succeeds.
A four-scale 48-cell matrix is run only if all four scales independently pass
the same Task-A gate under a genuinely matched learning protocol.

## Audit and stopping

Private envelopes record source, checkpoint, tokenizer, data and dependency
hashes; selected source IDs; parameter counts; exact cache identity; decoding;
software; and successful or failed cells. Outputs are atomic and never
overwritten. Stop only for failed invariants, OOM, timeout, non-finite values,
changed provenance, or a failed learnability gate. Do not alter the matrix based
on endpoint direction.

## Completion record

The released checkpoint obtained 784/1,319 (59.44%) strict exact match, close
to the published 58.5%. Across three optimization seeds, final full-test exact
match was 22.01±1.59% for sequential adaptation, 36.67±15.51% for low-cost GD,
52.31±2.35% for Rank-1+GD, and 49.23±3.28% for Diagonal+GD (mean±sample SD).
Rank-1+GD minus Diagonal+GD was +3.08±5.04 percentage points with two wins and
one loss, so the study does not establish a stable rank-1-specific advantage.

The low-cost replay audit triggered the prespecified released-decoder
sensitivity: its three caches had only 0–1.56% strict accuracy despite
79.69–85.94% fallback-numeric accuracy, whereas released two-pass replay had
82.81% strict accuracy and 100% delimiter rate. On that 128-question
sensitivity, Rank-1+GD and Diagonal+GD both obtained 54.69%, versus 53.12% for
GD. Exact aggregates, per-seed rows, paired discordance counts, provenance
hashes, and audit results are in `runs/data/gsm8k_rank1_behavior_results.json`.
