#!/usr/bin/env python3
"""Build a sanitized summary of the completed SMDM GSM8K behavior study."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SEEDS = (3407, 3408, 3409)
PRIMARY_METHODS = ("seq", "gd", "rank1_gd", "diag_gd")
METRICS = (
    "final_exact_match",
    "final_fallback_numeric_accuracy",
    "final_delimiter_rate",
    "conditional_retained_fraction",
    "later_average_loss",
    "later_average_answer_token_accuracy",
    "training_clip_fraction_mean",
    "training_ewc_loss_mean",
)


def _load(path: Path) -> dict:
    result = json.loads(path.read_text())
    if result.get("status") != "ok":
        raise ValueError(f"incomplete result: {path}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def _wilson(correct: int, count: int, z: float = 1.959963984540054) -> list[float]:
    proportion = correct / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = z * ((proportion * (1 - proportion) / count + z * z / (4 * count * count)) ** 0.5) / denominator
    return [center - radius, center + radius]


def _adapt_metrics(result: dict) -> dict:
    final = result["stages"][-1]["benchmark"]
    middle = result["stages"][1].get("benchmark")
    training = [stage["training"] for stage in result["stages"][1:]]
    return {
        "seed": result["seed"],
        "method": result["method"],
        "task_a_exact_match": result["stages"][0]["benchmark"]["exact_match"],
        "after_task_b_exact_match": middle["exact_match"] if middle else None,
        "final_correct": final["correct"],
        "final_count": final["count"],
        "final_exact_match": final["exact_match"],
        "final_wilson_95": _wilson(final["correct"], final["count"]),
        "final_fallback_numeric_accuracy": final["fallback_numeric_accuracy"],
        "final_delimiter_rate": final["delimiter_rate"],
        "conditional_retained_fraction": result["summary"][
            "gsm8k_retained_fraction_conditional_on_task_a_correct"
        ],
        "later_average_loss": result["summary"]["final_later_average_loss"],
        "later_average_answer_token_accuracy": result["summary"][
            "final_later_average_answer_token_accuracy"
        ],
        "training_clip_fraction_mean": statistics.fmean(
            stage["clip_fraction"] for stage in training
        ),
        "training_ewc_loss_mean": statistics.fmean(
            stage["ewc_loss_mean"] for stage in training
        ),
    }


def _records(result: dict) -> dict[str, dict]:
    return {
        row["example_id"]: row
        for row in result["stages"][-1]["benchmark"]["records"]
    }


def _discordance(left: dict, right: dict) -> dict:
    left_rows, right_rows = _records(left), _records(right)
    if left_rows.keys() != right_rows.keys():
        raise ValueError("paired benchmark rows differ")
    pairs = [(left_rows[key]["correct"], right_rows[key]["correct"]) for key in left_rows]
    return {
        "both_correct": sum(a and b for a, b in pairs),
        "left_only": sum(a and not b for a, b in pairs),
        "right_only": sum(not a and b for a, b in pairs),
        "neither_correct": sum(not a and not b for a, b in pairs),
    }


def _contrast(results: dict[tuple[int, str], dict], left: str, right: str) -> dict:
    differences = [
        100 * (
            results[(seed, left)]["summary"]["gsm8k_exact_match_final"]
            - results[(seed, right)]["summary"]["gsm8k_exact_match_final"]
        )
        for seed in SEEDS
    ]
    return {
        "left": left,
        "right": right,
        "unit": "percentage_points",
        "per_seed": dict(zip(map(str, SEEDS), differences)),
        **_mean_sd(differences),
        "left_wins_ties_losses": [
            sum(value > 0 for value in differences),
            sum(value == 0 for value in differences),
            sum(value < 0 for value in differences),
        ],
        "per_seed_discordance": {
            str(seed): _discordance(results[(seed, left)], results[(seed, right)])
            for seed in SEEDS
        },
    }


def _replay_quality(replay_path: Path, gsm_train: Path, tokenizer_path: Path) -> dict:
    # Import only for the real audit so --self-check stays dependency-light.
    from types import SimpleNamespace
    from experiments import smdm_gsm8k_rank1_benchmark as benchmark

    replay = json.loads(replay_path.read_text())
    tokenizer = benchmark._load_tokenizer(SimpleNamespace(tokenizer=tokenizer_path))
    wanted = set(replay["manifest"]["source_indices"])
    sources = {
        row["source_index"]: row
        for row in benchmark._read_gsm_sources(gsm_train)
        if row["source_index"] in wanted
    }
    strict = fallback = delimiters = 0
    lengths = []
    for source_index, row in zip(replay["manifest"]["source_indices"], replay["rows"]):
        final = benchmark._gsm_source_rows(sources[source_index])[1]
        target, target_marked = benchmark._extract_answer(final["answer"])
        if not target_marked:
            raise ValueError(f"missing target delimiter for source {source_index}")
        completion = tokenizer.decode(
            row["ids"][row["answer_start"] : row["answer_end"]],
            skip_special_tokens=True,
        ).strip()
        prediction, marked = benchmark._extract_answer(completion)
        delimiters += marked
        fallback += prediction == target
        strict += marked and prediction == target
        lengths.append(row["answer_end"] - row["answer_start"])
    count = len(replay["rows"])
    quality = {
        "count": count,
        "strict_exact_match": strict / count,
        "fallback_numeric_accuracy": fallback / count,
        "delimiter_rate": delimiters / count,
        "completion_tokens_mean": statistics.fmean(lengths),
    }
    embedded = replay["manifest"].get("quality")
    if embedded:
        for key in ("count", "strict_exact_match", "fallback_numeric_accuracy", "delimiter_rate"):
            if embedded[key] != quality[key]:
                raise ValueError(f"embedded replay audit differs for {key}")
    return quality


def _scale_screen(root: Path) -> list[dict]:
    prefixes = {
        170: "cache_m170_corrected_5000_s3407_v2",
        336: "cache_m336_corrected_5000_s3407_v2",
        472: "cache_m472_corrected_5000_s3407_v2",
        1028: "cache_m1028_corrected_5000_s3407_v2",
    }
    rows = []
    for config, prefix in prefixes.items():
        result = _load(root / "pilot" / f"{prefix}.prepare.json")
        benchmark = result["benchmark"]
        rows.append({
            "config": config,
            "parameter_count": result["parameter_count"],
            "updates": result["metadata"]["a_steps"],
            "questions": benchmark["count"],
            "strict_exact_match": benchmark["exact_match"],
            "fallback_numeric_accuracy": benchmark["fallback_numeric_accuracy"],
            "delimiter_rate": benchmark["delimiter_rate"],
            "passes_gate": benchmark["exact_match"] > 0.01,
        })
    return rows


def _calibration(root: Path) -> dict:
    baseline = _load(root / "calibration" / "s3407_gd_step100.json")
    gd_loss = baseline["summary"]["final_later_average_loss"]
    rows = []
    for rank_path in sorted((root / "calibration").glob("s3407_rank1_gd_c*_step100.json")):
        rank = _load(rank_path)
        c = rank["target_trace_per_parameter"]
        diag_path = root / "calibration" / rank_path.name.replace("rank1_gd", "diag_gd")
        diag = _load(diag_path)
        rank_loss = rank["summary"]["final_later_average_loss"]
        diag_loss = diag["summary"]["final_later_average_loss"]
        rows.append({
            "target_trace_per_parameter": c,
            "rank1_later_average_loss": rank_loss,
            "diagonal_later_average_loss": diag_loss,
            "eligible": max(rank_loss, diag_loss) <= 1.1 * gd_loss,
        })
    rows.sort(key=lambda row: row["target_trace_per_parameter"])
    selected = max(row["target_trace_per_parameter"] for row in rows if row["eligible"])
    return {
        "selection_rule": "largest c with both EWC losses <= 1.10 * GD loss",
        "gd_later_average_loss": gd_loss,
        "cells": rows,
        "selected_target_trace_per_parameter": selected,
    }


def _build(args) -> dict:
    root = args.run_root
    input_paths = []
    primary = {}
    for seed in SEEDS:
        for method in PRIMARY_METHODS:
            path = root / "formal" / f"s{seed}_{method}.json"
            input_paths.append(path)
            primary[(seed, method)] = _load(path)
    primary_rows = [_adapt_metrics(primary[(seed, method)]) for seed in SEEDS for method in PRIMARY_METHODS]
    aggregates = {
        method: {
            metric: _mean_sd([row[metric] for row in primary_rows if row["method"] == method])
            for metric in METRICS
        }
        for method in PRIMARY_METHODS
    }

    full_endpoint = {}
    for seed in SEEDS:
        for method in PRIMARY_METHODS:
            path = root / "full_endpoint" / f"s{seed}_{method}.json"
            input_paths.append(path)
            full_endpoint[(seed, method)] = _load(path)
            if full_endpoint[(seed, method)]["stages"][-1]["benchmark"]["count"] != 1319:
                raise ValueError(f"full endpoint is not the complete GSM8K test set: {path}")
    full_endpoint_rows = [
        _adapt_metrics(full_endpoint[(seed, method)])
        for seed in SEEDS for method in PRIMARY_METHODS
    ]
    full_endpoint_aggregates = {
        method: {
            metric: _mean_sd([
                row[metric] for row in full_endpoint_rows if row["method"] == method
            ])
            for metric in METRICS
        }
        for method in PRIMARY_METHODS
    }

    hq = {}
    for method in ("gd", "rank1_gd", "diag_gd"):
        path = root / "sensitivity" / f"s3407_{method}_hqreplay.json"
        input_paths.append(path)
        hq[method] = _load(path)
    hq_rows = [_adapt_metrics(hq[method]) for method in hq]
    hq_contrasts = []
    for left, right in (("rank1_gd", "gd"), ("rank1_gd", "diag_gd")):
        difference = 100 * (
            hq[left]["summary"]["gsm8k_exact_match_final"]
            - hq[right]["summary"]["gsm8k_exact_match_final"]
        )
        hq_contrasts.append({
            "left": left,
            "right": right,
            "strict_difference_percentage_points": difference,
            "later_loss_difference": (
                hq[left]["summary"]["final_later_average_loss"]
                - hq[right]["summary"]["final_later_average_loss"]
            ),
            "discordance": _discordance(hq[left], hq[right]),
        })

    ablation = []
    ablation_results = {}
    for seed in SEEDS:
        path = root / "ablation" / f"s{seed}_rank1.json"
        input_paths.append(path)
        ablation_results[(seed, "rank1")] = _load(path)
        ablation.append(_adapt_metrics(ablation_results[(seed, "rank1")]))

    full_path = root / "formal" / "official_m1028_eval_full1319x256_v2.json"
    repeat_path = root / "audit" / "s3407_diag_gd_repeat16.json"
    no_adaptation_path = root / "audit" / "s3407_no_adaptation.json"
    input_paths.extend((full_path, repeat_path, no_adaptation_path))
    full = _load(full_path)
    repeat = _load(repeat_path)
    no_adaptation = _load(no_adaptation_path)
    full_baseline_exact_match = full["benchmark"]["exact_match"]
    for row in full_endpoint_rows:
        row["full_test_change_from_released_checkpoint"] = (
            row["final_exact_match"] - full_baseline_exact_match
        )
    for method in PRIMARY_METHODS:
        full_endpoint_aggregates[method]["full_test_change_from_released_checkpoint"] = (
            _mean_sd([
                row["full_test_change_from_released_checkpoint"]
                for row in full_endpoint_rows if row["method"] == method
            ])
        )
    original_records = primary[(3407, "diag_gd")]["stages"][-1]["benchmark"]["records"][:16]
    repeat_records = repeat["stages"][-1]["benchmark"]["records"]
    if [row["example_id"] for row in original_records] != [row["example_id"] for row in repeat_records]:
        raise ValueError("repeatability audit benchmark rows differ")
    repeatability = {
        "questions": len(repeat_records),
        "records_exactly_equal": original_records == repeat_records,
        "output_match_fraction": statistics.fmean(
            left["output"] == right["output"]
            for left, right in zip(original_records, repeat_records)
        ),
        "prediction_match_fraction": statistics.fmean(
            left["prediction"] == right["prediction"]
            for left, right in zip(original_records, repeat_records)
        ),
        "correctness_match_fraction": statistics.fmean(
            left["correct"] == right["correct"]
            for left, right in zip(original_records, repeat_records)
        ),
        "primary_subset_exact_match": statistics.fmean(row["correct"] for row in original_records),
        "repeat_exact_match": statistics.fmean(row["correct"] for row in repeat_records),
        "primary_later_average_loss": primary[(3407, "diag_gd")]["summary"][
            "final_later_average_loss"
        ],
        "repeat_later_average_loss": repeat["summary"]["final_later_average_loss"],
    }

    low_replay_paths = {
        seed: root / "pilot" / f"cache_m1028_official_aug_s{seed}_v2.replay.json"
        for seed in SEEDS
    }
    hq_replay_path = root / "pilot" / "cache_m1028_official_aug_s3407_hqreplay_v1.replay.json"
    input_paths.extend((*low_replay_paths.values(), hq_replay_path))
    scale_paths = [
        root / "pilot" / f"cache_m{config}_corrected_5000_s3407_v2.prepare.json"
        for config in (170, 336, 472, 1028)
    ]
    calibration_paths = sorted((root / "calibration").glob("*.json"))
    input_paths.extend(scale_paths + calibration_paths)
    scale_screen = _scale_screen(root)
    calibration = _calibration(root)
    result = {
        "schema_version": 1,
        "study": "SMDM GSM8K behavioral retention under continual adaptation",
        "claim_boundary": (
            "Full-test, three-seed behavioral endpoints from one released 1.14B GSM8K checkpoint; "
            "the reduced-data four-scale screen is not a matched cross-scale retention comparison."
        ),
        "scale_feasibility_screen": scale_screen,
        "calibration": calibration,
        "released_checkpoint_full_gsm8k": {
            **{
                key: full["benchmark"][key]
                for key in (
                    "count", "correct", "exact_match", "fallback_numeric_accuracy",
                    "delimiter_rate", "second_prompt_truncations",
                )
            },
            "wilson_95": _wilson(full["benchmark"]["correct"], full["benchmark"]["count"]),
            "published_smdm_table_2_exact_match": 0.585,
        },
        "full_test_endpoint": {
            "rows": full_endpoint_rows,
            "aggregates": full_endpoint_aggregates,
            "contrasts": [
                _contrast(full_endpoint, "rank1_gd", "gd"),
                _contrast(full_endpoint, "rank1_gd", "diag_gd"),
            ],
        },
        "primary": {
            "rows": primary_rows,
            "aggregates": aggregates,
            "contrasts": [
                _contrast(primary, "rank1_gd", "gd"),
                _contrast(primary, "rank1_gd", "diag_gd"),
            ],
        },
        "replay_quality": {
            "low_cost_by_seed": {
                str(seed): _replay_quality(path, args.gsm_train, args.tokenizer)
                for seed, path in low_replay_paths.items()
            },
            "released_two_pass_seed_3407": _replay_quality(hq_replay_path, args.gsm_train, args.tokenizer),
            "two_pass_sensitivity_rows": hq_rows,
            "two_pass_sensitivity_contrasts": hq_contrasts,
        },
        "rank1_without_replay": {
            "rows": ablation,
            "aggregate": {
                metric: _mean_sd([row[metric] for row in ablation]) for metric in METRICS
            },
            "contrast_vs_rank1_gd": _contrast(
                {**primary, **ablation_results}, "rank1", "rank1_gd"
            ),
        },
        "plasticity_control": {
            "no_adaptation_later_average_loss": no_adaptation["summary"]["final_later_average_loss"],
            "no_adaptation_later_average_answer_token_accuracy": no_adaptation["summary"][
                "final_later_average_answer_token_accuracy"
            ],
            "primary_loss_reduction_vs_no_adaptation": {
                method: (
                    no_adaptation["summary"]["final_later_average_loss"]
                    - aggregates[method]["later_average_loss"]["mean"]
                )
                for method in PRIMARY_METHODS
            },
            "full_endpoint_loss_reduction_vs_no_adaptation": {
                method: (
                    no_adaptation["summary"]["final_later_average_loss"]
                    - full_endpoint_aggregates[method]["later_average_loss"]["mean"]
                )
                for method in PRIMARY_METHODS
            },
        },
        "audits": {
            "primary_matrix_cells": len(primary_rows),
            "full_endpoint_cells": len(full_endpoint_rows),
            "same_seed_repeatability": repeatability,
            "all_outputs_status_ok": True,
        },
        "input_sha256": {str(path.relative_to(root)): _sha256(path) for path in input_paths},
    }
    return result


def _percent(value: float) -> str:
    return f"{100 * value:.2f}"


def _tex_stat(stat: dict, scale: float = 1.0, digits: int = 2) -> str:
    return f'${stat["mean"] * scale:.{digits}f}\\pm{stat["sample_sd"] * scale:.{digits}f}$'


def _latex(result: dict) -> str:
    labels = {"seq": "Sequential", "gd": "GD", "rank1_gd": "Rank-1 + GD", "diag_gd": "Diagonal + GD"}
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{GSM8K behavioral retention from the released SMDM-1.14B math checkpoint. "
        r"Full-test endpoints are mean$\pm$sample SD over three optimization seeds on all 1,319 "
        r"test questions. Conditional retention is measured on the fixed 128-question audit subset "
        r"that the released checkpoint initially answered correctly. Rank-only and two-pass replay "
        r"rows are 128-question mechanism checks; the latter uses seed 3407. Higher exact match and "
        r"conditional retention are better; lower later-task loss is better. The EWC term is the "
        r"mean weighted penalty observed during the two adaptation stages.}",
        r"\label{tab:gsm8k-behavior}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}",
        r"\toprule",
        r"Method & Replay & EM (\%) & $\Delta$ ckpt. (pp) & Cond. ret. (\%) & Later loss & EWC term \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Full GSM8K endpoint (1,319 questions)}} \\",
    ]
    for method in PRIMARY_METHODS:
        aggregate = result["full_test_endpoint"]["aggregates"][method]
        ewc = "---" if method in ("seq", "gd") else _tex_stat(
            aggregate["training_ewc_loss_mean"], digits=4
        )
        replay = "No" if method == "seq" else "Low-cost"
        lines.append(
            f'{labels[method]} & {replay} & {_tex_stat(aggregate["final_exact_match"], 100)} & '
            f'{_tex_stat(aggregate["full_test_change_from_released_checkpoint"], 100)} & '
            f'{_tex_stat(aggregate["conditional_retained_fraction"], 100)} & '
            f'{_tex_stat(aggregate["later_average_loss"], digits=3)} & {ewc} \\\\'
        )
    rank = result["rank1_without_replay"]["aggregate"]
    lines.extend([
        r"\addlinespace[2pt]",
        r"\multicolumn{7}{l}{\textit{128-question mechanism checks}} \\",
        f'Rank-1 & No & {_tex_stat(rank["final_exact_match"], 100)} & --- & '
        f'{_tex_stat(rank["conditional_retained_fraction"], 100)} & '
        f'{_tex_stat(rank["later_average_loss"], digits=3)} & '
        f'{_tex_stat(rank["training_ewc_loss_mean"], digits=4)} \\\\',
        r"\addlinespace[2pt]",
        r"\multicolumn{7}{l}{\textit{128-question released two-pass replay sensitivity}} \\",
    ])
    for row in result["replay_quality"]["two_pass_sensitivity_rows"]:
        ewc = "---" if row["method"] == "gd" else f'${row["training_ewc_loss_mean"]:.4f}$'
        lines.append(
            f'{labels[row["method"]]} & Two-pass & ${100 * row["final_exact_match"]:.2f}$ & --- & '
            f'${100 * row["conditional_retained_fraction"]:.2f}$ & '
            f'${row["later_average_loss"]:.3f}$ & {ewc} \\\\'
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])
    return "\n".join(lines)


def _markdown(result: dict) -> str:
    full = result["released_checkpoint_full_gsm8k"]
    calibration = result["calibration"]
    endpoint = result["full_test_endpoint"]
    rank_vs_gd = endpoint["contrasts"][0]
    rank_vs_diag = endpoint["contrasts"][1]
    lines = [
        "# SMDM GSM8K behavioral retention results",
        "",
        result["claim_boundary"],
        "",
        "## Released-checkpoint validation",
        "",
        f'The released 1.14B GSM8K checkpoint scores {full["correct"]}/{full["count"]} '
        f'({_percent(full["exact_match"])}%) strict exact match on the full test set. Its Wilson 95% '
        f'interval is {_percent(full["wilson_95"][0])}%–{_percent(full["wilson_95"][1])}%; the '
        f'published SMDM Table 2 value is {_percent(full["published_smdm_table_2_exact_match"])}%.',
        "",
        "## Reduced-data scale feasibility screen",
        "",
        "| Config | Actual parameters | Updates | Questions | Strict | Fallback numeric | Delimiter | Gate |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in result["scale_feasibility_screen"]:
        lines.append(
            f'| {row["config"]}M | {row["parameter_count"]:,} | {row["updates"]} | '
            f'{row["questions"]} | {_percent(row["strict_exact_match"])} | '
            f'{_percent(row["fallback_numeric_accuracy"])} | {_percent(row["delimiter_rate"])} | '
            f'{"pass" if row["passes_gate"] else "fail"} |'
        )
    lines.extend([
        "",
        "This screen used only 5,000 updates on the unaugmented training set; it is not matched to the "
        "released 40-epoch augmented recipe.",
        "",
        "## Stiffness calibration",
        "",
        f'The held-out later-task-loss rule selected `c={calibration["selected_target_trace_per_parameter"]:g}`; '
        f'the 100-step GD reference loss was {calibration["gd_later_average_loss"]:.4f}.',
        "",
        "## Primary full-test endpoint",
        "",
        "All endpoint accuracies below use the complete 1,319-question GSM8K test set. Conditional "
        "retention remains a paired diagnostic on the fixed 128-question audit subset.",
        "",
        "| Seed | Method | Final strict accuracy | Change from released checkpoint | Conditional retained | Later loss |",
        "|---:|:---|---:|---:|---:|---:|",
    ])
    for row in endpoint["rows"]:
        lines.append(
            f'| {row["seed"]} | {row["method"]} | {_percent(row["final_exact_match"])} | '
            f'{100 * row["full_test_change_from_released_checkpoint"]:+.2f} pp | '
            f'{_percent(row["conditional_retained_fraction"])} | {row["later_average_loss"]:.4f} |'
        )
    lines.extend([
        "",
        "| Method | Final strict accuracy | Change from checkpoint | Conditional retained | Later loss | Mean EWC term |",
        "|:---|---:|---:|---:|---:|---:|",
    ])
    for method in PRIMARY_METHODS:
        aggregate = endpoint["aggregates"][method]
        lines.append(
            f'| {method} | {_percent(aggregate["final_exact_match"]["mean"])} '
            f'± {_percent(aggregate["final_exact_match"]["sample_sd"])} | '
            f'{100 * aggregate["full_test_change_from_released_checkpoint"]["mean"]:+.2f} '
            f'± {100 * aggregate["full_test_change_from_released_checkpoint"]["sample_sd"]:.2f} pp | '
            f'{_percent(aggregate["conditional_retained_fraction"]["mean"])} '
            f'± {_percent(aggregate["conditional_retained_fraction"]["sample_sd"])} | '
            f'{aggregate["later_average_loss"]["mean"]:.4f} ± '
            f'{aggregate["later_average_loss"]["sample_sd"]:.4f} | '
            f'{aggregate["training_ewc_loss_mean"]["mean"]:.4f} |'
        )
    lines.extend([
        "",
        f'Rank-1 + GD minus GD was {rank_vs_gd["mean"]:+.2f}±{rank_vs_gd["sample_sd"]:.2f} '
        f'percentage points across seeds (wins/ties/losses: '
        f'{rank_vs_gd["left_wins_ties_losses"]}); Rank-1 + GD minus diagonal + GD was '
        f'{rank_vs_diag["mean"]:+.2f}±{rank_vs_diag["sample_sd"]:.2f} percentage points '
        f'(wins/ties/losses: {rank_vs_diag["left_wins_ties_losses"]}).',
        "",
        "## Secondary 128-question trajectory audit",
        "",
        "| Seed | Method | Task A | After B | After C | Conditional retained | Later loss |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ])
    for row in result["primary"]["rows"]:
        lines.append(
            f'| {row["seed"]} | {row["method"]} | {_percent(row["task_a_exact_match"])} | '
            f'{_percent(row["after_task_b_exact_match"])} | {_percent(row["final_exact_match"])} | '
            f'{_percent(row["conditional_retained_fraction"])} | {row["later_average_loss"]:.4f} |'
        )
    lines.extend(["", "Values except later loss are percentages.", "", "## 128-question aggregate", "",
                  "| Method | Final strict accuracy | Conditional retained | Later loss | Mean EWC term | Clip rate |",
                  "|:---|---:|---:|---:|---:|---:|"])
    for method in PRIMARY_METHODS:
        aggregate = result["primary"]["aggregates"][method]
        lines.append(
            f'| {method} | {_percent(aggregate["final_exact_match"]["mean"])} '
            f'± {_percent(aggregate["final_exact_match"]["sample_sd"])} | '
            f'{_percent(aggregate["conditional_retained_fraction"]["mean"])} '
            f'± {_percent(aggregate["conditional_retained_fraction"]["sample_sd"])} | '
            f'{aggregate["later_average_loss"]["mean"]:.4f} ± '
            f'{aggregate["later_average_loss"]["sample_sd"]:.4f} | '
            f'{aggregate["training_ewc_loss_mean"]["mean"]:.4f} | '
            f'{_percent(aggregate["training_clip_fraction_mean"]["mean"])} |'
        )
    lines.extend(["", "## High-quality replay sensitivity (seed 3407)", "",
                  "| Method | Final strict accuracy | Conditional retained | Later loss |",
                  "|:---|---:|---:|---:|"])
    for row in result["replay_quality"]["two_pass_sensitivity_rows"]:
        lines.append(
            f'| {row["method"]} | {_percent(row["final_exact_match"])} | '
            f'{_percent(row["conditional_retained_fraction"])} | {row["later_average_loss"]:.4f} |'
        )
    lines.extend(["", "## Rank-1 without replay", "",
                  "| Seed | After B | After C | Conditional retained | Later loss |",
                  "|---:|---:|---:|---:|---:|"])
    for row in result["rank1_without_replay"]["rows"]:
        lines.append(
            f'| {row["seed"]} | {_percent(row["after_task_b_exact_match"])} | '
            f'{_percent(row["final_exact_match"])} | {_percent(row["conditional_retained_fraction"])} | '
            f'{row["later_average_loss"]:.4f} |'
        )
    low = list(result["replay_quality"]["low_cost_by_seed"].values())
    high = result["replay_quality"]["released_two_pass_seed_3407"]
    repeat = result["audits"]["same_seed_repeatability"]
    plasticity = result["plasticity_control"]
    rank_only_contrast = result["rank1_without_replay"]["contrast_vs_rank1_gd"]
    rank_aggregate = endpoint["aggregates"]["rank1_gd"]
    diagonal_aggregate = endpoint["aggregates"]["diag_gd"]
    gd_aggregate = endpoint["aggregates"]["gd"]
    rank_gd_fallback_delta = 100 * (
        rank_aggregate["final_fallback_numeric_accuracy"]["mean"]
        - gd_aggregate["final_fallback_numeric_accuracy"]["mean"]
    )
    rank_gd_delimiter_delta = 100 * (
        rank_aggregate["final_delimiter_rate"]["mean"]
        - gd_aggregate["final_delimiter_rate"]["mean"]
    )
    lines.extend([
        "",
        f'The three low-cost replay caches had {_percent(min(row["strict_exact_match"] for row in low))}%–'
        f'{_percent(max(row["strict_exact_match"] for row in low))}% strict accuracy, '
        f'{_percent(min(row["fallback_numeric_accuracy"] for row in low))}%–'
        f'{_percent(max(row["fallback_numeric_accuracy"] for row in low))}% fallback numeric accuracy, and '
        f'{_percent(min(row["delimiter_rate"] for row in low))}%–'
        f'{_percent(max(row["delimiter_rate"] for row in low))}% delimiter rate; released two-pass replay had '
        f'{_percent(high["strict_exact_match"])}% and {_percent(high["delimiter_rate"])}%, respectively.',
        "",
        "## Interpretation",
        "",
        "The full-test comparison is the behavioral endpoint; the 128-question trajectories are mechanism "
        "diagnostics rather than the headline estimate. A rank-1-specific claim requires a stable advantage "
        "over trace-matched diagonal EWC, not only an advantage over low-quality generated replay.",
        "",
        f'Rank-1 + GD improved strict exact match over low-cost GD by {rank_vs_gd["mean"]:.2f} points, '
        f'but its fallback-numeric margin was only {rank_gd_fallback_delta:.2f} points while its delimiter-rate '
        f'margin was {rank_gd_delimiter_delta:.2f} points. Much of the strict-score gap therefore tracks '
        "preservation of the required output marker, especially in the unstable GD seed.",
        "",
        f'Against trace-matched diagonal + GD, the Rank-1 + GD difference was '
        f'{rank_vs_diag["mean"]:+.2f}±{rank_vs_diag["sample_sd"]:.2f} points with per-seed signs '
        f'{rank_vs_diag["left_wins_ties_losses"]}. This three-seed result does not establish a stable '
        "rank-1-specific advantage.",
        "",
        f'Rank-1 without replay differed from Rank-1 + GD by '
        f'{rank_only_contrast["mean"]:.2f}±{rank_only_contrast["sample_sd"]:.2f} percentage points; '
        "the per-seed signs were mixed, so replay supplied no stable increment under the strong rank-1 penalty.",
        "",
        f'Equal implemented trace did not equal equal realized constraint: the mean weighted EWC term was '
        f'{rank_aggregate["training_ewc_loss_mean"]["mean"]:.4f} for rank-1 and '
        f'{diagonal_aggregate["training_ewc_loss_mean"]["mean"]:.4f} for diagonal. This accompanies '
        "higher later-task loss for rank-1 and limits utility claims from trace matching alone.",
        "",
        f'The unadapted checkpoint had later-task loss '
        f'{plasticity["no_adaptation_later_average_loss"]:.4f}; all full-endpoint methods reduced it by '
        f'{min(plasticity["full_endpoint_loss_reduction_vs_no_adaptation"].values()):.4f}–'
        f'{max(plasticity["full_endpoint_loss_reduction_vs_no_adaptation"].values()):.4f}. Retention therefore '
        "did not arise from completely refusing to learn the later tasks.",
        "",
        f'The same-seed GPU repeat was not bit-exact: outputs matched on '
        f'{_percent(repeat["output_match_fraction"])}% of 16 audited questions and correctness '
        f'matched on {_percent(repeat["correctness_match_fraction"])}%. Small percentage-point '
        "differences should therefore not be interpreted as stable method effects.",
        "",
        "The four reduced-data scale pilots all failed the predeclared Task-A gate. They therefore do not "
        "identify cross-scale retention, and they are not evidence that the smaller checkpoints cannot learn GSM8K.",
        "",
    ])
    return "\n".join(lines)


def _self_check() -> None:
    assert _mean_sd([1.0, 2.0, 3.0]) == {"mean": 2.0, "sample_sd": 1.0}
    assert [round(value, 6) for value in _wilson(1, 2)] == [0.094531, 0.905469]
    left = {"stages": [{}, {}, {"benchmark": {"records": [
        {"example_id": "a", "correct": True}, {"example_id": "b", "correct": False}
    ]}}]}
    right = {"stages": [{}, {}, {"benchmark": {"records": [
        {"example_id": "a", "correct": False}, {"example_id": "b", "correct": False}
    ]}}]}
    assert _discordance(left, right) == {
        "both_correct": 0, "left_only": 1, "right_only": 0, "neither_correct": 1
    }
    print("self-check ok")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/gsm8k_rank1_scale"))
    parser.add_argument("--gsm-train", type=Path, default=Path("SMDM/data/gsm8k/train_augmented.txt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer"))
    parser.add_argument("--output", type=Path, default=Path("runs/data/gsm8k_rank1_behavior_results.json"))
    parser.add_argument("--report", type=Path, default=Path("report/gsm8k_rank1_behavior_results.md"))
    parser.add_argument("--tex-table", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
        return
    result = _build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.report.write_text(_markdown(result))
    if args.tex_table:
        args.tex_table.parent.mkdir(parents=True, exist_ok=True)
        args.tex_table.write_text(_latex(result))
    print(json.dumps({
        "status": "ok", "output": str(args.output), "report": str(args.report),
        "tex_table": str(args.tex_table) if args.tex_table else None,
    }, indent=2))


if __name__ == "__main__":
    main()
