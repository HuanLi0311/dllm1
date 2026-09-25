#!/usr/bin/env python3
"""Validate and summarize the experiments requested by NOTICE.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED = {"219m": 5, "1140m": 3, "llada": 3, "update": 3}
PREFIXES = (16, 32, 64, 128, 256)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_std(values):
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _stability_threshold(rows):
    final = rows[-1]["winner"]
    return next(
        row["test_sample_count"]
        for index, row in enumerate(rows)
        if all(later["winner"] == final for later in rows[index:])
    )


def _load(path: Path):
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok":
        raise ValueError(f"non-ok result: {path}")
    return payload


def _convergence(directory: Path, expected: int):
    files = sorted(directory.glob("*.json"))
    if len(files) != expected:
        raise ValueError(f"{directory}: expected {expected} JSONs, found {len(files)}")
    payloads = [_load(path) for path in files]
    designs = [payload["convergence_design"] for payload in payloads]
    calibration_records = {design["calibration_records_sha256"] for design in designs}
    calibration_masks = {
        tuple(row["calibration_mask_sha256"] for row in design["mask_audit"])
        for design in designs
    }
    test_records = {design["test_records_sha256"] for design in designs}
    test_masks = {
        tuple(row["test_mask_sha256"] for row in design["mask_audit"])
        for design in designs
    }
    if len(calibration_records) != 1 or len(calibration_masks) != 1:
        raise ValueError(f"{directory}: calibration surrogate changed across test seeds")
    if len(test_records) != expected or len(test_masks) != expected:
        raise ValueError(f"{directory}: test order/masks do not differ across seeds")

    by_prefix, by_cell = defaultdict(list), defaultdict(list)
    thresholds = []
    reference_cells = None
    for payload in payloads:
        cells = {(row["parameter"], row["mask_condition"]) for row in payload["results"]}
        if reference_cells is None:
            reference_cells = cells
        elif cells != reference_cells:
            raise ValueError(f"{directory}: result grid differs across seeds")
        for result in payload["results"]:
            prefix_rows = result["test_prefix_results"]
            if tuple(row["test_sample_count"] for row in prefix_rows) != PREFIXES:
                raise ValueError(f"{directory}: incomplete nested test prefixes")
            thresholds.append(_stability_threshold(prefix_rows))
            cell = (result["parameter"], result["mask_condition"])
            for row in prefix_rows:
                by_prefix[row["test_sample_count"]].append(row)
                by_cell[cell, row["test_sample_count"]].append(row)

    aggregate = []
    for count in PREFIXES:
        rows = by_prefix[count]
        rank1 = [row["rank1_test_relative_frobenius_error"] for row in rows]
        diagonal = [row["diagonal_test_relative_frobenius_error"] for row in rows]
        margins = [row["paired_margin_diagonal_minus_rank1"] for row in rows]
        aggregate.append({
            "test_sample_count": count,
            "rank1": _mean_std(rank1),
            "diagonal": _mean_std(diagonal),
            "paired_margin_diagonal_minus_rank1": _mean_std(margins),
            "rank1_win_fraction": sum(value > 0 for value in margins) / len(margins),
            "cell_seed_count": len(rows),
            "aggregate_winner": "rank1" if statistics.fmean(margins) > 0 else "diagonal",
        })
    final_cells = []
    for cell in sorted(reference_cells):
        rows = by_cell[cell, PREFIXES[-1]]
        margins = [row["paired_margin_diagonal_minus_rank1"] for row in rows]
        final_cells.append({
            "parameter": cell[0],
            "mask_condition": cell[1],
            "paired_margin": _mean_std(margins),
            "rank1_wins": sum(value > 0 for value in margins),
            "diagonal_wins": sum(value < 0 for value in margins),
        })
    aggregate_threshold = next(
        row["test_sample_count"]
        for index, row in enumerate(aggregate)
        if all(later["aggregate_winner"] == aggregate[-1]["aggregate_winner"] for later in aggregate[index:])
    )
    return {
        "files": {str(path): _sha256(path) for path in files},
        "test_seed_count": expected,
        "cell_count": len(reference_cells),
        "fixed_calibration_verified": True,
        "distinct_test_order_and_masks_verified": True,
        "aggregate": aggregate,
        "aggregate_ranking_stable_from_nt": aggregate_threshold,
        "cell_seed_stability_threshold_counts": dict(sorted(Counter(thresholds).items())),
        "final_cells": final_cells,
    }


def _updates(directory: Path, expected: int):
    files = sorted(directory.glob("*.json"))
    if len(files) != expected:
        raise ValueError(f"{directory}: expected {expected} JSONs, found {len(files)}")
    rows = []
    for path in files:
        payload = _load(path)
        if payload["common_update"]["parameter_scope"] != "all trainable parameters":
            raise ValueError(f"{path}: fidelity/update parameter scopes do not match")
        rank1 = payload["update_weighted_fidelity"]["rank1"]
        diagonal = payload["update_weighted_fidelity"]["diagonal"]
        rows.append({
            "seed": payload["seed"],
            "delta_f_test_delta": payload["heldout_fisher"]["delta_f_test_delta"],
            "rank1_relative_update_error": rank1["relative_update_error"],
            "diagonal_relative_update_error": diagonal["relative_update_error"],
            "rank1_estimate_over_true": rank1["estimate_over_true"],
            "diagonal_estimate_over_true": diagonal["estimate_over_true"],
            "closer_surrogate": (
                "rank1" if rank1["relative_update_error"] < diagonal["relative_update_error"] else "diagonal"
            ),
            "gsm8k_exact_match_when_learned": payload["summary"]["gsm8k_exact_match_when_learned"],
            "gsm8k_exact_match_final": payload["summary"]["gsm8k_exact_match_final"],
            "gsm8k_retention_change": payload["summary"]["gsm8k_retention_change"],
            "matched_heldout_loss_increase": payload["matched_heldout_loss"]["increase"],
        })
    keys = (
        "rank1_relative_update_error",
        "diagonal_relative_update_error",
        "rank1_estimate_over_true",
        "diagonal_estimate_over_true",
        "gsm8k_exact_match_final",
        "gsm8k_retention_change",
        "matched_heldout_loss_increase",
    )
    return {
        "files": {str(path): _sha256(path) for path in files},
        "rows": rows,
        "aggregate": {key: _mean_std([row[key] for row in rows]) for key in keys},
        "closer_surrogate_counts": dict(Counter(row["closer_surrogate"] for row in rows)),
        "scope_match_verified": True,
        "correlation_not_estimated": "three seeds are insufficient and no prespecified fidelity intervention was varied",
    }


def _markdown(summary):
    lines = ["# NOTICE validation results", "", "## Test-gradient convergence", ""]
    lines += [
        "| Model | n_t | Rank-1 error | Diagonal error | Margin (diag-rank1) | Rank-1 win rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, result in summary["test_gradient_convergence"].items():
        for row in result["aggregate"]:
            lines.append(
                f"| {model} | {row['test_sample_count']} | {row['rank1']['mean']:.4f} ± {row['rank1']['std']:.4f} "
                f"| {row['diagonal']['mean']:.4f} ± {row['diagonal']['std']:.4f} "
                f"| {row['paired_margin_diagonal_minus_rank1']['mean']:+.4f} ± {row['paired_margin_diagonal_minus_rank1']['std']:.4f} "
                f"| {row['rank1_win_fraction']:.1%} |"
            )
        lines.append("")
        lines.append(
            f"{model}: fixed calibration and independent test/mask seeds verified; aggregate ordering is stable from "
            f"n_t={result['aggregate_ranking_stable_from_nt']}."
        )
        lines.append("")
    lines += [
        "## Common-update fidelity and forgetting", "",
        "| Seed | r_delta rank-1 | r_delta diagonal | Q_rank1/F | Q_diag/F | Final EM | Retention change | Held-out loss increase |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    updates = summary["common_update"]
    for row in updates["rows"]:
        lines.append(
            f"| {row['seed']} | {row['rank1_relative_update_error']:.4f} | {row['diagonal_relative_update_error']:.4f} "
            f"| {row['rank1_estimate_over_true']:.4f} | {row['diagonal_estimate_over_true']:.4f} "
            f"| {row['gsm8k_exact_match_final']:.4f} | {row['gsm8k_retention_change']:+.4f} "
            f"| {row['matched_heldout_loss_increase']:+.4f} |"
        )
    lines += [
        "",
        "All update-weighted quantities use the same GD/replay update and the same full trainable parameter set. "
        "No fidelity--forgetting correlation is reported from only three seeds.",
        "",
    ]
    return "\n".join(lines)


def _self_check():
    assert _stability_threshold([
        {"test_sample_count": 16, "winner": "rank1"},
        {"test_sample_count": 32, "winner": "diagonal"},
        {"test_sample_count": 64, "winner": "rank1"},
        {"test_sample_count": 128, "winner": "rank1"},
        {"test_sample_count": 256, "winner": "rank1"},
    ]) == 64
    assert _mean_std([1.0, 3.0]) == {"mean": 2.0, "std": 2**0.5}
    print(json.dumps({"self_check": "ok"}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    if any(value is None for value in (args.root, args.output_json, args.output_markdown)):
        parser.error("--root, --output-json, and --output-markdown are required")
    summary = {
        "test_gradient_convergence": {
            "SMDM-219M": _convergence(args.root / "test_convergence_219m", EXPECTED["219m"]),
            "SMDM-1.14B": _convergence(args.root / "test_convergence_1140m", EXPECTED["1140m"]),
            "LLaDA-8B": _convergence(args.root / "test_convergence_llada", EXPECTED["llada"]),
        },
        "common_update": _updates(args.root / "update_fidelity", EXPECTED["update"]),
        "summary_script_sha256": _sha256(Path(__file__)),
    }
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.output_markdown.write_text(_markdown(summary))
    print(json.dumps({"status": "ok", "output": str(args.output_json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
