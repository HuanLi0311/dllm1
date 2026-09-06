#!/usr/bin/env python3
"""Verify and summarize the reused R12/R16 evidence and new R18 dense runs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
ICLR2 = WORKSPACE / "iclr_2"
SEEDS = (3407, 3408, 3409)
METHODS = {
    "forward": ("seq", "gd", "rank1", "diagonal", "rank1_gd", "diag_gd", "joint"),
    "reverse": ("seq", "gd", "rank1_gd", "diag_gd"),
}
GEOMETRY_MASKS = ("fixed_0.1", "fixed_0.5", "fixed_0.9")
R12_MASKS = ("fixed_0.1", "fixed_0.3", "fixed_0.5", "fixed_0.7", "fixed_0.9")
R12_PARAMETERS = (
    "transformer.h.0.norm_1.weight",
    "transformer.h.8.norm_1.weight",
    "transformer.h.17.norm_1.weight",
)
DENSE_PARAMETERS = (
    "transformer.h.0.attn.proj.weight",
    "transformer.h.19.attn.proj.weight",
)
BASE_PROBE_SHA256 = "a3c08f89ea6dc6190686ae11175cc03c23cd59f97a67a13f2b72c66f5c39a84d"
AUDITED_PROBE_SHA256 = "991113e2856385fd5e85972825b7324177e7c238b2ad44075e77b9577da52cd2"
SMDM_219_SHA256 = "2d8c9b9a730715f2c772d5bc740e12951fc160e5e8511a16835f3537401ea9bb"
SMDM_1028_SHA256 = "ce96ce67a051613b6d7feb419c99c0b4db5bfcfaaa0833ed7f7ecbc6632841d6"
GSM8K_SHA256 = "a191018cd0cbe45b66f6bbb19d5ee1d6f932e56cb272133778d67a91eb0e0b99"
CORPORA_SHA256 = "918f497538950ecfe143a0b000746292379ea1f143d359c3915fd9bb4a4ebeb5"


def _load(path: Path) -> dict:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(left: float, right: float, message: str) -> None:
    if not math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"{message}: {left!r} != {right!r}")


def _mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _mean_sem(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sem": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0,
        "values": values,
    }


def _manifest_index(manifest_path: Path) -> dict[str, str]:
    manifest = _load(manifest_path)
    _require(manifest.get("status") == "ok", "release manifest is not successful")
    index = {}
    for row in manifest["raw_artifacts"] + manifest["sanitized_public_artifacts"]:
        index[row["release_artifact"]] = row["release_sha256"]
    for row in manifest["public_artifacts"]:
        index[row["path"]] = row["sha256"]
    return index


def _verify_release(path: Path, index: dict[str, str]) -> dict:
    relative = path.relative_to(ICLR2).as_posix()
    _require(relative in index, f"release artifact absent from manifest: {relative}")
    actual = _sha256(path)
    _require(actual == index[relative], f"release hash mismatch: {relative}")
    return {"path": relative, "sha256": actual}


def _geometry_summary(paths: list[Path], parameters: tuple[str, ...], masks: tuple[str, ...],
                      calibration_count: int, test_count: int, *, model_size: int,
                      checkpoint_sha256: str, data_sha256: str, probe_sha256: str,
                      task_id: int | None) -> dict:
    expected_cells = {(parameter, mask) for parameter in parameters for mask in masks}
    by_seed = {}
    inputs = []
    for path in paths:
        payload = _load(path)
        _require(payload.get("status") == "ok", f"failed geometry run: {path}")
        _require(payload.get("probe_sha256") == probe_sha256, f"wrong probe hash: {path}")
        if probe_sha256 == AUDITED_PROBE_SHA256:
            _require(payload.get("base_probe_sha256") == BASE_PROBE_SHA256,
                     f"wrong audited base-probe hash: {path}")
        _require(payload.get("checkpoint_sha256") == checkpoint_sha256,
                 f"wrong checkpoint hash: {path}")
        _require(payload.get("data_sha256") == data_sha256, f"wrong data hash: {path}")
        _require(payload["config"]["model_size"] == model_size, f"wrong model size: {path}")
        _require(payload["config"]["task_id"] == task_id, f"wrong task id: {path}")
        _require(payload["config"]["loss_mode"] == "native_conditional", f"wrong loss: {path}")
        _require(payload["config"]["shuffle_records"] is True, f"records were not shuffled: {path}")
        _require(payload["config"]["sequence_length"] == 64, f"wrong sequence length: {path}")
        seed = int(payload["config"]["seed"])
        _require(seed not in by_seed, f"duplicate geometry seed: {seed}")
        cells = {}
        for row in payload["results"]:
            key = (row["parameter"], row["mask_condition"])
            if key not in expected_cells:
                continue
            _require(key not in cells, f"duplicate geometry cell: {seed} {key}")
            _require(row["evaluation"] == "split_sample", f"non-held-out cell: {path} {key}")
            _require(row["calibration_sample_count"] == calibration_count, "wrong calibration count")
            _require(row["test_sample_count"] == test_count, "wrong test count")
            rank1 = float(row["mean_rank1_test_relative_frobenius_error"])
            diagonal = float(row["diagonal_test_relative_frobenius_error"])
            oracle = float(row["test_oracle_top1_relative_frobenius_error"])
            fitted_top = float(row["calibration_top1_test_relative_frobenius_error"])
            _require(all(math.isfinite(value) for value in (rank1, diagonal, oracle, fitted_top)),
                     f"non-finite geometry cell: {path} {key}")
            _require(oracle <= rank1 + 1e-10 and oracle <= fitted_top + 1e-10,
                     f"oracle ordering violation: {path} {key}")
            cells[key] = {"rank1": rank1, "diagonal": diagonal, "delta": diagonal - rank1,
                          "oracle_rank1": oracle}
        _require(set(cells) == expected_cells,
                 f"geometry grid mismatch for seed {seed}: missing={expected_cells - set(cells)}")
        by_seed[seed] = cells
        inputs.append({"path": str(path.relative_to(WORKSPACE)), "sha256": _sha256(path)})
    _require(set(by_seed) == {0, 1, 2}, f"geometry seeds must be 0--2, got {sorted(by_seed)}")

    seed_means = {}
    for seed, cells in sorted(by_seed.items()):
        seed_means[str(seed)] = {
            metric: statistics.fmean(cell[metric] for cell in cells.values())
            for metric in ("rank1", "diagonal", "delta", "oracle_rank1")
        }
    cells = {}
    for key in sorted(expected_cells):
        label = f"{key[0]}|{key[1]}"
        cells[label] = {
            metric: _mean_sd([by_seed[seed][key][metric] for seed in sorted(by_seed)])
            for metric in ("rank1", "diagonal", "delta", "oracle_rank1")
        }
    return {
        "inputs": inputs,
        "seed_means": seed_means,
        "aggregate_over_seed_means": {
            metric: _mean_sd([seed_means[str(seed)][metric] for seed in sorted(by_seed)])
            for metric in ("rank1", "diagonal", "delta", "oracle_rank1")
        },
        "rank1_favored_cell_means": sum(cell["delta"]["mean"] > 0 for cell in cells.values()),
        "cell_count": len(cells),
        "cells": cells,
    }


def _recompute_continual_summary(payload: dict, path: Path) -> None:
    sequence = [row["name"] for row in payload["metadata"]["sequence"]]
    if payload["metadata"]["method"] == "joint":
        final = payload["final_metrics"]
        losses = [float(final[task]["loss"]) for task in sequence]
        expected = {
            "final_average_loss": statistics.fmean(losses),
            "final_average_answer_token_accuracy": statistics.fmean(
                float(final[task]["answer_token_accuracy"]) for task in sequence
            ),
            "final_average_target_containment": statistics.fmean(
                float(final[task]["generation"]["accuracy"]) for task in sequence
            ),
            "final_average_exact_match": statistics.fmean(
                float(final[task]["generation"]["strict_accuracy"]) for task in sequence
            ),
            "final_average_rouge_l_f1": statistics.fmean(
                float(final[task]["generation"]["rouge_l_f1"]) for task in sequence
            ),
        }
    else:
        stages = payload["stages"]
        _require(len(stages) == len(sequence), f"wrong stage count: {path}")
        final = stages[-1]["metrics"]
        losses = [float(final[task]["loss"]) for task in sequence]
        learned = [float(stages[index]["metrics"][task]["loss"]) for index, task in enumerate(sequence)]
        forgetting = [after - before for after, before in zip(losses, learned)]
        expected = {
            "final_average_loss": statistics.fmean(losses),
            "past_task_forgetting": statistics.fmean(forgetting[:-1]),
            "final_average_answer_token_accuracy": statistics.fmean(
                float(final[task]["answer_token_accuracy"]) for task in sequence
            ),
            "final_average_target_containment": statistics.fmean(
                float(final[task]["generation"]["accuracy"]) for task in sequence
            ),
            "final_average_exact_match": statistics.fmean(
                float(final[task]["generation"]["strict_accuracy"]) for task in sequence
            ),
            "final_average_rouge_l_f1": statistics.fmean(
                float(final[task]["generation"]["rouge_l_f1"]) for task in sequence
            ),
        }
        for actual, wanted in zip(payload["summary"]["losses_when_learned"], learned):
            _close(float(actual), wanted, f"learned-loss mismatch: {path}")
        for actual, wanted in zip(payload["summary"]["task_forgetting"], forgetting):
            _close(float(actual), wanted, f"forgetting mismatch: {path}")
    for actual, wanted in zip(payload["summary"]["final_task_losses"], losses):
        _close(float(actual), wanted, f"final-task-loss mismatch: {path}")
    for key, wanted in expected.items():
        _close(float(payload["summary"][key]), wanted, f"summary mismatch {key}: {path}")


def _continual_summary(release_root: Path, manifest_index: dict[str, str],
                       existing_summary: Path) -> dict:
    runs = {}
    inputs = []
    for order, methods in METHODS.items():
        for method in methods:
            for seed in SEEDS:
                path = release_root / "public/runs/r16_native_mask/final" / order / f"s{seed}" / f"{method}.json.gz"
                _require(path.is_file(), f"missing R16 run: {path}")
                inputs.append(_verify_release(path, manifest_index))
                payload = _load(path)
                metadata = payload["metadata"]
                _require(payload.get("status") == "ok", f"failed R16 run: {path}")
                _require(metadata["protocol"] == "r16_native_mask_v1", f"wrong R16 protocol: {path}")
                _require(metadata["method"] == method and metadata["order"] == order and metadata["seed"] == seed,
                         f"R16 path/metadata mismatch: {path}")
                _require(metadata["trainable"] == "all" and metadata["steps_per_task"] == 1000,
                         f"R16 is not full-model/locked length: {path}")
                _require(metadata["mask_sampling"] == "independent_bernoulli_allow_empty_v1",
                         f"invalidated mask estimator: {path}")
                _require(metadata["minibatch_sampling"] == "separate_current_replay_rng_v1",
                         f"shared RNG protocol: {path}")
                _recompute_continual_summary(payload, path)
                runs[(order, method, seed)] = payload

    metrics = (
        "final_average_loss", "past_task_forgetting", "final_average_answer_token_accuracy",
        "final_average_target_containment", "final_average_exact_match", "final_average_rouge_l_f1",
    )
    aggregate = {}
    for order, methods in METHODS.items():
        aggregate[order] = {}
        for method in methods:
            rows = [runs[(order, method, seed)] for seed in SEEDS]
            aggregate[order][method] = {
                metric: _mean_sem([float(row["summary"][metric]) for row in rows])
                for metric in metrics if metric in rows[0]["summary"]
            }
    paired = {}
    for order in METHODS:
        paired[order] = {}
        for left, right in (("rank1_gd", "gd"), ("rank1_gd", "diag_gd"), ("diag_gd", "gd")):
            paired[order][f"{left}_minus_{right}"] = {}
            for metric in ("final_average_loss", "past_task_forgetting"):
                values = [
                    float(runs[(order, left, seed)]["summary"][metric])
                    - float(runs[(order, right, seed)]["summary"][metric])
                    for seed in SEEDS
                ]
                paired[order][f"{left}_minus_{right}"][metric] = {
                    **_mean_sem(values), "wins": sum(value < 0 for value in values)
                }

    _verify_release(existing_summary, manifest_index)
    reference = _load(existing_summary)
    for order, methods in aggregate.items():
        for method, values in methods.items():
            for metric, group in values.items():
                reference_group = reference["aggregate"][order][method][metric]
                _close(group["mean"], float(reference_group["mean"]), "R16 aggregate mean mismatch")
                _close(group["sem"], float(reference_group["sem"]), "R16 aggregate SEM mismatch")
        for comparison, values in paired[order].items():
            for metric, group in values.items():
                reference_group = reference["paired"][order][comparison][metric]
                _close(group["mean"], float(reference_group["mean"]), "R16 paired mean mismatch")
                _close(group["sem"], float(reference_group["sem"]), "R16 paired SEM mismatch")
                _require(group["wins"] == reference_group["wins"], "R16 paired win-count mismatch")
    return {
        "inputs": inputs,
        "run_count": len(runs),
        "raw_summaries_recomputed": True,
        "matches_locked_summary": True,
        "aggregate": aggregate,
        "paired": paired,
    }


def _self_check() -> None:
    assert _mean_sd([1.0, 2.0, 3.0]) == {"mean": 2.0, "sd": 1.0, "values": [1.0, 2.0, 3.0]}
    assert math.isclose(_mean_sem([1.0, 2.0, 3.0])["sem"], 1 / math.sqrt(3))
    try:
        _require(False, "sentinel")
    except ValueError as error:
        assert str(error) == "sentinel"
    else:
        raise AssertionError("fail-closed check did not fail")
    print(json.dumps({"self_check": "ok"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=ICLR2 / "release_evidence")
    parser.add_argument("--manifest", type=Path, default=ICLR2 / "release_evidence/release_manifest.json")
    parser.add_argument("--r16-summary", type=Path, default=ICLR2 / "runs/r16_native_mask/summary.json")
    parser.add_argument("--dense-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    _require(args.output is not None, "--output is required")
    manifest_index = _manifest_index(args.manifest)
    r12 = {}
    for corpus, stem in (("mt_bench", "mt_170_s"), ("reversal", "reversal_170_s")):
        paths = [
            args.release_root / "raw/runs/r12_audited_controls" / f"{stem}{seed}" / "benchmark.json.gz"
            for seed in range(3)
        ]
        for path in paths:
            _verify_release(path, manifest_index)
        r12[corpus] = _geometry_summary(
            paths, R12_PARAMETERS, R12_MASKS, 64, 64,
            model_size=170, checkpoint_sha256=SMDM_219_SHA256,
            data_sha256=CORPORA_SHA256, probe_sha256=AUDITED_PROBE_SHA256,
            task_id=0 if corpus == "mt_bench" else 1,
        )
    result = {
        "schema_version": 1,
        "status": "ok",
        "scope": "independent verification plus R18 dense extension; no causal corpus comparison",
        "cross_text_reuse": r12,
        "continual_reuse": _continual_summary(args.release_root, manifest_index, args.r16_summary),
    }
    if args.dense_root:
        paths = [args.dense_root / f"s{seed}.json" for seed in range(3)]
        result["dense_1028_new"] = _geometry_summary(
            paths, DENSE_PARAMETERS, GEOMETRY_MASKS, 64, 64,
            model_size=1028, checkpoint_sha256=SMDM_1028_SHA256,
            data_sha256=GSM8K_SHA256, probe_sha256=BASE_PROBE_SHA256, task_id=None,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output)}))


if __name__ == "__main__":
    main()
