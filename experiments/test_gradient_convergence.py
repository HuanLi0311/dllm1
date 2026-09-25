#!/usr/bin/env python3
"""Test-set convergence for the frozen DLLM Fisher geometry probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments import dllm_rank1_probe as probe


PREFIXES = (16, 32, 64, 128, 256)


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _read_selected(path: Path, split: str, calibration_count: int, calibration_seed: int, test_seed: int):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if split and record.get("split") != split:
                continue
            ids = [int(token) for token in record["input_ids"]]
            rows.append({"source_line": source_line, "input_ids": ids})
    random.Random(calibration_seed).shuffle(rows)
    calibration, test = rows[:calibration_count], rows[calibration_count:]
    random.Random(test_seed).shuffle(test)
    if len(calibration) != calibration_count or len(test) < PREFIXES[-1]:
        raise ValueError("selected data cannot provide the requested calibration/test split")
    test = test[: PREFIXES[-1]]
    return calibration, test


def _prefix_metrics(calibration_gradients, test_gradients):
    calibration = torch.stack(calibration_gradients).to(dtype=torch.float64)
    calibration_mean = calibration.mean(dim=0)
    calibration_diagonal = torch.mean(calibration.square(), dim=0)
    mean_norm_sq = torch.dot(calibration_mean, calibration_mean)
    mean_quadratic = torch.mean((calibration @ calibration_mean).square())
    epsilon = torch.tensor(1e-30, dtype=torch.float64)
    coefficient = mean_quadratic / torch.clamp(mean_norm_sq.square(), min=epsilon)
    rows = []
    for count in PREFIXES:
        test = torch.stack(test_gradients[:count]).to(dtype=torch.float64)
        test_gram = (test @ test.T) / count
        fisher_norm_sq = torch.sum(test_gram.square())
        fisher_norm = torch.sqrt(torch.clamp(fisher_norm_sq, min=epsilon))
        rank1_inner = coefficient * torch.mean((test @ calibration_mean).square())
        rank1_norm_sq = coefficient.square() * mean_norm_sq.square()
        rank1 = float(
            torch.sqrt(torch.clamp(fisher_norm_sq - 2 * rank1_inner + rank1_norm_sq, min=0))
            / fisher_norm
        )
        test_diagonal = torch.mean(test.square(), dim=0)
        diagonal_inner = torch.dot(calibration_diagonal, test_diagonal)
        diagonal_norm_sq = torch.dot(calibration_diagonal, calibration_diagonal)
        diagonal = float(
            torch.sqrt(torch.clamp(fisher_norm_sq - 2 * diagonal_inner + diagonal_norm_sq, min=0))
            / fisher_norm
        )
        rows.append({
            "test_sample_count": count,
            "rank1_test_relative_frobenius_error": rank1,
            "diagonal_test_relative_frobenius_error": diagonal,
            "paired_margin_diagonal_minus_rank1": diagonal - rank1,
            "winner": "rank1" if rank1 < diagonal else "diagonal" if diagonal < rank1 else "tie",
        })
    return rows


def _paper_calibration_masks(sample_masks, model_size, probabilities, count, length, device, seed):
    required = 256 if model_size == 170 else 128
    generator = torch.Generator(device=device).manual_seed(seed)
    # The accepted SMDM probe draws this unused tensor before sampling masks.
    torch.rand((required, length), device=device, generator=generator)
    rows = []
    for probability in probabilities:
        values = torch.full((required,), probability, device=device)
        rows.append(sample_masks(values, length, device, generator)[:count])
    return rows


def run(args) -> dict:
    calibration, test = _read_selected(
        args.data, args.split, args.calibration_samples, args.calibration_seed, args.test_seed
    )
    selected = calibration + test
    base_read_records = probe._read_records
    base_sample_masks = probe._sample_masks
    base_split_metrics = probe._split_metrics
    mask_audit = []
    condition_index = 0
    calibration_masks_by_condition = None

    def fixed_records(*_unused, **_unused_kw):
        return [row["input_ids"] for row in selected]

    def independent_masks(probabilities, length, device, _generator):
        nonlocal calibration_masks_by_condition, condition_index
        if calibration_masks_by_condition is None:
            calibration_masks_by_condition = _paper_calibration_masks(
                base_sample_masks,
                args.model_size,
                probe._parse_floats(args.mask_probabilities),
                args.calibration_samples,
                length,
                device,
                args.calibration_seed,
            )
        test_generator = torch.Generator(device=device).manual_seed(
            args.test_seed + 100_003 * condition_index
        )
        calibration_masks = calibration_masks_by_condition[condition_index]
        test_masks = base_sample_masks(
            probabilities[args.calibration_samples :], length, device, test_generator
        )
        mask_audit.append({
            "condition_index": condition_index,
            "calibration_mask_sha256": _tensor_hash(calibration_masks),
            "test_mask_sha256": _tensor_hash(test_masks),
        })
        condition_index += 1
        return torch.cat((calibration_masks, test_masks))

    def nested_metrics(calibration_gradients, test_gradients, calibration_losses, test_losses):
        prefixes = _prefix_metrics(calibration_gradients, test_gradients)
        return {
            "calibration_sample_count": len(calibration_gradients),
            "test_sample_count": len(test_gradients),
            "slice_numel": calibration_gradients[0].numel(),
            "mean_rank1_test_relative_frobenius_error": prefixes[-1]["rank1_test_relative_frobenius_error"],
            "diagonal_test_relative_frobenius_error": prefixes[-1]["diagonal_test_relative_frobenius_error"],
            "test_prefix_results": prefixes,
        }

    run_args = SimpleNamespace(
        checkpoint=str(args.checkpoint),
        model_size=args.model_size,
        code_root=str(args.code_root),
        data=str(args.data),
        split=args.split,
        task_id=None,
        mask_probabilities=args.mask_probabilities,
        samples=args.calibration_samples,
        sample_sizes=str(args.calibration_samples),
        test_samples=PREFIXES[-1],
        shuffle_records=False,
        loss_mode="native_conditional",
        include_native_schedule=False,
        native_eps=1e-3,
        sequence_length=args.sequence_length,
        parameter=args.parameter,
        seed=args.calibration_seed,
        device=args.device,
        output=None,
    )
    probe._read_records = fixed_records
    probe._sample_masks = independent_masks
    probe._split_metrics = nested_metrics
    try:
        result = probe._run(run_args)
    finally:
        probe._read_records = base_read_records
        probe._sample_masks = base_sample_masks
        probe._split_metrics = base_split_metrics

    if len(mask_audit) != len(probe._parse_floats(args.mask_probabilities)):
        raise AssertionError("mask audit does not match configured conditions")
    manifest = lambda rows: [
        {"source_line": row["source_line"], "input_ids_sha256": _json_hash(row["input_ids"])}
        for row in rows
    ]
    result.update({
        "schema_version": 2,
        "experiment": "dllm_test_gradient_convergence",
        "driver_sha256": probe._sha256(Path(__file__)),
        "convergence_design": {
            "calibration_samples": args.calibration_samples,
            "test_prefixes": list(PREFIXES),
            "calibration_seed": args.calibration_seed,
            "test_seed": args.test_seed,
            "calibration_mask_protocol": "replay accepted seed-0 primary probe RNG stream",
            "calibration_records": manifest(calibration),
            "test_records": manifest(test),
            "calibration_records_sha256": _json_hash(manifest(calibration)),
            "test_records_sha256": _json_hash(manifest(test)),
            "mask_audit": mask_audit,
        },
    })
    return result


def _self_check() -> None:
    calibration = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    test = [torch.tensor([1.0 + index / 1000, 1.0]) for index in range(256)]
    rows = _prefix_metrics(calibration, test)
    assert [row["test_sample_count"] for row in rows] == list(PREFIXES)
    assert all(row["winner"] in {"rank1", "diagonal", "tie"} for row in rows)
    for row in rows:
        count = row["test_sample_count"]
        expected = probe._split_metrics(calibration, test[:count], [1.0, 1.0], [1.0] * count)
        assert math.isclose(
            row["rank1_test_relative_frobenius_error"],
            expected["mean_rank1_test_relative_frobenius_error"],
            rel_tol=1e-12,
        )
        assert math.isclose(
            row["diagonal_test_relative_frobenius_error"],
            expected["diagonal_test_relative_frobenius_error"],
            rel_tol=1e-12,
        )
    masks1 = _paper_calibration_masks(probe._sample_masks, 170, [0.1, 0.5], 4, 8, "cpu", 0)
    masks2 = _paper_calibration_masks(probe._sample_masks, 170, [0.1, 0.5], 4, 8, "cpu", 0)
    assert [_tensor_hash(mask) for mask in masks1] == [_tensor_hash(mask) for mask in masks2]
    masks3 = _paper_calibration_masks(probe._sample_masks, 1028, [0.1, 0.5], 4, 8, "cpu", 0)
    assert _tensor_hash(masks1[0]) != _tensor_hash(masks3[0])
    print(json.dumps({"self_check": "ok"}))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-size", type=int, choices=(170, 1028))
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--mask-probabilities", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--test-seed", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--parameter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_check and any(
        value is None for value in (args.checkpoint, args.model_size, args.code_root, args.data, args.parameter, args.output)
    ):
        parser.error("checkpoint, model-size, code-root, data, parameter, and output are required")
    if args.calibration_samples < 2:
        parser.error("--calibration-samples must be at least 2")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    try:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        result = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": "ok", "output": str(args.output)}), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        }
        if args.output and not args.output.exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure, indent=2), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
