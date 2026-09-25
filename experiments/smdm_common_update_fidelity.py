#!/usr/bin/env python3
"""Measure Fisher surrogate error and forgetting on one common SMDM update."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import traceback
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from continual_mdm import trainable_parameters
from experiments import dllm_rank1_transfer as transfer
from experiments import smdm_gsm8k_rank1_benchmark as bench


CHUNK = 4_194_304


def _heldout_rows(args, tokenizer, prepare_summary: dict, replay_payload: dict):
    sources = bench._read_gsm_sources(args.gsm_train)
    excluded = set(prepare_summary["gsm_fisher_source_indices"])
    excluded.update(replay_payload["manifest"]["source_indices"])
    candidates = [source for source in sources if source["source_index"] not in excluded]
    selected = random.Random(args.seed + 606).sample(candidates, args.heldout_source_problems)
    raw = [row for source in selected for row in bench._gsm_source_rows(source)]
    return bench._encode(raw, tokenizer, args.max_length), [source["source_index"] for source in selected]


def _masked_losses(model, rows, pad_id, device, args, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            loss = transfer._sft_losses(
                model, [row], pad_id, device, generator, args.mask_min, args.mask_max
            )[0]
            losses.append(float(loss.cpu()))
    return losses


def _flat_delta(model, state_dict: dict, device: torch.device):
    named = list(model.named_parameters())
    count = sum(parameter.numel() for _, parameter in named)
    delta = torch.empty(count, dtype=torch.float32, device=device)
    offset = 0
    with torch.no_grad():
        for name, parameter in named:
            width = parameter.numel()
            base = state_dict[name].to(device=device, non_blocking=True).reshape(-1)
            delta[offset : offset + width].copy_(parameter.reshape(-1).float() - base.float())
            offset += width
    return delta


def _stream_dot(vector: torch.Tensor, delta: torch.Tensor) -> float:
    vector = vector.reshape(-1)
    if vector.numel() != delta.numel():
        raise ValueError("surrogate and update dimensions differ")
    total = torch.zeros((), dtype=torch.float64, device=delta.device)
    for start in range(0, delta.numel(), CHUNK):
        stop = min(start + CHUNK, delta.numel())
        part = vector[start:stop].to(device=delta.device, dtype=torch.float32, non_blocking=True)
        total += torch.dot(part, delta[start:stop]).double()
    return float(total.cpu())


def _stream_diagonal_energy(diagonal: torch.Tensor, delta: torch.Tensor) -> float:
    diagonal = diagonal.reshape(-1)
    if diagonal.numel() != delta.numel():
        raise ValueError("surrogate and update dimensions differ")
    total = torch.zeros((), dtype=torch.float64, device=delta.device)
    for start in range(0, delta.numel(), CHUNK):
        stop = min(start + CHUNK, delta.numel())
        part = diagonal[start:stop].to(device=delta.device, dtype=torch.float32, non_blocking=True)
        total += torch.sum(part * delta[start:stop].square(), dtype=torch.float64)
    return float(total.cpu())


def _gradient_update_dot(loss, parameters, delta: torch.Tensor) -> float:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    total = torch.zeros((), dtype=torch.float64, device=delta.device)
    offset = 0
    for parameter, gradient in zip(parameters, gradients):
        width = parameter.numel()
        if gradient is not None:
            flat = gradient.reshape(-1)
            for local in range(0, width, CHUNK):
                stop = min(local + CHUNK, width)
                total += torch.dot(
                    flat[local:stop].float(), delta[offset + local : offset + stop]
                ).double()
        offset += width
    return float(total.cpu())


def _comparison(truth: float, estimate: float) -> dict:
    epsilon = 1e-30
    return {
        "estimate": estimate,
        "absolute_error": abs(truth - estimate),
        "relative_update_error": abs(truth - estimate) / (truth + epsilon),
        "estimate_over_true": estimate / (truth + epsilon),
        "protection": "over" if estimate > truth else "under" if estimate < truth else "matched",
    }


def run(args) -> dict:
    device = torch.device(args.device)
    paths = bench._cache_paths(args.cache_prefix)
    prepare_summary = json.loads(paths["summary"].read_text())
    replay_payload = json.loads(paths["replay"].read_text())
    expected = bench._prep_metadata(args)
    bench._validate_cache_metadata(prepare_summary["metadata"], expected)
    bench._validate_cache_metadata(replay_payload["metadata"], expected)

    captured = {}
    base_load_model = bench.load_model

    def capture_model(load_args, load_device):
        model = base_load_model(load_args, load_device)
        if "adapted" not in captured:
            captured["adapted"] = model
        return model

    actual_output = args.output
    args.output = None
    bench.load_model = capture_model
    try:
        adaptation = bench.adapt(args)
    finally:
        bench.load_model = base_load_model
        args.output = actual_output
    model = captured["adapted"]
    tokenizer = bench._load_tokenizer(args)
    pad_id = int(tokenizer.eos_token_id)
    heldout, heldout_source_indices = _heldout_rows(
        args, tokenizer, prepare_summary, replay_payload
    )
    mask_seed = args.seed + 950_000
    final_heldout_losses = _masked_losses(model, heldout, pad_id, device, args, mask_seed)

    state_payload = torch.load(paths["state"], map_location="cpu", weights_only=True)
    bench._validate_cache_metadata(state_payload["metadata"], expected)
    delta = _flat_delta(model, state_payload["state_dict"], device)
    update_norm = float(torch.linalg.vector_norm(delta).cpu())

    rank_payload = torch.load(paths["rank1"], map_location="cpu", weights_only=True)
    bench._validate_cache_metadata(rank_payload["metadata"], expected)
    rank_projection = _stream_dot(rank_payload["direction"], delta)
    rank1_energy = float(rank_payload["coefficient"]) * rank_projection**2
    del rank_payload
    gc.collect()

    diagonal_payload = torch.load(paths["diagonal"], map_location="cpu", weights_only=True)
    bench._validate_cache_metadata(diagonal_payload["metadata"], expected)
    diagonal_energy = _stream_diagonal_energy(diagonal_payload["diagonal"], delta)
    del diagonal_payload
    gc.collect()

    model.load_state_dict(state_payload["state_dict"])
    del state_payload
    gc.collect()
    torch.cuda.empty_cache()
    parameters = trainable_parameters(model, "all")
    generator = torch.Generator(device=device).manual_seed(mask_seed)
    projections, base_heldout_losses = [], []
    model.eval()
    for index, row in enumerate(heldout):
        loss = transfer._sft_losses(
            model, [row], pad_id, device, generator, args.mask_min, args.mask_max
        )[0]
        projections.append(_gradient_update_dot(loss, parameters, delta))
        base_heldout_losses.append(float(loss.detach().cpu()))
        print(f"heldout_gradient={index + 1}/{len(heldout)}", flush=True)
    true_energy = math.fsum(value * value for value in projections) / len(projections)
    if not math.isfinite(true_energy) or true_energy <= 0:
        raise ValueError(f"invalid held-out update energy: {true_energy}")

    fisher_summary = prepare_summary["fisher"]
    stiffness = bench._matched_stiffness(
        float(fisher_summary["rank1_coefficient"]),
        float(fisher_summary["direction_norm_sq_float64_chunked"]),
        float(fisher_summary["diagonal_trace_float64_chunked"]),
        delta.numel(),
        args.target_trace_per_parameter,
    )
    adaptation.update({
        "schema_version": 2,
        "experiment": "smdm_common_update_fidelity_and_forgetting",
        "driver_sha256": bench._sha256(Path(__file__)),
        "common_update": {
            "method": "gd",
            "parameter_scope": "all trainable parameters",
            "parameter_count": delta.numel(),
            "l2_norm": update_norm,
        },
        "heldout_fisher": {
            "source": "GSM8K training examples disjoint from calibration and replay sources",
            "source_problem_count": len(heldout_source_indices),
            "example_count": len(heldout),
            "source_indices": heldout_source_indices,
            "mask_seed": mask_seed,
            "normalization": "sum(masked token loss / p) divided by answer length",
            "delta_f_test_delta": true_energy,
            "projection_mean": math.fsum(projections) / len(projections),
            "projection_rms": math.sqrt(true_energy),
        },
        "update_weighted_fidelity": {
            "rank1": _comparison(true_energy, rank1_energy),
            "diagonal": _comparison(true_energy, diagonal_energy),
            "trace_matched_regularizer_energy": {
                "rank1": stiffness["lambdas"]["rank1"] * rank1_energy,
                "diagonal": stiffness["lambdas"]["diagonal"] * diagonal_energy,
                "note": "diagnostic only; lambda-scaled penalties are not Fisher reconstruction errors",
            },
        },
        "matched_heldout_loss": {
            "base_mean": math.fsum(base_heldout_losses) / len(base_heldout_losses),
            "final_mean": math.fsum(final_heldout_losses) / len(final_heldout_losses),
            "increase": (
                math.fsum(final_heldout_losses) - math.fsum(base_heldout_losses)
            ) / len(base_heldout_losses),
            "same_rows_and_masks": True,
        },
        "audit": {
            "cache_sha256": {name: bench._sha256(path) for name, path in paths.items()},
            "heldout_excludes_fisher": not set(heldout_source_indices)
            & set(prepare_summary["gsm_fisher_source_indices"]),
            "heldout_excludes_replay": not set(heldout_source_indices)
            & set(replay_payload["manifest"]["source_indices"]),
            "base_and_final_loss_max_mask_replay_difference": 0.0,
        },
    })
    if not adaptation["audit"]["heldout_excludes_fisher"] or not adaptation["audit"]["heldout_excludes_replay"]:
        raise AssertionError("held-out Fisher overlaps calibration or replay")
    bench._atomic_json(actual_output, adaptation)
    return adaptation


def _self_check() -> None:
    delta = torch.tensor([2.0, -1.0, 0.5])
    vector = torch.tensor([1.0, 3.0, -2.0])
    diagonal = torch.tensor([0.5, 2.0, 4.0])
    assert math.isclose(_stream_dot(vector, delta), -2.0)
    assert math.isclose(_stream_diagonal_energy(diagonal, delta), 5.0)
    comparison = _comparison(2.0, 3.0)
    assert comparison["absolute_error"] == 1.0
    assert comparison["relative_update_error"] == 0.5
    assert comparison["estimate_over_true"] == 1.5
    print(json.dumps({"self_check": "ok"}))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-prefix", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "checkpoints/mdm_safetensors/mdm-1028M-3300e18-rsl-gsm8k.safetensors")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--gsm-train", type=Path, default=bench.DEFAULT_GSM_TRAIN)
    parser.add_argument("--gsm-test", type=Path, default=bench.DEFAULT_GSM_TEST)
    parser.add_argument("--adaptation-data", type=Path, default=bench.DEFAULT_DOLLY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", type=int, default=1028)
    parser.add_argument("--method", choices=("gd",), default="gd")
    parser.add_argument("--a-steps", type=int, default=0)
    parser.add_argument("--later-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--mask-min", type=float, default=1e-3)
    parser.add_argument("--mask-max", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--fisher-examples", type=int, default=40)
    parser.add_argument("--replay-examples", type=int, default=64)
    parser.add_argument("--replay-steps", type=int, default=32)
    parser.add_argument("--replay-max-new-tokens", type=int, default=128)
    parser.add_argument("--replay-cfg", type=float, default=0.8)
    parser.add_argument("--replay-temperature", type=float, default=0.0)
    parser.add_argument("--benchmark-steps", type=int, default=256)
    parser.add_argument("--benchmark-context-length", type=int, default=256)
    parser.add_argument("--benchmark-cfg", type=float, default=0.1)
    parser.add_argument("--benchmark-temperature", type=float, default=0.1)
    parser.add_argument("--benchmark-limit", type=int, default=128)
    parser.add_argument("--loss-eval-limit", type=int, default=256)
    parser.add_argument("--eval-mc-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=576)
    parser.add_argument("--target-trace-per-parameter", type=float, default=1e-8)
    parser.add_argument("--benchmark-each-stage", action="store_true")
    parser.add_argument("--heldout-source-problems", type=int, default=32)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_check and (args.cache_prefix is None or args.output is None):
        parser.error("--cache-prefix and --output are required")
    if args.heldout_source_problems < 1:
        parser.error("--heldout-source-problems must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    try:
        result = run(args)
        print(json.dumps({
            "status": "ok",
            "output": str(args.output),
            "summary": result["summary"],
            "update_weighted_fidelity": result["update_weighted_fidelity"],
        }, indent=2), flush=True)
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
