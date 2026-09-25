#!/usr/bin/env python3
"""Test-set convergence for fixed LLaDA-8B Fisher surrogates."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import dllm_rank1_probe as base


PREFIXES = (16, 32, 64, 128, 256)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tensor_hash(value) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _records(path, tokenizer, sequence_length, calibration_count, calibration_seed, test_seed):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record["question"].strip() + "\n" + record["answer"].strip()
            tokens = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
            if len(tokens) >= sequence_length:
                rows.append({"source_line": source_line, "tokens": tokens[:sequence_length]})
    random.Random(calibration_seed).shuffle(rows)
    calibration, pool = rows[:calibration_count], rows[calibration_count:]
    random.Random(test_seed).shuffle(pool)
    test = pool[: PREFIXES[-1]]
    if len(calibration) != calibration_count or len(test) != PREFIXES[-1]:
        raise ValueError("not enough full-length records for calibration and test")
    return calibration, test


def _prefix_rows(calibration_gradients, test_gradients):
    import torch

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


def _paper_calibration_masks(sample_masks, probabilities, count, length, device, seed):
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    rows = []
    for probability in probabilities:
        values = torch.full((128,), probability, device=device)
        rows.append(sample_masks(values, length, device, generator)[:count])
    return rows


def run(args):
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint = args.checkpoint.resolve()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    calibration, test = _records(
        args.data,
        tokenizer,
        args.sequence_length,
        args.calibration_samples,
        args.calibration_seed,
        args.test_seed,
    )
    selected = calibration + test
    ids = torch.tensor([row["tokens"] for row in selected], dtype=torch.long, device=device)
    model = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named = dict(model.named_parameters())
    parameter_names = [name.strip() for name in args.parameter.split(",") if name.strip()]
    missing = [name for name in parameter_names if name not in named]
    if missing:
        raise KeyError("parameter not found: " + ", ".join(missing))
    targets = [named[name].requires_grad_(True) for name in parameter_names]

    results, mask_audit = [], []
    configured_probabilities = base._parse_floats(args.mask_probabilities)
    calibration_masks_by_condition = _paper_calibration_masks(
        base._sample_masks,
        configured_probabilities,
        args.calibration_samples,
        args.sequence_length,
        device,
        args.calibration_seed,
    )
    for condition_index, probability in enumerate(configured_probabilities):
        probabilities = torch.full((len(selected),), probability, device=device)
        test_generator = torch.Generator(device=device).manual_seed(
            args.test_seed + 100_003 * condition_index
        )
        calibration_masks = calibration_masks_by_condition[condition_index]
        test_masks = base._sample_masks(
            probabilities[args.calibration_samples :],
            args.sequence_length,
            device,
            test_generator,
        )
        masks = torch.cat((calibration_masks, test_masks))
        mask_audit.append({
            "condition_index": condition_index,
            "calibration_mask_sha256": _tensor_hash(calibration_masks),
            "test_mask_sha256": _tensor_hash(test_masks),
        })
        gradients = {name: [] for name in parameter_names}
        for index, clean in enumerate(ids):
            noisy = clean.clone()
            noisy[masks[index]] = model.config.mask_token_id
            logits = model(noisy.unsqueeze(0)).logits[0].float()
            token_losses = F.cross_entropy(
                logits[masks[index]], clean[masks[index]], reduction="none"
            )
            loss = token_losses.sum() / (probability * args.sequence_length)
            example_gradients = torch.autograd.grad(loss, targets, allow_unused=False)
            if not torch.isfinite(loss) or any(
                not torch.isfinite(gradient).all() for gradient in example_gradients
            ):
                raise FloatingPointError(f"non-finite value at p={probability:g}")
            for name, gradient in zip(parameter_names, example_gradients):
                gradients[name].append(gradient.detach().float().cpu().reshape(-1))
            del logits, loss, example_gradients
        for name in parameter_names:
            results.append({
                "model": "LLaDA-8B-Base",
                "model_size_m": 8016,
                "mask_probability": probability,
                "mask_condition": f"fixed_{probability:g}",
                "loss_mode": "native_conditional",
                "evaluation": "split_sample_test_convergence",
                "parameter": name,
                "calibration_sample_count": args.calibration_samples,
                "calibration_seed": args.calibration_seed,
                "test_seed": args.test_seed,
                "sequence_length": args.sequence_length,
                "test_prefix_results": _prefix_rows(
                    gradients[name][: args.calibration_samples],
                    gradients[name][args.calibration_samples :],
                ),
            })

    checkpoint_files = [
        checkpoint / "config.json",
        checkpoint / "model.safetensors.index.json",
        checkpoint / "modeling_llada.py",
        *sorted(checkpoint.glob("model-*.safetensors")),
    ]
    checkpoint_hashes = {path.name: _sha256(path) for path in checkpoint_files}
    manifest = lambda rows: [
        {"source_line": row["source_line"], "tokens_sha256": _json_hash(row["tokens"])}
        for row in rows
    ]
    return {
        "schema_version": 1,
        "status": "ok",
        "experiment": "llada_test_gradient_convergence",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _json_hash(checkpoint_hashes),
        "checkpoint_files_sha256": checkpoint_hashes,
        "driver_sha256": _sha256(Path(__file__)),
        "base_probe_sha256": _sha256(Path(base.__file__)),
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "data": str(args.data.resolve()),
        "data_sha256": _sha256(args.data),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "convergence_design": {
            "test_prefixes": list(PREFIXES),
            "calibration_mask_protocol": "replay accepted seed-0 primary probe RNG stream",
            "calibration_records": manifest(calibration),
            "test_records": manifest(test),
            "calibration_records_sha256": _json_hash(manifest(calibration)),
            "test_records_sha256": _json_hash(manifest(test)),
            "mask_audit": mask_audit,
        },
        "results": results,
    }


def _self_check():
    import torch

    calibration = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    test = [torch.tensor([1.0 + index / 1000, 1.0]) for index in range(256)]
    rows = _prefix_rows(calibration, test)
    assert [row["test_sample_count"] for row in rows] == list(PREFIXES)
    assert all(row["winner"] in {"rank1", "diagonal", "tie"} for row in rows)
    for row in rows:
        count = row["test_sample_count"]
        expected = base._split_metrics(calibration, test[:count], [1.0, 1.0], [1.0] * count)
        assert abs(
            row["rank1_test_relative_frobenius_error"]
            - expected["mean_rank1_test_relative_frobenius_error"]
        ) < 1e-12
        assert abs(
            row["diagonal_test_relative_frobenius_error"]
            - expected["diagonal_test_relative_frobenius_error"]
        ) < 1e-12
    masks1 = _paper_calibration_masks(base._sample_masks, [0.1, 0.5], 4, 8, "cpu", 0)
    masks2 = _paper_calibration_masks(base._sample_masks, [0.1, 0.5], 4, 8, "cpu", 0)
    assert [_tensor_hash(mask) for mask in masks1] == [_tensor_hash(mask) for mask in masks2]
    print(json.dumps({"self_check": "ok"}))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--mask-probabilities", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--test-seed", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--parameter", default="model.transformer.blocks.0.attn_norm.weight,model.transformer.blocks.15.attn_norm.weight,model.transformer.blocks.31.attn_norm.weight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_check and any(value is None for value in (args.checkpoint, args.data, args.output)):
        parser.error("--checkpoint, --data, and --output are required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    try:
        result = run(args)
        status = 0
    except Exception as exc:
        result = {
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        }
        status = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
