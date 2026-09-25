#!/usr/bin/env python3
"""Train the source-paper MNIST UNet and audit held-out Fisher reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_TIMESTEPS = tuple(range(100, 1000, 100))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_model(source_root: Path, device):
    source_root = source_root.resolve()
    if not (source_root / "src" / "ddim.py").is_file():
        raise FileNotFoundError(f"missing source model: {source_root / 'src/ddim.py'}")
    sys.path.insert(0, str(source_root))
    from src.ddim import get_model

    return get_model(1, 32, device, 10, model_size="small-big")


def _mnist(root: Path, train: bool, download: bool):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return datasets.MNIST(root=root, train=train, download=download, transform=transform)


def _train(model, dataset, *, seed: int, epochs: int, batch_size: int,
           workers: int, device, learning_rate: float) -> list[float]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    started = time.monotonic()
    for epoch in range(epochs):
        total_loss = 0.0
        total_count = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            timesteps, noise, _, prediction = model.diffusion_loss(images, labels)
            del timesteps
            loss = F.mse_loss(prediction, noise)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * images.shape[0]
            total_count += images.shape[0]
        losses.append(total_loss / total_count)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(json.dumps({
                "phase": "train",
                "epoch": epoch + 1,
                "epochs": epochs,
                "loss": losses[-1],
                "elapsed_seconds": time.monotonic() - started,
            }), flush=True)
    return losses


def _selected_examples(dataset, *, seed: int, calibration_count: int,
                       test_count: int):
    import torch

    if calibration_count + test_count > len(dataset):
        raise ValueError("calibration and test samples exceed the dataset size")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:calibration_count + test_count]
    images = torch.stack([dataset[int(index)][0] for index in indices])
    labels = torch.tensor([dataset[int(index)][1] for index in indices], dtype=torch.long)
    return (
        images[:calibration_count],
        labels[:calibration_count],
        images[calibration_count:],
        labels[calibration_count:],
        indices.tolist(),
    )


def _collect_gradients(model, images, labels, *, timestep: int, noise_seed: int,
                       device, label: str):
    import torch
    import torch.nn.functional as F

    model.eval()
    unet = model.unet
    parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    dimension = sum(parameter.numel() for parameter in parameters)
    gradients = torch.empty((images.shape[0], dimension), device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    t = torch.tensor([timestep], device=device, dtype=torch.long)
    started = time.monotonic()
    for index in range(images.shape[0]):
        image = images[index:index + 1].to(device, non_blocking=True)
        target = labels[index:index + 1].to(device, non_blocking=True)
        noise = torch.randn(image.shape, generator=generator, device=device, dtype=image.dtype)
        noisy_image = model.scheduler.add_noise(image, noise, t)
        unet.zero_grad(set_to_none=True)
        prediction = unet(noisy_image, t, target).sample
        loss = F.mse_loss(prediction, noise)
        loss.backward()
        pieces = []
        for parameter in parameters:
            if parameter.grad is None:
                raise RuntimeError("a trainable UNet parameter has no gradient")
            pieces.append(parameter.grad.reshape(-1))
        gradients[index].copy_(torch.cat(pieces))
        if index == 0 or (index + 1) % 128 == 0 or index + 1 == images.shape[0]:
            print(json.dumps({
                "phase": "gradients",
                "split": label,
                "timestep": timestep,
                "completed": index + 1,
                "total": images.shape[0],
                "elapsed_seconds": time.monotonic() - started,
            }), flush=True)
    unet.zero_grad(set_to_none=True)
    return gradients


def _relative_error(norm_sq, cross, surrogate_norm_sq):
    import torch

    residual_sq = torch.clamp(norm_sq - 2 * cross + surrogate_norm_sq, min=0)
    return float(torch.sqrt(residual_sq / norm_sq))


def _reconstruction_metrics(calibration, test) -> dict:
    import torch

    calibration = calibration.to(torch.float64)
    test = test.to(torch.float64)
    calibration_count = calibration.shape[0]
    test_count = test.shape[0]

    calibration_gram = calibration @ calibration.T / calibration_count
    test_gram = test @ test.T / test_count
    calibration_norm_sq = torch.sum(calibration_gram * calibration_gram)
    test_norm_sq = torch.sum(test_gram * test_gram)

    mean = calibration.mean(dim=0)
    mean_norm_sq = torch.dot(mean, mean)
    calibration_quadratic = torch.mean((calibration @ mean) ** 2)
    coefficient = calibration_quadratic / torch.clamp(mean_norm_sq * mean_norm_sq, min=1e-30)
    rank1_norm_sq = coefficient * coefficient * mean_norm_sq * mean_norm_sq
    test_quadratic = torch.mean((test @ mean) ** 2)

    calibration_diagonal = torch.mean(calibration * calibration, dim=0)
    test_diagonal = torch.mean(test * test, dim=0)
    diagonal_norm_sq = torch.dot(calibration_diagonal, calibration_diagonal)

    calibration_eigenvalues = torch.linalg.eigvalsh(calibration_gram)
    test_eigenvalues = torch.linalg.eigvalsh(test_gram)
    calibration_lambda1 = torch.clamp(calibration_eigenvalues[-1], min=0)
    test_lambda1 = torch.clamp(test_eigenvalues[-1], min=0)
    calibration_oracle = torch.sqrt(torch.clamp(1 - calibration_lambda1 ** 2 / calibration_norm_sq, min=0))
    test_oracle = torch.sqrt(torch.clamp(1 - test_lambda1 ** 2 / test_norm_sq, min=0))

    return {
        "calibration_rank1_error": _relative_error(
            calibration_norm_sq, coefficient * calibration_quadratic, rank1_norm_sq),
        "calibration_diagonal_error": _relative_error(
            calibration_norm_sq, diagonal_norm_sq, diagonal_norm_sq),
        "calibration_oracle_error": float(calibration_oracle),
        "test_rank1_error": _relative_error(
            test_norm_sq, coefficient * test_quadratic, rank1_norm_sq),
        "test_diagonal_error": _relative_error(
            test_norm_sq,
            torch.dot(calibration_diagonal, test_diagonal),
            diagonal_norm_sq,
        ),
        "test_oracle_error": float(test_oracle),
        "calibration_lambda2_over_lambda1": float(
            calibration_eigenvalues[-2] / torch.clamp(calibration_lambda1, min=1e-30)),
        "test_lambda2_over_lambda1": float(
            test_eigenvalues[-2] / torch.clamp(test_lambda1, min=1e-30)),
        "rank1_coefficient": float(coefficient),
        "mean_gradient_norm": float(torch.sqrt(mean_norm_sq)),
    }


def _self_check() -> None:
    import torch

    calibration = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 2.0, 1.0], [1.0, 1.0, -1.0]],
        dtype=torch.float64,
    )
    test = torch.tensor(
        [[2.0, 0.0, 1.0], [0.0, 1.0, 2.0], [1.0, -1.0, 0.0]],
        dtype=torch.float64,
    )
    measured = _reconstruction_metrics(calibration, test)
    fisher_calibration = calibration.T @ calibration / calibration.shape[0]
    fisher_test = test.T @ test / test.shape[0]
    mean = calibration.mean(dim=0)
    coefficient = (mean @ fisher_calibration @ mean) / (mean @ mean) ** 2
    rank1 = coefficient * torch.outer(mean, mean)
    diagonal = torch.diag(torch.diag(fisher_calibration))
    expected = {
        "calibration_rank1_error": torch.linalg.matrix_norm(fisher_calibration - rank1) / torch.linalg.matrix_norm(fisher_calibration),
        "calibration_diagonal_error": torch.linalg.matrix_norm(fisher_calibration - diagonal) / torch.linalg.matrix_norm(fisher_calibration),
        "test_rank1_error": torch.linalg.matrix_norm(fisher_test - rank1) / torch.linalg.matrix_norm(fisher_test),
        "test_diagonal_error": torch.linalg.matrix_norm(fisher_test - diagonal) / torch.linalg.matrix_norm(fisher_test),
    }
    for key, value in expected.items():
        assert math.isclose(measured[key], float(value), rel_tol=1e-12, abs_tol=1e-12), key
    assert measured["calibration_oracle_error"] <= measured["calibration_rank1_error"] + 1e-12
    assert measured["test_oracle_error"] <= 1.0
    print(json.dumps({"self_check": "ok"}))


def _source_commit(source_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "probe", "all"), default="all")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--calibration-count", type=int, default=1024)
    parser.add_argument("--test-count", type=int, default=1024)
    parser.add_argument("--timesteps", type=int, nargs="+", default=DEFAULT_TIMESTEPS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=Path("iclr_1/runs/data/mnist"))
    parser.add_argument("--output-dir", type=Path, default=Path("iclr_1/runs/r19_mnist_unet_heldout"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("iclr_1/third_party/iclr2026-rank1-fisher"),
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    if args.self_check:
        _self_check()
        return
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch size must be positive; workers cannot be negative")
    if min(args.calibration_count, args.test_count) < 2:
        parser.error("calibration and test counts must be at least two")
    if not args.timesteps or any(timestep < 0 or timestep >= 1000 for timestep in args.timesteps):
        parser.error("timesteps must lie in [0, 1000)")

    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _set_seed(args.seed)
    model = _load_model(args.source_root, device)
    parameter_count = sum(parameter.numel() for parameter in model.unet.parameters())
    if parameter_count != 152497:
        raise RuntimeError(f"unexpected small-big UNet parameter count: {parameter_count}")

    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    if args.mode in ("train", "all") and not checkpoint_path.exists():
        training_data = _mnist(args.data_root, train=True, download=True)
        started = time.monotonic()
        losses = _train(
            model,
            training_data,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            learning_rate=args.learning_rate,
        )
        checkpoint = {
            "state_dict": model.state_dict(),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "architecture": "small-big",
            "parameter_count": parameter_count,
            "losses": losses,
            "elapsed_seconds": time.monotonic() - started,
        }
        temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary_checkpoint)
        os.replace(temporary_checkpoint, checkpoint_path)
        print(json.dumps({"phase": "checkpoint", "path": str(checkpoint_path)}), flush=True)
    elif args.mode in ("train", "all"):
        print(json.dumps({"phase": "checkpoint", "status": "reuse", "path": str(checkpoint_path)}), flush=True)

    if args.mode == "train":
        return
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"probe requires checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "small-big" or checkpoint.get("parameter_count") != parameter_count:
        raise RuntimeError("checkpoint architecture does not match the requested model")
    model.load_state_dict(checkpoint["state_dict"])

    test_data = _mnist(args.data_root, train=False, download=True)
    selection_seed = args.seed + 10_000_000
    calibration_images, calibration_labels, test_images, test_labels, indices = _selected_examples(
        test_data,
        seed=selection_seed,
        calibration_count=args.calibration_count,
        test_count=args.test_count,
    )
    results = []
    probe_started = time.monotonic()
    for timestep in args.timesteps:
        calibration_noise_seed = args.seed * 1_000_000 + timestep * 10 + 1
        test_noise_seed = args.seed * 1_000_000 + timestep * 10 + 2
        calibration_gradients = _collect_gradients(
            model,
            calibration_images,
            calibration_labels,
            timestep=timestep,
            noise_seed=calibration_noise_seed,
            device=device,
            label="calibration",
        )
        test_gradients = _collect_gradients(
            model,
            test_images,
            test_labels,
            timestep=timestep,
            noise_seed=test_noise_seed,
            device=device,
            label="test",
        )
        metrics = _reconstruction_metrics(calibration_gradients, test_gradients)
        del calibration_gradients, test_gradients
        if device.type == "cuda":
            torch.cuda.empty_cache()
        result = {
            "timestep": timestep,
            "calibration_noise_seed": calibration_noise_seed,
            "test_noise_seed": test_noise_seed,
            **metrics,
        }
        results.append(result)
        print(json.dumps({"phase": "result", **result}), flush=True)

    source_root = args.source_root.resolve()
    payload = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "mnist_unet_heldout_fisher_reconstruction",
        "seed": args.seed,
        "selection_seed": selection_seed,
        "dataset": "MNIST-test",
        "calibration_count": args.calibration_count,
        "test_count": args.test_count,
        "calibration_indices": indices[:args.calibration_count],
        "test_indices": indices[args.calibration_count:],
        "model": {
            "architecture": "source-paper-small-big-unet",
            "parameter_count": parameter_count,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "training_epochs": checkpoint["epochs"],
            "training_batch_size": checkpoint["batch_size"],
            "training_learning_rate": checkpoint["learning_rate"],
        },
        "source": {
            "repository": "Teachable-AI-Lab/iclr2026-rank1-fisher",
            "commit": _source_commit(source_root),
            "ddim_sha256": _sha256(source_root / "src" / "ddim.py"),
        },
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "device": str(device),
        "command": [sys.executable, *sys.argv],
        "script_sha256": _sha256(Path(__file__)),
        "probe_elapsed_seconds": time.monotonic() - probe_started,
        "results": results,
    }
    output_path = run_dir / "audit.json"
    _write_json_atomic(output_path, payload)
    print(json.dumps({"status": "ok", "output": str(output_path)}), flush=True)


if __name__ == "__main__":
    main()
