#!/usr/bin/env python3
"""Render qualitative samples from the three audited MNIST UNet checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


SEEDS = (0, 1, 2)
TIMESTEPS = (100, 500, 900)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(source_root: Path, checkpoint: Path, audit: Path, device):
    import torch

    payload = json.loads(audit.read_text(encoding="utf-8"))
    if payload.get("seed") not in SEEDS or payload.get("status") != "ok":
        raise ValueError(f"invalid audit: {audit}")
    if _sha256(checkpoint) != payload["model"]["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
    sys.path.insert(0, str(source_root.resolve()))
    from src.ddim import get_model

    model = get_model(1, 32, device, 10, model_size="small-big")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def _mnist_example(root: Path):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return datasets.MNIST(root=root, train=False, download=False, transform=transform)[0]


def _display(tensor):
    return ((tensor.detach().cpu().squeeze() + 1) / 2).clamp(0, 1).numpy()


def _render(models, example, output_stem: Path, device) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from matplotlib import font_manager

    for font in Path("/usr/share/fonts/opentype/urw-base35").glob("NimbusRoman-*.otf"):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": "Nimbus Roman",
        "font.size": 7.5,
        "text.color": "#26313B",
        "axes.labelcolor": "#26313B",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    labels = torch.arange(10, device=device)
    sample_generator = torch.Generator(device=device).manual_seed(20270925)
    initial_noise = torch.randn((10, 1, 32, 32), generator=sample_generator, device=device)
    samples = [
        model.sample_with_noise(initial_noise.clone(), labels, num_inference_steps=50)
        for model in models
    ]

    image, label = example
    image = image.unsqueeze(0).to(device)
    target = torch.tensor([label], device=device)
    noise_generator = torch.Generator(device=device).manual_seed(20270926)
    noise = torch.randn(image.shape, generator=noise_generator, device=device)
    noisy = []
    reconstructions = [[] for _ in models]
    for timestep in TIMESTEPS:
        t = torch.tensor([timestep], device=device, dtype=torch.long)
        x_t = models[0].scheduler.add_noise(image, noise, t)
        noisy.append(x_t)
        alpha = models[0].scheduler.alphas_cumprod[timestep].to(device=device, dtype=image.dtype)
        for model_index, model in enumerate(models):
            with torch.no_grad():
                predicted_noise = model.unet(x_t, t, target).sample
            x0 = (x_t - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
            reconstructions[model_index].append(x0)

    fig = plt.figure(figsize=(7.1, 4.4), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=(1.25, 2.0), hspace=0.42)

    sample_grid = outer[0].subgridspec(3, 10, wspace=0.04, hspace=0.08)
    for row, seed in enumerate(SEEDS):
        for column in range(10):
            axis = fig.add_subplot(sample_grid[row, column])
            axis.imshow(_display(samples[row][column]), cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(str(column), pad=1.5, fontsize=7)
            if column == 0:
                axis.set_ylabel(f"seed {seed}", rotation=0, ha="right", va="center", labelpad=12)

    reconstruction_grid = outer[1].subgridspec(5, 3, wspace=0.09, hspace=0.08)
    row_labels = [f"clean ({label})", "noisy", "seed 0", "seed 1", "seed 2"]
    for row in range(5):
        for column, timestep in enumerate(TIMESTEPS):
            axis = fig.add_subplot(reconstruction_grid[row, column])
            value = image if row == 0 else noisy[column] if row == 1 else reconstructions[row - 2][column]
            axis.imshow(_display(value), cmap="gray", vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(f"$t={timestep}$", pad=2, fontsize=7.5)
            if column == 0:
                axis.set_ylabel(row_labels[row], rotation=0, ha="right", va="center", labelpad=12)

    fig.text(0.07, 0.985, "(a) Conditional DDIM samples (shared initial noise)",
             ha="left", va="top", weight="bold", fontsize=8.5)
    fig.text(0.07, 0.565, r"(b) Fixed MNIST example: noisy $x_t$ and one-step $\hat{x}_0$",
             ha="left", va="top", weight="bold", fontsize=8.5)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _self_check() -> None:
    import torch

    x0 = torch.tensor([[-0.7, 0.2], [0.4, 0.9]])
    noise = torch.tensor([[0.3, -0.8], [0.5, 0.1]])
    alpha = torch.tensor(0.37)
    x_t = torch.sqrt(alpha) * x0 + torch.sqrt(1 - alpha) * noise
    recovered = (x_t - torch.sqrt(1 - alpha) * noise) / torch.sqrt(alpha)
    assert torch.allclose(recovered, x0, atol=1e-6)
    print(json.dumps({"self_check": "ok"}))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("iclr_1/runs/r19_mnist_unet_heldout"))
    parser.add_argument("--source-root", type=Path, default=Path("iclr_1/third_party/iclr2026-rank1-fisher"))
    parser.add_argument("--data-root", type=Path, default=Path("iclr_1/runs/data/mnist"))
    parser.add_argument("--output-stem", type=Path, default=Path("assets/iclr_1/figures/mnist_unet_demo"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        _self_check()
        return

    import torch

    device = torch.device(args.device)
    models = [
        _model(
            args.source_root,
            args.run_dir / f"seed_{seed}" / "checkpoint.pt",
            args.run_dir / f"seed_{seed}" / "audit.json",
            device,
        )
        for seed in SEEDS
    ]
    _render(models, _mnist_example(args.data_root), args.output_stem, device)
    print(json.dumps({"status": "ok", "output": str(args.output_stem)}))


if __name__ == "__main__":
    main()
