#!/usr/bin/env python3
"""Validate, aggregate, and plot the three-seed MNIST UNet Fisher audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path


EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_TIMESTEPS = tuple(range(100, 1000, 100))
EXPECTED_SOURCE_COMMIT = "c7577f22551941e4bf58e33405fc78e8fcb608aa"
EXPECTED_MODEL = {
    "architecture": "source-paper-small-big-unet",
    "parameter_count": 152497,
    "training_epochs": 200,
    "training_batch_size": 128,
    "training_learning_rate": 0.0002,
}
METRICS = (
    "calibration_rank1_error",
    "calibration_diagonal_error",
    "calibration_oracle_error",
    "test_rank1_error",
    "test_diagonal_error",
    "test_oracle_error",
    "calibration_lambda2_over_lambda1",
    "test_lambda2_over_lambda1",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean_sd(values) -> dict:
    values = list(values)
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
    }


def _read_json(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _load(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    payloads = [_read_json(path) for path in paths]
    if len(payloads) != len(EXPECTED_SEEDS):
        raise ValueError(f"expected three audit files, received {len(payloads)}")
    by_seed = {payload["seed"]: payload for payload in payloads}
    if set(by_seed) != set(EXPECTED_SEEDS):
        raise ValueError(f"expected seeds {EXPECTED_SEEDS}, received {sorted(by_seed)}")
    reference = by_seed[EXPECTED_SEEDS[0]]
    records = []
    for seed in EXPECTED_SEEDS:
        payload = by_seed[seed]
        if payload.get("status") != "ok":
            raise ValueError(f"seed {seed} is not successful")
        if payload.get("dataset") != "MNIST-test":
            raise ValueError(f"seed {seed} uses the wrong dataset")
        if payload.get("calibration_count") != 1024 or payload.get("test_count") != 1024:
            raise ValueError(f"seed {seed} uses the wrong split sizes")
        if any(payload.get("model", {}).get(key) != value for key, value in EXPECTED_MODEL.items()):
            raise ValueError(f"seed {seed} uses the wrong model or training protocol")
        if payload.get("source", {}).get("commit") != EXPECTED_SOURCE_COMMIT:
            raise ValueError(f"seed {seed} uses the wrong source commit")
        for key in ("dataset", "calibration_count", "test_count"):
            if payload[key] != reference[key]:
                raise ValueError(f"seed {seed} mismatches {key}")
        if payload["source"] != reference["source"]:
            raise ValueError(f"seed {seed} uses different source code")
        if payload["model"]["architecture"] != reference["model"]["architecture"]:
            raise ValueError(f"seed {seed} uses a different architecture")
        calibration_indices = payload["calibration_indices"]
        test_indices = payload["test_indices"]
        if len(calibration_indices) != payload["calibration_count"]:
            raise ValueError(f"seed {seed} calibration index count is incomplete")
        if len(test_indices) != payload["test_count"]:
            raise ValueError(f"seed {seed} test index count is incomplete")
        if set(calibration_indices) & set(test_indices):
            raise ValueError(f"seed {seed} reuses examples across calibration and test")
        by_timestep = {row["timestep"]: row for row in payload["results"]}
        if tuple(sorted(by_timestep)) != EXPECTED_TIMESTEPS:
            raise ValueError(f"seed {seed} has an incomplete timestep grid")
        for timestep in EXPECTED_TIMESTEPS:
            row = by_timestep[timestep]
            if any(not math.isfinite(float(row[metric])) for metric in METRICS):
                raise ValueError(f"seed {seed}, timestep {timestep} contains a nonfinite metric")
            if row["calibration_oracle_error"] > row["calibration_rank1_error"] + 1e-10:
                raise ValueError(f"seed {seed}, timestep {timestep} violates calibration oracle ordering")
            if row["test_oracle_error"] > row["test_rank1_error"] + 1e-10:
                raise ValueError(f"seed {seed}, timestep {timestep} violates test oracle ordering")
            records.append({"seed": seed, **row})
    files = [{"path": str(path), "sha256": _sha256(path)} for path in paths]
    return records, files


def _aggregate(records: list[dict]) -> list[dict]:
    output = []
    for timestep in EXPECTED_TIMESTEPS:
        rows = sorted(
            (row for row in records if row["timestep"] == timestep),
            key=lambda row: row["seed"],
        )
        output.append({
            "timestep": timestep,
            "seeds": [row["seed"] for row in rows],
            **{metric: _mean_sd(row[metric] for row in rows) for metric in METRICS},
        })
    return output


def _plot(rows: list[dict], output_stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    blue, orange, ink, gray = "#326F95", "#C78343", "#26313B", "#747D84"
    for font in Path("/usr/share/fonts/opentype/urw-base35").glob("NimbusRoman-*.otf"):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({
        "font.family": "Nimbus Roman",
        "mathtext.fontset": "stix",
        "font.size": 7.2,
        "axes.titlesize": 7.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "text.color": ink,
        "axes.labelcolor": ink,
        "axes.edgecolor": "#A0A7AD",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.color": "#E6E9EC",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    x = [row["timestep"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.21, top=0.76, wspace=0.18)
    methods = (
        ("rank1", "Rank-1", blue, "o", "-"),
        ("diagonal", "Diagonal", orange, "s", "-"),
        ("oracle", "Oracle rank-1", gray, "^", "--"),
    )
    for axis, split, title, letter in zip(
        axes,
        ("calibration", "test"),
        ("Calibration scoring", "Held-out scoring"),
        "ab",
    ):
        for key, label, color, marker, linestyle in methods:
            groups = [row[f"{split}_{key}_error"] for row in rows]
            axis.errorbar(
                x,
                [group["mean"] for group in groups],
                yerr=[group["sample_sd"] for group in groups],
                color=color,
                marker=marker,
                linestyle=linestyle,
                capsize=2,
                elinewidth=0.8,
                label=label,
            )
        axis.axhline(1, color="#838B92", linestyle=":", linewidth=0.8)
        axis.set_xticks(x[::2])
        axis.grid(axis="y")
        axis.set_title(f"({letter}) {title}", loc="left", pad=6)
    axes[0].set_ylabel("Relative Frobenius error")
    fig.supxlabel("Diffusion timestep", fontsize=8, y=0.03)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.54, 1.01), ncol=3)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _write_table(rows: list[dict], output: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{MNIST UNet held-out relative Frobenius error.  Values are mean$\pm$sample SD over three independent training seeds; the final column counts seeds with lower rank-1 than diagonal error.}",
        r"\label{tab:mnist-unet-heldout}",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ccccc@{}}",
        r"\toprule",
        r"\textbf{$t$} & \textbf{rank-1} & \textbf{diagonal} & \textbf{oracle} & \textbf{rank-1 seeds} \\",
        r"\midrule",
    ]
    for row in rows:
        rank1 = row["test_rank1_error"]
        diagonal = row["test_diagonal_error"]
        oracle = row["test_oracle_error"]
        wins = sum(a < b for a, b in zip(rank1["values"], diagonal["values"]))
        lines.append(
            f'{row["timestep"]} & '
            f'${rank1["mean"]:.4f}\\pm{rank1["sample_sd"]:.4f}$ & '
            f'${diagonal["mean"]:.4f}\\pm{diagonal["sample_sd"]:.4f}$ & '
            f'${oracle["mean"]:.4f}\\pm{oracle["sample_sd"]:.4f}$ & '
            f'{wins}/3 \\\\'
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audits", type=Path, nargs="+")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure-stem", type=Path, required=True)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args(argv)

    records, files = _load(args.audits)
    aggregate = _aggregate(records)
    payload = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "mnist_unet_heldout_fisher_reconstruction_summary",
        "inputs": files,
        "seeds": list(EXPECTED_SEEDS),
        "timesteps": list(EXPECTED_TIMESTEPS),
        "aggregate": aggregate,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _plot(aggregate, args.figure_stem)
    if args.table:
        _write_table(aggregate, args.table)
    print(json.dumps({
        "status": "ok",
        "summary": str(args.summary),
        "figure": str(args.figure_stem),
        "table": str(args.table) if args.table else None,
    }))


if __name__ == "__main__":
    main()
