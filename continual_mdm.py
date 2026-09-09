"""Small, reproducible A -> B -> A study for SMDM masked diffusion LMs."""

from __future__ import annotations

import argparse
import json
import os
import random
import site
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


USER_SITE = site.getusersitepackages()
if USER_SITE in sys.path:
    sys.path.remove(USER_SITE)
os.environ.setdefault("PYTHONNOUSERSITE", "1")


ROOT = Path(__file__).resolve().parent
SMDM = ROOT / "SMDM"
if str(SMDM) not in sys.path:
    sys.path.insert(0, str(SMDM))

MASK_ID = 32000
RANKK_DIRECTION_BLOCK_SIZE = 16_777_216
# ponytail: 32K chunk lowers the largest temporary to ~0.5MiB for 1028M; raise only if profiling proves headroom.
RANKK_PROJECTION_CHUNK_SIZE = 32_768


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_examples(task: str, train_size: int, eval_size: int) -> tuple[list[dict], list[dict]]:
    if task == "arithmetic":
        pairs = [(i, 17 + (i * 7) % 83) for i in range(train_size + eval_size)]
        rows = [
            {
                "prompt": f"Question: What is {a} + {b}?\nAnswer:",
                "answer": f" {a + b}",
            }
            for a, b in pairs
        ]
    elif task == "reverse":
        rows = [
            {
                "prompt": f"Task: Reverse the word codeword{i:03d}.\nAnswer:",
                "answer": f" {f'codeword{i:03d}'[::-1]}",
            }
            for i in range(train_size + eval_size)
        ]
    else:
        raise ValueError(f"unknown task: {task}")
    return rows[:train_size], rows[train_size:]


def encode_rows(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    encoded = []
    for row in rows:
        prompt = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        answer = tokenizer(row["answer"], add_special_tokens=False)["input_ids"]
        ids = prompt + answer + [tokenizer.eos_token_id]
        if len(ids) <= max_length and answer:
            encoded.append({"ids": ids, "answer_start": len(prompt), "answer_end": len(prompt) + len(answer)})
    if not encoded:
        raise ValueError("all examples were removed by max_length")
    return encoded


def collate(rows: list[dict], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    width = max(len(row["ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    spans = []
    for i, row in enumerate(rows):
        length = len(row["ids"])
        ids[i, :length] = torch.tensor(row["ids"], dtype=torch.long)
        valid[i, :length] = True
        spans.append((row["answer_start"], row["answer_end"]))
    return ids, valid, spans


def noisy_batch(
    ids: torch.Tensor,
    valid: torch.Tensor,
    generator: torch.Generator,
    eps: float = 1e-3,
    mask_min: float = 1e-3,
    mask_max: float = 1.0,
    force_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, width = ids.shape
    t = mask_min + (mask_max - mask_min) * torch.rand(batch, device=ids.device, generator=generator)
    p_mask = ((1.0 - eps) * t + eps)[:, None].expand(batch, width)
    mask = (torch.rand((batch, width), device=ids.device, generator=generator) < p_mask) & valid
    if force_mask:
        for i in range(batch):
            if not mask[i].any():
                mask[i, valid[i].nonzero(as_tuple=False)[0, 0]] = True
    noisy = torch.where(mask, torch.full_like(ids, MASK_ID), ids)
    return noisy, mask, p_mask


def per_example_loss(
    model: torch.nn.Module,
    ids: torch.Tensor,
    valid: torch.Tensor,
    generator: torch.Generator,
    autocast: bool,
    mask_min: float = 1e-3,
    mask_max: float = 1.0,
    spans: list[tuple[int, int]] | None = None,
    answer_only: bool = False,
    normalization: str = "mean",
    force_mask: bool = True,
) -> torch.Tensor:
    if normalization not in ("mean", "sequence"):
        raise ValueError("normalization must be 'mean' or 'sequence'")
    noisy, mask, p_mask = noisy_batch(
        ids, valid, generator, mask_min=mask_min, mask_max=mask_max, force_mask=force_mask
    )
    loss_mask = mask
    denominator_mask = valid
    if answer_only:
        if spans is None:
            raise ValueError("answer_only loss requires answer spans")
        answer_valid = torch.zeros_like(valid)
        for index, (left, right) in enumerate(spans):
            answer_valid[index, left:right] = True
        loss_mask = mask & answer_valid
        denominator_mask = answer_valid
        if force_mask:
            for index, row_mask in enumerate(loss_mask):
                if not row_mask.any():
                    left, right = spans[index]
                    if left >= right:
                        raise ValueError("answer span must be non-empty")
                    loss_mask[index, left] = True
                    noisy[index, left] = MASK_ID
    with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16, enabled=autocast):
        logits = model(noisy)
        if loss_mask.any():
            token_loss = F.cross_entropy(logits[loss_mask], ids[loss_mask], reduction="none") / p_mask[loss_mask]
        else:
            token_loss = logits.new_empty((0,))
    losses = []
    offset = 0
    for index, row_mask in enumerate(loss_mask):
        count = int(row_mask.sum())
        if count:
            numerator = token_loss[offset : offset + count].sum()
            denominator = count if normalization == "mean" else int(denominator_mask[index].sum())
            losses.append(numerator / max(denominator, 1))
        else:
            losses.append(logits[index].reshape(-1)[0] * 0.0)
        offset += count
    return torch.stack(losses)


@torch.no_grad()
def evaluate_loss(model, rows, device, batch_size, seed, pad_id, mask_min=1e-3, mask_max=1.0) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    for start in range(0, len(rows), batch_size):
        ids, valid, _ = collate(rows[start : start + batch_size], pad_id=pad_id)
        losses.append(
            per_example_loss(
                model, ids.to(device), valid.to(device), generator, device.type == "cuda", mask_min, mask_max
            )
        )
    return float(torch.cat(losses).mean().cpu())


@torch.no_grad()
def answer_accuracy(model, rows, device, batch_size, pad_id) -> float:
    model.eval()
    correct = 0
    total = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        ids, valid, spans = collate(batch_rows, pad_id)
        ids, valid = ids.to(device), valid.to(device)
        masked = ids.clone()
        for i, (left, right) in enumerate(spans):
            masked[i, left:right] = MASK_ID
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predictions = model(masked).argmax(dim=-1)
        for i, (left, right) in enumerate(spans):
            correct += int(torch.equal(predictions[i, left:right], ids[i, left:right]))
            total += 1
    return correct / total


@torch.no_grad()
def answer_token_accuracy(model, rows, device, batch_size, pad_id) -> float:
    model.eval()
    correct = 0
    total = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        ids, valid, spans = collate(batch_rows, pad_id)
        ids, valid = ids.to(device), valid.to(device)
        masked = ids.clone()
        for i, (left, right) in enumerate(spans):
            masked[i, left:right] = MASK_ID
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predictions = model(masked).argmax(dim=-1)
        for i, (left, right) in enumerate(spans):
            target = ids[i, left:right]
            correct += int((predictions[i, left:right] == target).sum())
            total += right - left
    return correct / total


def trainable_parameters(model, mode: str) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    last = len(model.transformer.h) - 1
    # ponytail: final-block modes keep per-example Fisher cheap; all mode is an
    # explicit full-model experiment and needs enough host memory for gradients.
    if mode == "all":
        selected = list(model.parameters())
    elif mode == "last_mlp":
        names = {
            f"transformer.h.{last}.mlp.swiglu.w3.weight",
            f"transformer.h.{last}.norm_2.weight",
        }
    elif mode == "last_block":
        names = {name for name, _ in model.named_parameters() if name.startswith(f"transformer.h.{last}.")}
    else:
        raise ValueError(f"unknown trainable subspace: {mode}")
    if mode != "all":
        selected = [parameter for name, parameter in model.named_parameters() if name in names]
    if not selected:
        raise RuntimeError(f"no parameters selected for {mode}")
    for parameter in selected:
        parameter.requires_grad_(True)
    return selected


def flat_parameters(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([parameter.reshape(-1) for parameter in parameters])


def parameter_delta_projection(
    parameters, theta_ref, direction, direction_scales=None, block_size=RANKK_DIRECTION_BLOCK_SIZE
) -> torch.Tensor:
    result = theta_ref.new_zeros(())
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        local = 0
        while local < size:
            start = offset + local
            width = size - local
            if direction_scales is not None:
                width = min(width, block_size - start % block_size, RANKK_PROJECTION_CHUNK_SIZE)
            stop = start + width
            direction_slice = direction[start:stop]
            if direction_scales is not None:
                work_dtype = torch.float16
                delta = parameter.reshape(-1)[local : local + width].to(work_dtype) - theta_ref[start:stop]
                direction_slice = direction_slice.to(device=delta.device, dtype=work_dtype, non_blocking=True)
                scale = direction_scales[start // block_size].to(
                    device=delta.device, dtype=work_dtype, non_blocking=True
                )
                direction_slice = direction_slice * scale
                result = result + torch.sum(delta * direction_slice, dtype=torch.float32)
            else:
                delta = parameter.reshape(-1)[local : local + width] - theta_ref[start:stop]
                if direction_slice.device != delta.device or direction_slice.dtype != delta.dtype:
                    direction_slice = direction_slice.to(device=delta.device, dtype=delta.dtype, non_blocking=True)
                result = result + delta.dot(direction_slice)
            local += width
        offset += size
    return result


def parameter_diagonal_penalty(parameters, theta_ref, diagonal) -> torch.Tensor:
    result = theta_ref.new_zeros(())
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        delta = parameter.reshape(-1) - theta_ref[offset : offset + size]
        result = result + (diagonal[offset : offset + size] * delta.square()).sum()
        offset += size
    return result


def parameter_delta_stats(
    parameters,
    theta_ref,
    directions: dict[str, torch.Tensor],
    direction_scales: dict[str, torch.Tensor] | None = None,
    block_size=RANKK_DIRECTION_BLOCK_SIZE,
) -> dict:
    norm_sq = theta_ref.new_zeros(())
    abs_sum = theta_ref.new_zeros(())
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            size = parameter.numel()
            delta = parameter.detach().reshape(-1) - theta_ref[offset : offset + size]
            norm_sq = norm_sq + delta.square().sum()
            abs_sum = abs_sum + delta.abs().sum()
            offset += size
        projections = {
            name: parameter_delta_projection(
                parameters,
                theta_ref,
                direction,
                None if direction_scales is None else direction_scales.get(name),
                block_size,
            )
            for name, direction in directions.items()
            if direction is not None
        }
    count = max(offset, 1)
    return {
        "parameter_displacement_l2": float(norm_sq.sqrt().cpu()),
        "parameter_displacement_mean_abs": float((abs_sum / count).cpu()),
        **{
            f"parameter_displacement_{name}_projection": float(value.cpu())
            for name, value in projections.items()
            if directions[name] is not None
        },
        **{
            f"parameter_displacement_{name}_projection_abs": abs(float(value.cpu()))
            for name, value in projections.items()
            if directions[name] is not None
        },
    }


def parameter_rankk_penalty(
    parameters, theta_ref, directions, lambdas, direction_scales=None, block_size=RANKK_DIRECTION_BLOCK_SIZE
) -> torch.Tensor:
    result = theta_ref.new_zeros(())
    for index, (direction, coefficient) in enumerate(zip(directions, lambdas)):
        scales = None if direction_scales is None else direction_scales[index]
        projection = parameter_delta_projection(parameters, theta_ref, direction, scales, block_size)
        result = result + coefficient * projection.square()
    return 0.5 * result


@torch.no_grad()
def _rankk_direction_chunk(
    direction, direction_scales, start: int, stop: int, device: torch.device, work_dtype: torch.dtype, block_size: int
) -> torch.Tensor:
    chunk = direction[:, start:stop].to(device=device, dtype=work_dtype, non_blocking=True)
    if direction_scales is not None:
        scale = direction_scales[:, start // block_size].to(device=device, dtype=work_dtype, non_blocking=True)
        chunk.mul_(scale[:, None])
    return chunk


@torch.no_grad()
def add_parameter_direction_gradient_(
    parameters, direction, multiplier, direction_scales=None, block_size=RANKK_DIRECTION_BLOCK_SIZE
) -> None:
    offset = 0
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        grad = parameter.grad.reshape(-1)
        size = parameter.numel()
        local = 0
        while local < size:
            start = offset + local
            width = size - local
            if direction_scales is not None:
                width = min(width, block_size - start % block_size, RANKK_PROJECTION_CHUNK_SIZE)
            stop = start + width
            direction_slice = direction[start:stop].to(device=grad.device, dtype=grad.dtype, non_blocking=True)
            if direction_scales is not None:
                scale = direction_scales[start // block_size].to(
                    device=grad.device, dtype=grad.dtype, non_blocking=True
                )
                direction_slice.mul_(scale)
            grad[local : local + width].add_(direction_slice, alpha=float(multiplier))
            local += width
        offset += size


def parameter_rankk_penalty_and_add_gradient_(
    parameters,
    theta_ref,
    directions,
    lambdas,
    ewc_lambda,
    direction_scales=None,
    block_size=RANKK_DIRECTION_BLOCK_SIZE,
) -> torch.Tensor:
    rank = directions.shape[0]
    work_dtype = torch.float16 if direction_scales is not None else theta_ref.dtype
    projections = theta_ref.new_zeros(rank, dtype=torch.float32)
    offset = 0
    for parameter in parameters:
        flat = parameter.reshape(-1)
        size = parameter.numel()
        local = 0
        while local < size:
            start = offset + local
            width = size - local
            if direction_scales is not None:
                width = min(width, block_size - start % block_size, RANKK_PROJECTION_CHUNK_SIZE)
            stop = start + width
            delta = flat[local : local + width] - theta_ref[start:stop]
            if direction_scales is not None:
                delta = delta.to(work_dtype)
            direction_chunk = _rankk_direction_chunk(
                directions, direction_scales, start, stop, delta.device, work_dtype, block_size
            )
            projections.add_(torch.matmul(direction_chunk, delta.to(work_dtype)).to(torch.float32))
            local += width
        offset += size
    penalty = 0.5 * torch.dot(lambdas, projections.square())
    coeffs = ewc_lambda * lambdas * projections
    coeffs_work = coeffs.to(dtype=work_dtype)
    offset = 0
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        grad = parameter.grad.reshape(-1)
        size = parameter.numel()
        local = 0
        while local < size:
            start = offset + local
            width = size - local
            if direction_scales is not None:
                width = min(width, block_size - start % block_size, RANKK_PROJECTION_CHUNK_SIZE)
            stop = start + width
            direction_chunk = _rankk_direction_chunk(
                directions, direction_scales, start, stop, grad.device, work_dtype, block_size
            )
            grad_update = torch.matmul(coeffs_work, direction_chunk)
            grad[local : local + width].add_(grad_update)
            local += width
        offset += size
    return penalty


def save_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_cache(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def estimate_fisher(
    model,
    rows,
    parameters,
    device,
    batch_size,
    max_examples,
    pad_id,
    seed,
    mask_min,
    mask_max,
    answer_only=False,
):
    model.train()
    generator = torch.Generator(device=device).manual_seed(seed)
    vectors = []
    seen = 0
    for start in range(0, len(rows), batch_size):
        if seen >= max_examples:
            break
        batch_rows = rows[start : start + batch_size]
        ids, valid, spans = collate(batch_rows, pad_id)
        ids, valid = ids.to(device), valid.to(device)
        losses = per_example_loss(
            model, ids, valid, generator, device.type == "cuda", mask_min, mask_max,
            spans, answer_only,
        )
        for i, loss in enumerate(losses):
            if seen >= max_examples:
                break
            grads = torch.autograd.grad(loss, parameters, retain_graph=i + 1 < len(losses), allow_unused=True)
            vectors.append(torch.cat([
                (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1).detach().float().cpu()
                for parameter, gradient in zip(parameters, grads)
            ]))
            seen += 1
        model.zero_grad(set_to_none=True)
    if len(vectors) < 2:
        raise RuntimeError("Fisher estimation needs at least two examples")
    gradients = torch.stack(vectors)
    covariance = gradients @ gradients.T / gradients.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    left = eigenvectors[:, order[0]]
    top = gradients.T @ left
    top = top / top.norm().clamp_min(1e-12)
    mean_gradient = gradients.mean(dim=0)
    mean_direction = mean_gradient / mean_gradient.norm().clamp_min(1e-12)
    fisher_diag = gradients.square().mean(dim=0)
    trace = float(fisher_diag.sum())
    lambda1 = float(eigenvalues[0].clamp_min(0))
    lambda2 = float(eigenvalues[1].clamp_min(0))
    spectrum = eigenvalues.clamp_min(0)
    lambda_mean = float((gradients @ mean_direction).square().mean())
    full_frobenius_sq = float(spectrum.square().sum())
    diagonal_frobenius_sq = float(fisher_diag.square().sum())
    full_frobenius = max(full_frobenius_sq, 1e-24) ** 0.5
    cosine = gradients @ gradients.T
    norms = gradients.norm(dim=1).clamp_min(1e-12)
    cosine = cosine / (norms[:, None] * norms[None, :])
    pairwise = cosine[~torch.eye(len(gradients), dtype=torch.bool)]
    mean_cosine = (gradients @ mean_direction) / norms
    stats = {
        "fisher_examples": len(gradients),
        "fisher_dim": gradients.shape[1],
        "fisher_trace": trace,
        "fisher_lambda1": lambda1,
        "fisher_lambda2": lambda2,
        "fisher_lambda_mean": lambda_mean,
        "lambda1_over_trace": lambda1 / max(trace, 1e-12),
        "lambda2_over_lambda1": lambda2 / max(lambda1, 1e-12),
        "lambda_mean_over_trace": lambda_mean / max(trace, 1e-12),
        "rank1_relative_frobenius_error": max(full_frobenius_sq - lambda1**2, 0.0) ** 0.5 / full_frobenius,
        "mean_rank1_relative_frobenius_error": max(full_frobenius_sq - lambda_mean**2, 0.0) ** 0.5 / full_frobenius,
        "diagonal_relative_frobenius_error": max(full_frobenius_sq - diagonal_frobenius_sq, 0.0) ** 0.5 / full_frobenius,
        "gradient_pairwise_cosine": float(pairwise.mean()),
        "gradient_pairwise_cosine_abs": float(pairwise.abs().mean()),
        "gradient_mean_cosine": float(mean_cosine.mean()),
        "gradient_mean_cosine_abs": float(mean_cosine.abs().mean()),
        "top_mean_direction_cosine_abs": float(torch.abs(top.dot(mean_direction)).clamp(0.0, 1.0)),
        "fisher_spectrum": [float(value) for value in spectrum],
    }
    return {
        "u": top.to(device),
        "u_top": top.to(device),
        "u_mean": mean_direction.to(device),
        "diag": fisher_diag.to(device),
        "lambda1": lambda1,
        "lambda_mean": lambda_mean,
    }, stats


def train_stage(
    model,
    rows,
    parameters,
    device,
    batch_size,
    steps,
    lr,
    method,
    fisher,
    theta_ref,
    clip,
    seed,
    pad_id,
    ewc_lambda,
    mask_min=1e-3,
    mask_max=1.0,
    replay_rows=None,
    replay_weight: float = 0.0,
):
    model.train()
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    generator = torch.Generator(device=device).manual_seed(seed)
    replay_generator = torch.Generator(device=device).manual_seed(seed + 17)
    task_loss_sum = 0.0
    replay_loss_sum = 0.0
    penalty_sum = 0.0
    penalty_max = 0.0
    penalty_last = 0.0
    for step in range(steps):
        batch_rows = [rows[(step * batch_size + i) % len(rows)] for i in range(batch_size)]
        ids, valid, _ = collate(batch_rows, pad_id)
        ids, valid = ids.to(device), valid.to(device)
        losses = per_example_loss(model, ids, valid, generator, device.type == "cuda", mask_min, mask_max)
        task_loss = losses.mean()
        task_loss_sum += float(task_loss.detach().cpu())
        loss = task_loss
        replay_loss = None
        if replay_rows is not None and replay_weight > 0.0:
            replay_batch_rows = [
                replay_rows[(step * batch_size + i) % len(replay_rows)] for i in range(batch_size)
            ]
            replay_ids, replay_valid, _ = collate(replay_batch_rows, pad_id)
            replay_ids, replay_valid = replay_ids.to(device), replay_valid.to(device)
            replay_losses = per_example_loss(
                model, replay_ids, replay_valid, replay_generator, device.type == "cuda", mask_min, mask_max
            )
            replay_loss = replay_losses.mean()
            replay_loss_sum += float(replay_loss.detach().cpu())
            loss = loss + replay_weight * replay_loss
        manual_rankk = method == "rankk"
        if method != "plain" and not manual_rankk:
            if method in ("rank1", "rank1_low_snr"):
                direction = fisher["u_top"] if "u_top" in fisher else fisher["u"]
                coefficient = fisher["lambda1"]
                projection = parameter_delta_projection(parameters, theta_ref, direction)
                penalty = 0.5 * coefficient * projection ** 2
            elif method in ("mean_rank1", "mean_rank1_low_snr"):
                direction = fisher["u_mean"]
                projection = parameter_delta_projection(parameters, theta_ref, direction)
                penalty = 0.5 * fisher["lambda_mean"] * projection ** 2
            elif method == "diagonal":
                penalty = 0.5 * parameter_diagonal_penalty(parameters, theta_ref, fisher["diag"])
            elif method == "rankk":
                penalty = parameter_rankk_penalty(
                    parameters,
                    theta_ref,
                    fisher["u_top_k"],
                    fisher["lambda_k"],
                    fisher.get("u_top_k_scales"),
                    fisher.get("u_top_k_block_size", RANKK_DIRECTION_BLOCK_SIZE),
                )
            else:
                raise ValueError(f"unknown method: {method}")
            penalty_last = float(penalty.detach().cpu())
            penalty_sum += penalty_last
            penalty_max = max(penalty_max, penalty_last)
            loss = loss + ewc_lambda * penalty
        optimizer.zero_grad(set_to_none=True)
        if manual_rankk:
            loss.backward()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            penalty = parameter_rankk_penalty_and_add_gradient_(
                parameters,
                theta_ref,
                fisher["u_top_k"],
                fisher["lambda_k"],
                ewc_lambda,
                fisher.get("u_top_k_scales"),
                fisher.get("u_top_k_block_size", RANKK_DIRECTION_BLOCK_SIZE),
            )
            penalty_last = float(penalty.detach().cpu())
            penalty_sum += penalty_last
            penalty_max = max(penalty_max, penalty_last)
            loss = loss.detach() + ewc_lambda * penalty
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, clip)
        optimizer.step()
        if step == 0 or (step + 1) == steps or (step + 1) % max(steps // 5, 1) == 0:
            print(f"stage_step={step + 1}/{steps} loss={float(loss.detach().cpu()):.6f}", flush=True)
    return {
        "steps": steps,
        "task_loss_mean": task_loss_sum / max(steps, 1),
        "replay_loss_mean": replay_loss_sum / max(steps, 1),
        "ewc_penalty_mean": penalty_sum / max(steps, 1),
        "ewc_penalty_max": penalty_max,
        "ewc_penalty_last": penalty_last,
        "ewc_loss_term_mean": ewc_lambda * penalty_sum / max(steps, 1),
        "ewc_to_task_loss_ratio": (ewc_lambda * penalty_sum) / max(task_loss_sum, 1e-12),
    }


def load_model(args, device):
    from lit_gpt.config import Config
    from lit_gpt.diffmodel import TransEncoder

    config = Config.from_name(f"Diff_LLaMA_{args.model}M")
    model = TransEncoder(config)
    model.load_state_dict(load_file(str(args.checkpoint), device="cpu"))
    return model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else None)


def run(args) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_float32_matmul_precision("high")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    a_train_raw, a_eval_raw = task_examples("arithmetic", args.train_size, args.eval_size)
    b_train_raw, b_eval_raw = task_examples("reverse", args.train_size, args.eval_size)
    a_train = encode_rows(a_train_raw, tokenizer, args.max_length)
    a_eval = encode_rows(a_eval_raw, tokenizer, args.max_length)
    b_train = encode_rows(b_train_raw, tokenizer, args.max_length)
    b_eval = encode_rows(b_eval_raw, tokenizer, args.max_length)
    for rows in (a_train, a_eval, b_train, b_eval):
        for row in rows:
            row["pad_id"] = pad_id

    model = load_model(args, device)
    parameters = trainable_parameters(model, args.trainable)
    print(f"device={device} trainable={sum(p.numel() for p in parameters):,}", flush=True)

    train_stage(
        model,
        a_train,
        parameters,
        device,
        args.batch_size,
        args.a_steps,
        args.lr,
        "plain",
        None,
        None,
        args.clip,
        args.seed + 1,
        pad_id,
        0.0,
    )
    a_after_a = evaluate_loss(model, a_eval, device, args.batch_size, args.seed + 11, pad_id)
    b_before_b = evaluate_loss(model, b_eval, device, args.batch_size, args.seed + 12, pad_id)
    a_answer_after_a = answer_accuracy(model, a_eval, device, args.batch_size, pad_id)
    b_answer_before_b = answer_accuracy(model, b_eval, device, args.batch_size, pad_id)
    a_token_after_a = answer_token_accuracy(model, a_eval, device, args.batch_size, pad_id)
    b_token_before_b = answer_token_accuracy(model, b_eval, device, args.batch_size, pad_id)
    theta_ref = flat_parameters(parameters).detach().clone()
    fisher_mask_min = args.fisher_mask_min
    fisher_mask_max = args.fisher_mask_max
    if args.method in ("rank1_low_snr", "mean_rank1_low_snr") and fisher_mask_min is None:
        fisher_mask_min = 0.8
    if fisher_mask_min is None:
        fisher_mask_min = 1e-3
    if fisher_mask_max is None:
        fisher_mask_max = 1.0
    if not 0.0 <= fisher_mask_min < fisher_mask_max <= 1.0:
        raise ValueError("fisher mask window must satisfy 0 <= min < max <= 1")
    if args.method == "plain":
        fisher = None
        fisher_stats = {}
    else:
        fisher, fisher_stats = estimate_fisher(
            model,
            a_train,
            parameters,
            device,
            args.fisher_batch_size,
            args.fisher_examples,
            pad_id,
            args.seed + 21,
            fisher_mask_min,
            fisher_mask_max,
            args.fisher_answer_only,
        )
    train_stage(
        model,
        b_train,
        parameters,
        device,
        args.batch_size,
        args.b_steps,
        args.lr,
        args.method,
        fisher,
        theta_ref,
        args.clip,
        args.seed + 2,
        pad_id,
        args.ewc_lambda,
    )
    a_after_b = evaluate_loss(model, a_eval, device, args.batch_size, args.seed + 11, pad_id)
    b_after_b = evaluate_loss(model, b_eval, device, args.batch_size, args.seed + 12, pad_id)
    a_answer_after_b = answer_accuracy(model, a_eval, device, args.batch_size, pad_id)
    b_answer_after_b = answer_accuracy(model, b_eval, device, args.batch_size, pad_id)
    a_token_after_b = answer_token_accuracy(model, a_eval, device, args.batch_size, pad_id)
    b_token_after_b = answer_token_accuracy(model, b_eval, device, args.batch_size, pad_id)
    result = {
        "method": args.method,
        "seed": args.seed,
        "model": f"Diff_LLaMA_{args.model}M",
        "trainable": args.trainable,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "a_loss_after_a": a_after_a,
        "a_loss_after_b": a_after_b,
        "b_loss_before_b": b_before_b,
        "b_loss_after_b": b_after_b,
        "a_loss_relative_increase": (a_after_b - a_after_a) / max(abs(a_after_a), 1e-12),
        "b_loss_relative_decrease": (b_before_b - b_after_b) / max(abs(b_before_b), 1e-12),
        "a_answer_exact_after_a": a_answer_after_a,
        "a_answer_exact_after_b": a_answer_after_b,
        "b_answer_exact_before_b": b_answer_before_b,
        "b_answer_exact_after_b": b_answer_after_b,
        "a_answer_token_accuracy_after_a": a_token_after_a,
        "a_answer_token_accuracy_after_b": a_token_after_b,
        "b_answer_token_accuracy_before_b": b_token_before_b,
        "b_answer_token_accuracy_after_b": b_token_after_b,
        "ewc_lambda": 0.0 if args.method == "plain" else args.ewc_lambda,
        "fisher_mask_min": fisher_mask_min,
        "fisher_mask_max": fisher_mask_max,
        **fisher_stats,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def self_check() -> None:
    ids = torch.tensor([[1, 2, 3], [4, 5, 2]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    generator = torch.Generator().manual_seed(3)
    noisy, mask, p_mask = noisy_batch(ids, valid, generator)
    assert noisy.shape == ids.shape and mask.shape == ids.shape
    assert torch.all(mask <= valid)
    assert torch.all(mask.any(dim=1))
    assert torch.all((p_mask > 0) & (p_mask <= 1))
    parameters = [torch.nn.Parameter(torch.tensor([1.0, 2.0]))]
    theta_ref = torch.zeros(2)
    directions = torch.eye(2)
    lambdas = torch.tensor([2.0, 3.0])
    penalty = parameter_rankk_penalty(parameters, theta_ref, directions, lambdas)
    penalty.backward()
    assert abs(float(penalty) - 7.0) < 1e-6
    assert torch.allclose(parameters[0].grad, torch.tensor([2.0, 6.0]))
    scaled_projection = parameter_delta_projection(
        parameters, theta_ref, torch.tensor([0.5, 0.25]), torch.tensor([2.0]), 2
    )
    assert abs(float(scaled_projection) - 2.0) < 1e-6

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(MASK_ID + 1, 4)
            self.proj = torch.nn.Linear(4, MASK_ID + 1)

        def forward(self, x):
            return self.proj(self.emb(x))

    toy_model = ToyModel()
    toy_rows = [
        {"ids": [1, 2, 3], "answer_start": 1, "answer_end": 2},
        {"ids": [4, 5, 6], "answer_start": 1, "answer_end": 2},
    ]
    toy_stats = train_stage(
        toy_model,
        toy_rows,
        list(toy_model.parameters()),
        torch.device("cpu"),
        1,
        1,
        1e-3,
        "plain",
        None,
        None,
        1.0,
        7,
        0,
        0.0,
        replay_rows=toy_rows,
        replay_weight=0.5,
    )
    assert toy_stats["steps"] == 1
    assert toy_stats["replay_loss_mean"] >= 0.0
    print("self-check: ok")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--model", type=int, default=170)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--method",
        choices=("plain", "rank1", "rank1_low_snr", "mean_rank1", "mean_rank1_low_snr", "diagonal"),
        default="plain",
    )
    parser.add_argument("--trainable", choices=("all", "last_mlp", "last_block"), default="last_mlp")
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--eval-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fisher-batch-size", type=int, default=2)
    parser.add_argument("--fisher-examples", type=int, default=16)
    parser.add_argument("--a-steps", type=int, default=100)
    parser.add_argument("--b-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ewc-lambda", type=float, default=1e4)
    parser.add_argument("--fisher-mask-min", type=float)
    parser.add_argument("--fisher-mask-max", type=float)
    parser.add_argument("--fisher-answer-only", action="store_true")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        run(args)
