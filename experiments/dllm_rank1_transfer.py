#!/usr/bin/env python3
"""Faithful two-task transfer of rank-1 EWC + generative distillation to SMDM."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
import re
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from continual_benchmark import MASK_ID, diff_generate_batch, encode_benchmark_rows  # noqa: E402
from continual_mdm import (  # noqa: E402
    answer_token_accuracy,
    flat_parameters,
    load_model,
    set_seed,
    trainable_parameters,
)
from continual_reverse import fact_rows, measure  # noqa: E402


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable_args(args) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _cache_paths(prefix: Path) -> dict[str, Path]:
    base = str(prefix)
    return {
        "state": Path(base + ".state.pt"),
        "rank1": Path(base + ".rank1.pt"),
        "diagonal": Path(base + ".diagonal.pt"),
        "replay": Path(base + ".replay.json"),
        "summary": Path(base + ".prepare.json"),
    }


def _protocol_metadata(args) -> dict:
    return {
        "mask_sampling": "independent_bernoulli_allow_empty_v1",
        "minibatch_sampling": "separate_current_replay_rng_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "tokenizer": str(args.tokenizer.resolve()),
        "reverse_dir": str(args.reverse_dir.resolve()),
        "model": args.model,
        "task_a": args.task_a,
        "task_b": args.task_b,
        "a_group_start": args.a_group_start,
        "b_group_start": args.b_group_start,
        "group_count": args.group_count,
        "calibration_per_fact": args.calibration_per_fact,
        "fisher_source": "task_a_training_examples",
        "trainable": args.trainable,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "a_steps": args.a_steps,
        "lr": args.lr,
        "mask_min": args.mask_min,
        "mask_max": args.mask_max,
        "seed": args.seed,
        "replay_prompts": args.replay_prompts,
        "replay_steps": args.replay_steps,
        "replay_length": args.replay_length,
        "replay_cfg": args.replay_cfg,
        "replay_temperature": args.replay_temperature,
    }


def _split_calibration(rows: list[dict], per_fact: int) -> tuple[list[dict], list[dict]]:
    if per_fact < 1:
        raise ValueError("calibration_per_fact must be positive")
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["fact_id"]), []).append(row)
    train, calibration = [], []
    for fact_id in sorted(grouped):
        group = sorted(grouped[fact_id], key=lambda row: int(row["template_id"]))
        if len(group) <= per_fact:
            raise ValueError(f"fact {fact_id} has too few rows for a held-out calibration split")
        train.extend(group[:-per_fact])
        calibration.extend(group[-per_fact:])
    return train, calibration


def _collate(rows: list[dict], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row["ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    answer = torch.zeros((len(rows), width), dtype=torch.bool)
    valid = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        length = len(row["ids"])
        left, right = int(row["answer_start"]), int(row["answer_end"])
        if not 0 <= left < right <= length:
            raise ValueError("invalid answer span")
        ids[index, :length] = torch.tensor(row["ids"], dtype=torch.long)
        valid[index, :length] = True
        answer[index, left:right] = True
    return ids, valid, answer


def _corrupt_answers(
    ids: torch.Tensor,
    answer: torch.Tensor,
    generator: torch.Generator,
    mask_min: float,
    mask_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, width = ids.shape
    t = mask_min + (mask_max - mask_min) * torch.rand(batch, device=ids.device, generator=generator)
    probability = t[:, None].expand(batch, width)
    mask = (torch.rand((batch, width), device=ids.device, generator=generator) < probability) & answer
    noisy = ids.masked_fill(mask, MASK_ID)
    return noisy, mask, probability


def _reduce_tokens(values: torch.Tensor, mask: torch.Tensor, probability: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
    losses = []
    offset = 0
    for index in range(mask.shape[0]):
        count = int(mask[index].sum())
        denominator = max(int(answer[index].sum()), 1)
        weighted = values[offset : offset + count] / probability[index][mask[index]]
        losses.append(weighted.sum() / denominator)
        offset += count
    return torch.stack(losses)


def _sft_losses(model, rows, pad_id, device, generator, mask_min, mask_max) -> torch.Tensor:
    ids, _, answer = _collate(rows, pad_id)
    ids, answer = ids.to(device), answer.to(device)
    noisy, mask, probability = _corrupt_answers(ids, answer, generator, mask_min, mask_max)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(noisy)
        token_losses = F.cross_entropy(logits[mask].float(), ids[mask], reduction="none")
    return _reduce_tokens(token_losses, mask, probability, answer)


def _distillation_losses(student, teacher, rows, pad_id, device, generator, mask_min, mask_max, temperature) -> torch.Tensor:
    ids, _, answer = _collate(rows, pad_id)
    ids, answer = ids.to(device), answer.to(device)
    noisy, mask, probability = _corrupt_answers(ids, answer, generator, mask_min, mask_max)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        teacher_logits = teacher(noisy)[mask].float() / temperature
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        student_logits = student(noisy)[mask].float() / temperature
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        token_kl = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True).sum(dim=-1)
        token_kl = token_kl * (temperature * temperature)
    return _reduce_tokens(token_kl, mask, probability, answer)


def _flat_gradient(loss, parameters, retain_graph=False) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([
        (gradient if gradient is not None else torch.zeros_like(parameter)).detach().float().cpu().reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def _fisher_pass(model, rows, parameters, pad_id, device, seed, mask_min, mask_max, projection=None):
    generator = torch.Generator(device=device).manual_seed(seed)
    gradient_sum = diagonal_sum = None
    projected_squares = []
    losses = []
    model.eval()
    for index, row in enumerate(rows):
        per_example = _sft_losses(model, [row], pad_id, device, generator, mask_min, mask_max)[0]
        vector = _flat_gradient(per_example, parameters)
        losses.append(float(per_example.detach().cpu()))
        if projection is None:
            if gradient_sum is None:
                gradient_sum = torch.zeros_like(vector)
                diagonal_sum = torch.zeros_like(vector)
            gradient_sum.add_(vector)
            diagonal_sum.addcmul_(vector, vector)
        else:
            projected_squares.append(float(torch.dot(vector, projection).square()))
        model.zero_grad(set_to_none=True)
        print(f"fisher_example={index + 1}/{len(rows)} pass={'projection' if projection is not None else 'moments'}", flush=True)
    if projection is None:
        return gradient_sum, diagonal_sum, losses
    return projected_squares, losses


def _estimate_mean_and_diagonal_fisher(model, rows, parameters, pad_id, device, args) -> tuple[dict, dict]:
    if len(rows) < 2:
        raise ValueError("Fisher calibration requires at least two held-out examples")
    gradient_sum, diagonal_sum, first_losses = _fisher_pass(
        model, rows, parameters, pad_id, device, args.seed + 2101, args.mask_min, args.mask_max
    )
    count = len(rows)
    mean = gradient_sum / count
    mean_norm = float(mean.norm())
    direction = mean / max(mean_norm, 1e-30)
    diagonal = diagonal_sum / count
    projected_squares, second_losses = _fisher_pass(
        model, rows, parameters, pad_id, device, args.seed + 2101,
        args.mask_min, args.mask_max, projection=direction,
    )
    coefficient = sum(projected_squares) / count
    stats = {
        "calibration_examples": count,
        "source": "task_a_training_examples",
        "parameter_count": int(mean.numel()),
        "mean_gradient_norm": mean_norm,
        "rank1_coefficient": coefficient,
        "diagonal_trace": float(diagonal.sum()),
        "calibration_loss_mean": sum(first_losses) / count,
        "repeat_loss_max_abs_difference": max(abs(a - b) for a, b in zip(first_losses, second_losses)),
        "normalization": "sum(masked token loss / p) divided by answer length",
    }
    del gradient_sum, diagonal_sum, mean
    return {"direction": direction, "coefficient": coefficient, "diagonal": diagonal}, stats


def _projection(parameters, reference, direction, chunk_size=1_048_576) -> torch.Tensor:
    result = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    offset = 0
    for parameter in parameters:
        flat = parameter.reshape(-1)
        for local in range(0, flat.numel(), chunk_size):
            width = min(chunk_size, flat.numel() - local)
            delta = flat[local : local + width].float() - reference[offset + local : offset + local + width].float()
            result = result + torch.dot(delta, direction[offset + local : offset + local + width])
        offset += flat.numel()
    return result


def _diagonal_penalty(parameters, reference, diagonal, chunk_size=1_048_576) -> torch.Tensor:
    result = torch.zeros((), device=parameters[0].device, dtype=torch.float32)
    offset = 0
    for parameter in parameters:
        flat = parameter.reshape(-1)
        for local in range(0, flat.numel(), chunk_size):
            width = min(chunk_size, flat.numel() - local)
            delta = flat[local : local + width].float() - reference[offset + local : offset + local + width].float()
            result = result + torch.sum(diagonal[offset + local : offset + local + width] * delta.square())
        offset += flat.numel()
    return 0.5 * result


def _train(
    model,
    rows,
    parameters,
    pad_id,
    device,
    args,
    steps,
    seed,
    teacher=None,
    replay_rows=None,
    reference=None,
    fisher=None,
    ewc_kind=None,
    ewc_lambda=0.0,
) -> dict:
    started = time.monotonic()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    current_rng = random.Random(seed)
    replay_rng = random.Random(seed + 303)
    current_generator = torch.Generator(device=device).manual_seed(seed + 101)
    replay_generator = torch.Generator(device=device).manual_seed(seed + 202)
    totals = {"current": 0.0, "distill": 0.0, "penalty": 0.0, "total": 0.0}
    maxima = {"penalty": 0.0, "gradient_norm": 0.0}
    clipped_steps = 0
    model.train()
    for step in range(steps):
        batch = [rows[current_rng.randrange(len(rows))] for _ in range(args.batch_size)]
        current_loss = _sft_losses(
            model, batch, pad_id, device, current_generator, args.mask_min, args.mask_max
        ).mean()
        distill_loss = torch.zeros((), device=device)
        if teacher is not None:
            replay_batch = [replay_rows[replay_rng.randrange(len(replay_rows))] for _ in range(args.batch_size)]
            distill_loss = _distillation_losses(
                model, teacher, replay_batch, pad_id, device, replay_generator,
                args.mask_min, args.mask_max, args.distill_temperature,
            ).mean()
        penalty = torch.zeros((), device=device)
        if ewc_kind == "rank1":
            projection = _projection(parameters, reference, fisher["direction"])
            penalty = 0.5 * fisher["coefficient"] * projection.square()
        elif ewc_kind == "diagonal":
            penalty = _diagonal_penalty(parameters, reference, fisher["diagonal"])
        total = current_loss + args.distill_weight * distill_loss + ewc_lambda * penalty
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.clip))
        clipped_steps += gradient_norm > args.clip
        optimizer.step()
        values = {
            "current": float(current_loss.detach().cpu()),
            "distill": float(distill_loss.detach().cpu()),
            "penalty": float(penalty.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        for key, value in values.items():
            totals[key] += value
        maxima["penalty"] = max(maxima["penalty"], values["penalty"])
        maxima["gradient_norm"] = max(maxima["gradient_norm"], gradient_norm)
        if step == 0 or step + 1 == steps or (step + 1) % max(steps // 10, 1) == 0:
            print(
                f"train_step={step + 1}/{steps} current={values['current']:.5f} "
                f"distill={values['distill']:.5f} penalty={values['penalty']:.6g} total={values['total']:.5f}",
                flush=True,
            )
    return {
        "steps": steps,
        "wall_time_seconds": time.monotonic() - started,
        **{f"{key}_mean": value / max(steps, 1) for key, value in totals.items()},
        "penalty_max": maxima["penalty"],
        "gradient_norm_max": maxima["gradient_norm"],
        "clip_fraction": clipped_steps / max(steps, 1),
        "ewc_loss_mean": ewc_lambda * totals["penalty"] / max(steps, 1),
        "distill_loss_weighted_mean": args.distill_weight * totals["distill"] / max(steps, 1),
    }


@torch.no_grad()
def _evaluate_loss(model, rows, pad_id, device, args, seed) -> float:
    model.eval()
    values = []
    for repeat in range(args.eval_mc_samples):
        generator = torch.Generator(device=device).manual_seed(seed + repeat)
        for start in range(0, len(rows), args.eval_batch_size):
            values.extend(
                float(value)
                for value in _sft_losses(
                    model, rows[start : start + args.eval_batch_size], pad_id, device,
                    generator, args.mask_min, args.mask_max,
                ).cpu()
            )
    return sum(values) / len(values)


def _generate_replay(model, tokenizer, prompts, device, args) -> list[dict]:
    rows = []
    model.eval()
    for index, prompt in enumerate(prompts):
        set_seed(args.seed + 700_000 + index)
        output, _ = diff_generate_batch(
            model, tokenizer, [prompt], device, args.replay_steps, args.replay_length,
            args.replay_cfg, args.replay_temperature,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        generated = output[0].detach().cpu().tolist()
        end = len(generated)
        if tokenizer.eos_token_id in generated[len(prompt_ids) :]:
            end = len(prompt_ids) + generated[len(prompt_ids) :].index(tokenizer.eos_token_id) + 1
        if end <= len(prompt_ids):
            raise RuntimeError("teacher generated an empty replay completion")
        rows.append({
            "ids": generated[:end],
            "answer_start": len(prompt_ids),
            "answer_end": end,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        })
        print(f"replay_generated={index + 1}/{len(prompts)} tokens={end - len(prompt_ids)}", flush=True)
    return rows


def _load_data(args, tokenizer):
    if args.task_a == args.task_b:
        raise ValueError("task_a and task_b must differ")
    a_all = fact_rows(args.reverse_dir, args.task_a, "train", args.a_group_start, args.group_count)
    b_raw = fact_rows(args.reverse_dir, args.task_b, "train", args.b_group_start, args.group_count)
    _, calibration_raw = _split_calibration(a_all, args.calibration_per_fact)
    a_train_raw = a_all
    a_eval = fact_rows(args.reverse_dir, args.task_a, "test", args.a_group_start, args.group_count)
    b_eval = fact_rows(args.reverse_dir, args.task_b, "test", args.b_group_start, args.group_count)
    return {
        "a_train_raw": a_train_raw,
        "calibration_raw": calibration_raw,
        "b_train_raw": b_raw,
        "a_train": encode_benchmark_rows(a_train_raw, tokenizer, args.max_length),
        "calibration": encode_benchmark_rows(calibration_raw, tokenizer, args.max_length),
        "b_train": encode_benchmark_rows(b_raw, tokenizer, args.max_length),
        "a_eval": a_eval,
        "b_eval": b_eval,
    }


def _measure_all(model, tokenizer, data, pad_id, device, args, suffix) -> dict:
    a_generation, a_items = measure(model, tokenizer, data["a_eval"], args, device)
    b_generation, b_items = measure(model, tokenizer, data["b_eval"], args, device)
    a_loss_rows = encode_benchmark_rows(
        [{"prompt": row["prompt"], "answer": row["target"]} for row in data["a_eval"]],
        tokenizer,
        args.max_length,
    )
    b_loss_rows = encode_benchmark_rows(
        [{"prompt": row["prompt"], "answer": row["target"]} for row in data["b_eval"]],
        tokenizer,
        args.max_length,
    )
    a_generation.update(_overlap_metrics(a_items))
    b_generation.update(_overlap_metrics(b_items))
    return {
        f"a_generation_{suffix}": a_generation,
        f"b_generation_{suffix}": b_generation,
        f"a_generation_items_{suffix}": a_items,
        f"b_generation_items_{suffix}": b_items,
        f"a_answer_token_accuracy_{suffix}": answer_token_accuracy(
            model, a_loss_rows, device, args.eval_batch_size, pad_id
        ),
        f"b_answer_token_accuracy_{suffix}": answer_token_accuracy(
            model, b_loss_rows, device, args.eval_batch_size, pad_id
        ),
        f"a_loss_{suffix}": _evaluate_loss(model, a_loss_rows, pad_id, device, args, args.seed + 8101),
        f"b_loss_{suffix}": _evaluate_loss(model, b_loss_rows, pad_id, device, args, args.seed + 8201),
    }


def _overlap_metrics(items: list[dict]) -> dict:
    rouge_l, recalls = [], []
    for item in items:
        target = re.findall(r"\w+", item["target"].lower())
        prediction = re.findall(r"\w+", item["prediction"].lower())
        # ponytail: O(n*m) LCS is simplest and bounded by the 52-token generation cap.
        previous = [0] * (len(prediction) + 1)
        for target_token in target:
            current = [0]
            for index, prediction_token in enumerate(prediction, 1):
                current.append(
                    previous[index - 1] + 1
                    if target_token == prediction_token
                    else max(previous[index], current[-1])
                )
            previous = current
        overlap = previous[-1]
        precision = overlap / max(len(prediction), 1)
        recall = overlap / max(len(target), 1)
        rouge_l.append(2 * precision * recall / max(precision + recall, 1e-30))
        recalls.append(recall)
    return {
        "rouge_l_f1": sum(rouge_l) / len(rouge_l),
        "lcs_recall": sum(recalls) / len(recalls),
    }


def _validate_cache_metadata(actual: dict, expected: dict):
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError("cache metadata mismatch: " + ", ".join(mismatches))


def prepare(args) -> dict:
    started = time.monotonic()
    set_seed(args.seed)
    device = torch.device(args.device)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    data = _load_data(args, tokenizer)
    model = load_model(args, device)
    parameters = trainable_parameters(model, args.trainable)
    print(f"prepare trainable_parameters={sum(p.numel() for p in parameters):,}", flush=True)
    a_training = _train(
        model, data["a_train"], parameters, pad_id, device, args, args.a_steps, args.seed + 1
    )
    baseline_metrics = _measure_all(model, tokenizer, data, pad_id, device, args, "after_a")
    fisher, fisher_stats = _estimate_mean_and_diagonal_fisher(
        model, data["calibration"], parameters, pad_id, device, args
    )
    prompt_pool = [row["prompt"] for row in data["a_train_raw"]]
    prompts = [prompt_pool[index % len(prompt_pool)] for index in range(args.replay_prompts)]
    replay = _generate_replay(model, tokenizer, prompts, device, args)
    paths = _cache_paths(args.cache_prefix)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _protocol_metadata(args)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"kind": "dllm_transfer_state_v1", "metadata": metadata, "state_dict": state}, paths["state"])
    torch.save({
        "kind": "dllm_transfer_rank1_v1", "metadata": metadata,
        "direction": fisher["direction"], "coefficient": fisher["coefficient"], "stats": fisher_stats,
    }, paths["rank1"])
    torch.save({
        "kind": "dllm_transfer_diagonal_v1", "metadata": metadata,
        "diagonal": fisher["diagonal"], "stats": fisher_stats,
    }, paths["diagonal"])
    paths["replay"].write_text(json.dumps({
        "kind": "dllm_transfer_replay_v1", "metadata": metadata, "rows": replay,
    }, indent=2) + "\n")
    summary = {
        "schema_version": 2,
        "status": "ok",
        "experiment": "dllm_rank1_transfer_prepare",
        "created_utc": _utc_now(),
        "wall_time_seconds": time.monotonic() - started,
        "host": __import__("os").uname().nodename,
        "metadata": metadata,
        "a_training": a_training,
        "fisher": fisher_stats,
        "replay_examples": len(replay),
        "replay_answer_tokens_mean": sum(row["answer_end"] - row["answer_start"] for row in replay) / len(replay),
        "cache_paths": {key: str(value) for key, value in paths.items()},
        **baseline_metrics,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    if args.prepare_output:
        args.prepare_output.parent.mkdir(parents=True, exist_ok=True)
        args.prepare_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_method(args) -> dict:
    started = time.monotonic()
    set_seed(args.seed)
    device = torch.device(args.device)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    pad_id = int(tokenizer.eos_token_id)
    data = _load_data(args, tokenizer)
    paths = _cache_paths(args.cache_prefix)
    expected = _protocol_metadata(args)
    state_payload = torch.load(paths["state"], map_location="cpu", weights_only=True)
    _validate_cache_metadata(state_payload["metadata"], expected)
    model = load_model(args, device)
    model.load_state_dict(state_payload["state_dict"])
    parameters = trainable_parameters(model, args.trainable)
    reference = flat_parameters(parameters).detach().clone()
    uses_gd = args.method in ("gd", "diag_gd", "rank1_gd")
    teacher = replay_rows = None
    if uses_gd:
        teacher = load_model(args, device)
        teacher.load_state_dict(state_payload["state_dict"])
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        replay_payload = json.loads(paths["replay"].read_text())
        _validate_cache_metadata(replay_payload["metadata"], expected)
        replay_rows = replay_payload["rows"]
    del state_payload
    ewc_kind = None
    fisher = None
    ewc_lambda = args.ewc_lambda
    if args.method in ("rank1", "rank1_gd"):
        payload = torch.load(paths["rank1"], map_location="cpu", weights_only=True)
        _validate_cache_metadata(payload["metadata"], expected)
        fisher = {"direction": payload["direction"].to(device), "coefficient": float(payload["coefficient"])}
        ewc_kind = "rank1"
        if args.ewc_target_stiffness is not None:
            ewc_lambda = args.ewc_target_stiffness / max(fisher["coefficient"], 1e-30)
    elif args.method in ("diagonal", "diag_gd"):
        payload = torch.load(paths["diagonal"], map_location="cpu", weights_only=True)
        _validate_cache_metadata(payload["metadata"], expected)
        fisher = {"diagonal": payload["diagonal"].to(device)}
        ewc_kind = "diagonal"
    before_b = _measure_all(model, tokenizer, data, pad_id, device, args, "after_a")
    training = _train(
        model, data["b_train"], parameters, pad_id, device, args, args.b_steps, args.seed + 2,
        teacher=teacher, replay_rows=replay_rows, reference=reference, fisher=fisher,
        ewc_kind=ewc_kind, ewc_lambda=ewc_lambda,
    )
    after_b = _measure_all(model, tokenizer, data, pad_id, device, args, "after_b")
    prepare_summary = json.loads(paths["summary"].read_text())
    result = {
        "schema_version": 2,
        "status": "ok",
        "experiment": "dllm_rank1_transfer",
        "created_utc": _utc_now(),
        "wall_time_seconds": time.monotonic() - started,
        "host": __import__("os").uname().nodename,
        "method": args.method,
        "config": _jsonable_args(args),
        "metadata": expected,
        "source_sha256": _sha256(Path(__file__)),
        "a_training": prepare_summary["a_training"],
        "b_training": training,
        "fisher": prepare_summary["fisher"],
        **before_b,
        **after_b,
        "ewc_lambda_effective": ewc_lambda if ewc_kind else 0.0,
        "ewc_effective_stiffness": (
            ewc_lambda * fisher["coefficient"] if ewc_kind == "rank1" else None
        ),
    }
    result["b_generation_before_b"] = result["b_generation_after_a"]
    result["b_generation_items_before_b"] = result["b_generation_items_after_a"]
    result["b_answer_token_accuracy_before_b"] = result["b_answer_token_accuracy_after_a"]
    result["b_loss_before_b"] = result["b_loss_after_a"]
    result["a_generation_forgetting"] = (
        result["a_generation_after_a"]["accuracy"] - result["a_generation_after_b"]["accuracy"]
    )
    result["a_loss_forgetting"] = result["a_loss_after_b"] - result["a_loss_after_a"]
    result["b_generation_gain"] = (
        result["b_generation_after_b"]["accuracy"] - result["b_generation_before_b"]["accuracy"]
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def _self_check():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    reference = torch.tensor([1.0, 2.0])
    direction = torch.tensor([1.0, 0.0])
    diagonal = torch.tensor([2.0, 3.0])
    assert float(_projection([parameter], reference, direction)) == 0.0
    assert float(_diagonal_penalty([parameter], reference, diagonal)) == 0.0
    with torch.no_grad():
        parameter[0] += 0.5
    assert math.isclose(float(_projection([parameter], reference, direction)), 0.5)
    assert math.isclose(float(_diagonal_penalty([parameter], reference, diagonal)), 0.25)
    logits = torch.tensor([[1.0, 2.0]])
    log_probs = F.log_softmax(logits, dim=-1)
    assert abs(float(F.kl_div(log_probs, log_probs, log_target=True, reduction="sum"))) < 1e-7
    train, calibration = _split_calibration([
        {"fact_id": 0, "template_id": index} for index in range(4)
    ], 1)
    assert len(train) == 3 and len(calibration) == 1
    overlap = _overlap_metrics([{"target": "a b c", "prediction": "a x c"}])
    assert math.isclose(overlap["rouge_l_f1"], 2 / 3)
    assert math.isclose(overlap["lcs_recall"], 2 / 3)
    trials = 50_000
    ids = torch.zeros((trials, 3), dtype=torch.long)
    answer = torch.ones_like(ids, dtype=torch.bool)
    generator = torch.Generator().manual_seed(7)
    _, mask, probability = _corrupt_answers(ids, answer, generator, 0.01, 0.01)
    rates = mask.float().mean(dim=0)
    assert torch.all(torch.abs(rates - 0.01) < 1.5e-3)
    assert (~mask.any(dim=1)).any()
    constant_loss = _reduce_tokens(torch.ones(int(mask.sum())), mask, probability, answer).mean()
    assert abs(float(constant_loss) - 1.0) < 0.08
    print(json.dumps({"self_check": "ok"}))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run"), default="run")
    parser.add_argument("--method", choices=("seq", "gd", "diagonal", "diag_gd", "rank1", "rank1_gd"), default="gd")
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "checkpoints/mdm_safetensors/mdm-170M-100e18.safetensors")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer")
    parser.add_argument("--reverse-dir", type=Path, default=ROOT / "SMDM/data/reverse_experiments/june_version_7921032488")
    parser.add_argument("--cache-prefix", type=Path)
    parser.add_argument("--prepare-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=int, default=170)
    parser.add_argument("--task-a", choices=("p2d", "d2p"), default="d2p")
    parser.add_argument("--task-b", choices=("p2d", "d2p"), default="p2d")
    parser.add_argument("--a-group-start", type=int, default=0)
    parser.add_argument("--b-group-start", type=int, default=5)
    parser.add_argument("--group-count", type=int, default=5)
    parser.add_argument(
        "--fisher-per-fact", "--calibration-per-fact",
        dest="calibration_per_fact", type=int, default=10,
    )
    parser.add_argument("--trainable", choices=("all", "last_block", "last_mlp"), default="all")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-mc-samples", type=int, default=4)
    parser.add_argument("--a-steps", type=int, default=1000)
    parser.add_argument("--b-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--mask-min", type=float, default=1e-3)
    parser.add_argument("--mask-max", type=float, default=1.0)
    parser.add_argument("--replay-prompts", type=int, default=64)
    parser.add_argument("--replay-steps", type=int, default=32)
    parser.add_argument("--replay-length", type=int, default=52)
    parser.add_argument("--replay-cfg", type=float, default=0.8)
    parser.add_argument("--replay-temperature", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--ewc-lambda", type=float, default=30.0)
    parser.add_argument("--ewc-target-stiffness", type=float)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--reverse-steps", type=int, default=32)
    parser.add_argument("--reverse-length", type=int, default=52)
    parser.add_argument("--reverse-cfg", type=float, default=0.8)
    parser.add_argument("--reverse-temperature", type=float, default=0.0)
    parser.add_argument("--generation-seed", type=int, default=3407)
    parser.add_argument("--show-predictions", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        return args
    if args.cache_prefix is None:
        parser.error("--cache-prefix is required")
    if args.mode == "run" and args.output is None:
        parser.error("--output is required in run mode")
    if not 0.0 < args.mask_min < args.mask_max <= 1.0:
        parser.error("mask range must satisfy 0 < min < max <= 1")
    if args.distill_temperature <= 0:
        parser.error("distillation temperature must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_check:
        _self_check()
        return 0
    try:
        prepare(args) if args.mode == "prepare" else run_method(args)
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "experiment": "dllm_rank1_transfer",
            "created_utc": _utc_now(),
            "config": _jsonable_args(args),
            "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure, indent=2), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
