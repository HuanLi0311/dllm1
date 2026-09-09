#!/usr/bin/env python3
"""GSM8K behavioral retention for mean-rank-1 versus diagonal Fisher EWC.

The runner has three deliberately small modes: ``evaluate`` checks a released
checkpoint, ``prepare`` learns GSM8K once and caches the shared Task-A state,
Fisher, and replay rows, and ``adapt`` starts every method from that cache.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    # SMDM's vendored package imports ``iclr_1`` by name.
    sys.path.insert(0, str(ROOT.parent))

from continual_mdm import flat_parameters, load_model, set_seed, trainable_parameters  # noqa: E402
from experiments import dllm_rank1_transfer as transfer  # noqa: E402


MASK_ID = 32000
METHODS = ("seq", "gd", "rank1", "rank1_gd", "diag_gd")
LATER_TASKS = ("summarization", "creative_writing")
PARAMETER_COUNTS = {170: 219_050_496, 336: 401_123_328, 472: 553_827_840, 1028: 1_142_367_744}
TRACE_CHUNK = 1_048_576
NUMBER = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?")
DEFAULT_GSM_TRAIN = ROOT / "SMDM/data/gsm8k/train_no_aug.txt"
DEFAULT_GSM_TEST = ROOT / "SMDM/data/gsm8k/test.jsonl"
DEFAULT_DOLLY = ROOT / "runs/data/dolly_natural_stream.jsonl"
DEFAULT_TOKENIZER = ROOT / "tokenizer"
GSM_TRAIN_SOURCE_COUNTS = {
    "52ebf7c73927f7434abbb2f7b705a82fb3dbdd4695438b7654de78b701c23b36": 5_249,
    "6f9a20bc1476ca65eee9bc5117c2d0582b1f1733d5148ffc0ff29cad2a9e9c6b": 384_620,
}


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _runtime(device: torch.device) -> dict:
    import transformers

    return {
        "host": os.uname().nodename,
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
    }


def _dependencies() -> dict[str, str]:
    paths = (Path(__file__), ROOT / "continual_mdm.py", ROOT / "experiments/dllm_rank1_transfer.py")
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_sha256(rows: list) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(temporary, path)


def _cache_paths(prefix: Path) -> dict[str, Path]:
    return {
        "state": Path(str(prefix) + ".state.pt"),
        "rank1": Path(str(prefix) + ".rank1.pt"),
        "diagonal": Path(str(prefix) + ".diagonal.pt"),
        "replay": Path(str(prefix) + ".replay.json"),
        "summary": Path(str(prefix) + ".prepare.json"),
    }


def _canonical_number(value: str) -> str | None:
    value = value.replace(",", "").strip()
    try:
        number = Fraction(value) if "/" in value else Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return str(number) if number.denominator != 1 else str(number.numerator)


def _extract_answer(text: str) -> tuple[str | None, bool]:
    marked = re.findall(r"####\s*([^\n]+)", text)
    numbers = NUMBER.findall(marked[-1] if marked else text)
    return (_canonical_number(numbers[-1]) if numbers else None, bool(marked))


def _lf_lines(text: str) -> list[str]:
    return [line for line in text.split("\n") if line.strip()]


def _read_gsm_sources(train_path: Path) -> list[dict]:
    sources = []
    # Python's line iterator splits on LF, not the literal U+2028 in one question.
    with train_path.open() as handle:
        for source_index, line in enumerate(handle):
            line = line.removesuffix("\n")
            if "||" in line and "####" in line:
                sources.append({"source_index": source_index, "line": line})
    expected = GSM_TRAIN_SOURCE_COUNTS.get(_sha256(train_path))
    if expected is not None and len(sources) != expected:
        raise ValueError(f"unexpected GSM8K source count: {len(sources)} != {expected}")
    if len(sources) < 3:
        raise ValueError("GSM8K training source has fewer than three valid problems")
    return sources


def _gsm_source_rows(source: dict) -> tuple[dict, dict]:
    question, solution = source["line"].split("||", 1)
    thought, answer = solution.split("####", 1)
    source_index = source["source_index"]
    question_segment = f"Question: {question}"
    thought_segment = "Answer: " + thought
    common = {"source_index": source_index}
    # The released SMDM recipe turns every problem into two conditionals.
    return (
        {
            **common,
            "example_id": f"gsm8k:train:{source_index}:thought",
            "phase": "thought",
            "prompt": question_segment,
            "answer": thought_segment,
            "prompt_segments": [question_segment],
            "answer_segments": [thought_segment],
        },
        {
            **common,
            "example_id": f"gsm8k:train:{source_index}:answer",
            "phase": "answer",
            "prompt": question_segment + thought_segment,
            "answer": "####" + answer,
            "prompt_segments": [question_segment, thought_segment],
            "answer_segments": ["####" + answer],
        },
    )


def _read_gsm_test(test_path: Path) -> list[dict]:
    test = []
    for source_index, line in enumerate(_lf_lines(test_path.read_text())):
        row = json.loads(line)
        test.append({
            "source_index": source_index,
            "example_id": f"gsm8k:test:{source_index}",
            "prompt": f"Question: {row['question'].strip()}",
            "answer": "Answer: " + row["answer"].strip(),
            "target": _canonical_number(str(row["target"])),
            "question": row["question"],
        })
    if len(test) != 1319:
        raise ValueError(f"unexpected GSM8K test count: {len(test)}")
    return test


def _read_later_tasks(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tasks = []
    for task_index, name in enumerate(LATER_TASKS, 1):
        train = [row for row in rows if row.get("task") == name and row.get("split") == "train"]
        test = [row for row in rows if row.get("task") == name and row.get("split") == "test"]
        if len(train) != 120 or len(test) != 40:
            raise ValueError(f"{name}: expected 120 train and 40 test rows")
        tasks.append({"task_index": task_index, "name": name, "train_raw": train, "test_raw": test})
    return tasks


def _encode(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    encoded = []
    for row in rows:
        prompt_segments = row.get("prompt_segments", [row["prompt"]])
        answer_segments = row.get("answer_segments", [row["answer"]])
        prompt_ids = [
            token
            for segment in prompt_segments
            for token in tokenizer(segment, add_special_tokens=True)["input_ids"]
        ]
        answer_ids = [
            token
            for segment in answer_segments
            for token in tokenizer(segment, add_special_tokens=True)["input_ids"]
        ]
        ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        if not len(prompt_ids) < len(ids) <= max_length:
            raise ValueError(f"{row.get('example_id', row.get('source_index'))}: invalid encoded length")
        encoded.append({
            **row,
            "ids": ids,
            "prompt_ids": prompt_ids,
            "answer_start": len(prompt_ids),
            "answer_end": len(ids) - 1,
        })
    return encoded


def _split_source_ids(source_ids: list[int], fisher_examples: int, seed: int) -> tuple[list[int], list[int]]:
    source_ids = sorted(source_ids)
    if not 2 <= fisher_examples < len(source_ids):
        raise ValueError("fisher_examples must leave at least one GSM8K source problem")
    random.Random(seed + 404).shuffle(source_ids)
    return source_ids[fisher_examples:], source_ids[:fisher_examples]


def _sample_replay_prompts(rows: list[dict], count: int, seed: int) -> tuple[list[dict], dict]:
    rows = [row for row in rows if row.get("phase") in (None, "thought")]
    if not 1 <= count <= len(rows):
        raise ValueError("replay_examples is outside the GSM8K training pool")
    selected = random.Random(seed + 505).sample(rows, count)
    manifest = {
        "count": count,
        "source_indices": [row["source_index"] for row in selected],
        "prompt_sha256": _records_sha256([row["prompt"] for row in selected]),
    }
    return selected, manifest


def _gumbel_argmax(logits: torch.Tensor, temperature: float, generator: torch.Generator) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    uniform = torch.rand(logits.shape, device=logits.device, dtype=torch.float64, generator=generator)
    gumbel = -torch.log(-torch.log(uniform.clamp(1e-12, 1 - 1e-12)))
    score = logits.double() + temperature * gumbel
    return score.argmax(dim=-1)


@torch.no_grad()
def _diffuse_equal_prompt_length(
    model,
    prompt_ids: list[list[int]],
    device: torch.device,
    steps: int,
    max_new_tokens: int,
    cfg: float,
    temperature: float,
    seed: int,
) -> torch.Tensor:
    if not prompt_ids or len({len(row) for row in prompt_ids}) != 1:
        raise ValueError("a generation batch must have one exact prompt length")
    prompt_length = len(prompt_ids[0])
    x = torch.full(
        (len(prompt_ids), prompt_length + max_new_tokens), MASK_ID,
        dtype=torch.long, device=device,
    )
    for index, row in enumerate(prompt_ids):
        x[index, :prompt_length] = torch.tensor(row, dtype=torch.long, device=device)
    timesteps = torch.linspace(1, 1e-5, steps + 1, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for step in range(steps):
        mask = x == MASK_ID
        if cfg > 0:
            unconditional = x.clone()
            unconditional[:, :prompt_length] = MASK_ID
            logits = model(torch.cat((x, unconditional), dim=0))
            logits, unconditional_logits = torch.chunk(logits, 2, dim=0)
            logits = unconditional_logits + (cfg + 1) * (logits - unconditional_logits)
        else:
            logits = model(x)
        fraction = float(1 - timesteps[step + 1] / timesteps[step])
        for row_index in range(len(prompt_ids)):
            positions = mask[row_index].nonzero(as_tuple=False).flatten()
            if not len(positions):
                continue
            selected_logits = logits[row_index, positions]
            predictions = _gumbel_argmax(selected_logits, temperature, generator)
            transfer = len(positions) if step + 1 == steps else int(len(positions) * fraction)
            if transfer:
                if transfer == len(positions):
                    chosen = torch.arange(len(positions), device=device)
                else:
                    confidence = selected_logits.float().softmax(dim=-1).gather(
                        1, predictions[:, None]
                    ).squeeze(1)
                    chosen = confidence.topk(transfer).indices
                x[row_index, positions[chosen]] = predictions[chosen]
    return x


def _generate(
    model,
    rows: list[dict],
    tokenizer,
    device: torch.device,
    batch_size: int,
    steps: int,
    max_new_tokens: int,
    cfg: float,
    temperature: float,
    seed: int,
    benchmark_records: bool,
) -> list[dict]:
    grouped = defaultdict(list)
    for output_index, row in enumerate(rows):
        grouped[len(row["prompt_ids"])].append((output_index, row))
    outputs = [None] * len(rows)
    completed = 0
    batch_index = 0
    for prompt_length in sorted(grouped):
        group = grouped[prompt_length]
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            generated = _diffuse_equal_prompt_length(
                model, [row["prompt_ids"] for _, row in batch], device,
                steps, max_new_tokens, cfg, temperature, seed + batch_index,
            ).cpu().tolist()
            batch_index += 1
            for (output_index, row), token_ids in zip(batch, generated):
                completion = token_ids[prompt_length:]
                if tokenizer.eos_token_id in completion:
                    completion = completion[:completion.index(tokenizer.eos_token_id)]
                text = tokenizer.decode(completion, skip_special_tokens=True).strip()
                if benchmark_records:
                    prediction, marked = _extract_answer(text)
                    outputs[output_index] = {
                        "example_id": row["example_id"],
                        "source_index": row["source_index"],
                        "target": row["target"],
                        "prediction": prediction,
                        "has_delimiter": marked,
                        "correct": marked and prediction == row["target"],
                        "fallback_numeric_correct": prediction == row["target"],
                        "output": text,
                    }
                else:
                    if not completion:
                        raise RuntimeError("teacher generated an empty replay completion")
                    outputs[output_index] = {
                        "ids": row["prompt_ids"] + completion,
                        "answer_start": prompt_length,
                        "answer_end": prompt_length + len(completion),
                        "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                    }
            completed += len(batch)
            print(f"generated={completed}/{len(rows)} prompt_tokens={prompt_length}", flush=True)
    return outputs


def _diffusion_pass(
    model,
    prompt_ids: list[list[int]],
    device: torch.device,
    batch_size: int,
    steps: int,
    context_length: int,
    cfg: float,
    temperature: float,
    seed: int,
    pass_index: int,
) -> list[list[int]]:
    outputs = []
    completed = 0
    for start in range(0, len(prompt_ids), batch_size):
        batch = prompt_ids[start : start + batch_size]
        generated = _diffuse_variable_prompt_length(
            model, batch, device, steps, context_length, cfg, temperature,
            [seed + index for index in range(start, start + len(batch))],
        ).cpu().tolist()
        outputs.extend(generated)
        completed += len(batch)
        lengths = [len(ids) for ids in batch]
        print(
            f"benchmark_pass={pass_index}/2 generated={completed}/{len(prompt_ids)} "
            f"prompt_tokens={min(lengths)}-{max(lengths)}",
            flush=True,
        )
    return outputs


@torch.no_grad()
def _diffuse_variable_prompt_length(
    model,
    prompt_ids: list[list[int]],
    device: torch.device,
    steps: int,
    context_length: int,
    cfg: float,
    temperature: float,
    seeds: list[int],
) -> torch.Tensor:
    if not prompt_ids or len(prompt_ids) != len(seeds):
        raise ValueError("generation rows and seeds must be nonempty and aligned")
    if any(len(ids) > context_length for ids in prompt_ids):
        raise ValueError("a benchmark prompt exceeds the generation context")
    x = torch.full(
        (len(prompt_ids), context_length), MASK_ID, dtype=torch.long, device=device,
    )
    for row_index, ids in enumerate(prompt_ids):
        x[row_index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    timesteps = torch.linspace(1, 1e-5, steps + 1, device=device)
    model.eval()
    for step in range(steps):
        mask = x == MASK_ID
        if cfg > 0:
            unconditional = x.clone()
            for row_index, ids in enumerate(prompt_ids):
                unconditional[row_index, :len(ids)] = MASK_ID
            logits = model(torch.cat((x, unconditional), dim=0))
            logits, unconditional_logits = torch.chunk(logits, 2, dim=0)
            logits = unconditional_logits + (cfg + 1) * (logits - unconditional_logits)
        else:
            logits = model(x)
        fraction = float(1 - timesteps[step + 1] / timesteps[step])
        for row_index, generator in enumerate(generators):
            positions = mask[row_index].nonzero(as_tuple=False).flatten()
            if not len(positions):
                continue
            selected_logits = logits[row_index, positions]
            predictions = _gumbel_argmax(selected_logits, temperature, generator)
            transfer = len(positions) if step + 1 == steps else int(len(positions) * fraction)
            if transfer:
                if transfer == len(positions):
                    chosen = torch.arange(len(positions), device=device)
                else:
                    confidence = selected_logits.float().softmax(dim=-1).gather(
                        1, predictions[:, None]
                    ).squeeze(1)
                    chosen = confidence.topk(transfer).indices
                x[row_index, positions[chosen]] = predictions[chosen]
    return x


def _official_gsm8k_generate(
    model,
    rows: list[dict],
    tokenizer,
    device: torch.device,
    batch_size: int,
    steps: int,
    context_length: int,
    cfg: float,
    temperature: float,
    seed: int,
) -> list[dict]:
    """Mirror the released evaluator's two successive length-256 passes."""
    first = _diffusion_pass(
        model, [row["prompt_ids"] for row in rows], device, batch_size,
        steps, context_length, cfg, temperature, seed, 1,
    )
    first_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in first]
    retokenized = [
        tokenizer(text, add_special_tokens=True)["input_ids"] for text in first_text
    ]
    second_prompts = [ids[:context_length] for ids in retokenized]
    second = _diffusion_pass(
        model, second_prompts, device, batch_size, steps, context_length,
        cfg, temperature, seed + 1_000_000, 2,
    )
    records = []
    for row, ids, retokenized_ids in zip(rows, second, retokenized):
        full_text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        prompt_text = tokenizer.decode(row["prompt_ids"], skip_special_tokens=True).strip()
        completion = (
            full_text[len(prompt_text):].strip()
            if full_text.startswith(prompt_text)
            else full_text
        )
        prediction, marked = _extract_answer(completion)
        records.append({
            "example_id": row["example_id"],
            "source_index": row["source_index"],
            "target": row["target"],
            "prediction": prediction,
            "has_delimiter": marked,
            "correct": marked and prediction == row["target"],
            "fallback_numeric_correct": prediction == row["target"],
            "output": completion,
            "full_output": full_text,
            "retokenized_first_pass_tokens": len(retokenized_ids),
            "second_prompt_truncated": len(retokenized_ids) > context_length,
        })
    return records


def _benchmark(model, rows, tokenizer, device, args, seed: int) -> dict:
    started = time.monotonic()
    records = _official_gsm8k_generate(
        model, rows, tokenizer, device, args.generation_batch_size,
        args.benchmark_steps, args.benchmark_context_length,
        args.benchmark_cfg, args.benchmark_temperature, seed,
    )
    return {
        "count": len(records),
        "exact_match": sum(row["correct"] for row in records) / len(records),
        "correct": sum(row["correct"] for row in records),
        "fallback_numeric_accuracy": sum(
            row["fallback_numeric_correct"] for row in records
        ) / len(records),
        "delimiter_rate": sum(row["has_delimiter"] for row in records) / len(records),
        "second_prompt_truncations": sum(row["second_prompt_truncated"] for row in records),
        "wall_time_seconds": time.monotonic() - started,
        "records": records,
    }


def _task_metrics(model, rows, pad_id: int, device: torch.device, args, seed: int) -> dict:
    return {
        "loss": transfer._evaluate_loss(model, rows, pad_id, device, args, seed),
        "answer_token_accuracy": transfer.answer_token_accuracy(
            model, rows, device, args.eval_batch_size, pad_id
        ),
    }


def _direction_norm_sq(direction: torch.Tensor, chunk_size: int = TRACE_CHUNK) -> float:
    flat = direction.detach().cpu().reshape(-1)
    parts = []
    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start : start + chunk_size].double()
        parts.append(float(torch.dot(chunk, chunk)))
    value = math.fsum(parts)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid stored direction norm squared: {value}")
    return value


def _float64_sum(vector: torch.Tensor, chunk_size: int = TRACE_CHUNK) -> float:
    flat = vector.detach().cpu().reshape(-1)
    parts = [
        float(flat[start : start + chunk_size].sum(dtype=torch.float64))
        for start in range(0, flat.numel(), chunk_size)
    ]
    value = math.fsum(parts)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid stored diagonal trace: {value}")
    return value


def _matched_stiffness(
    coefficient: float,
    direction_norm_sq: float,
    diagonal_trace: float,
    parameter_count: int,
    target_trace_per_parameter: float,
) -> dict:
    values = (coefficient, direction_norm_sq, diagonal_trace, target_trace_per_parameter)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"invalid trace-matching quantities: {values}")
    target = target_trace_per_parameter * parameter_count
    rank1_trace = coefficient * direction_norm_sq
    lambdas = {"rank1": target / rank1_trace, "diagonal": target / diagonal_trace}
    checks = {
        "rank1": lambdas["rank1"] * rank1_trace,
        "diagonal": lambdas["diagonal"] * diagonal_trace,
    }
    if not all(math.isclose(value, target, rel_tol=1e-12) for value in checks.values()):
        raise AssertionError("implemented weighted traces do not match")
    return {
        "target_trace_per_parameter": target_trace_per_parameter,
        "weighted_trace_target": target,
        "rank1_unweighted_trace": rank1_trace,
        "diagonal_unweighted_trace": diagonal_trace,
        "direction_norm_sq_float64_chunked": direction_norm_sq,
        "lambdas": lambdas,
        "weighted_trace_checks": checks,
    }


def _prep_metadata(args) -> dict:
    return {
        "seed": args.seed,
        "model": args.model,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "tokenizer_sha256": _tree_sha256(args.tokenizer),
        "gsm_train_sha256": _sha256(args.gsm_train),
        "gsm_test_sha256": _sha256(args.gsm_test),
        "gsm_sft_encoding": "released_two_conditionals_segment_tokenization_v1",
        "max_length": args.max_length,
        "a_steps": args.a_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "clip": args.clip,
        "mask_min": args.mask_min,
        "mask_max": args.mask_max,
        "fisher_examples": args.fisher_examples,
        "replay_examples": args.replay_examples,
        "replay_steps": args.replay_steps,
        "replay_max_new_tokens": args.replay_max_new_tokens,
        "replay_cfg": args.replay_cfg,
        "replay_temperature": args.replay_temperature,
    }


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _load_tokenizer(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    return tokenizer


def _load_gsm_test_encoded(args, tokenizer) -> list[dict]:
    return _encode(_read_gsm_test(args.gsm_test), tokenizer, args.max_length)


def _load_gsm_for_prepare(args, tokenizer) -> tuple[list[dict], list[dict], list[dict], dict]:
    sources = _read_gsm_sources(args.gsm_train)
    source_by_id = {row["source_index"]: row for row in sources}
    train_ids, fisher_ids = _split_source_ids(
        list(source_by_id), args.fisher_examples, args.seed
    )
    fisher_raw = [row for source_id in fisher_ids for row in _gsm_source_rows(source_by_id[source_id])]

    if args.a_steps:
        if len(sources) > 50_000:
            raise ValueError(
                "full augmented-data SFT belongs in the released distributed trainer; "
                "use --a-steps 0 with its learned checkpoint here"
            )
        train_raw = [row for source_id in train_ids for row in _gsm_source_rows(source_by_id[source_id])]
    else:
        if args.replay_examples > len(train_ids):
            raise ValueError("replay_examples exceeds the non-Fisher GSM8K source pool")
        selected_ids = random.Random(args.seed + 505).sample(train_ids, args.replay_examples)
        train_raw = [_gsm_source_rows(source_by_id[source_id])[0] for source_id in selected_ids]

    stats = {
        "source_problems": len(sources),
        "non_fisher_source_pool": len(train_ids),
        "optimization_source_problems": len(train_ids) if args.a_steps else 0,
        "fisher_source_problems": len(fisher_ids),
        "fisher_source_indices": fisher_ids,
        "task_a_loaded_from_checkpoint": args.a_steps == 0,
    }
    return (
        _encode(train_raw, tokenizer, args.max_length),
        _encode(fisher_raw, tokenizer, args.max_length),
        _load_gsm_test_encoded(args, tokenizer),
        stats,
    )


def evaluate(args) -> dict:
    started = time.monotonic()
    dependencies = _dependencies()
    checkpoint_sha256 = _sha256(args.checkpoint)
    gsm_test_sha256 = _sha256(args.gsm_test)
    set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = _load_tokenizer(args)
    gsm_test = _load_gsm_test_encoded(args, tokenizer)
    benchmark_rows = gsm_test[: args.benchmark_limit or None]
    model = load_model(args, device)
    evaluated_state = None
    if args.cache_prefix is not None:
        state_path = _cache_paths(args.cache_prefix)["state"]
        state_payload = torch.load(state_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_payload["state_dict"])
        evaluated_state = {
            "path": str(state_path),
            "sha256": _sha256(state_path),
            "metadata": state_payload.get("metadata"),
        }
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "smdm_gsm8k_checkpoint_evaluation",
        "created_utc": _utc_now(),
        "wall_time_seconds": time.monotonic() - started,
        "runtime": _runtime(device),
        "dependencies": dependencies,
        "model": args.model,
        "checkpoint_sha256": checkpoint_sha256,
        "gsm_test_sha256": gsm_test_sha256,
        "evaluated_state_cache": evaluated_state,
        "decoding": _decoding_settings(args),
        "benchmark": _benchmark(model, benchmark_rows, tokenizer, device, args, args.seed + 90_000),
    }
    result["wall_time_seconds"] = time.monotonic() - started
    if (
        _dependencies() != dependencies
        or _sha256(args.checkpoint) != checkpoint_sha256
        or _sha256(args.gsm_test) != gsm_test_sha256
    ):
        raise RuntimeError("source or evaluation input changed during execution")
    _write_result(args, result)
    return result


def prepare(args) -> dict:
    started = time.monotonic()
    dependencies = _dependencies()
    metadata = _prep_metadata(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = _load_tokenizer(args)
    pad_id = int(tokenizer.eos_token_id)
    gsm_train, gsm_fisher, gsm_test, gsm_stats = _load_gsm_for_prepare(args, tokenizer)
    benchmark_rows = gsm_test[: args.benchmark_limit or None]
    loss_rows = gsm_test[: args.loss_eval_limit or None]
    model = load_model(args, device)
    parameters = trainable_parameters(model, "all")
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count != PARAMETER_COUNTS[args.model]:
        raise RuntimeError(f"unexpected parameter count: {parameter_count}")
    print(f"prepare model={args.model} parameters={parameter_count:,}", flush=True)
    training = transfer._train(
        model, gsm_train, parameters, pad_id, device, args,
        args.a_steps, args.seed + 1_000,
    )
    metrics = _task_metrics(model, loss_rows, pad_id, device, args, args.seed + 81_000)
    benchmark = _benchmark(model, benchmark_rows, tokenizer, device, args, args.seed + 90_000)
    if benchmark["exact_match"] < args.min_gsm8k_exact_match:
        raise RuntimeError(
            f"Task-A gate failed: exact_match={benchmark['exact_match']:.6f} "
            f"< {args.min_gsm8k_exact_match:.6f}"
        )
    fisher, fisher_stats = transfer._estimate_mean_and_diagonal_fisher(
        model, gsm_fisher, parameters, pad_id, device, args
    )
    direction_norm_sq = _direction_norm_sq(fisher["direction"])
    diagonal_trace = _float64_sum(fisher["diagonal"])
    replay_source, replay_manifest = _sample_replay_prompts(
        gsm_train, args.replay_examples, args.seed
    )
    replay = _generate(
        model, replay_source, tokenizer, device, args.generation_batch_size,
        args.replay_steps, args.replay_max_new_tokens,
        args.replay_cfg, args.replay_temperature, args.seed + 700_000, False,
    )
    paths = _cache_paths(args.cache_prefix)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    _atomic_torch(paths["state"], {"kind": "gsm8k_task_a_state_v1", "metadata": metadata, "state_dict": state})
    _atomic_torch(paths["rank1"], {
        "kind": "gsm8k_mean_rank1_v1", "metadata": metadata,
        "direction": fisher["direction"], "coefficient": float(fisher["coefficient"]),
        "direction_norm_sq": direction_norm_sq, "stats": fisher_stats,
    })
    _atomic_torch(paths["diagonal"], {
        "kind": "gsm8k_diagonal_v1", "metadata": metadata,
        "diagonal": fisher["diagonal"], "diagonal_trace": diagonal_trace,
        "stats": fisher_stats,
    })
    _atomic_json(paths["replay"], {
        "kind": "gsm8k_generated_replay_v1", "metadata": metadata,
        "manifest": replay_manifest, "rows": replay,
    })
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "smdm_gsm8k_rank1_prepare",
        "created_utc": _utc_now(),
        "wall_time_seconds": time.monotonic() - started,
        "runtime": _runtime(device),
        "dependencies": dependencies,
        "metadata": metadata,
        "parameter_count": parameter_count,
        "gsm_source": gsm_stats,
        "gsm_loaded_training_instances": len(gsm_train),
        "gsm_fisher_instances": len(gsm_fisher),
        "gsm_fisher_source_problems": len({row["source_index"] for row in gsm_fisher}),
        "gsm_loaded_training_source_indices_sha256": _records_sha256(
            [row["source_index"] for row in gsm_train]
        ),
        "gsm_fisher_source_indices": gsm_stats["fisher_source_indices"],
        "training": training,
        "metrics": metrics,
        "benchmark": benchmark,
        "fisher": {
            **fisher_stats,
            "direction_norm_sq_float64_chunked": direction_norm_sq,
            "diagonal_trace_float64_chunked": diagonal_trace,
        },
        "replay": {"manifest": replay_manifest, "rows_sha256": _records_sha256(replay)},
        "cache": {name: str(path) for name, path in paths.items()},
    }
    if _dependencies() != dependencies or _prep_metadata(args) != metadata:
        raise RuntimeError("source or Task-A input changed during preparation")
    _atomic_json(paths["summary"], result)
    if args.output and args.output != paths["summary"]:
        _atomic_json(args.output, result)
    print(json.dumps({
        "status": "ok", "cache_prefix": str(args.cache_prefix),
        "exact_match": benchmark["exact_match"], "parameter_count": parameter_count,
    }, indent=2), flush=True)
    return result


def _validate_cache_metadata(actual: dict, expected: dict) -> None:
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError("Task-A cache metadata mismatch: " + ", ".join(mismatches))


def _load_later_encoded(args, tokenizer) -> list[dict]:
    tasks = _read_later_tasks(args.adaptation_data)
    for task in tasks:
        task["train"] = _encode(task["train_raw"], tokenizer, args.max_length)
        task["test"] = _encode(task["test_raw"], tokenizer, args.max_length)
    return tasks


def adapt(args) -> dict:
    started = time.monotonic()
    dependencies = _dependencies()
    if args.method not in METHODS:
        raise ValueError(f"unknown method: {args.method}")
    set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = _load_tokenizer(args)
    pad_id = int(tokenizer.eos_token_id)
    gsm_test = _load_gsm_test_encoded(args, tokenizer)
    benchmark_rows = gsm_test[: args.benchmark_limit or None]
    tasks = _load_later_encoded(args, tokenizer)
    paths = _cache_paths(args.cache_prefix)
    expected = _prep_metadata(args)
    adaptation_data_sha256 = _sha256(args.adaptation_data)
    prepare_summary = json.loads(paths["summary"].read_text())
    _validate_cache_metadata(prepare_summary["metadata"], expected)
    if prepare_summary.get("dependencies") != dependencies:
        raise ValueError("Task-A cache was produced by different source code")
    state_payload = torch.load(paths["state"], map_location="cpu", weights_only=True)
    _validate_cache_metadata(state_payload["metadata"], expected)
    model = load_model(args, device)
    model.load_state_dict(state_payload["state_dict"])
    parameters = trainable_parameters(model, "all")
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count != PARAMETER_COUNTS[args.model]:
        raise RuntimeError(f"unexpected parameter count: {parameter_count}")
    reference = (
        flat_parameters(parameters).detach().clone()
        if args.method in ("rank1", "rank1_gd", "diag_gd")
        else None
    )
    uses_gd = args.method in ("gd", "rank1_gd", "diag_gd")
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

    fisher_summary = prepare_summary["fisher"]
    stiffness = _matched_stiffness(
        float(fisher_summary["rank1_coefficient"]),
        float(fisher_summary["direction_norm_sq_float64_chunked"]),
        float(fisher_summary["diagonal_trace_float64_chunked"]), parameter_count,
        args.target_trace_per_parameter,
    )
    fisher = ewc_kind = None
    ewc_lambda = 0.0
    if args.method in ("rank1", "rank1_gd"):
        rank_payload = torch.load(paths["rank1"], map_location="cpu", weights_only=True)
        _validate_cache_metadata(rank_payload["metadata"], expected)
        fisher = {
            "direction": rank_payload["direction"].to(device),
            "coefficient": float(rank_payload["coefficient"]),
        }
        ewc_kind = "rank1"
        ewc_lambda = stiffness["lambdas"]["rank1"]
        del rank_payload
    elif args.method == "diag_gd":
        diagonal_payload = torch.load(paths["diagonal"], map_location="cpu", weights_only=True)
        _validate_cache_metadata(diagonal_payload["metadata"], expected)
        fisher = {"diagonal": diagonal_payload["diagonal"].to(device)}
        ewc_kind = "diagonal"
        ewc_lambda = stiffness["lambdas"]["diagonal"]
        del diagonal_payload

    stages = [{
        "stage": 0,
        "task": "gsm8k",
        "benchmark": {
            key: value for key, value in prepare_summary["benchmark"].items() if key != "records"
        },
    }]
    for stage_index, task in enumerate(tasks, 1):
        print(f"stage={stage_index}/{len(tasks)} task={task['name']} method={args.method}", flush=True)
        training = transfer._train(
            model, task["train"], parameters, pad_id, device, args,
            args.later_steps, args.seed + 1_000 * (stage_index + 1),
            teacher=teacher, replay_rows=replay_rows,
            reference=reference, fisher=fisher, ewc_kind=ewc_kind,
            ewc_lambda=ewc_lambda,
        )
        metrics = _task_metrics(
            model, task["test"], pad_id, device, args,
            args.seed + 82_000 + 100 * stage_index,
        )
        benchmark = None
        if args.benchmark_each_stage or stage_index == len(tasks):
            benchmark = _benchmark(
                model, benchmark_rows, tokenizer, device, args,
                args.seed + 90_000,
            )
        stages.append({
            "stage": stage_index, "task": task["name"],
            "training": training, "metrics": metrics, "benchmark": benchmark,
        })

    final_later_metrics = {
        task["name"]: _task_metrics(
            model, task["test"], pad_id, device, args,
            args.seed + 83_000 + 100 * task["task_index"],
        )
        for task in tasks
    }
    learned = prepare_summary["benchmark"]["exact_match"]
    final = stages[-1]["benchmark"]["exact_match"]
    learned_correct = {
        row["example_id"] for row in prepare_summary["benchmark"]["records"] if row["correct"]
    }
    final_correct = {
        row["example_id"] for row in stages[-1]["benchmark"]["records"] if row["correct"]
    }
    retained_fraction = (
        len(learned_correct & final_correct) / len(learned_correct) if learned_correct else None
    )
    result = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "smdm_gsm8k_rank1_behavior",
        "created_utc": _utc_now(),
        "wall_time_seconds": time.monotonic() - started,
        "runtime": _runtime(device),
        "dependencies": dependencies,
        "method": args.method,
        "seed": args.seed,
        "model": args.model,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "parameter_count": parameter_count,
        "task_sequence": ["gsm8k", *LATER_TASKS],
        "adaptation_data_sha256": adaptation_data_sha256,
        "decoding": _decoding_settings(args),
        "target_trace_per_parameter": args.target_trace_per_parameter,
        "stiffness_match": stiffness,
        "ewc_kind": ewc_kind,
        "ewc_lambda_effective": ewc_lambda,
        "replay": {"enabled": uses_gd, "examples": len(replay_rows or [])},
        "stages": stages,
        "final_later_task_metrics": final_later_metrics,
        "summary": {
            "gsm8k_exact_match_when_learned": learned,
            "gsm8k_exact_match_final": final,
            "gsm8k_retention_change": final - learned,
            "gsm8k_retained_fraction_conditional_on_task_a_correct": retained_fraction,
            "final_later_average_loss": sum(
                metrics["loss"] for metrics in final_later_metrics.values()
            ) / len(final_later_metrics),
            "final_later_average_answer_token_accuracy": sum(
                metrics["answer_token_accuracy"] for metrics in final_later_metrics.values()
            ) / len(final_later_metrics),
        },
    }
    if (
        _dependencies() != dependencies
        or _prep_metadata(args) != expected
        or _sha256(args.adaptation_data) != adaptation_data_sha256
    ):
        raise RuntimeError("source or adaptation input changed during execution")
    _write_result(args, result)
    return result


def _decoding_settings(args) -> dict:
    return {
        "algorithm": "released_smdm_two_pass_per_example_prompt_v2",
        "passes": 2,
        "steps_per_pass": args.benchmark_steps,
        "context_length_per_pass": args.benchmark_context_length,
        "cfg": args.benchmark_cfg,
        "temperature": args.benchmark_temperature,
        "retokenized_second_prompt_overflow": "truncate_right_and_record",
        "completion_only_scoring": True,
        "answer_extraction": "last_number_after_last_####_else_last_number",
    }


def _write_result(args, result: dict) -> None:
    if args.output:
        _atomic_json(args.output, result)
    print(json.dumps({
        "status": result["status"], "experiment": result["experiment"],
        "output": str(args.output) if args.output else None,
        "summary": result.get("summary", {
            "exact_match": result.get("benchmark", {}).get("exact_match")
        }),
    }, indent=2), flush=True)


def _self_check() -> None:
    assert _extract_answer("work\n#### 1,234") == ("1234", True)
    assert _extract_answer("therefore -2.50") == ("-5/2", False)
    assert _lf_lines("first\u2028still first\nsecond\n") == ["first\u2028still first", "second"]
    direction = torch.tensor([3.0, 4.0], dtype=torch.float32)
    diagonal = torch.tensor([0.5, 0.125, 1.5, 0.25], dtype=torch.float32)
    norm_sq = _direction_norm_sq(direction, chunk_size=1)
    trace = _float64_sum(diagonal, chunk_size=2)
    match = _matched_stiffness(2.5, norm_sq, trace, 4, 0.25)
    assert norm_sq == 25.0 and trace == 2.375
    assert match["weighted_trace_checks"] == {"rank1": 1.0, "diagonal": 1.0}

    class ConstantModel:
        def eval(self):
            return self

        def __call__(self, ids):
            logits = torch.zeros((*ids.shape, 8))
            logits[..., 7] = 1
            return logits

    generated = _diffuse_equal_prompt_length(
        ConstantModel(), [[1, 2], [3, 4]], torch.device("cpu"),
        2, 3, 0.0, 0.0, 7,
    )
    assert generated.tolist() == [[1, 2, 7, 7, 7], [3, 4, 7, 7, 7]]
    passed = _diffusion_pass(
        ConstantModel(), [[1, 2], [3]], torch.device("cpu"),
        2, 2, 5, 0.0, 0.0, 7, 1,
    )
    assert passed == [[1, 2, 7, 7, 7], [3, 7, 7, 7, 7]]
    batched = _diffuse_variable_prompt_length(
        ConstantModel(), [[1, 2], [3]], torch.device("cpu"),
        2, 5, 0.0, 0.5, [17, 18],
    ).tolist()
    singles = [
        _diffuse_variable_prompt_length(
            ConstantModel(), [ids], torch.device("cpu"),
            2, 5, 0.0, 0.5, [seed],
        )[0].tolist()
        for ids, seed in zip([[1, 2], [3]], [17, 18])
    ]
    assert batched == singles

    class SegmentTokenizer:
        eos_token_id = 0

        def __call__(self, text, add_special_tokens=True):
            assert add_special_tokens
            return {"input_ids": [1, len(text)]}

    encoded = _encode([{
        "source_index": 0, "prompt": "ignored", "answer": "ignored",
        "prompt_segments": ["p", "qq"], "answer_segments": ["aaa"],
    }], SegmentTokenizer(), 10)[0]
    assert encoded["prompt_ids"] == [1, 1, 1, 2]
    assert encoded["ids"] == [1, 1, 1, 2, 1, 3, 0]
    thought_row, answer_row = _gsm_source_rows({"source_index": 3, "line": "q||t #### 4"})
    assert thought_row["phase"] == "thought" and answer_row["phase"] == "answer"
    assert answer_row["prompt_segments"] == ["Question: q", "Answer: t "]
    assert answer_row["answer_segments"] == ["#### 4"]

    train1, held1 = _split_source_ids(list(range(20)), 4, 9)
    train2, held2 = _split_source_ids(list(range(20)), 4, 9)
    assert train1 == train2 and held1 == held2 and len(held1) == 4
    assert not set(train1) & set(held1)
    print(json.dumps({"self_check": "ok"}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("evaluate", "prepare", "adapt"), default="evaluate")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--model", type=int, choices=tuple(PARAMETER_COUNTS), default=170)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gsm-train", type=Path, default=DEFAULT_GSM_TRAIN)
    parser.add_argument("--gsm-test", type=Path, default=DEFAULT_GSM_TEST)
    parser.add_argument("--adaptation-data", type=Path, default=DEFAULT_DOLLY)
    parser.add_argument("--cache-prefix", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", choices=METHODS, default="seq")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--a-steps", type=int, default=5000)
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
    parser.add_argument("--benchmark-limit", type=int, default=0)
    parser.add_argument("--loss-eval-limit", type=int, default=256)
    parser.add_argument("--eval-mc-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=576)
    parser.add_argument("--min-gsm8k-exact-match", type=float, default=0.01)
    parser.add_argument("--target-trace-per-parameter", type=float, default=1e-8)
    parser.add_argument("--benchmark-each-stage", action="store_true", default=True)
    parser.add_argument(
        "--final-benchmark-only", action="store_false", dest="benchmark_each_stage",
        help="development shortcut; omit the post-Task-B generation pass",
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return args
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    if args.mode in ("prepare", "adapt") and args.cache_prefix is None:
        parser.error("--cache-prefix is required for prepare/adapt")
    if args.mode in ("evaluate", "adapt") and args.output is None:
        parser.error("--output is required for evaluate/adapt")
    if args.benchmark_limit < 0 or args.loss_eval_limit < 0:
        parser.error("evaluation limits cannot be negative")
    if args.a_steps < 0 or args.later_steps < 0:
        parser.error("training steps cannot be negative")
    if args.benchmark_context_length < 1:
        parser.error("--benchmark-context-length must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "prepare":
        prepare(args)
    else:
        adapt(args)


if __name__ == "__main__":
    main()
