#!/usr/bin/env python3
"""Derive a cache with released two-pass GSM8K replay without recomputing Fisher."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import smdm_gsm8k_rank1_benchmark as benchmark  # noqa: E402


def _new_metadata(metadata: dict, args) -> dict:
    result = copy.deepcopy(metadata)
    result.update({
        "replay_steps": args.steps,
        "replay_max_new_tokens": args.context_length,
        "replay_cfg": args.cfg,
        "replay_temperature": args.temperature,
    })
    return result


def _replay_row(prompt_ids: list[int], completion: str, tokenizer, max_length: int) -> tuple[dict, bool]:
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    available = max_length - len(prompt_ids)
    if available < 1 or not completion_ids:
        raise ValueError("two-pass replay produced an empty or overlength completion")
    truncated = len(completion_ids) > available
    completion_ids = completion_ids[:available]
    ids = prompt_ids + completion_ids
    return {
        "ids": ids,
        "answer_start": len(prompt_ids),
        "answer_end": len(ids),
    }, truncated


def build(args) -> dict:
    source_paths = benchmark._cache_paths(args.source_prefix)
    output_paths = benchmark._cache_paths(args.output_prefix)
    if any(path.exists() for path in output_paths.values()):
        raise FileExistsError("refusing to overwrite an output cache")
    summary = json.loads(source_paths["summary"].read_text())
    if summary.get("dependencies") != benchmark._dependencies():
        raise ValueError("source cache was produced by different benchmark code")
    old_metadata = summary["metadata"]
    if old_metadata["model"] != args.model or old_metadata["max_length"] != args.max_length:
        raise ValueError("model or maximum length differs from the source cache")
    if old_metadata["checkpoint_sha256"] != benchmark._sha256(args.checkpoint):
        raise ValueError("checkpoint differs from the source cache")
    if old_metadata["gsm_train_sha256"] != benchmark._sha256(args.gsm_train):
        raise ValueError("GSM8K training source differs from the source cache")

    selected_ids = summary["replay"]["manifest"]["source_indices"]
    source_by_id = {
        row["source_index"]: row for row in benchmark._read_gsm_sources(args.gsm_train)
    }
    raw_rows = []
    for source_id in selected_ids:
        conditionals = benchmark._gsm_source_rows(source_by_id[source_id])
        thought, final = conditionals
        target, marked = benchmark._extract_answer(final["answer"])
        if not marked:
            raise ValueError(f"source {source_id} has no marked final answer")
        raw_rows.append({**thought, "target": target})

    tokenizer = benchmark._load_tokenizer(SimpleNamespace(tokenizer=args.tokenizer))
    rows = benchmark._encode(raw_rows, tokenizer, args.max_length)
    device = torch.device(args.device)
    model = benchmark.load_model(args, device)
    state_payload = torch.load(source_paths["state"], map_location="cpu", weights_only=True)
    benchmark._validate_cache_metadata(state_payload["metadata"], old_metadata)
    model.load_state_dict(state_payload["state_dict"])
    del state_payload
    records = benchmark._official_gsm8k_generate(
        model, rows, tokenizer, device, args.batch_size, args.steps,
        args.context_length, args.cfg, args.temperature,
        old_metadata["seed"] + 700_000,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    replay_rows = []
    truncations = 0
    for row, record in zip(rows, records):
        replay, truncated = _replay_row(
            row["prompt_ids"], record["output"], tokenizer, args.max_length
        )
        replay["prompt_sha256"] = hashlib.sha256(row["prompt"].encode()).hexdigest()
        replay_rows.append(replay)
        truncations += truncated

    settings = {
        "algorithm": "released_smdm_two_pass_per_example_prompt_v2",
        "steps_per_pass": args.steps,
        "context_length_per_pass": args.context_length,
        "cfg": args.cfg,
        "temperature": args.temperature,
        "completion_token_truncations_at_training_max_length": truncations,
    }
    quality = {
        "count": len(records),
        "strict_exact_match": sum(row["correct"] for row in records) / len(records),
        "fallback_numeric_accuracy": sum(row["fallback_numeric_correct"] for row in records) / len(records),
        "delimiter_rate": sum(row["has_delimiter"] for row in records) / len(records),
        "second_prompt_truncations": sum(row["second_prompt_truncated"] for row in records),
    }
    new_metadata = _new_metadata(old_metadata, args)
    source_summary_sha256 = benchmark._sha256(source_paths["summary"])
    derivation = {
        "source_cache_prefix": str(args.source_prefix),
        "source_summary_sha256": source_summary_sha256,
        "unchanged": ["state_dict", "rank1_direction_and_coefficient", "diagonal_fisher"],
    }

    # ponytail: resave the three tensors with truthful replay metadata; recomputing
    # the unchanged 80-example Fisher would add GPU-hours without adding evidence.
    for name in ("state", "rank1", "diagonal"):
        payload = torch.load(source_paths[name], map_location="cpu", weights_only=True)
        benchmark._validate_cache_metadata(payload["metadata"], old_metadata)
        payload["metadata"] = new_metadata
        payload["derived_from"] = derivation
        benchmark._atomic_torch(output_paths[name], payload)
        del payload
        gc.collect()

    replay_manifest = {
        **summary["replay"]["manifest"],
        "generation": settings,
        "quality": quality,
    }
    benchmark._atomic_json(output_paths["replay"], {
        "kind": "gsm8k_generated_replay_two_pass_v1",
        "metadata": new_metadata,
        "manifest": replay_manifest,
        "rows": replay_rows,
        "derived_from": derivation,
    })
    result = copy.deepcopy(summary)
    result["created_utc"] = benchmark._utc_now()
    result["metadata"] = new_metadata
    result["replay"] = {
        "manifest": replay_manifest,
        "rows_sha256": benchmark._records_sha256(replay_rows),
    }
    result["cache"] = {name: str(path) for name, path in output_paths.items()}
    result["cache_derivation"] = derivation
    benchmark._atomic_json(output_paths["summary"], result)
    print(json.dumps({
        "status": "ok",
        "cache_prefix": str(args.output_prefix),
        "replay_quality": quality,
    }, indent=2), flush=True)
    return result


def _self_check() -> None:
    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return {"input_ids": list(range(len(text)))}

    row, truncated = _replay_row([7, 8], "abcd", Tokenizer(), 5)
    assert row == {"ids": [7, 8, 0, 1, 2], "answer_start": 2, "answer_end": 5}
    assert truncated
    args = SimpleNamespace(steps=256, context_length=256, cfg=0.1, temperature=0.1)
    metadata = _new_metadata({"replay_steps": 32}, args)
    assert metadata["replay_steps"] == 256 and metadata["replay_cfg"] == 0.1
    print("self-check ok")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gsm-train", type=Path, default=benchmark.DEFAULT_GSM_TRAIN)
    parser.add_argument("--tokenizer", type=Path, default=benchmark.DEFAULT_TOKENIZER)
    parser.add_argument("--model", type=int, choices=tuple(benchmark.PARAMETER_COUNTS), default=1028)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--cfg", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=576)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return args
    for name in ("source_prefix", "output_prefix", "checkpoint"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.batch_size < 1 or args.steps < 1 or args.context_length < 1:
        parser.error("generation sizes must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check()
    else:
        build(args)


if __name__ == "__main__":
    main()
