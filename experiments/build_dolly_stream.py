#!/usr/bin/env python3
"""Build the fixed, prompt-disjoint Dolly stream used by the natural-task study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path


DATASET = "databricks/databricks-dolly-15k"
REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
TASKS = ("closed_qa", "summarization", "creative_writing")


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format_prompt(row: dict) -> str:
    prompt = f"Instruction: {row['instruction'].strip()}"
    context = row.get("context", "").strip()
    if context:
        prompt += f"\nContext: {context}"
    return prompt + "\nResponse:"


def _token_lengths(tokenizer, prompt: str, answer: str) -> tuple[int, int]:
    prefix = ([] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id])
    prompt_ids = prefix + tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(" " + answer.strip(), add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    return len(prompt_ids), len(prompt_ids) + len(answer_ids) + 1


def _selection_key(seed: int, task: str, source_index: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{task}\0{source_index}".encode()).digest()


def _select(candidates: list[dict], task: str, seed: int, train: int, test: int) -> list[dict]:
    ordered = sorted(candidates, key=lambda row: _selection_key(seed, task, row["source_index"]))
    if len(ordered) < train + test:
        raise ValueError(f"{task}: need {train + test} eligible rows, found {len(ordered)}")
    selected = ordered[: train + test]
    for index, row in enumerate(selected):
        row["split"] = "train" if index < train else "test"
        row["example_id"] = f"{task}:{row['split']}:{index if index < train else index - train}"
    return selected


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build(args) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    smdm = AutoTokenizer.from_pretrained(args.smdm_tokenizer, local_files_only=True, use_fast=True)
    qwen = AutoTokenizer.from_pretrained(args.qwen_tokenizer, local_files_only=True, use_fast=True)
    source = load_dataset(DATASET, split="train", revision=REVISION)
    candidates = {task: [] for task in TASKS}
    rejected = Counter()
    source_records = []
    for source_index, raw in enumerate(source):
        category = raw["category"]
        if category not in candidates:
            continue
        prompt = _format_prompt(raw)
        answer = raw["response"].strip()
        if not answer:
            rejected[(category, "empty_answer")] += 1
            continue
        lengths = {
            "smdm": _token_lengths(smdm, prompt, answer),
            "qwen": _token_lengths(qwen, prompt, answer),
        }
        if any(prompt_length > args.max_prompt_tokens for prompt_length, _ in lengths.values()):
            rejected[(category, "prompt_length")] += 1
            continue
        if any(total_length > args.max_length for _, total_length in lengths.values()):
            rejected[(category, "total_length")] += 1
            continue
        record = {
            "task": category,
            "source_index": source_index,
            "prompt": prompt,
            "answer": " " + answer,
            "token_lengths": {
                name: {"prompt": values[0], "total": values[1]} for name, values in lengths.items()
            },
        }
        candidates[category].append(record)
        source_records.append({
            "source_index": source_index,
            "category": category,
            "instruction": raw["instruction"],
            "context": raw.get("context", ""),
            "response": raw["response"],
        })

    rows = []
    for task in TASKS:
        rows.extend(_select(candidates[task], task, args.seed, args.train_per_task, args.test_per_task))
    prompts = [row["prompt"] for row in rows]
    if len(prompts) != len(set(prompts)):
        raise ValueError("selected prompts are not globally disjoint")

    jsonl = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    selected_source_indices = {row["source_index"] for row in rows}
    manifest = {
        "schema_version": 1,
        "status": "ok",
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "dataset_fingerprint": source._fingerprint,
        "license": "CC BY-SA 3.0",
        "builder_sha256": _file_digest(Path(__file__)),
        "selection_seed": args.seed,
        "task_order": list(TASKS),
        "train_per_task": args.train_per_task,
        "test_per_task": args.test_per_task,
        "max_length": args.max_length,
        "max_prompt_tokens": args.max_prompt_tokens,
        "tokenizers": {
            "smdm": str(args.smdm_tokenizer),
            "qwen": str(args.qwen_tokenizer),
        },
        "eligible_counts": {task: len(candidates[task]) for task in TASKS},
        "rejected_counts": {f"{task}:{reason}": count for (task, reason), count in sorted(rejected.items())},
        "selected_counts": {
            task: dict(Counter(row["split"] for row in rows if row["task"] == task)) for task in TASKS
        },
        "prompt_disjoint": len(prompts) == len(set(prompts)),
        "selected_rows_sha256": _digest(rows),
        "selected_source_sha256": _digest([
            record for record in source_records if record["source_index"] in selected_source_indices
        ]),
        "per_split_sha256": {
            f"{task}:{split}": _digest([
                row for row in rows if row["task"] == task and row["split"] == split
            ])
            for task in TASKS for split in ("train", "test")
        },
    }
    _write_atomic(args.output, jsonl)
    _write_atomic(args.manifest, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def _self_check() -> None:
    rows = [{"source_index": index} for index in range(12)]
    first = _select([dict(row) for row in rows], "task", 7, 4, 2)
    second = _select([dict(row) for row in rows], "task", 7, 4, 2)
    assert first == second and len(first) == 6
    assert Counter(row["split"] for row in first) == {"train": 4, "test": 2}
    assert len({row["source_index"] for row in first}) == len(first)
    assert _format_prompt({"instruction": "Do it", "context": "Given this"}) == (
        "Instruction: Do it\nContext: Given this\nResponse:"
    )
    print(json.dumps({"self_check": "ok"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/data/dolly_natural_stream.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/data/dolly_natural_stream_manifest.json"))
    parser.add_argument("--smdm-tokenizer", type=Path, default=Path("tokenizer"))
    parser.add_argument("--qwen-tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--train-per-task", type=int, default=120)
    parser.add_argument("--test-per-task", type=int, default=40)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=160)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    manifest = build(args)
    print(json.dumps({
        "status": "ok",
        "output": str(args.output),
        "manifest": str(args.manifest),
        "selected_counts": manifest["selected_counts"],
    }))


if __name__ == "__main__":
    main()
