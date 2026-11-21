"""Deterministic generation-data manifests and answer-preserving tokenization."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from random import Random
from statistics import mean
from typing import Iterable

import torch


def canonical_row_hash(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_id(row: dict, row_index: int):
    for key in ("id", "uuid", "example_id", "problem_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    return None


def file_checksums(dataset) -> list[dict]:
    """Hash locally cached dataset files when the datasets library exposes them."""
    results = []
    for item in getattr(dataset, "cache_files", []) or []:
        filename = item.get("filename")
        if not filename or not Path(filename).is_file():
            results.append({"path": filename, "sha256": None, "status": "not_local"})
            continue
        path = Path(filename)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        results.append({"path": str(path), "bytes": path.stat().st_size,
                        "sha256": digest.hexdigest(), "status": "hashed"})
    return results


def create_or_load_manifest(
    dataset,
    path: str | Path,
    *,
    repo: str,
    revision: str,
    split: str,
    train_count: int = 100_000,
    validation_count: int = 2_000,
    seed: int = 42,
    validation_seed: int = 43,
) -> dict:
    """Persist exact source indices, row IDs/hashes, and data provenance."""
    path = Path(path)
    dataset_size = len(dataset)
    fingerprint = getattr(dataset, "_fingerprint", None)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = (repo, revision, split, dataset_size, train_count,
                    validation_count, seed, validation_seed)
        actual = (manifest.get("repo"), manifest.get("revision"), manifest.get("split"),
                  manifest.get("dataset_rows"), manifest.get("train_count"),
                  manifest.get("validation_count"), manifest.get("sampling_seed"),
                  manifest.get("validation_seed"))
        if actual != expected:
            raise ValueError(f"Existing manifest does not match requested data: {actual} != {expected}")
        if manifest.get("dataset_fingerprint") != fingerprint:
            raise ValueError("Dataset fingerprint changed since manifest creation")
        return manifest

    if train_count <= 0 or validation_count < 0 or train_count + validation_count > dataset_size:
        raise ValueError(
            f"Need train_count + validation_count <= {dataset_size}; got "
            f"{train_count} + {validation_count}"
        )
    train_indices = Random(seed).sample(range(dataset_size), train_count)
    train_set = set(train_indices)
    remaining = [index for index in range(dataset_size) if index not in train_set]
    validation_indices = Random(validation_seed).sample(remaining, validation_count)

    def describe(indices):
        rows = []
        for index in indices:
            row = dataset[int(index)]
            rows.append({
                "row_index": int(index),
                "sample_id": _sample_id(row, int(index)),
                "row_sha256": canonical_row_hash(row),
            })
        return rows

    manifest = {
        "manifest_version": 1,
        "repo": repo,
        "revision": revision,
        "split": split,
        "dataset_rows": dataset_size,
        "dataset_fingerprint": fingerprint,
        "cache_file_checksums": file_checksums(dataset),
        "sampling_seed": seed,
        "validation_seed": validation_seed,
        "train_count": train_count,
        "validation_count": validation_count,
        "train": describe(train_indices),
        "validation": describe(validation_indices),
    }
    train_ids = {item["row_index"] for item in manifest["train"]}
    val_ids = {item["row_index"] for item in manifest["validation"]}
    if len(train_ids) != train_count or len(val_ids) != validation_count or train_ids & val_ids:
        raise RuntimeError("Manifest has duplicate or overlapping row indices")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return manifest


def math_prompt_answer(row: dict, eos_token: str | None = None) -> tuple[str, str]:
    if "query" not in row or "response" not in row:
        raise KeyError("MetaMathQA row must contain 'query' and 'response' fields")
    prompt = str(row["query"]).rstrip() + "\n"
    answer = str(row["response"]).strip()
    if not prompt.strip() or not answer:
        raise ValueError("MetaMathQA query/response must both be non-empty")
    return prompt, answer


def tokenize_math_example(tokenizer, row: dict) -> dict:
    prompt, answer = math_prompt_answer(row)
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False, truncation=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = list(answer_ids) + [int(tokenizer.eos_token_id)]
    return {
        "prompt_ids": list(prompt_ids),
        "answer_ids": list(answer_ids),
        "prompt_length": len(prompt_ids),
        "answer_length": len(answer_ids),
        "total_length": len(prompt_ids) + len(answer_ids),
    }


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return int(round(ordered[lower] * (upper - position) + ordered[upper] * (position - lower)))


def analyze_math_lengths(
    tokenizer, rows: Iterable[dict], *, max_examples: int | None = None,
    candidate_max_lengths: list[int] | None = None,
) -> dict:
    prompt_lengths, answer_lengths, combined_lengths = [], [], []
    answer_only = []
    count = 0
    for row in rows:
        item = tokenize_math_example(tokenizer, row)
        prompt_lengths.append(item["prompt_length"])
        answer_lengths.append(item["answer_length"])
        combined_lengths.append(item["total_length"])
        answer_only.append(item["answer_length"])
        count += 1
        if max_examples is not None and count >= max_examples:
            break

    def stats(values):
        return {
            "count": len(values),
            "min": min(values) if values else None,
            "mean": mean(values) if values else None,
            "p50": _percentile(values, 0.50),
            "p90": _percentile(values, 0.90),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "max": max(values) if values else None,
        }

    truncation_candidates = []
    for candidate in candidate_max_lengths or []:
        if candidate < 2:
            raise ValueError("Candidate max sequence lengths must be at least 2")
        # The collator never truncates answer tokens; it raises when the answer
        # alone will not fit. This table reports that risk before selecting a cap.
        # With ordinary right truncation, any sequence longer than the cap can
        # lose response tokens because the response follows the prompt.
        right_truncation_risk = sum(total > candidate for total in combined_lengths)
        answer_cannot_fit = sum(length >= candidate for length in answer_lengths)
        prompt_would_need_trimming = sum(
            total > candidate and answer < candidate
            for total, answer in zip(combined_lengths, answer_lengths)
        )
        truncation_candidates.append({
            "candidate_max_seq_length": candidate,
            "examples_with_answer_truncated_by_naive_right_truncation": right_truncation_risk,
            "answer_truncation_rate_if_naively_right_truncated": right_truncation_risk / count if count else None,
            "answer_alone_too_long_to_fit_with_prompt": answer_cannot_fit,
            "examples_requiring_prompt_left_truncation": prompt_would_need_trimming,
            "answer_preserving_collator_would_raise": answer_cannot_fit > 0,
        })

    return {
        "examples_analyzed": count,
        "prompt_tokens": stats(prompt_lengths),
        "answer_tokens_including_eos": stats(answer_lengths),
        "combined_tokens": stats(combined_lengths),
        "minimum_max_seq_length_to_fit_all_answers": max(answer_lengths) + 1 if answer_lengths else None,
        "candidate_length_truncation_checks": truncation_candidates,
        "max_seq_length_selected": None,
        "note": "No max sequence length has been selected; choose only after reviewing these distributions and candidate truncation checks.",
    }


def collate_math(tokenizer, rows: list[dict], *, max_seq_length: int) -> dict:
    """Right-pad batches; preserve all target answer tokens, trimming prompt from the left."""
    if max_seq_length < 2:
        raise ValueError("max_seq_length must be at least 2")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    features = []
    for row in rows:
        item = tokenize_math_example(tokenizer, row)
        prompt_ids = item["prompt_ids"]
        answer_ids = item["answer_ids"]
        if len(answer_ids) >= max_seq_length:
            raise ValueError(
                f"Answer length {len(answer_ids)} leaves no room for prompt at max_seq_length={max_seq_length}"
            )
        prompt_truncated = max(0, len(prompt_ids) + len(answer_ids) - max_seq_length)
        if prompt_truncated:
            prompt_ids = prompt_ids[-(max_seq_length - len(answer_ids)):]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        features.append((input_ids, labels, prompt_truncated, len(answer_ids)))

    max_batch_length = max(len(item[0]) for item in features)
    pad_id = int(tokenizer.pad_token_id)
    input_ids, attention_mask, labels = [], [], []
    for ids, target, _, _ in features:
        padding = max_batch_length - len(ids)
        input_ids.append(ids + [pad_id] * padding)
        attention_mask.append([1] * len(ids) + [0] * padding)
        labels.append(target + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "target_tokens": sum(item[3] for item in features),
        "prompt_tokens_truncated": sum(item[2] for item in features),
        "answer_tokens_truncated": 0,
    }
