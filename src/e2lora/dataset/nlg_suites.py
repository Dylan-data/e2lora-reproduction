"""Task datasets for Table 2 and Appendix L.

MetaMathQA is pinned by the repository. The public Code-Feedback snapshot listed
in the repository has fewer than the 100K code-only examples described by the
paper; this loader refuses to silently train on a smaller/duplicated set.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from random import Random
from typing import Any

from e2lora.dataset.loading import load_split
from e2lora.dataset.commonsense_reasoning import normalize_commonsense_row

METAMATH = {
    "repo": "meta-math/MetaMathQA",
    "revision": "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18",
    "split": "train",
}
CODE_FEEDBACK = {
    "repo": "m-a-p/Code-Feedback",
    "revision": "f411b16a97c910ac9acf8b0d0948e340aa77cc34",
    "file": "Code-Feedback.jsonl",
    "listed_approx_rows": 66400,
    "paper_required_code_rows": 100000,
}


def _records_from_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _normalize_code_row(row: dict[str, Any]) -> dict[str, str] | None:
    prompt = row.get("query") or row.get("instruction") or row.get("prompt") or row.get("input")
    target = row.get("answer") or row.get("response") or row.get("output") or row.get("completion")
    if prompt is None or target is None:
        return None
    prompt, target = str(prompt).strip(), str(target).strip()
    if not prompt or not target:
        return None
    # The source repo may mix code and non-code instruction examples. Do not
    # guess code-only membership based on keywords; retain source-provided flags.
    if row.get("is_code") is False or str(row.get("category", "")).lower() in {"math", "general", "writing"}:
        return None
    sample_id = row.get("id") or row.get("uuid") or row.get("task_id")
    return {"prompt": prompt, "target": target, "sample_id": str(sample_id) if sample_id is not None else ""}


def load_metamath_records(*, train_count: int = 100000, validation_count: int = 2000,
                          seed: int = 42, validation_seed: int = 43, cache_dir=None):
    dataset = load_split(METAMATH["repo"], None, split=METAMATH["split"],
                         revision=METAMATH["revision"], cache_dir=cache_dir)
    if len(dataset) < train_count + validation_count:
        raise ValueError(f"MetaMathQA has only {len(dataset)} rows; need {train_count + validation_count}")
    train_ids = Random(seed).sample(range(len(dataset)), train_count)
    remaining = [i for i in range(len(dataset)) if i not in set(train_ids)]
    val_ids = Random(validation_seed).sample(remaining, validation_count)
    train = []
    validation = []
    for dest, ids in ((train, train_ids), (validation, val_ids)):
        for idx in ids:
            row = dataset[int(idx)]
            if "query" not in row or "response" not in row:
                raise KeyError("MetaMathQA row requires query/response fields")
            dest.append({"prompt": str(row["query"]).strip() + "\n",
                         "target": str(row["response"]).strip(),
                         "sample_id": str(row.get("id", idx)), "source_index": int(idx)})
    return train, validation, {"repo": METAMATH["repo"], "revision": METAMATH["revision"],
                               "split": "train", "train_count": len(train),
                               "validation_count": len(validation), "sampling_seed": seed,
                               "validation_seed": validation_seed,
                               "validation_protocol": "disjoint held-out rows from MetaMathQA train (team choice)"}


def load_codefeedback_records(path: str | Path, *, train_count: int = 100000,
                              validation_count: int = 2000, seed: int = 42,
                              validation_seed: int = 43, code_only_confirmed: bool = False):
    if not code_only_confirmed:
        raise ValueError("Provide a pre-filtered code-only JSONL and pass code_only_confirmed=True; the repo does not specify a reliable universal row-level code filter.")
    source_rows = _records_from_jsonl(path)
    usable = []
    for source_index, row in enumerate(source_rows):
        item = _normalize_code_row(row)
        if item is not None:
            item["source_index"] = source_index
            if not item.get("sample_id"):
                item["sample_id"] = f"source-row:{source_index}"
            usable.append(item)
    if len(usable) < train_count + validation_count:
        raise ValueError(
            f"Code-Feedback source {path} provides {len(usable)} usable code-like rows; "
            f"need {train_count + validation_count} (100K train + 2K held-out validation). "
            "The repository-pinned public snapshot is listed as ~66.4K rows and does not "
            "satisfy the paper's 100K code-only training subset. Supply the actual 100K+ "
            "code-only source/manifest; do not duplicate rows or relabel the shortfall."
        )
    ids = Random(seed).sample(range(len(usable)), train_count)
    remaining = [i for i in range(len(usable)) if i not in set(ids)]
    val_ids = Random(validation_seed).sample(remaining, validation_count)
    train = [usable[i] for i in ids]
    validation = [usable[i] for i in val_ids]
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return train, validation, {"repo": CODE_FEEDBACK["repo"], "revision": CODE_FEEDBACK["revision"],
                               "source_file": str(path), "source_file_sha256": digest,
                               "source_rows": len(source_rows),
                               "usable_rows": len(usable), "train_count": len(train),
                               "validation_count": len(validation), "sampling_seed": seed,
                               "validation_seed": validation_seed,
                               "source_revision_caveat": "Local file must be checksum-recorded; actual 100K paper subset not identified in public snapshot."}


def load_humaneval_test(*, cache_dir=None, allow_unpinned=False):
    if not allow_unpinned:
        raise ValueError("HumanEval dataset revision is not pinned in this repo; pass --allow-unpinned-datasets after recording cache SHA256.")
    # The repository's canonical list points to the official HumanEval release.
    return load_split("openai/openai_humaneval", None, split="test", revision=None, cache_dir=cache_dir)


def normalize_humaneval_row(row: dict[str, Any]) -> dict[str, str]:
    prompt = row.get("prompt")
    task_id = row.get("task_id") or row.get("id")
    if prompt is None or task_id is None:
        raise ValueError("HumanEval test rows require prompt and task_id/id")
    return {"prompt": str(prompt), "target": "", "task_id": str(task_id)}
