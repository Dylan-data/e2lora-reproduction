"""Dataset adapters for Appendix K commonsense reasoning.

The paper lists the six benchmarks but does not specify prompt wording or the
validation/test split protocol. This module makes both explicit and records that
the default public validation split is a team protocol, not a recovered paper detail.
"""
from __future__ import annotations

from typing import Any

from e2lora.dataset.loading import load_split

DATASETS = {
    "arc_challenge": {"repo": "allenai/ai2_arc", "subset": "ARC-Challenge",
                      "revision": "210d026faf9955653af8916fad021475a3f00453",
                      "paper_name": "ARC-Challenge", "short_name": "ARC-C"},
    "arc_easy": {"repo": "allenai/ai2_arc", "subset": "ARC-Easy",
                  "revision": "210d026faf9955653af8916fad021475a3f00453",
                  "paper_name": "ARC-Easy", "short_name": "ARC-E"},
    "boolq": {"repo": "google/boolq", "subset": None,
              "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
              "paper_name": "BoolQ", "short_name": "BoolQ"},
    "openbookqa": {"repo": "allenai/openbookqa", "subset": "main",
                   "revision": "388097ea7776314e93a529163e0fea805b8a6454",
                   "paper_name": "OpenBookQA", "short_name": "OBQA"},
    "piqa": {"repo": "ybisk/piqa", "subset": "plain_text",
        "revision": "f7af4e534d3934fcce5511153df45ba340543d9f",
        "paper_name": "PIQA", "short_name": "PIQA",},
    "siqa": {"repo": "allenai/social_i_qa", "subset": "default",
        "revision": "a8e45ea64a25b66aba3da431e50cca860d1a982b",
        "paper_name": "Social IQa", "short_name": "SIQA",},
    }


def _choice_label(index: int) -> str:
    return chr(ord("A") + index)


def _normalize_label(value: Any, labels: list[str]) -> str:
    raw = str(value).strip()
    if raw in labels:
        return raw
    if raw.isdigit():
        number = int(raw)
        # ARC/OBQA label lists can be 1-based or 0-based; use exact numeric label first.
        if raw in labels:
            return raw
        if 0 <= number < len(labels):
            return labels[number]
        if 1 <= number <= len(labels):
            return labels[number - 1]
    raise ValueError(f"Could not map answer label {value!r} to {labels}")


def normalize_commonsense_row(task: str, row: dict[str, Any]) -> dict[str, str]:
    """Return prompt, target label and stable source fields for supported tasks."""
    if task not in DATASETS:
        raise ValueError(f"Unknown commonsense task {task!r}; choose from {tuple(DATASETS)}")
    if task in ("arc_challenge", "arc_easy", "openbookqa"):
        question = row.get("question", {})
        if task == "openbookqa":
            stem = row.get("question_stem") or question.get("stem") or question.get("question")
        else:
            stem = question.get("stem") if isinstance(question, dict) else question
        choices_obj = row.get("choices") or question.get("choices") or {}
        if isinstance(choices_obj, dict):
            labels = list(choices_obj.get("label", []))
            texts = list(choices_obj.get("text", []))
        else:
            labels = [str(item.get("label", _choice_label(i))) for i, item in enumerate(choices_obj)]
            texts = [str(item.get("text", "")) for item in choices_obj]
        if not stem or not labels or len(labels) != len(texts):
            raise ValueError(f"Malformed {task} row: missing stem or choices")
        rendered = "\n".join(f"{_choice_label(i)}. {text}" for i, text in enumerate(texts))
        answer_key = row.get("answerKey") or row.get("answer_key")
        if answer_key is None:
            raise KeyError(f"{task} row missing answerKey")
        try:
            answer_index = labels.index(str(answer_key))
        except ValueError:
            # ARC-Style data sometimes uses numeric answerKey with 1-based indexing.
            answer_index = int(str(answer_key)) - 1 if str(answer_key).isdigit() else -1
        if not 0 <= answer_index < len(labels):
            raise ValueError(f"Invalid {task} answer key {answer_key!r}, labels={labels}")
        target = _choice_label(answer_index)
        question_text = str(stem)
    elif task == "boolq":
        passage, question_text = row.get("passage"), row.get("question")
        if passage is None or question_text is None:
            raise ValueError("BoolQ row requires passage and question")
        rendered = "A. Yes\nB. No"
        answer = row.get("answer")
        target = "A" if answer is True or str(answer).lower() == "true" else "B"
        question_text = f"Passage: {passage}\nQuestion: {question_text}"
    elif task == "piqa":
        goal = row.get("goal")
        choices = [row.get("sol1"), row.get("sol2")]
        label = row.get("label")
        if goal is None or any(x is None for x in choices) or label is None:
            raise ValueError("PIQA row requires goal, sol1, sol2 and label")
        rendered = f"A. {choices[0]}\nB. {choices[1]}"
        target = "A" if int(label) == 0 else "B"
        question_text = str(goal)
    else:  # Social IQa
        context = row.get("context")
        question_text = row.get("question")
        choices = [row.get("answerA"), row.get("answerB"), row.get("answerC")]
        label = row.get("label")
        if context is None or question_text is None or any(x is None for x in choices) or label is None:
            raise ValueError("SIQA row requires context, question, answerA/B/C and label")
        rendered = "\n".join(f"{_choice_label(i)}. {answer}" for i, answer in enumerate(choices))
        target = _choice_label(int(label) - 1)
        question_text = f"Context: {context}\nQuestion: {question_text}"
    prompt = (
        "Answer the following commonsense reasoning question. Choose the best option.\n"
        f"{question_text}\n{rendered}\n"
        "Respond with only the option letter.\nAnswer:"
    )
    return {"prompt": prompt, "target": target, "question": str(question_text)}


def load_commonsense_split(task: str, split: str, *, cache_dir=None, allow_unpinned: bool = False,):
    if task not in DATASETS:
        raise ValueError(
            f"Unsupported task {task}; "
            f"supported: {tuple(DATASETS)}"
        )

    cfg = DATASETS[task]
    revision = cfg["revision"]

    if not revision and not allow_unpinned:
        raise ValueError(
            f"{task} has no commit-pinned revision."
        )

    return load_split(
        cfg["repo"],
        cfg["subset"],
        split=split,
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )


def evaluate_split_for_public_labels(task: str) -> str:
    """Public labeled validation is the default reproducible protocol for Table 14."""
    return "validation"
