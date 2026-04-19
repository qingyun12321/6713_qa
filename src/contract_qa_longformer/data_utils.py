from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import ijson


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_json_array_items(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        yield from ijson.items(f, "item")


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def answers_list_to_hf(answer_items: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "text": [item["text"] for item in answer_items],
        "answer_start": [item["answer_start"] for item in answer_items],
    }


def normalize_examples(raw_examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for example in raw_examples:
        answers = answers_list_to_hf(example.get("answers", []))
        normalized.append(
            {
                "id": example["id"],
                "title": example["title"],
                "context": example["context"],
                "question": example["question"],
                "answers": answers,
                "is_impossible": bool(example.get("is_impossible", False)),
            }
        )
    return normalized


def grouped_train_val_split(
    examples: list[dict[str, Any]],
    group_key: str = "title",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    group_to_examples: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        group_to_examples.setdefault(example[group_key], []).append(example)

    groups = list(group_to_examples)
    rng = random.Random(seed)
    rng.shuffle(groups)

    n_val_groups = max(1, int(round(len(groups) * val_ratio)))
    val_groups = set(groups[:n_val_groups])

    train_examples: list[dict[str, Any]] = []
    val_examples: list[dict[str, Any]] = []
    for group_name, group_examples in group_to_examples.items():
        if group_name in val_groups:
            val_examples.extend(group_examples)
        else:
            train_examples.extend(group_examples)

    manifest = {
        "seed": seed,
        "group_key": group_key,
        "val_ratio": val_ratio,
        "train_groups": sorted(set(example[group_key] for example in train_examples)),
        "val_groups": sorted(val_groups),
    }
    return train_examples, val_examples, manifest


def summarize_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    title_counts = Counter(example["title"] for example in examples)
    no_answer = sum(1 for example in examples if example["is_impossible"])
    answer_counts = Counter(len(example["answers"]["text"]) for example in examples)
    context_lengths = [len(example["context"]) for example in examples]
    question_lengths = [len(example["question"]) for example in examples]
    return {
        "rows": len(examples),
        "unique_titles": len(title_counts),
        "no_answer_rows": no_answer,
        "no_answer_ratio": no_answer / len(examples) if examples else 0.0,
        "answer_count_distribution": dict(sorted(answer_counts.items())),
        "context_chars": summarize_numeric(context_lengths),
        "question_chars": summarize_numeric(question_lengths),
        "top_titles": title_counts.most_common(10),
    }


def summarize_numeric(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "median": sorted_values[len(sorted_values) // 2],
        "p90": sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * 0.90))],
        "p95": sorted_values[min(len(sorted_values) - 1, int(len(sorted_values) * 0.95))],
        "max": sorted_values[-1],
    }
