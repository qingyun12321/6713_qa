from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from datasets import load_dataset

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contract_qa.data_utils import save_json, summarize_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download SQuAD v2, normalize it into the project's QA JSON format, "
            "and split the official validation set 1:1 into validation and test."
        )
    )
    parser.add_argument("--dataset-name", type=str, default="rajpurkar/squad_v2")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / "squad_v2_prepared")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_squad_example(example: dict) -> dict:
    answers = example.get("answers", {}) or {}
    texts = list(answers.get("text", []) or [])
    starts = list(answers.get("answer_start", []) or [])
    if len(texts) != len(starts):
        raise ValueError(f"Answer text/start length mismatch for example {example['id']}")

    is_impossible = bool(example.get("is_impossible", len(texts) == 0)) or len(texts) == 0
    return {
        "id": example["id"],
        "title": example.get("title", "").strip(),
        "context": example["context"],
        "question": example["question"].strip(),
        "answers": {
            "text": texts,
            "answer_start": starts,
        },
        "is_impossible": is_impossible,
    }


def split_validation_test(examples: list[dict], seed: int) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    validation_examples: list[dict] = []
    test_examples: list[dict] = []
    bucket_manifest: dict[str, dict[str, int]] = {}

    for is_impossible, bucket_name in ((False, "has_answer"), (True, "no_answer")):
        bucket = [example for example in examples if example["is_impossible"] == is_impossible]
        rng.shuffle(bucket)
        split_index = len(bucket) // 2
        validation_bucket = bucket[:split_index]
        test_bucket = bucket[split_index:]
        validation_examples.extend(validation_bucket)
        test_examples.extend(test_bucket)
        bucket_manifest[bucket_name] = {
            "source_examples": len(bucket),
            "validation_examples": len(validation_bucket),
            "test_examples": len(test_bucket),
        }

    rng.shuffle(validation_examples)
    rng.shuffle(test_examples)

    manifest = {
        "seed": seed,
        "split_strategy": "1:1 split from official validation, stratified by is_impossible",
        "source_validation_examples": len(examples),
        "validation_examples": len(validation_examples),
        "test_examples": len(test_examples),
        "buckets": bucket_manifest,
    }
    return validation_examples, test_examples, manifest


def main() -> None:
    args = parse_args()

    dataset = load_dataset(args.dataset_name)
    train_examples = [normalize_squad_example(example) for example in dataset["train"]]
    official_validation_examples = [normalize_squad_example(example) for example in dataset["validation"]]
    validation_examples, test_examples, split_manifest = split_validation_test(
        official_validation_examples,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "train.json", train_examples)
    save_json(args.output_dir / "validation.json", validation_examples)
    save_json(args.output_dir / "test.json", test_examples)
    save_json(
        args.output_dir / "split_manifest.json",
        {
            "dataset_name": args.dataset_name,
            "source_splits": {
                "train": len(train_examples),
                "official_validation": len(official_validation_examples),
            },
            **split_manifest,
        },
    )
    save_json(
        args.output_dir / "summary.json",
        {
            "dataset_name": args.dataset_name,
            "train": summarize_examples(train_examples),
            "validation": summarize_examples(validation_examples),
            "test": summarize_examples(test_examples),
        },
    )

    print(f"Saved prepared SQuAD v2 data to {args.output_dir}")
    print(f"Train rows: {len(train_examples)}")
    print(f"Validation rows: {len(validation_examples)}")
    print(f"Test rows: {len(test_examples)}")


if __name__ == "__main__":
    main()
