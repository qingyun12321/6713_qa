from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contract_qa_longformer.data_utils import (
    grouped_train_val_split,
    load_json,
    normalize_examples,
    save_json,
    summarize_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare grouped train/validation splits for contract QA.")
    parser.add_argument("--raw-train-path", type=Path, required=True)
    parser.add_argument("--raw-test-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_examples = normalize_examples(load_json(args.raw_train_path))
    test_examples = normalize_examples(load_json(args.raw_test_path))

    split_train, split_val, manifest = grouped_train_val_split(
        train_examples,
        group_key="title",
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "train.json", split_train)
    save_json(args.output_dir / "validation.json", split_val)
    save_json(args.output_dir / "test.json", test_examples)
    save_json(args.output_dir / "split_manifest.json", manifest)
    save_json(
        args.output_dir / "summary.json",
        {
            "train": summarize_examples(split_train),
            "validation": summarize_examples(split_val),
            "test": summarize_examples(test_examples),
        },
    )

    print(f"Saved prepared data to {args.output_dir}")
    print(f"Train rows: {len(split_train)}")
    print(f"Validation rows: {len(split_val)}")
    print(f"Test rows: {len(test_examples)}")


if __name__ == "__main__":
    main()

