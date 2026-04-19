from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contract_qa.data_utils import load_json
from contract_qa.qa_utils import prepare_validation_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile token lengths and window expansion factors.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--max-seq-lengths", type=int, nargs="+", default=[2048, 3072, 4096])
    parser.add_argument("--doc-strides", type=int, nargs="+", default=[256, 512, 1024])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    examples = load_json(args.data_path)[: args.sample_size]
    dataset = Dataset.from_list(examples)

    for max_seq_length, doc_stride in itertools.product(args.max_seq_lengths, args.doc_strides):
        tokenized = dataset.map(
            lambda batch: prepare_validation_features(batch, tokenizer, max_seq_length, doc_stride),
            batched=True,
            remove_columns=dataset.column_names,
        )
        expansion = len(tokenized) / max(1, len(dataset))
        context_windows = [record["example_id"] for record in tokenized]
        print(
            f"max_seq_length={max_seq_length} doc_stride={doc_stride} "
            f"features={len(tokenized)} examples={len(dataset)} expansion={expansion:.2f}"
        )


if __name__ == "__main__":
    main()
