from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contract_qa.qa_utils import postprocess_qa_predictions, prepare_validation_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run command-line inference for contract QA.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--context", type=str, default=None)
    parser.add_argument("--context-file", type=Path, default=None)
    parser.add_argument("--title", type=str, default="cli_input")
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--doc-stride", type=int, default=512)
    parser.add_argument("--max-answer-length", type=int, default=256)
    parser.add_argument("--n-best-size", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.context) == bool(args.context_file):
        raise SystemExit("Provide exactly one of --context or --context-file.")

    context = args.context
    if args.context_file is not None:
        context = args.context_file.read_text(encoding="utf-8")

    example = {
        "id": "cli-example",
        "title": args.title,
        "context": context,
        "question": args.question,
        "answers": {"text": [], "answer_start": []},
        "is_impossible": False,
    }

    dataset = Dataset.from_list([example])
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    tokenized = dataset.map(
        lambda batch: prepare_validation_features(batch, tokenizer, args.max_seq_length, args.doc_stride),
        batched=True,
        remove_columns=dataset.column_names,
    )
    feature_records = tokenized.to_list()
    model_dataset = Dataset.from_list(
        [
            {key: value for key, value in record.items() if key not in {"offset_mapping", "example_id"}}
            for record in feature_records
        ]
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(PROJECT_DIR / "outputs" / "tmp_predict"),
            per_device_eval_batch_size=1,
            report_to=[],
            do_train=False,
            do_eval=False,
            remove_unused_columns=False,
        ),
        processing_class=tokenizer,
        data_collator=DefaultDataCollator(),
    )
    raw_predictions = trainer.predict(model_dataset).predictions
    records_by_id = postprocess_qa_predictions(
        examples=[example],
        features=feature_records,
        raw_predictions=raw_predictions,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
    )
    record = records_by_id["cli-example"]
    prediction_text = "" if record["score_diff"] > args.threshold else record["best_text"]
    payload = {
        "question": args.question,
        "answer": prediction_text,
        "best_span_text": record["best_text"],
        "score_diff": record["score_diff"],
        "null_score": record["null_score"],
        "best_span_score": record["best_span_score"],
        "best_offsets": record["best_offsets"],
        "threshold": args.threshold,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
