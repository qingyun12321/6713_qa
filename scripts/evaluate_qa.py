from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, DefaultDataCollator, Trainer, TrainingArguments

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contract_qa_longformer.data_utils import iter_json_array_items, save_json
from contract_qa_longformer.metrics import evaluate_prediction_records, search_best_threshold
from contract_qa_longformer.qa_utils import postprocess_qa_predictions, prepare_validation_features

LOG_PATH: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a QA checkpoint on validation or test data.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--doc-stride", type=int, default=512)
    parser.add_argument("--max-answer-length", type=int, default=256)
    parser.add_argument("--n-best-size", type=int, default=20)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--search-threshold", action="store_true")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--preprocess-chunk-size", type=int, default=100)
    return parser.parse_args()


def iter_example_chunks(path: Path, chunk_size: int, limit: int | None):
    chunk: list[dict] = []
    yielded = 0
    for example in iter_json_array_items(path):
        chunk.append(example)
        yielded += 1
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
        if limit is not None and yielded >= limit:
            break
    if chunk:
        yield chunk


def compact_example(example: dict) -> dict:
    return {
        "id": example["id"],
        "title": example["title"],
        "answers": example["answers"],
        "is_impossible": example["is_impossible"],
    }


def current_rss_mb() -> float | None:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except OSError:
        return None
    return None


def log_event(message: str, **payload) -> None:
    event = {"message": message, **payload}
    line = json.dumps(event, ensure_ascii=False)
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def main() -> None:
    global LOG_PATH
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    LOG_PATH = args.output_dir / "eval.log"
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output_dir / "tmp_eval"),
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            report_to=[],
            do_train=False,
            do_eval=False,
            remove_unused_columns=False,
            bf16_full_eval=torch.cuda.is_available(),
        ),
        processing_class=tokenizer,
        data_collator=DefaultDataCollator(),
    )
    log_event(
        "evaluation_started",
        pid=os.getpid(),
        device=str(trainer.args.device),
        n_gpu=trainer.args.n_gpu,
        bf16_full_eval=trainer.args.bf16_full_eval,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        preprocess_chunk_size=args.preprocess_chunk_size,
        rss_mb=current_rss_mb(),
    )

    all_examples: list[dict] = []
    records_by_id: dict[str, dict] = {}
    num_features = 0

    for chunk_index, chunk_examples in enumerate(
        iter_example_chunks(
            args.data_path,
            chunk_size=args.preprocess_chunk_size,
            limit=args.max_eval_samples,
        ),
        start=1,
    ):
        chunk_start = time.time()
        log_event(
            "chunk_started",
            chunk_index=chunk_index,
            chunk_examples=len(chunk_examples),
            processed_examples=len(all_examples),
            rss_mb=current_rss_mb(),
        )
        dataset = Dataset.from_list(chunk_examples)
        tokenized = dataset.map(
            lambda batch: prepare_validation_features(batch, tokenizer, args.max_seq_length, args.doc_stride),
            batched=True,
            remove_columns=dataset.column_names,
        )
        feature_records = tokenized.to_list()
        chunk_feature_count = len(feature_records)
        num_features += chunk_feature_count
        log_event(
            "chunk_tokenized",
            chunk_index=chunk_index,
            chunk_examples=len(chunk_examples),
            chunk_features=chunk_feature_count,
            total_features=num_features,
            elapsed_sec=round(time.time() - chunk_start, 2),
            rss_mb=current_rss_mb(),
        )
        model_dataset = Dataset.from_list(
            [
                {key: value for key, value in record.items() if key not in {"offset_mapping", "example_id"}}
                for record in feature_records
            ]
        )

        log_event(
            "chunk_predict_started",
            chunk_index=chunk_index,
            chunk_examples=len(chunk_examples),
            chunk_features=chunk_feature_count,
            rss_mb=current_rss_mb(),
        )
        raw_predictions = trainer.predict(model_dataset).predictions
        chunk_records_by_id = postprocess_qa_predictions(
            examples=chunk_examples,
            features=feature_records,
            raw_predictions=raw_predictions,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
        )
        all_examples.extend(compact_example(example) for example in chunk_examples)
        records_by_id.update(chunk_records_by_id)

        del dataset
        del tokenized
        del feature_records
        del model_dataset
        del raw_predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_event(
            "chunk_finished",
            chunk_index=chunk_index,
            chunk_examples=len(chunk_examples),
            chunk_features=chunk_feature_count,
            processed_examples=len(all_examples),
            elapsed_sec=round(time.time() - chunk_start, 2),
            total_elapsed_sec=round(time.time() - start_time, 2),
            rss_mb=current_rss_mb(),
        )

    if args.search_threshold:
        log_event("threshold_search_started", num_examples=len(all_examples), total_features=num_features)
        best_exact_threshold, best_exact_metrics = search_best_threshold(all_examples, records_by_id, metric_name="exact")
        best_f1_threshold, best_f1_metrics = search_best_threshold(all_examples, records_by_id, metric_name="f1")
        selected_threshold = best_f1_threshold
    else:
        best_exact_threshold = None
        best_exact_metrics = None
        best_f1_threshold = None
        best_f1_metrics = None
        selected_threshold = args.threshold

    selected_metrics = evaluate_prediction_records(all_examples, records_by_id, selected_threshold)

    predictions = []
    for example in all_examples:
        record = records_by_id[example["id"]]
        prediction_text = "" if record["score_diff"] > selected_threshold else record["best_text"]
        predictions.append(
            {
                "id": example["id"],
                "title": example["title"],
                "prediction_text": prediction_text,
                "best_text": record["best_text"],
                "score_diff": record["score_diff"],
                "null_score": record["null_score"],
                "best_span_score": record["best_span_score"],
                "best_offsets": record["best_offsets"],
                "references": example["answers"]["text"],
                "is_impossible": example["is_impossible"],
            }
        )

    summary = {
        "data_path": str(args.data_path),
        "model_path": str(args.model_path),
        "num_examples": len(all_examples),
        "num_features": num_features,
        "max_seq_length": args.max_seq_length,
        "doc_stride": args.doc_stride,
        "max_answer_length": args.max_answer_length,
        "preprocess_chunk_size": args.preprocess_chunk_size,
        "selected_threshold": selected_threshold,
        "selected_metrics": selected_metrics,
        "best_exact_threshold": best_exact_threshold,
        "best_exact_metrics": best_exact_metrics,
        "best_f1_threshold": best_f1_threshold,
        "best_f1_metrics": best_f1_metrics,
    }

    save_json(args.output_dir / "predictions.json", predictions)
    save_json(args.output_dir / "evaluation_summary.json", summary)
    log_event(
        "evaluation_finished",
        num_examples=len(all_examples),
        num_features=num_features,
        total_elapsed_sec=round(time.time() - start_time, 2),
        selected_threshold=selected_threshold,
        rss_mb=current_rss_mb(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
