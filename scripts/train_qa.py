from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import logging
import math
import os
import platform
import resource
import shutil
import signal
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DefaultDataCollator,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contract_qa.data_utils import iter_json_array_items
from contract_qa.qa_utils import prepare_train_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Longformer on grouped contract QA data.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--doc-stride", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--preprocess-chunk-size", type=int, default=100)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def read_meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    try:
        with Path("/proc/meminfo").open("r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":", 1)
                info[key] = int(value.strip().split()[0])
    except OSError:
        return {}
    return info


def get_runtime_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "pid": os.getpid(),
        "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
    }

    meminfo = read_meminfo()
    if meminfo:
        snapshot["system_mem_total_mb"] = round(meminfo.get("MemTotal", 0) / 1024, 2)
        snapshot["system_mem_available_mb"] = round(meminfo.get("MemAvailable", 0) / 1024, 2)

    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        snapshot.update(
            {
                "cuda_device": torch.cuda.get_device_name(device_idx),
                "cuda_total_mb": round(props.total_memory / (1024**2), 2),
                "cuda_allocated_mb": round(torch.cuda.memory_allocated(device_idx) / (1024**2), 2),
                "cuda_reserved_mb": round(torch.cuda.memory_reserved(device_idx) / (1024**2), 2),
                "cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated(device_idx) / (1024**2), 2),
                "cuda_max_reserved_mb": round(torch.cuda.max_memory_reserved(device_idx) / (1024**2), 2),
            }
        )

    return snapshot


def setup_logger(log_path: Path) -> tuple[logging.Logger, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("a", encoding="utf-8", buffering=1)
    logger = logging.getLogger(f"train_qa.{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger, log_stream


def log_json(logger: logging.Logger, event: str, payload: dict[str, Any]) -> None:
    logger.info("%s | %s", event, json.dumps(payload, ensure_ascii=False, sort_keys=True))


class ResourceLoggingCallback(TrainerCallback):
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def on_train_begin(self, args, state, control, **kwargs):
        log_json(
            self.logger,
            "train_begin",
            {
                "global_step": state.global_step,
                "max_steps": state.max_steps,
                "num_train_epochs": args.num_train_epochs,
                "snapshot": get_runtime_snapshot(),
            },
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        log_json(
            self.logger,
            "trainer_log",
            {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "logs": logs or {},
                "snapshot": get_runtime_snapshot(),
            },
        )

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        log_json(
            self.logger,
            "checkpoint_saved",
            {
                "global_step": state.global_step,
                "checkpoint_dir": str(checkpoint_dir),
                "snapshot": get_runtime_snapshot(),
            },
        )

    def on_train_end(self, args, state, control, **kwargs):
        log_json(
            self.logger,
            "train_end",
            {
                "global_step": state.global_step,
                "snapshot": get_runtime_snapshot(),
            },
        )


def iter_example_chunks(path: Path, chunk_size: int, limit: int | None):
    chunk: list[dict[str, Any]] = []
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


def preprocess_json_chunks_to_disk(
    *,
    json_path: Path,
    output_dir: Path,
    tokenizer,
    max_seq_length: int,
    doc_stride: int,
    chunk_size: int,
    limit: int | None,
    logger: logging.Logger,
) -> tuple[Dataset, int, int]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_dirs: list[Path] = []
    raw_example_total = 0
    mapped_feature_total = 0

    for chunk_idx, chunk_examples in enumerate(
        iter_example_chunks(json_path, chunk_size=chunk_size, limit=limit),
        start=1,
    ):
        raw_example_total += len(chunk_examples)
        chunk_dataset = Dataset.from_list(chunk_examples)
        mapped = chunk_dataset.map(
            lambda batch: prepare_train_features(batch, tokenizer, max_seq_length, doc_stride),
            batched=True,
            remove_columns=chunk_dataset.column_names,
        )
        chunk_dir = output_dir / f"chunk_{chunk_idx:04d}"
        mapped.save_to_disk(chunk_dir)
        chunk_dirs.append(chunk_dir)
        mapped_feature_total += len(mapped)

        log_json(
            logger,
            "train_preprocess_chunk_done",
            {
                "chunk_index": chunk_idx,
                "raw_rows_in_chunk": len(chunk_examples),
                "raw_rows_cumulative": raw_example_total,
                "mapped_features_in_chunk": len(mapped),
                "mapped_features_cumulative": mapped_feature_total,
                "chunk_dir": str(chunk_dir),
                "snapshot": get_runtime_snapshot(),
            },
        )

        del chunk_examples
        del chunk_dataset
        del mapped
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not chunk_dirs:
        return Dataset.from_list([]), 0, 0

    tokenized_chunks = [load_from_disk(str(chunk_dir)) for chunk_dir in chunk_dirs]
    tokenized_train = tokenized_chunks[0] if len(tokenized_chunks) == 1 else concatenate_datasets(tokenized_chunks)
    return tokenized_train, raw_example_total, mapped_feature_total


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or (args.output_dir / "train.log")
    logger, log_stream = setup_logger(log_path)
    faulthandler.enable(log_stream, all_threads=True)

    def handle_signal(signum, frame) -> None:
        log_json(
            logger,
            "signal_received",
            {
                "signal": signum,
                "snapshot": get_runtime_snapshot(),
            },
        )
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        log_json(
            logger,
            "startup",
            {
                "argv": sys.argv,
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "cwd": os.getcwd(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "snapshot": get_runtime_snapshot(),
            },
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        log_json(
            logger,
            "tokenizer_loaded",
            {
                "snapshot": get_runtime_snapshot(),
            },
        )

        cache_root = args.output_dir / "preprocessed_cache"
        t0 = time.time()
        tokenized_train, train_example_count, train_feature_count = preprocess_json_chunks_to_disk(
            json_path=args.train_path,
            output_dir=cache_root / "train",
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            chunk_size=args.preprocess_chunk_size,
            limit=args.max_train_samples,
            logger=logger,
        )
        log_json(
            logger,
            "train_tokenization_done",
            {
                "train_examples": train_example_count,
                "train_features": train_feature_count,
                "elapsed_sec": round(time.time() - t0, 2),
                "snapshot": get_runtime_snapshot(),
            },
        )

        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        fp16 = torch.cuda.is_available() and not bf16
        micro_batches_per_epoch = math.ceil(len(tokenized_train) / args.per_device_train_batch_size)
        optimizer_steps_per_epoch = math.ceil(micro_batches_per_epoch / args.gradient_accumulation_steps)
        total_train_steps = args.max_steps if args.max_steps > 0 else max(
            1, math.ceil(optimizer_steps_per_epoch * args.num_train_epochs)
        )
        warmup_steps = int(total_train_steps * args.warmup_ratio)

        log_json(
            logger,
            "training_plan",
            {
                "train_examples": train_example_count,
                "train_features": train_feature_count,
                "micro_batches_per_epoch": micro_batches_per_epoch,
                "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                "resolved_total_train_steps": total_train_steps,
                "warmup_steps": warmup_steps,
                "bf16": bf16,
                "fp16": fp16,
                "snapshot": get_runtime_snapshot(),
            },
        )

        model = AutoModelForQuestionAnswering.from_pretrained(args.model_path)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        training_args = TrainingArguments(
            output_dir=str(args.output_dir),
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            weight_decay=args.weight_decay,
            warmup_steps=warmup_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_strategy="steps",
            save_total_limit=args.save_total_limit,
            eval_strategy="no",
            report_to=[],
            seed=args.seed,
            dataloader_num_workers=0,
            max_steps=args.max_steps,
            remove_unused_columns=False,
            bf16=bf16,
            fp16=fp16,
            gradient_checkpointing=args.gradient_checkpointing,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            processing_class=tokenizer,
            data_collator=DefaultDataCollator(),
            callbacks=[ResourceLoggingCallback(logger)],
        )

        resume_from_checkpoint = str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

        run_summary = {
            "train_examples": train_example_count,
            "train_features": train_feature_count,
            "max_seq_length": args.max_seq_length,
            "doc_stride": args.doc_stride,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "max_steps": args.max_steps,
            "resume_from_checkpoint": resume_from_checkpoint,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "resolved_total_train_steps": total_train_steps,
            "warmup_steps": warmup_steps,
            "preprocess_chunk_size": args.preprocess_chunk_size,
            "gradient_checkpointing": args.gradient_checkpointing,
            "bf16": bf16,
            "fp16": fp16,
            "train_metrics": train_result.metrics,
        }
        with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)

        log_json(logger, "run_summary", run_summary)
        print(json.dumps(run_summary, indent=2))
    except BaseException as exc:
        log_json(
            logger,
            "train_failed",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "snapshot": get_runtime_snapshot(),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        log_stream.flush()
        log_stream.close()


if __name__ == "__main__":
    main()
