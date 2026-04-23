# Contract QA Longformer Experiments

This repository contains preprocessing, training, evaluation, and inference scripts for extractive question answering with no-answer detection on two datasets:

- `CUAD`, prepared from local flattened JSON files
- `SQuAD`, prepared from the Hugging Face `rajpurkar/squad_v2` dataset

The codebase uses a shared Longformer-based QA pipeline across both datasets, with dataset-specific preparation scripts and common training, evaluation, and post-processing utilities.

## Overview

- Task: extractive QA with no-answer detection
- Model family: local `longformer-base-4096`
- Runtime: Python `3.11+`, managed through `uv`
- Outputs: written under `outputs/` when scripts are run

Both datasets are normalized into the same JSON schema:

```json
{
  "id": "...",
  "title": "...",
  "context": "...",
  "question": "...",
  "answers": {
    "text": ["..."],
    "answer_start": [123]
  },
  "is_impossible": false
}
```

## Project Layout

```text
<repo-root>/
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   └── experiment_runbook.md
├── docs/
│   ├── cuad_finetuning_report.md
│   └── squad_finetuning_report.md
├── scripts/
│   ├── prepare_cuad.py
│   ├── prepare_squad.py
│   ├── profile_lengths.py
│   ├── train_qa.py
│   ├── evaluate_qa.py
│   └── predict.py
└── src/contract_qa/
    ├── __init__.py
    ├── data_utils.py
    ├── metrics.py
    └── qa_utils.py
```

## Core Scripts

- `scripts/prepare_cuad.py`: prepares grouped train and validation splits from the local CUAD-style flat JSON files and keeps the provided test split.
- `scripts/prepare_squad.py`: downloads and normalizes SQuAD into the shared project format, then splits the official validation set into validation and test.
- `scripts/profile_lengths.py`: profiles sliding-window expansion for different `max_seq_length` and `doc_stride` combinations.
- `scripts/train_qa.py`: fine-tunes a QA checkpoint on one prepared training split.
- `scripts/evaluate_qa.py`: evaluates a checkpoint on a prepared validation or test split and writes metrics, predictions, and logs.
- `scripts/predict.py`: runs single-example inference from `--context` or `--context-file`.

## Quick Start

Install dependencies from the repository root:

```bash
uv sync
```

Prepare CUAD:

```bash
uv run python "scripts/prepare_cuad.py" \
  --raw-train-path "dataset/data/flat_train.json" \
  --raw-test-path "dataset/data/flat_test.json" \
  --output-dir "outputs/cuad_prepared"
```

Prepare SQuAD:

```bash
uv run python "scripts/prepare_squad.py" \
  --output-dir "outputs/squad_prepared"
```

Train a checkpoint on a prepared split:

```bash
uv run python "scripts/train_qa.py" \
  --train-path "outputs/cuad_prepared/train.json" \
  --model-path "models/longformer-base-4096" \
  --output-dir "outputs/train_example"
```

Evaluate a checkpoint:

```bash
uv run python "scripts/evaluate_qa.py" \
  --data-path "outputs/cuad_prepared/validation.json" \
  --model-path "outputs/train_example" \
  --output-dir "outputs/eval_example"
```

Run single-document inference:

```bash
uv run python "scripts/predict.py" \
  --model-path "outputs/train_example" \
  --question "What is the governing law?" \
  --context-file "/path/to/contract.txt"
```

For tested parameter profiles, dataset-specific runbooks, and command variants, use [configs/experiment_runbook.md](configs/experiment_runbook.md).

## Documentation

- [configs/experiment_runbook.md](configs/experiment_runbook.md): parameter choices, profiling notes, runtime guidance, and reproducible command examples for CUAD and SQuAD.
- [docs/cuad_finetuning_report.md](docs/cuad_finetuning_report.md): detailed write-up of the CUAD fine-tuning experiments.
- [docs/squad_finetuning_report.md](docs/squad_finetuning_report.md): detailed write-up of the SQuAD fine-tuning experiments.
