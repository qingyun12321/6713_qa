# Experiment Runbook and Parameter Notes

This document collects the parameter choices, profiling notes, runtime guidance, and command examples that were previously embedded in the README. The README now stays focused on repository orientation; use this file when reproducing or extending experiments.

## Shared Pipeline Decisions

- Task formulation: extractive QA with no-answer detection
- Model family: local `longformer-base-4096`
- Long-context handling:
  - `truncation="only_second"`
  - `return_overflowing_tokens=True`
  - configurable `doc_stride`
  - global attention on question-side tokens and special tokens
- Evaluation:
  - report `Exact Match` and `F1`
  - also track no-answer metrics
  - select the no-answer threshold on validation, then reuse it on test
- Inference:
  - `predict.py` supports `--question` together with either `--context` or `--context-file`
  - output includes both the thresholded answer and raw span-vs-null score information

## CUAD Runbook

### Data Preparation Decisions

- Keep the provided `flat_test.json` as the final test set.
- Split only `flat_train.json` into train and validation.
- Split by `title`, not by row, to avoid contract leakage across train and validation.
- Normalize answers into:
  - `{"text": [...], "answer_start": [...]}`
- Preserve `is_impossible` as the no-answer label.

### Baseline Parameters

These were the selected baseline settings after the first smoke-test round.

| Parameter | Initial value | Why |
| --- | --- | --- |
| `max_seq_length` | `3072` | Smoke test runs cleanly on the local 16GB GPU and keeps feature expansion manageable |
| `doc_stride` | `256` | Current full-data baseline that reduces overlap and total feature count |
| `max_answer_length` | `256` | Better fit than short-answer defaults for long contractual spans |
| `learning_rate` | `2e-5` | Stable first-pass full fine-tuning default |
| `per_device_train_batch_size` | `1` | Required by Longformer memory cost |
| `gradient_accumulation_steps` | `8` | Reasonable effective batch without pushing VRAM too hard |
| `num_train_epochs` | `1` | Best first full-data run target on this machine |
| `gradient_checkpointing` | `on` | Confirmed useful and compatible in smoke testing |
| `precision` | `bf16` when available | Confirmed active on the local GPU during smoke testing |

### Smoke-Test Findings

#### Data split

- Prepared split saved to `outputs/cuad_prepared`.
- Split result:
  - train: `20150` rows across `367` titles
  - validation: `2300` rows across `41` titles
  - test: `4182` rows across `102` titles
- Validation no-answer ratio is `47.65%`.
- Test no-answer ratio is `70.25%`.

#### Window expansion profile

On a validation sample of `128` examples:

| `max_seq_length` | `doc_stride` | features | expansion |
| --- | --- | --- | --- |
| `2048` | `256` | `370` | `2.89x` |
| `2048` | `512` | `414` | `3.23x` |
| `2048` | `1024` | `569` | `4.45x` |
| `3072` | `256` | `288` | `2.25x` |
| `3072` | `512` | `288` | `2.25x` |
| `3072` | `1024` | `328` | `2.56x` |
| `4096` | `256` | `248` | `1.94x` |
| `4096` | `512` | `248` | `1.94x` |
| `4096` | `1024` | `248` | `1.94x` |

Interpretation:

- `4096` reduces feature count, but each feature is more expensive.
- `3072 + 256` is the current default because it keeps runtime lower while still fitting comfortably.
- The final decision between `3072` and `4096` should be based on a longer real training run, not on preprocessing alone.

### Resource Guidance

- Do not run full-dataset preprocessing and model training at the same time on this machine.
- Do not launch more than one training job concurrently.
- The machine has shown WSL instability when memory reaches roughly `85%` to `90%`.
- Prefer staged runs:
  - first `--max-steps 50`
  - then `--max-steps 200`
  - then a full epoch only after memory and stability are confirmed
- Prefer `--preprocess-chunk-size 100` as the safe default on this machine.
- Keep checkpoint saving relatively infrequent during real training, for example every `500` or `1000` optimizer steps, to avoid extra I/O pressure.
- If the machine becomes unstable, lower load before changing the model:
  - close notebooks or browsers
  - avoid parallel CPU-heavy scripts
  - stop extra GPU workloads

### Training Budget

This estimate is based on the later `1000`-example real training run, not just the initial smoke tests.

Observed reference run:

- `1000` training examples
- `max_seq_length=3072`
- `doc_stride=256`
- `num_train_epochs=1`
- `train_features=4921`
- `resolved_total_train_steps=616`
- `train_runtime ~= 1580.57s ~= 26.3 minutes`

Practical extrapolation:

- Full train split size is `20150` examples.
- The `1000`-example run suggests about `20.15x` more raw examples.
- At the same `3072 + 256` setting, that implies a rough full-epoch training budget near `9` hours.

Recommended planning numbers for the current full-data baseline:

- `1` epoch at `3072 + 256`: budget about `9 to 11 hours`
- `2` epochs at `3072 + 256`: budget about `18 to 22 hours`

### `max_steps` vs `num_train_epochs`

- `num_train_epochs` means how many times the training loop should pass over the full expanded training set.
- `max_steps` means how many optimizer update steps to run before stopping.
- If `max_steps > 0`, it takes priority and overrides epoch-based stopping.
- `per_device_train_batch_size=1` means one training feature is processed at a time.
- `gradient_accumulation_steps=8` means the optimizer updates once every `8` micro-batches.

Recommended staged plan:

- smoke check: `max_steps=50`, `num_train_epochs=1`
- medium check: `max_steps=200`, `num_train_epochs=1`
- first real run: `max_steps=-1`, `num_train_epochs=1`
- only consider `num_train_epochs=2` if the first full epoch improves answerable examples and the machine stays stable

### Commands

Run everything from the repository root directory.

#### 1. Prepare CUAD splits

```bash
uv run python "scripts/prepare_cuad.py" \
  --raw-train-path "dataset/data/flat_train.json" \
  --raw-test-path "dataset/data/flat_test.json" \
  --output-dir "outputs/cuad_prepared"
```

This command writes:

- `train.json`
- `validation.json`
- `test.json`
- `split_manifest.json`
- `summary.json`

#### 2. Profile window expansion

```bash
uv run python "scripts/profile_lengths.py" \
  --data-path "outputs/cuad_prepared/validation.json" \
  --model-path "models/longformer-base-4096" \
  --sample-size 256 \
  --max-seq-lengths 2048 3072 4096 \
  --doc-strides 256 512 1024
```

#### 3. Full-data training for 1 epoch

```bash
uv run python "scripts/train_qa.py" \
  --train-path "outputs/cuad_prepared/train.json" \
  --model-path "models/longformer-base-4096" \
  --output-dir "outputs/train_full_epoch1_3072_stride256" \
  --max-seq-length 3072 \
  --doc-stride 256 \
  --learning-rate 2e-5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --max-steps -1 \
  --save-steps 500 \
  --save-total-limit 2 \
  --logging-steps 20 \
  --preprocess-chunk-size 100 \
  --gradient-checkpointing
```

#### 4. Validation evaluation for a new checkpoint

```bash
uv run python "scripts/evaluate_qa.py" \
  --data-path "outputs/cuad_prepared/validation.json" \
  --model-path "outputs/train_full_epoch1_3072_stride256" \
  --output-dir "outputs/eval_train_full_epoch1_3072_stride256_val" \
  --max-seq-length 3072 \
  --doc-stride 256 \
  --max-answer-length 256 \
  --per-device-eval-batch-size 1 \
  --search-threshold \
  --preprocess-chunk-size 100
```

Every evaluation run writes `evaluation_summary.json`, `predictions.json`, and `eval.log` under `--output-dir`.

#### 5. Single-document prediction

```bash
uv run python "scripts/predict.py" \
  --model-path "outputs/train_full_epoch1_3072_stride256" \
  --question 'Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer.' \
  --context-file "/path/to/contract.txt" \
  --max-seq-length 3072 \
  --doc-stride 256 \
  --max-answer-length 256
```

## SQuAD Runbook

The SQuAD path in this repository uses the Hugging Face `rajpurkar/squad_v2` dataset but keeps the document and script naming short as `SQuAD`.

### Dataset Preparation

```bash
uv run python "scripts/prepare_squad.py" \
  --output-dir "outputs/squad_prepared" \
  --seed 42
```

This command:

- downloads `rajpurkar/squad_v2`
- writes `train.json`, `validation.json`, `test.json`
- keeps the official `train` split as training data
- splits the official `validation` split `1:1` into new validation and test splits
- stratifies that split on `is_impossible`
- writes `summary.json` and `split_manifest.json`

### Selected Parameters

Recommended first-run settings for this setup:

- `max_seq_length=512`
- `doc_stride=128`
- `max_answer_length=64`
- `learning_rate=3e-5`
- `per_device_train_batch_size=4`
- `gradient_accumulation_steps=2`
- `num_train_epochs=1`

Why `3e-5` here:

- the run starts from raw pretrained Longformer weights, not from an already fine-tuned QA checkpoint
- with only `1` epoch, `3e-5` is a stronger and more standard first-pass QA fine-tuning rate than `1e-5`
- if training is unstable, the first fallback to try is `2e-5`

### Commands

#### 1. Train on SQuAD for 1 epoch

```bash
uv run python "scripts/train_qa.py" \
  --train-path "outputs/squad_prepared/train.json" \
  --model-path "models/longformer-base-4096" \
  --output-dir "outputs/squad_epoch1_seq512_stride128_bs4_ga2" \
  --max-seq-length 512 \
  --doc-stride 128 \
  --learning-rate 3e-5 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --num-train-epochs 1 \
  --max-steps -1 \
  --save-steps 1000 \
  --save-total-limit 2 \
  --logging-steps 50 \
  --preprocess-chunk-size 100
```

#### 2. Search the best no-answer threshold on validation

```bash
uv run python "scripts/evaluate_qa.py" \
  --data-path "outputs/squad_prepared/validation.json" \
  --model-path "outputs/squad_epoch1_seq512_stride128_bs4_ga2" \
  --output-dir "outputs/squad_epoch1_seq512_stride128_bs4_ga2_val" \
  --max-seq-length 512 \
  --doc-stride 128 \
  --max-answer-length 64 \
  --per-device-eval-batch-size 4 \
  --search-threshold \
  --preprocess-chunk-size 100
```

#### 3. Evaluate on test with the frozen validation threshold

```bash
VAL_THRESHOLD=$(uv run python - <<'PY'
import json
from pathlib import Path

summary = json.loads(
    Path("outputs/squad_epoch1_seq512_stride128_bs4_ga2_val/evaluation_summary.json").read_text(
        encoding="utf-8"
    )
)
print(summary["selected_threshold"])
PY
)

uv run python "scripts/evaluate_qa.py" \
  --data-path "outputs/squad_prepared/test.json" \
  --model-path "outputs/squad_epoch1_seq512_stride128_bs4_ga2" \
  --output-dir "outputs/squad_epoch1_seq512_stride128_bs4_ga2_test" \
  --max-seq-length 512 \
  --doc-stride 128 \
  --max-answer-length 64 \
  --per-device-eval-batch-size 4 \
  --threshold "$VAL_THRESHOLD" \
  --preprocess-chunk-size 100
```
