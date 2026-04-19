# Contract QA Longformer Fine-Tuning

This repository contains the executable plan and starter code for the Part C task: fine-tuning a model on the contract QA dataset without using SQuAD.

## Scope

- Task definition: extractive question answering with no-answer detection in contract review.
- Model family: local `longformer-base-4096`.
- Data sources:
  - `dataset/data/flat_train.json`
  - `dataset/data/flat_test.json`
- Environment: managed from `pyproject.toml` and run with `uv`.

## Project Layout

```text
contract_qa/
├── README.md
├── configs/
├── docs/
├── outputs/
├── scripts/
│   ├── prepare_data.py
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

## Decisions Locked In

### Data splitting

- Keep the provided `flat_test.json` as the final test set.
- Split only `flat_train.json` into train and validation.
- Split by `title`, not by row, to avoid contract leakage across train and validation.

### Data format

- Convert `answers` from list-of-dicts into the Hugging Face QA format:
  - `{"text": [...], "answer_start": [...]}`
- Preserve `is_impossible` as the no-answer label.

### Long document handling

- Use Longformer with sliding window preprocessing.
- Use `truncation="only_second"` because question is short and context is long.
- Use `return_overflowing_tokens=True` and a configurable `doc_stride`.
- Apply global attention to question-side tokens and special tokens.

### Evaluation

- Report `Exact Match` and `F1`.
- Also report no-answer metrics:
  - `no_answer_accuracy`
  - `no_answer_precision`
  - `no_answer_recall`
  - `no_answer_f1`
- Support multiple reference answers by taking the best match.
- Search the no-answer threshold on validation, then freeze it for test.

### Command-line inference

- Support:
  - `--question`
  - `--context`
  - `--context-file`
- Return answer text plus null-vs-span score information.

## Baseline Parameters

These are the current baseline settings after the first smoke-test round.

| Parameter | Initial value | Why |
| --- | --- | --- |
| `max_seq_length` | `3072` | Smoke test runs cleanly on the local 16GB GPU and keeps feature expansion manageable |
| `doc_stride` | `256` | Current full-data baseline that reduces overlap and total feature count |
| `max_answer_length` | `256` | Better fit than SQuAD-style defaults for long contractual spans |
| `learning_rate` | `2e-5` | Stable first-pass full fine-tuning default |
| `per_device_train_batch_size` | `1` | Required by Longformer memory cost |
| `gradient_accumulation_steps` | `8` | Reasonable effective batch without pushing VRAM too hard |
| `num_train_epochs` | `1` | Best first full-data run target on this machine |
| `gradient_checkpointing` | `on` | Confirmed useful and compatible in smoke testing |
| `precision` | `bf16` when available | Confirmed active on the local GPU during smoke testing |

## Smoke-Test Findings

### Data split

- Prepared split saved to `outputs/prepared_data`.
- Split result:
  - train: `20150` rows across `367` titles
  - validation: `2300` rows across `41` titles
  - test: `4182` rows across `102` titles
- Validation no-answer ratio is `47.65%`.
- Test no-answer ratio is `70.25%`.

### Window expansion profile

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

## Resource Safety

- Do not run full-dataset preprocessing and model training at the same time on this machine.
- Do not launch more than one training job concurrently.
- The machine has shown WSL instability when memory reaches roughly `85% to 90%`.
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

## Time Estimate

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
- At the same `3072 + 256` setting, that implies a rough full-epoch training budget near `9 hours`.
Recommended planning numbers for the current full-data baseline:

- `1` epoch at `3072 + 256`: budget about `9 to 11 hours`
- `2` epochs at `3072 + 256`: budget about `18 to 22 hours`

Recommendation:

- Treat `1 epoch` as the first real full-data target.
- Only consider `2` epochs after checking validation quality and machine stability.
- For safety, start the full run only when the machine can be left undisturbed for at least half a day.

## Speed Trade-Offs

Changing `max_seq_length` and `doc_stride` can absolutely change runtime, but the effect is not identical.

- Lower `max_seq_length` usually makes each training feature cheaper:
  - less GPU memory per forward/backward pass
  - less compute per step
- But lower `max_seq_length` can also create more sliding-window features for the same document.
- So lowering `max_seq_length` does not guarantee faster end-to-end training. It often reduces per-step cost, but may increase the number of steps needed.

- Lower `doc_stride` means less overlap between windows.
- Less overlap usually means fewer total training features.
- Fewer total features usually means shorter preprocessing time and shorter total training time.
- But if `doc_stride` is too small, answers near window boundaries are more likely to be missed, which can hurt model quality.

Practical interpretation for this project:

- Reducing `doc_stride` is the more direct way to reduce total runtime.
- Reducing `max_seq_length` is a trade-off:
  - faster per step
  - but potentially more total windows
- Good runtime tuning should optimize both:
  - keep `max_seq_length` large enough to cover long contractual clauses
  - keep `doc_stride` only as large as needed to avoid losing answer spans at boundaries

Current default remains:

- `max_seq_length=3072`
- `doc_stride=256`

These are currently the safest quality-first settings that still fit the machine.

## `max_steps` vs `num_train_epochs`

These two settings are related, but they are not the same thing.

- `num_train_epochs` means how many times the training loop should pass over the full expanded training set.
- `max_steps` means how many optimizer update steps to run before stopping.
- In the current training script, if `max_steps > 0`, it takes priority and effectively overrides the epoch-based stopping rule.

How that works with gradient accumulation:

- `per_device_train_batch_size=1` means one training feature is processed at a time.
- `gradient_accumulation_steps=8` means the optimizer updates once every `8` micro-batches.
- So one optimizer step is not one raw example. It is one accumulated parameter update after several forward and backward passes.

Practical interpretation:

- Use `max_steps` for smoke tests and safe staged checks.
- Use `num_train_epochs` for the real full training run.
- For the final full run, set `max_steps=-1` so training stops based on epochs rather than an artificial step cap.

Recommended staged plan:

- smoke check: `max_steps=50`, `num_train_epochs=1`
- medium check: `max_steps=200`, `num_train_epochs=1`
- first real run: `max_steps=-1`, `num_train_epochs=1`
- only consider `num_train_epochs=2` if the first full epoch improves answerable examples and the machine stays stable

## Commands

Run everything from the `contract_qa` root directory.

### 1. Prepare grouped data splits

```bash
uv run python "scripts/prepare_data.py" \
  --raw-train-path "dataset/data/flat_train.json" \
  --raw-test-path "dataset/data/flat_test.json" \
  --output-dir "outputs/prepared_data"
```

### 2. Profile window expansion before training

```bash
uv run python "scripts/profile_lengths.py" \
  --data-path "outputs/prepared_data/validation.json" \
  --model-path "models/longformer-base-4096" \
  --sample-size 256 \
  --max-seq-lengths 2048 3072 4096 \
  --doc-strides 256 512 1024
```

### 3. Full-data training for 1 epoch

Note:

- `--max-answer-length` is not a training argument.
- It is only used by `evaluate_qa.py` and `predict.py` during answer postprocessing.
- `train_qa.py` now writes a line-buffered log file at `OUTPUT_DIR/train.log` by default.
- If training crashes before completion, check `train.log` first, then any `checkpoint-*` folders, then `run_summary.json` if it exists.
- `train_qa.py` now preprocesses data in chunks and logs a memory snapshot after each chunk.
- You can control chunk size with `--preprocess-chunk-size`. Use a smaller value such as `100` or `200` if WSL is unstable.
- `train_qa.py` now streams training examples from JSON instead of loading the whole train set into memory at once.
- `train_qa.py` no longer accepts validation input or runs validation during training.

```bash
uv run python "scripts/train_qa.py" \
  --train-path "outputs/prepared_data/train.json" \
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

This run:

- uses the full grouped training split of `20150` examples
- keeps the current full-data baseline `max_seq_length=3072`, `doc_stride=256`
- runs a real `1` epoch because `--max-steps -1`
- keeps the streamed low-memory preprocessing path
- should be budgeted at about `9 to 11 hours` on this machine

### 3b. One epoch on 1000 samples with lower overlap

```bash
uv run python "scripts/train_qa.py" \
  --train-path "outputs/prepared_data/train.json" \
  --model-path "models/longformer-base-4096" \
  --output-dir "outputs/train_1000_epoch1_stride256" \
  --max-seq-length 3072 \
  --doc-stride 256 \
  --learning-rate 2e-5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 1 \
  --max-steps -1 \
  --max-train-samples 1000 \
  --save-steps 100 \
  --save-total-limit 2 \
  --logging-steps 10 \
  --preprocess-chunk-size 100 \
  --gradient-checkpointing
```

This run:

- uses only the first `1000` training examples
- runs a real `1` epoch because `--max-steps -1`
- reduces overlap to `256`
- keeps the streamed low-memory preprocessing path

Resume from a saved checkpoint:

```bash
uv run python "scripts/train_qa.py" \
  --train-path "outputs/prepared_data/train.json" \
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
  --gradient-checkpointing \
  --resume-from-checkpoint "outputs/train_full_epoch1_3072_stride256/checkpoint-500"
```

### 4. Full validation evaluation for a new checkpoint

```bash
uv run python "scripts/evaluate_qa.py" \
  --data-path "outputs/prepared_data/validation.json" \
  --model-path "outputs/train_full_epoch1_3072_stride256" \
  --output-dir "outputs/eval_train_full_epoch1_3072_stride256_val" \
  --max-seq-length 3072 \
  --doc-stride 256 \
  --max-answer-length 256 \
  --per-device-eval-batch-size 1 \
  --search-threshold \
  --preprocess-chunk-size 100
```

- `evaluate_qa.py` now evaluates in chunks instead of tokenizing the full validation file at once.
- It only keeps the minimum example metadata needed for metrics and prediction export, rather than retaining full `context` and `question` strings in memory after each chunk finishes.
- For this machine, keep `--preprocess-chunk-size 100` as the safe default unless evaluation memory usage is clearly low.
- `evaluate_qa.py` now prints chunk-level progress logs in JSON so long-running evaluation is easier to monitor.
- When CUDA is available, evaluation now enables `bf16_full_eval` to reduce GPU-side evaluation cost.
- For a brand-new checkpoint, keep `--search-threshold` on so the best validation threshold is found automatically.

### 4b. Full validation evaluation for the current 1000-sample checkpoint

```bash
uv run python "scripts/evaluate_qa.py" \
  --data-path "outputs/prepared_data/validation.json" \
  --model-path "outputs/train_1000_epoch1_stride256" \
  --output-dir "outputs/eval_train_1000_epoch1_stride256_val_chunked" \
  --max-seq-length 3072 \
  --doc-stride 256 \
  --max-answer-length 256 \
  --per-device-eval-batch-size 1 \
  --search-threshold \
  --preprocess-chunk-size 100
```

- This command evaluates all `2300` validation examples.
- It reuses the same window settings as the current checkpoint: `max_seq_length=3072` and `doc_stride=256`.
- Outputs will be written to:
  - `evaluation_summary.json`
  - `predictions.json`

### 4c. Re-run validation for the same checkpoint with a fixed threshold

For the current `1000`-sample checkpoint, the best validation threshold already found is `7.921875`.

```bash
uv run python "scripts/evaluate_qa.py" \
  --data-path "outputs/prepared_data/validation.json" \
  --model-path "outputs/train_1000_epoch1_stride256" \
  --output-dir "outputs/eval_train_1000_epoch1_stride256_val_fixed_threshold" \
  --max-seq-length 3072 \
  --doc-stride 256 \
  --max-answer-length 256 \
  --per-device-eval-batch-size 1 \
  --threshold 7.921875 \
  --preprocess-chunk-size 100
```

- Use this fixed-threshold version only for repeated evaluation of the same checkpoint.
- If the checkpoint changes, run section `4` or `4b` with `--search-threshold` again.

### 5. Command-line prediction

```bash
uv run python "scripts/predict.py" \
  --model-path "outputs/train_1000_epoch1_stride256" \
  --question 'Highlight the parts (if any) of this contract related to "Parties" that should be reviewed by a lawyer.' \
  --context-file "/path/to/contract.txt" \
  --max-seq-length 3072 \
  --doc-stride 256 \
  --max-answer-length 256 \
  --threshold 7.921875
```

## What To Lock Next

- Compare a longer run of `3072 + 256` against one run of `4096 + 256`.
- Tune the final validation threshold only after a real checkpoint exists.
- Check whether answerable examples improve enough to keep `max_answer_length=256`; if not, test `384`.
- Keep `batch_size=1`, `gradient_accumulation_steps=8`, `gradient_checkpointing=on` unless a longer run shows instability.

## Planned Deliverables

- Grouped split manifest and summary stats.
- Fine-tuning script.
- Evaluation script with threshold tuning.
- CLI prediction script.
- Smoke-test results and final parameter table in Markdown.
