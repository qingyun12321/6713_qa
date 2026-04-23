# CUAD Fine-Tuning Report for Longformer

## 1. Objective

This report documents the CUAD fine-tuning experiment in this repository. The goal was to adapt a pre-trained long-context Transformer model to extractive contract question answering with no-answer detection.

The practical objective was:

- take a pre-trained long-document model,
- fine-tune it on the provided contract QA training split,
- produce a runnable checkpoint,
- and evaluate whether the fine-tuned model can extract answer spans from long contracts while correctly abstaining on no-answer cases.

## 2. Model Choice

The selected model family was **Longformer**, using the local checkpoint:

- `models/longformer-base-4096`

Longformer was chosen because contract documents are often much longer than the context length supported by standard BERT-style encoders. A short-context model would require aggressive truncation and would likely miss relevant clauses that appear far from the beginning of the document.

Longformer is therefore a better fit for this task because it supports long input sequences while keeping the extractive QA setup relatively standard.

## 3. Fine-Tuning Design

### 3.1 Task formulation

The task was framed as **extractive QA with no-answer detection**:

- if the answer exists in the contract, the model predicts a start token and end token span
- if the answer does not exist, the model is trained to point to the CLS position, which is later converted into an abstention decision using a threshold

### 3.2 Long-document preprocessing

Because many contracts exceed even Longformer’s practical working context in a single pass, the training pipeline used a **sliding-window approach**:

- `truncation="only_second"` so that the question is preserved
- `return_overflowing_tokens=True`
- a configurable `doc_stride`
- global attention on question-side tokens and special tokens

This setup allows the model to see long contracts through multiple overlapping windows while still training in a standard extractive QA format.

### 3.3 Memory-safe training pipeline

Earlier experiments showed that naive preprocessing could cause memory pressure. To make the fine-tuning stage stable, the final training pipeline:

- streamed examples from JSON,
- tokenized training data in chunks,
- cached chunked features to disk,
- and used BF16 on the available GPU.

This implementation is in [train_qa.py](../scripts/train_qa.py).

## 4. Experiments and Configuration Tuning

### 4.1 Early experiments

Before the final full run, multiple sequence-length and stride configurations were profiled on a validation sample. The main observations were:

- `4096` reduced the number of windows, but each window was more expensive
- `3072` preserved large context while lowering per-step cost
- `doc_stride=256` reduced overlap and total feature count compared with larger stride settings

The profiled feature counts are documented in [configs/experiment_runbook.md](../configs/experiment_runbook.md). The most relevant combinations were:

| `max_seq_length` | `doc_stride` | Features on sample | Expansion |
| --- | ---: | ---: | ---: |
| 3072 | 256 | 288 | 2.25x |
| 3072 | 512 | 288 | 2.25x |
| 4096 | 256 | 248 | 1.94x |

### 4.2 Earlier training baseline

The project first used a conservative baseline centered around:

- `max_seq_length=3072`
- `doc_stride=512`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=8`
- `gradient_checkpointing=true`

This was safe, but it under-used the available GPU.

### 4.3 Final selected fine-tuning configuration

After GPU utilization and memory behavior were tested on the RTX 5090, the final selected training configuration became:

| Parameter | Final value |
| --- | --- |
| Model | `longformer-base-4096` |
| `max_seq_length` | `3072` |
| `doc_stride` | `256` |
| `learning_rate` | `2e-5` |
| `num_train_epochs` | `1.0` |
| `per_device_train_batch_size` | `4` |
| `gradient_accumulation_steps` | `2` |
| Effective batch size | `8` |
| `warmup_steps` | `1556` |
| `preprocess_chunk_size` | `500` |
| Gradient checkpointing | `false` |
| Precision | `bf16` |

This configuration is recorded in [run_summary.json](../outputs/train_full_epoch1_3072_stride256_bs4_ga2/run_summary.json).

## 5. Final Fine-Tuning Run

The final model checkpoint is stored in:

- [outputs/train_full_epoch1_3072_stride256_bs4_ga2](../outputs/train_full_epoch1_3072_stride256_bs4_ga2)

Training completed successfully through the final step:

- `global_step = 15562`

The final checkpoint and model artifacts include:

- `checkpoint-15562`
- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `run_summary.json`

### 5.1 Training scale

The final run used:

- `20,150` training examples
- `124,491` expanded training features
- `15,562` optimizer steps

### 5.2 Training runtime and throughput

Final training statistics from [run_summary.json](../outputs/train_full_epoch1_3072_stride256_bs4_ga2/run_summary.json):

- `train_runtime = 12022.2124s`
- `train_samples_per_second = 10.355`
- `train_steps_per_second = 1.294`
- `train_loss = 0.5027364569887255`
- `epoch = 1.0`

### 5.3 Why this configuration was selected

This final configuration was selected because it provided a practical balance between:

- long-context coverage,
- manageable feature expansion,
- strong GPU utilization,
- and stable end-to-end training.

Compared with the earlier conservative configuration, the final setup:

- increased throughput by using `batch_size=4`,
- reduced accumulation overhead with `grad_accum=2`,
- removed gradient checkpointing to improve speed,
- and still remained within the available GPU memory budget on the RTX 5090.

## 6. Validation and Test Performance of the Fine-Tuned Model

Although this report focuses on the CUAD fine-tuning experiment, the quality of the resulting model still needs to be demonstrated quantitatively.

### 6.1 Validation results

Validation results are stored in:

- [evaluation_summary.json](../outputs/eval_train_full_epoch1_3072_stride256_bs4_ga2_val/evaluation_summary.json)

Best validation threshold:

- `5.015625`

Validation metrics:

| Metric | Value |
| --- | ---: |
| Exact Match | 61.09 |
| F1 | 68.53 |
| Has-answer Exact Match | 29.15 |
| Has-answer F1 | 43.36 |
| No-answer Accuracy | 96.17 |
| No-answer Precision | 81.90 |
| No-answer Recall | 96.17 |
| No-answer F1 | 88.46 |

### 6.2 Test results

Test results are stored in:

- [evaluation_summary.json](../outputs/eval_train_full_epoch1_3072_stride256_bs4_ga2_test/evaluation_summary.json)

The test run reused the threshold selected on validation (`5.015625`).

Test metrics:

| Metric | Value |
| --- | ---: |
| Exact Match | 85.70 |
| F1 | 88.61 |
| Has-answer Exact Match | 61.66 |
| Has-answer F1 | 71.44 |
| No-answer Accuracy | 95.85 |
| No-answer Precision | 92.30 |
| No-answer Recall | 95.85 |
| No-answer F1 | 94.04 |

## 7. Interpretation of the Fine-Tuning Outcome

The fine-tuned model clearly learned useful task-specific behavior. Several conclusions stand out:

1. **The model learned strong no-answer behavior.**  
   Both validation and test results show high no-answer accuracy and no-answer F1. This means the fine-tuned model is effective at abstaining when a requested field is not present.

2. **The model also improved on answerable extraction, but less strongly.**  
   The biggest remaining weakness is extracting exact spans for answerable examples. This is especially visible on validation, where `has_answer_f1` remains much lower than no-answer performance.

3. **The test set scores are much higher than the validation set scores.**  
   This is not only because the model improved, but also because the test split is much more no-answer-heavy (`70.25%`) than the validation split (`47.65%`). Since the model is particularly strong at no-answer decisions, the overall test metrics are boosted by this label distribution.

## 8. Error Analysis of the Fine-Tuned Model

Even though the fine-tuned model performs well overall, the residual errors reveal where future CUAD improvements should focus.

### 8.1 Validation error profile

Total validation errors:

- `895 / 2300` (`38.91%`)

Main validation error categories:

| Error type | Count | Share of validation errors |
| --- | ---: | ---: |
| Wrong span | 458 | 51.17% |
| False negative on answerable example | 233 | 26.03% |
| Overlong span / superset | 105 | 11.73% |
| False positive on no-answer example | 42 | 4.69% |
| Too-short substring | 31 | 3.46% |
| Partial overlap | 26 | 2.91% |

Hardest validation labels included:

- `Parties`
- `Audit Rights`
- `Insurance`
- `License Grant`
- `Post-Termination Services`
- `Revenue/Profit Sharing`

### 8.2 Test error profile

Total test errors:

- `599 / 4182` (`14.32%`)

Main test error categories:

| Error type | Count | Share of test errors |
| --- | ---: | ---: |
| False negative on answerable example | 235 | 39.23% |
| False positive on no-answer example | 122 | 20.37% |
| Overlong span / superset | 95 | 15.86% |
| Wrong span | 77 | 12.85% |
| Too-short substring | 58 | 9.68% |
| Partial overlap | 12 | 2.00% |

Hardest test labels included:

- `Effective Date`
- `Parties`
- `Uncapped Liability`
- `Post-Termination Services`
- `Cap On Liability`
- `Minimum Commitment`

### 8.3 Main recurring failure modes

#### False abstention on answerable examples

The most important remaining weakness is that the model sometimes predicts no answer when the answer is actually present. This usually happens when the answer is embedded in dense legal language or expressed indirectly rather than in a short, obvious clause.

#### Overlong answer spans

The model often finds the correct region but predicts a span that is too broad, such as a full sentence, clause block, or title section. This is a boundary problem rather than a complete topic failure.

#### Date confusion

For labels such as `Agreement Date` and `Effective Date`, the model often selects the wrong date when multiple plausible dates appear in the same contract.

#### Semantically related false positives

For labels like `Uncapped Liability` and `Post-Termination Services`, the model sometimes extracts a semantically related clause even though the gold label is no-answer. This means it recognizes the general topic but not the exact annotation target.

### 8.4 Example patterns

Representative examples from the held-out analysis include:

- missing a staffing-based `Minimum Commitment` clause because it was not phrased as a clean purchase floor
- missing a `Change Of Control` answer expressed through assignment/successor language
- extracting a liability cap clause for `Uncapped Liability`
- returning overly long title or party blocks instead of the exact gold span
- selecting prominent but incorrect dates for `Agreement Date` and `Effective Date`

### 8.5 What the error analysis implies

The fine-tuned model is often close to the correct clause region, even when it fails EM. This suggests that future improvements should focus less on basic retrieval and more on:

- span boundary precision,
- no-answer calibration for subtle answerable cases,
- and better semantic disambiguation for dates and liability-related clauses.

## 9. Limitations of the Current Fine-Tuning Work

The CUAD fine-tuning work is strong overall, but it still has several limitations:

- only one final full-epoch configuration was selected as the main model
- the repository does not currently preserve a separate simple baseline implementation
- the model is still much stronger on no-answer detection than on precise answerable extraction
- the test set’s no-answer-heavy distribution makes the overall test metrics look more favorable than the validation metrics

## 10. Future Fine-Tuning Improvements

The most useful CUAD follow-up work would likely be:

1. continue fine-tuning from the best checkpoint with a lower learning rate
2. improve answer-span boundary precision
3. test label-aware or date-aware heuristics during post-processing
4. compare additional long-context model variants if compute permits
5. add a retained simple baseline for cleaner modelling comparison

## 11. Conclusion

The CUAD fine-tuning experiment was successful. A pre-trained Longformer model was adapted to the contract QA task using long-context sliding-window preprocessing and a memory-safe training pipeline. The final selected configuration used `3072` tokens, `doc_stride=256`, `batch_size=4`, `gradient_accumulation_steps=2`, BF16 precision, and no gradient checkpointing.

The resulting fine-tuned model achieved:

- **Validation:** `EM 61.09`, `F1 68.53`
- **Test:** `EM 85.70`, `F1 88.61`

The model’s strongest capability is no-answer detection, while its main remaining weakness is precise extraction of answerable spans from complex legal language. Overall, the fine-tuning work demonstrates that a Longformer-based approach is effective for long-document contract QA and provides a strong CUAD baseline for the project.
