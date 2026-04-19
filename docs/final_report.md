# Contract QA with Longformer: Fine-Tuning, Evaluation, and Error Analysis

## Abstract

This report documents the end-to-end development of a contract question answering (QA) system for long legal documents. The task is extractive QA with no-answer detection: given a contract and a natural-language question such as "What is the effective date?" or "Who are the parties?", the model must either extract the relevant text span from the document or abstain when the information is not present. The final system fine-tunes a Longformer model with sliding-window preprocessing so that long contracts can be processed without truncating away critical evidence.

The main contributions of the project are: a grouped train/validation split that avoids contract-level leakage, a preprocessing pipeline that converts the raw annotations into Hugging Face QA format, a memory-safe long-document fine-tuning workflow, a command-line inference interface, and a chunked evaluation pipeline with threshold tuning for no-answer decisions. The final selected model uses `max_seq_length=3072`, `doc_stride=256`, `per_device_train_batch_size=4`, `gradient_accumulation_steps=2`, `bf16`, and no gradient checkpointing. On the selected validation threshold of `5.015625`, the model achieved `EM 61.09 / F1 68.53` on validation and `EM 85.70 / F1 88.61` on the held-out test set.

The results show that the model is particularly strong at no-answer detection, while the main residual weakness is answerable-span extraction, especially for dense legal clauses, exact span boundaries, and date fields containing multiple plausible candidates.

## 1. Problem Definition

### 1.1 NLP problem

The project addresses long-document contract question answering. This is a structured information extraction problem framed as extractive QA with optional abstention. For each `(question, contract)` pair, the system must:

1. identify whether the contract contains the requested information, and
2. if it does, return the exact supporting text span.

This is a realistic contract review task because legal practitioners often need to locate clauses such as:

- parties
- effective date
- change of control
- audit rights
- insurance
- cap on liability
- minimum commitment

Unlike short-document QA benchmarks, contract QA is challenging because evidence is often buried in long, repetitive, and highly formal text. A single contract may contain multiple candidate clauses or several dates, which makes exact extraction difficult even when the model understands the topic.

### 1.2 Domain and text source

The text source is a corpus of contracts and contract-like legal agreements. The project works on the provided flattened JSON data files:

- `dataset/data/flat_train.json`
- `dataset/data/flat_test.json`

The documents are long, clause-heavy, and domain-specific. This motivated the use of a long-context Transformer rather than a standard BERT-style model with a short sequence limit.

## 2. Dataset Selection and Preparation

### 2.1 Dataset choice

The project uses an existing contract QA dataset that was already provided in flattened JSON form. Each example contains:

- `id`
- `title`
- `context`
- `question`
- `answers`
- `is_impossible`

This project does not create a new human-labelled dataset and therefore does not report inter-annotator agreement. It also does not rely on an external lexicon or ontology.

### 2.2 Split strategy

To avoid information leakage, the project does not randomly split rows. Instead, it splits the original training data by `title`, so that all examples from the same contract stay in the same partition. This is important because row-level splitting would allow near-duplicate language from the same contract to appear in both training and validation.

The split policy was:

- keep `flat_test.json` as the final untouched test set
- split only `flat_train.json` into train and validation
- group by `title`
- use `val_ratio=0.1`
- use `seed=42`

The split manifest is recorded in [split_manifest.json](/workspace/contract_qa/outputs/prepared_data/split_manifest.json).

### 2.3 Final data sizes

The prepared data summary is recorded in [summary.json](/workspace/contract_qa/outputs/prepared_data/summary.json).

| Split | Rows | Unique Titles | No-answer Rows | No-answer Ratio |
| --- | ---: | ---: | ---: | ---: |
| Train | 20,150 | 367 | 10,174 | 50.49% |
| Validation | 2,300 | 41 | 1,096 | 47.65% |
| Test | 4,182 | 102 | 2,938 | 70.25% |

Two observations matter for later evaluation:

- the dataset contains many no-answer cases, so abstention is a core part of the problem
- the test set is much more no-answer-heavy than the validation set, which affects the interpretation of overall EM/F1

### 2.4 Normalization and output format

The raw `answers` field was converted from a list-of-dictionaries into Hugging Face QA format:

```json
{
  "text": ["..."],
  "answer_start": [123]
}
```

The pipeline preserves `is_impossible` so that no-answer examples remain explicit. The generated files are:

- `outputs/prepared_data/train.json`
- `outputs/prepared_data/validation.json`
- `outputs/prepared_data/test.json`

The test file was verified to match the original raw test set after normalization only; it was not resplit or altered semantically.

## 3. Modelling Approach

### 3.1 Modelling choice

The final system fine-tunes a Longformer-based extractive QA model. Longformer was chosen because it supports much longer input sequences than standard short-context Transformer encoders. This is necessary for contracts, where critical information may appear far from the beginning of the document.

The implemented system uses:

- sliding-window tokenization
- question-first formatting with `truncation="only_second"`
- overflow handling across long documents
- global attention on question tokens and special tokens
- chunked preprocessing for memory safety
- threshold-based no-answer decision at evaluation time

### 3.2 Why Longformer was appropriate

Long legal contracts often exceed the practical input limit of standard BERT-like models. A naive truncation strategy would throw away evidence and make many questions unanswerable even when the answer exists. Longformer allows the system to keep much more context while still using a Transformer encoder architecture familiar from extractive QA.

### 3.3 Rule-based/statistical baseline

The assignment guideline expects a simple baseline. In the current repository state, the implemented and fully documented deliverable is the fine-tuned Longformer system rather than a separate retained baseline script. This should be treated as a limitation of the current code drop. A minimal baseline can still be described conceptually as a future extension, for example:

- keyword matching plus nearest-clause extraction
- TF-IDF retrieval plus answer-span heuristic
- majority-class abstention for selected labels

However, the reproducible code artifacts in this repository are centered on the Longformer fine-tuning system.

### 3.4 Command-line testing interface

The repository includes a command-line inference script:

- [predict.py](/workspace/contract_qa/scripts/predict.py)

This satisfies the command-line testing requirement from the assignment. It supports:

- `--question`
- `--context`
- `--context-file`
- thresholded answer output

This makes the system directly testable from the terminal without needing a notebook.

## 4. System Design and Implementation

### 4.1 Preprocessing pipeline

The preprocessing script is [prepare_data.py](/workspace/contract_qa/scripts/prepare_data.py). Its responsibilities are:

- load raw train and test JSON
- normalize the QA annotation format
- create a grouped train/validation split by `title`
- write summary statistics and split metadata

### 4.2 Long-document tokenization

The QA feature preparation code is in [qa_utils.py](/workspace/contract_qa/src/contract_qa/qa_utils.py). The system uses:

- `max_seq_length`
- `doc_stride`
- overflowing windows
- alignment of character offsets to token positions
- CLS-based no-answer labeling for impossible examples

For evaluation, offset mappings are retained so that span predictions can be projected back to character positions in the original document.

### 4.3 Memory-safe training pipeline

Early experiments showed that full preprocessing could stress system memory. The final training script therefore uses several mitigation strategies:

- stream examples from JSON instead of loading everything into RAM
- preprocess examples in chunks and save to disk
- release intermediate objects aggressively
- optionally empty CUDA cache between stages

This design is implemented in [train_qa.py](/workspace/contract_qa/scripts/train_qa.py). It was important because earlier attempts suggested that preprocessing, not GPU training, was the primary source of instability.

### 4.4 Chunked evaluation

The evaluation script [evaluate_qa.py](/workspace/contract_qa/scripts/evaluate_qa.py) also processes data in chunks. This prevents validation or test evaluation from holding the entire feature-expanded dataset in memory at once.

This evaluation design supports:

- exact match (EM)
- token-level F1
- no-answer precision / recall / F1
- threshold search on the validation set

## 5. Experimental Process

### 5.1 Initial profiling and smoke tests

Before the final full run, several smaller experiments were used to understand runtime and memory behavior.

Window profiling on a 128-example validation sample gave the following feature counts:

| `max_seq_length` | `doc_stride` | Features | Expansion |
| --- | ---: | ---: | ---: |
| 2048 | 256 | 370 | 2.89x |
| 2048 | 512 | 414 | 3.23x |
| 2048 | 1024 | 569 | 4.45x |
| 3072 | 256 | 288 | 2.25x |
| 3072 | 512 | 288 | 2.25x |
| 3072 | 1024 | 328 | 2.56x |
| 4096 | 256 | 248 | 1.94x |
| 4096 | 512 | 248 | 1.94x |
| 4096 | 1024 | 248 | 1.94x |

Interpretation:

- `4096` creates fewer features, but each feature is more expensive
- `3072` preserves a large context window while reducing per-step cost
- `doc_stride=256` reduces overlap and total feature count

The repository documentation also records:

- a smoke training run with `3072 + 512`
- a streamed debug run on 1,000 examples
- memory-mitigation changes to both training and evaluation

### 5.2 Final selected training configuration

The final selected model is stored in [outputs/train_full_epoch1_3072_stride256_bs4_ga2](/workspace/contract_qa/outputs/train_full_epoch1_3072_stride256_bs4_ga2), with the final summary recorded in [run_summary.json](/workspace/contract_qa/outputs/train_full_epoch1_3072_stride256_bs4_ga2/run_summary.json).

| Parameter | Final value |
| --- | --- |
| `max_seq_length` | `3072` |
| `doc_stride` | `256` |
| `learning_rate` | `2e-5` |
| `num_train_epochs` | `1.0` |
| `per_device_train_batch_size` | `4` |
| `gradient_accumulation_steps` | `2` |
| Effective batch size | `8` |
| `warmup_steps` | `1556` |
| `preprocess_chunk_size` | `500` |
| `gradient_checkpointing` | `false` |
| Precision | `bf16` |

Final training statistics:

- train examples: `20,150`
- train features: `124,491`
- optimizer steps: `15,562`
- training runtime: `12,022.21s`
- training loss: `0.5027`
- training steps per second: `1.294`

### 5.3 Why this final configuration was selected

The final configuration reflects both modelling and systems considerations:

- `3072` tokens preserved enough long-context evidence without the higher per-step cost of `4096`
- `doc_stride=256` reduced redundant overlap
- `batch_size=4, grad_accum=2` used more of the available GPU than the earlier `1 x 8` setup
- disabling gradient checkpointing improved throughput once the RTX 5090 memory budget proved large enough
- `bf16` reduced training cost while remaining stable on the available hardware

## 6. Evaluation Methodology

### 6.1 Quantitative metrics

The project reports:

- Exact Match (EM)
- token-level F1
- no-answer accuracy
- no-answer precision
- no-answer recall
- no-answer F1

These metrics are appropriate because the task requires both:

- correct extraction of answer spans
- correct abstention when the contract does not contain the answer

### 6.2 Threshold search

The system does not hard-code a no-answer threshold. Instead:

1. the threshold is searched on the validation set
2. the best validation threshold is frozen
3. the frozen threshold is applied to the test set

This is an appropriate evaluation design because it avoids tuning directly on the test set.

## 7. Quantitative Results

### 7.1 Validation results

Validation results are stored in [evaluation_summary.json](/workspace/contract_qa/outputs/eval_train_full_epoch1_3072_stride256_bs4_ga2_val/evaluation_summary.json).

Selected threshold on validation:

- `5.015625`

Validation metrics at the selected threshold:

| Metric | Value |
| --- | ---: |
| Exact Match | 61.09 |
| F1 | 68.53 |
| Has-answer Exact Match | 29.15 |
| Has-answer F1 | 43.36 |
| No-answer Exact Match | 96.17 |
| No-answer Accuracy | 96.17 |
| No-answer Precision | 81.90 |
| No-answer Recall | 96.17 |
| No-answer F1 | 88.46 |

Interpretation:

- the system is already very strong on no-answer detection
- answerable examples remain much harder
- the overall validation result is limited mainly by answer-span extraction quality

### 7.2 Test results

Test results are stored in [evaluation_summary.json](/workspace/contract_qa/outputs/eval_train_full_epoch1_3072_stride256_bs4_ga2_test/evaluation_summary.json).

The same threshold selected on validation (`5.015625`) was applied to the test set.

| Metric | Value |
| --- | ---: |
| Exact Match | 85.70 |
| F1 | 88.61 |
| Has-answer Exact Match | 61.66 |
| Has-answer F1 | 71.44 |
| No-answer Exact Match | 95.88 |
| No-answer Accuracy | 95.85 |
| No-answer Precision | 92.30 |
| No-answer Recall | 95.85 |
| No-answer F1 | 94.04 |

### 7.3 Interpreting the validation-test gap

The test result is substantially stronger than the validation result. This should not be interpreted as a purely uniform performance gain. The more likely explanation is the difference in label distribution:

- validation no-answer ratio: `47.65%`
- test no-answer ratio: `70.25%`

Since the model is particularly strong on no-answer predictions, a more no-answer-heavy split naturally yields higher aggregate EM/F1. This is why the report should discuss both:

- overall metrics, and
- answerable versus no-answer breakdowns

## 8. Qualitative Error Analysis

The error analysis below is based on the model predictions in:

- [validation predictions](/workspace/contract_qa/outputs/eval_train_full_epoch1_3072_stride256_bs4_ga2_val/predictions.json)
- [test predictions](/workspace/contract_qa/outputs/eval_train_full_epoch1_3072_stride256_bs4_ga2_test/predictions.json)

The buckets were inferred heuristically by aligning predictions to gold examples and grouping non-exact cases into interpretable types.

### 8.1 Validation error profile

Total validation errors: `895 / 2300` (`38.91%`).

| Error type | Count | Share of validation errors |
| --- | ---: | ---: |
| Wrong span | 458 | 51.17% |
| False negative on answerable example | 233 | 26.03% |
| Overlong span / superset | 105 | 11.73% |
| False positive on no-answer example | 42 | 4.69% |
| Too-short substring | 31 | 3.46% |
| Partial overlap | 26 | 2.91% |

The hardest validation labels by error rate were:

- Parties
- Audit Rights
- Insurance
- License Grant
- Post-Termination Services
- Revenue/Profit Sharing

### 8.2 Test error profile

Total test errors: `599 / 4182` (`14.32%`).

| Error type | Count | Share of test errors |
| --- | ---: | ---: |
| False negative on answerable example | 235 | 39.23% |
| False positive on no-answer example | 122 | 20.37% |
| Overlong span / superset | 95 | 15.86% |
| Wrong span | 77 | 12.85% |
| Too-short substring | 58 | 9.68% |
| Partial overlap | 12 | 2.00% |

The hardest test labels by error rate were:

- Effective Date
- Parties
- Uncapped Liability
- Post-Termination Services
- Cap On Liability
- Minimum Commitment

### 8.3 Main recurring error categories

#### 8.3.1 False abstention on answerable clauses

The model often predicts an empty string even though the answer is present in the contract. This tends to happen when:

- the evidence is phrased indirectly
- the clause is long and embedded in dense legal prose
- the answer is semantically implied rather than expressed as a short literal phrase

This is the single biggest residual failure mode on the test set.

#### 8.3.2 Overlong span extraction

In many cases the model finds the correct region of the contract but returns too much text, such as a full sentence, paragraph, or title block instead of the gold subspan. This hurts EM substantially and still reduces F1.

#### 8.3.3 Date confusion

For `Agreement Date` and `Effective Date`, contracts often contain several salient dates:

- original agreement date
- amendment date
- execution date
- signature date

The model frequently selects the wrong date among these plausible candidates.

#### 8.3.4 Semantic false positives on no-answer cases

For labels such as `Uncapped Liability` or `Post-Termination Services`, the model sometimes extracts a semantically related clause even when the dataset label is no-answer. This means the model partially understands the topic but not the exact annotation target.

#### 8.3.5 Shortened title or entity predictions

For `Document Name` and `Parties`, the model sometimes predicts a shortened or generic form rather than the exact full gold span. This suggests that it identifies the correct concept but not the exact required boundary.

### 8.4 Representative examples

Representative test-time examples include:

- `Minimum Commitment`: the model abstained when the gold answer described a staffing commitment, suggesting difficulty with operational commitments not phrased as direct purchase floors
- `Change Of Control`: the model abstained on assignment/successor language, suggesting incomplete mapping from business concept to legal phrasing
- `Uncapped Liability`: the model extracted a liability-cap clause instead of an uncapped-liability exception
- `Post-Termination Services`: the model extracted termination-effect language that was related but not label-consistent
- `Parties`: the model returned an overly long block of boilerplate plus one party name
- `Document Name`: the model often returned the title region but not the exact annotated title span
- `Effective Date` and `Agreement Date`: the model selected prominent but incorrect dates in documents containing several alternatives

### 8.5 What the error analysis implies

The error analysis suggests that the model's main remaining weakness is not basic topic recognition. Instead, the main limitations are:

- deciding when to abstain versus answer on subtle answerable examples
- selecting exact span boundaries
- disambiguating between multiple nearby dates or legal subclauses

This is encouraging, because it means the model is often close to the correct region even when it does not achieve exact-match success.

## 9. Discussion

### 9.1 Strengths

- The project solves a realistic long-document legal NLP task.
- The final pipeline is fully runnable from the command line.
- The grouped split avoids title-level leakage.
- The system handles no-answer questions effectively.
- The final training and evaluation workflows are memory-aware and reproducible.
- The model achieves strong test performance overall.

### 9.2 Limitations

- The repository does not currently include a retained simple baseline implementation, even though the assignment scope expects one.
- No Gradio or equivalent demo is included in the current codebase.
- The model is still noticeably weaker on answerable span extraction than on no-answer rejection.
- Date disambiguation and exact span boundaries remain difficult.
- The very strong test result should be interpreted in light of the no-answer-heavy test distribution.

## 10. Future Work

Based on the experiments and error analysis, the most promising next steps are:

1. add a simple rule-based or retrieval-style baseline for explicit comparison
2. perform controlled second-stage fine-tuning with a lower learning rate
3. improve span-boundary precision, especially for titles and party lists
4. add features or heuristics for date disambiguation
5. analyze label-specific thresholds or calibration for abstention
6. build a lightweight demo interface for interactive inspection

## 11. Deliverable Mapping to the Assignment Guideline

### Part A: Problem Definition

- NLP problem: contract QA with no-answer detection
- domain: long-form legal contracts and agreements

### Part B: Dataset Selection

- uses an existing provided dataset in train/test JSON form
- no new annotation was created in this repository
- therefore no inter-annotator agreement is applicable

### Part C: Modelling

- main implemented system: fine-tuned Longformer for extractive QA
- command-line inference supported
- a simple retained baseline is not currently present in the repository and should be acknowledged as missing

### Part D: Evaluation

- quantitative evaluation: EM, F1, and no-answer metrics
- qualitative evaluation: detailed error analysis
- command-line testing: implemented via `predict.py`
- demo: not implemented in the current repository

## 12. Conclusion

This project developed a complete long-document contract QA pipeline built around Longformer fine-tuning, grouped data splitting, threshold-based abstention, and chunked evaluation. The final selected system trained successfully for one epoch on 20,150 grouped training examples and achieved `EM 61.09 / F1 68.53` on validation and `EM 85.70 / F1 88.61` on test.

The strongest part of the system is no-answer detection, which is especially valuable in a contract review setting where many requested fields are absent. The main remaining challenge is precise extraction of answerable spans, particularly for dense legal clauses, document titles, party names, and date fields with multiple plausible candidates. Overall, the project demonstrates that a well-engineered long-context QA pipeline can produce strong, reproducible results on contract analysis while still leaving clear avenues for future improvement.
