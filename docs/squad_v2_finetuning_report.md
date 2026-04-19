# SQuAD v2 Fine-Tuning Report for Longformer

## 1. Objective

This report documents the full experimental process for fine-tuning the local `longformer-base-4096` checkpoint on **SQuAD v2**.

The goal of this experiment was to answer a practical migration question:

- the project had originally been built around a contract QA dataset with very long contexts
- the same codebase now needed to be adapted to a much shorter, more standard extractive QA benchmark
- the model should be trained from the **raw pretrained Longformer weights**
- the final system should still support both extractive answer prediction and **no-answer detection**

This report therefore covers:

- dataset preparation
- parameter selection
- training behavior
- validation and test performance
- error analysis
- limitations and next steps

## 2. Model and Task Setup

### 2.1 Model

The model used in this experiment was:

- `models/longformer-base-4096`

This is a Longformer checkpoint with a maximum position embedding size of `4098`, which makes it suitable for long-document QA. Although SQuAD v2 does not require such long context in practice, reusing this model keeps the pipeline consistent with the project codebase.

### 2.2 Task formulation

The task was framed as **extractive QA with no-answer detection**:

- if an answer exists, the model predicts a start token and end token
- if no answer exists, the model is trained to point to the CLS token
- during evaluation, the model compares the null score and best span score
- a threshold selected on validation is then frozen and reused on test

This is the same core QA formulation already implemented in:

- [train_qa.py](/workspace/contract_qa/scripts/train_qa.py)
- [evaluate_qa.py](/workspace/contract_qa/scripts/evaluate_qa.py)
- [qa_utils.py](/workspace/contract_qa/src/contract_qa/qa_utils.py)

## 3. Dataset Preparation

### 3.1 Source dataset

The source dataset was:

- `rajpurkar/squad_v2`

This dataset contains:

- answerable questions
- explicitly unanswerable questions
- multiple reference answers for many validation examples

That makes it a strong fit for the current codebase, which already supports both extractive span scoring and no-answer thresholding.

### 3.2 Why a new data script was needed

The original project data preparation path was written for the contract dataset and assumed:

- a separate raw train file and raw test file already existed
- examples used a contract-specific flat format

SQuAD v2 does not ship in that exact structure, so a dedicated script was added:

- [prepare_squad_v2.py](/workspace/contract_qa/scripts/prepare_squad_v2.py)

This script:

- downloads SQuAD v2 through Hugging Face `datasets`
- converts each example into the project’s standard JSON schema
- normalizes answers into:
  - `{"text": [...], "answer_start": [...]}`
- preserves `is_impossible`
- writes `train.json`, `validation.json`, `test.json`
- also writes `summary.json` and `split_manifest.json`

### 3.3 Validation/test split strategy

Following the requested protocol, the official SQuAD v2 `validation` split was divided `1:1` into a new validation split and a new test split.

To avoid a skewed no-answer distribution, the split was **stratified by `is_impossible`** rather than done as a single naive shuffle.

The split manifest is stored in:

- [split_manifest.json](/workspace/contract_qa/outputs/squad_v2_prepared/split_manifest.json)

The resulting split sizes were:

| Split | Rows |
| --- | ---: |
| Train | 130,319 |
| Validation | 5,936 |
| Test | 5,937 |

The answerability balance was:

| Split | Has-answer | No-answer | No-answer ratio |
| --- | ---: | ---: | ---: |
| Train | 86,821 | 43,498 | 33.38% |
| Validation | 2,964 | 2,972 | 50.07% |
| Test | 2,964 | 2,973 | 50.08% |

This is a good split for evaluation because validation and test have almost identical class balance.

### 3.4 Input length profile

The key dataset statistics are stored in:

- [summary.json](/workspace/contract_qa/outputs/squad_v2_prepared/summary.json)

Important observations:

- train context median length: `692` characters
- validation context median length: `718` characters
- test context median length: `713` characters
- train question median length: `55` characters
- validation question median length: `57` characters
- test question median length: `56` characters

Compared with the original contract dataset, SQuAD v2 is dramatically shorter. This was the main reason for revising the preprocessing strategy away from extremely large windows.

### 3.5 Data quality issue discovered during setup

During the first SQuAD training attempt, tokenization failed with:

```text
Exception: Truncation error: Sequence to truncate too short to respect the provided max_length
```

The root cause was a malformed training example:

- `id = 572fdefb947a6a140053cd8d`
- title: `Antenna_(radio)`
- the question contained a very long run of whitespace characters

This made the question abnormally long and broke `truncation="only_second"` because only the context was allowed to be truncated.

The issue was fixed in two places:

- [prepare_squad_v2.py](/workspace/contract_qa/scripts/prepare_squad_v2.py:31) now strips question and title text during dataset generation
- [qa_utils.py](/workspace/contract_qa/src/contract_qa/qa_utils.py:9) now also sanitizes questions before tokenization

This fix is important because it makes the pipeline robust to malformed whitespace-only or whitespace-heavy questions in future datasets as well.

## 4. Parameter Selection

### 4.1 Final configuration

The final training configuration used for SQuAD v2 was:

| Parameter | Value |
| --- | --- |
| Model | `longformer-base-4096` |
| `max_seq_length` | `512` |
| `doc_stride` | `128` |
| `max_answer_length` | `64` |
| `learning_rate` | `3e-5` |
| `num_train_epochs` | `1` |
| `per_device_train_batch_size` | `4` |
| `gradient_accumulation_steps` | `2` |
| Effective batch size | `8` |
| `preprocess_chunk_size` | `500` |
| Warmup | `10%` of total steps |
| Precision | `bf16` |
| Gradient checkpointing | `false` |

The exact run record is stored in:

- [run_summary.json](/workspace/contract_qa/outputs/squad_v2_epoch1_seq512_stride128_bs4_ga2/run_summary.json)

### 4.2 Why `max_seq_length = 512`

The original contract QA experiments used much larger contexts, but SQuAD v2 is short enough that this is unnecessary.

In this run:

- train features: `130,550` from `130,319` examples
- validation features: `5,975` from `5,936` examples
- test features: `5,980` from `5,937` examples

That means the expansion factors were:

| Split | Feature expansion |
| --- | ---: |
| Train | `1.0018x` |
| Validation | `1.0066x` |
| Test | `1.0072x` |

This is a strong signal that almost all examples fit in a single window already. So `512` is large enough to preserve nearly all contexts without wasting compute.

### 4.3 Why keep `doc_stride = 128`

Even though sliding windows were barely used in practice, keeping a moderate stride of `128` is still reasonable because:

- it safely handles the small number of overflow cases
- it keeps the code path consistent with the long-context QA pipeline
- it introduces negligible expansion overhead on this dataset

### 4.4 Why `max_answer_length = 64`

The original contract QA setup used a much larger answer length because contractual answers can be long clauses. On SQuAD v2, answers are much shorter.

`64` was chosen because it:

- is much tighter than the contract setting
- still provides enough slack to avoid cutting off longer SQuAD spans
- reduces the risk of selecting implausibly long answer spans during postprocessing

### 4.5 Why `learning_rate = 3e-5`

This run started from the **raw pretrained Longformer weights**, not from a previously fine-tuned QA checkpoint.

For that setting:

- `3e-5` is a standard and reasonable first-pass QA learning rate
- `1e-5` would likely be too conservative for a single-epoch run
- `2e-5` would be a sensible fallback if instability appeared

In this experiment, training was stable enough that `3e-5` did not need to be reduced.

### 4.6 Why `batch_size = 4`, `accumulation = 2`

The original long-contract experiments had to be conservative because of large sequences. Once SQuAD switched to `512` token windows, the GPU had much more headroom.

This allowed:

- `per_device_train_batch_size = 4`
- `gradient_accumulation_steps = 2`
- effective batch size `8`

The logs show this was well within budget:

- peak allocated GPU memory stayed around `5.12 GB`
- reserved memory stayed around `6.74 GB`
- the device was an `NVIDIA GeForce RTX 5090` with `~32 GB` total memory

So for this SQuAD setup, gradient checkpointing was not needed.

## 5. Training Process

### 5.1 Command used

The SQuAD training run used the command documented in:

- [README.md](/workspace/contract_qa/README.md:445)

Equivalent command:

```bash
uv run python "scripts/train_qa.py" \
  --train-path "outputs/squad_v2_prepared/train.json" \
  --model-path "models/longformer-base-4096" \
  --output-dir "outputs/squad_v2_epoch1_seq512_stride128_bs4_ga2" \
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
  --preprocess-chunk-size 500
```

### 5.2 Scale of the run

From [run_summary.json](/workspace/contract_qa/outputs/squad_v2_epoch1_seq512_stride128_bs4_ga2/run_summary.json:1):

- raw training examples: `130,319`
- tokenized training features: `130,550`
- total optimizer steps: `16,319`
- warmup steps: `1,631`

### 5.3 Runtime and throughput

Final training statistics were:

| Metric | Value |
| --- | ---: |
| `train_runtime` | `4819.72s` |
| Runtime in minutes | `80.33` |
| `train_samples_per_second` | `27.09` |
| `train_steps_per_second` | `3.39` |
| `train_loss` | `1.1004` |

This is a very practical runtime for a full-epoch experiment on the full SQuAD training split.

### 5.4 Training stability

Training was stable throughout:

- no OOM occurred
- no tokenizer failure remained after data sanitization
- no crash occurred during checkpoint saves
- the final checkpoint reached `global_step = 16319`

Late-stage training logs show losses generally staying in a healthy range around `0.70` to `0.92` in the final stretch, rather than exploding.

## 6. Validation Evaluation

### 6.1 Validation setup

Validation used:

- the new `validation.json`
- `max_seq_length = 512`
- `doc_stride = 128`
- `max_answer_length = 64`
- `per_device_eval_batch_size = 4`
- threshold search enabled

### 6.2 Validation results

Validation summary is stored in:

- [evaluation_summary.json](/workspace/contract_qa/outputs/squad_v2_epoch1_seq512_stride128_bs4_ga2_val/evaluation_summary.json)

Best validation threshold for F1:

- `-0.796875`

Validation metrics:

| Metric | Value |
| --- | ---: |
| Exact Match | `80.61` |
| F1 | `83.65` |
| Has-answer EM | `75.51` |
| Has-answer F1 | `81.60` |
| No-answer Accuracy | `85.70` |
| No-answer Precision | `87.17` |
| No-answer Recall | `85.70` |
| No-answer F1 | `86.43` |

### 6.3 Validation runtime note

The validation log shows two very different phases:

- chunked tokenization and forward prediction were fast
- threshold search took much longer than the forward pass itself

From [eval.log](/workspace/contract_qa/outputs/squad_v2_epoch1_seq512_stride128_bs4_ga2_val/eval.log):

- the chunked forward pass finished at about `97.91s`
- the total validation run took about `4004.98s`

So the threshold search dominated runtime on validation.

This is not a correctness problem, but it is a useful engineering observation:

- fixed-threshold evaluation is cheap
- exhaustive threshold search is the main evaluation bottleneck

## 7. Test Evaluation

### 7.1 Test setup

The test run reused the threshold selected on validation:

- `threshold = -0.796875`

This follows the correct evaluation protocol:

- tune threshold on validation
- freeze threshold
- evaluate once on test

### 7.2 Test results

Test summary is stored in:

- [evaluation_summary.json](/workspace/contract_qa/outputs/squad_v2_epoch1_seq512_stride128_bs4_ga2_test/evaluation_summary.json)

Test metrics were:

| Metric | Value |
| --- | ---: |
| Exact Match | `80.70` |
| F1 | `83.62` |
| Has-answer EM | `76.05` |
| Has-answer F1 | `81.90` |
| No-answer Accuracy | `85.33` |
| No-answer Precision | `87.63` |
| No-answer Recall | `85.33` |
| No-answer F1 | `86.47` |

### 7.3 Test runtime

The test run did not search thresholds, so it was much faster.

From [eval.log](/workspace/contract_qa/outputs/squad_v2_epoch1_seq512_stride128_bs4_ga2_test/eval.log):

- total test runtime: `91.02s`

This confirms that most of the extra validation runtime came from threshold tuning rather than model inference.

## 8. Interpretation of the Results

### 8.1 Overall conclusion

This experiment should be considered a **successful fine-tuning run**.

The strongest evidence is that validation and test performance are almost identical:

| Metric | Validation | Test | Difference |
| --- | ---: | ---: | ---: |
| EM | `80.61` | `80.70` | `+0.09` |
| F1 | `83.65` | `83.62` | `-0.03` |
| Has-answer EM | `75.51` | `76.05` | `+0.54` |
| Has-answer F1 | `81.60` | `81.90` | `+0.31` |
| No-answer Accuracy | `85.70` | `85.33` | `-0.37` |
| No-answer F1 | `86.43` | `86.47` | `+0.04` |

These differences are very small. That indicates:

- the validation threshold generalized well
- the new validation/test split was balanced and sensible
- the model did not overfit validation-specific quirks in any obvious way

### 8.2 What the model is good at

The trained model shows two clear strengths:

1. It handles no-answer cases reliably.
   The no-answer metrics are consistently in the mid-80s, with precision around `87%`.

2. It extracts answerable spans reasonably well for a first-pass one-epoch run from raw pretrained weights.
   A has-answer F1 around `81.6` to `81.9` is a strong outcome given that this was not a long hyperparameter sweep.

### 8.3 What remains imperfect

The model is not yet saturating the benchmark. The most obvious gaps are:

- abstaining when it should answer
- answering when it should abstain
- selecting spans that are semantically close but slightly too long or too short

So the residual error is less about catastrophic misunderstanding and more about:

- boundary precision
- null-vs-span calibration
- occasional semantic confusion between nearby entities or dates

## 9. Error Analysis

### 9.1 Error counts

Using [predictions.json](/workspace/contract_qa/outputs/squad_v2_epoch1_seq512_stride128_bs4_ga2_test/predictions.json), the test split had:

- total examples: `5,937`
- non-exact predictions: `1,146`
- exact-match error rate: `19.30%`

Validation was nearly the same:

- total examples: `5,936`
- non-exact predictions: `1,151`
- exact-match error rate: `19.39%`

Again, this confirms stable generalization.

### 9.2 Error type breakdown

The test errors can be grouped as follows:

| Error type | Count | Share of all test errors |
| --- | ---: | ---: |
| False positive on no-answer example | 436 | `38.05%` |
| False negative on answerable example | 357 | `31.15%` |
| Too-long superspan | 155 | `13.53%` |
| Too-short subspan | 101 | `8.81%` |
| Wrong span / semantic miss | 90 | `7.85%` |
| Partial overlap | 7 | `0.61%` |

The validation breakdown was very similar:

| Error type | Count |
| --- | ---: |
| False positive on no-answer example | 425 |
| False negative on answerable example | 375 |
| Too-long superspan | 142 |
| Too-short subspan | 127 |
| Wrong span | 76 |
| Partial overlap | 6 |

### 9.3 Main takeaway from the error mix

The dominant failure modes are calibration-related:

- predicting an answer where there should be none
- predicting no answer where a real answer exists

Together these two categories account for:

- `793 / 1146 = 69.20%` of all test errors

This means the biggest remaining weakness is **null-vs-span decision quality**, not raw extractive ability alone.

### 9.4 Span boundary errors

The next biggest group is boundary mismatch:

- too long: `155`
- too short: `101`

These are often near-miss predictions where the model finds the right region but not the exact official boundary.

Representative examples:

1. `571a50df4faf5e1900b8a962` from `Oxygen`
   Prediction: `Combustion hazards also apply to compounds of oxygen with a high oxidative potential`
   Reference: `compounds of oxygen with a high oxidative potential`
   Interpretation: the model found the right answer but included extra leading words.

2. `57286d4f2ca10214002da329` from `Yuan_dynasty`
   Prediction: `1268 and 1273`
   Reference: `between 1268 and 1273`
   Interpretation: the answer is semantically correct but slightly too short.

3. `5733140a4776f419006606e4` from `Warsaw`
   Prediction: `it has survived many wars, conflicts and invasions throughout its long history`
   Reference set includes `survived many wars, conflicts and invasions`
   Interpretation: the model over-selected a closely related but longer phrase.

### 9.5 False positives on no-answer examples

These cases happen when the model becomes too confident in a plausible-looking span even though the correct action is abstention.

Representative examples:

1. `5ad258b4d7d075001a428ddc` from `Oxygen`
   Predicted answer: `Reactive oxygen species`
   Reference: no answer

2. `5a6395e668151a001a9223a2` from `Victoria_(Australia)`
   Predicted answer: `32 °C`
   Reference: no answer

3. `5a892d303b2508001a72a4f0` from `Prime_number`
   Predicted answer: `deterministic algorithms`
   Reference: no answer

These errors suggest that the model sometimes treats a locally relevant noun phrase or number as an answer simply because it fits the question surface form.

### 9.6 False negatives on answerable examples

These happen when the model abstains even though an answer exists.

Representative examples:

1. `5737a7351c456719005744f1` from `Force`
   Model abstained.
   Best span candidate: `kinetic`
   Reference: `kinetic`

2. `573755afc3c5551400e51eb5` from `Force`
   Model abstained.
   Best span candidate: `the mass of the system`
   Reference: `mass of the system`

3. `5728ebcb3acd2414000e01dd` from `Civil_disobedience`
   Model abstained.
   Best span candidate: `not guilty`
   Reference set centered on `creative plea` / `no contest`

The first two examples are especially revealing because the model had effectively found the right phrase family but still chose the null option. That points again to calibration rather than complete comprehension failure.

### 9.7 Wrong-span semantic misses

These are the cleanest content-understanding failures, where the predicted span is genuinely about the wrong entity, date, cause, or relation.

Representative examples:

1. `5710e9f8a58dae1900cd6b33` from `Huguenot`
   Prediction: `French`
   Reference: `Spanish`

2. `5725cbb289a1e219009abed5` from `Amazon_rainforest`
   Prediction: `between AD 0–1250`
   Reference: `1970s`

3. `57339c16d058e614000b5ec6` from `Warsaw`
   Prediction: `Ogród Saski`
   Reference: `Saxon Garden`

The last example is especially interesting because it is not fully wrong semantically; it is a translation mismatch. That shows one limitation of exact surface-form evaluation.

### 9.8 Harder topics

The lowest-scoring test titles by F1 among topics with at least 50 examples included:

| Title | Count | F1 | EM |
| --- | ---: | ---: | ---: |
| `Packet_switching` | 179 | 69.15 | 64.80 |
| `Civil_disobedience` | 166 | 70.18 | 67.47 |
| `Force` | 167 | 72.55 | 71.26 |
| `Victoria_(Australia)` | 126 | 73.46 | 67.46 |
| `Jacksonville,_Florida` | 84 | 73.97 | 70.24 |

These topics may be harder because they contain:

- more dense factual detail
- more closely competing candidate spans
- more dates, place names, technical terms, or paraphrased references

## 10. Strengths of This Experimental Setup

This run had several practical strengths:

1. The codebase was successfully adapted from a contract-specific QA workflow to a standard benchmark workflow without changing the core training architecture.

2. The final configuration was efficient.
   Training on the full SQuAD train split completed in about `80` minutes, which is very manageable.

3. The chosen `512 / 128` setup was empirically justified.
   Feature expansion stayed close to `1.0x`, so the configuration was efficient rather than wasteful.

4. Validation and test were extremely consistent.
   That is one of the strongest indicators that the final reported result is trustworthy.

5. The pipeline is now more robust after fixing malformed whitespace-heavy questions.

## 11. Limitations and Shortcomings

Despite the strong outcome, there are still important limitations.

### 11.1 No hyperparameter sweep

This report covers one primary final configuration, not a full search over:

- multiple learning rates
- multiple epoch counts
- different warmup schedules
- threshold calibration alternatives

So the result is strong, but it is not necessarily optimal.

### 11.2 Only one epoch was tested in the final configuration

`1` epoch worked well, but this leaves open a useful question:

- would `2` epochs improve answerable-example extraction
- or would they mainly worsen no-answer calibration

That remains untested here.

### 11.3 Threshold search is expensive

The validation threshold search is much slower than the actual inference pass. This is not a modeling weakness, but it is a pipeline inefficiency that would matter in repeated experiments.

### 11.4 Longformer may be overkill for SQuAD v2

The model worked, but SQuAD v2 rarely needed the long-context capability:

- train expansion was only `1.0018x`
- validation and test expansion were about `1.007x`

This suggests a shorter-context QA encoder could potentially reach similar quality with lower compute cost.

### 11.5 Residual calibration errors

The largest error groups were:

- false positives on no-answer examples
- false negatives on answerable examples

That suggests there is still room to improve:

- null-score calibration
- threshold robustness
- confidence separation between abstention and span prediction

## 12. Recommended Next Steps

The next experiments should be prioritized in this order:

### 12.1 Try a small learning-rate sweep

Recommended values:

- `2e-5`
- `3e-5`
- `4e-5`

Reason:

- calibration errors dominate the residual error
- a small LR sweep may improve null-vs-span behavior without requiring major pipeline changes

### 12.2 Test a second epoch carefully

Recommended comparison:

- current configuration at `1` epoch
- same configuration at `2` epochs

Key question:

- does has-answer F1 improve without degrading no-answer precision too much

### 12.3 Consider calibration-focused postprocessing

Because null-vs-span mistakes dominate the errors, useful follow-up work could include:

- more efficient threshold search
- threshold selection targeted to a weighted objective
- calibration analysis of score distributions on answerable vs no-answer examples

### 12.4 Consider a smaller QA backbone for SQuAD-specific efficiency

Longformer works, but it is not necessarily the most compute-efficient choice for this dataset. If the goal shifts from codebase consistency to benchmark efficiency, a shorter-context model would be worth testing.

## 13. Final Conclusion

The SQuAD v2 fine-tuning experiment was successful.

The final model:

- trained stably from raw pretrained Longformer weights
- used a revised short-context configuration better matched to SQuAD v2
- achieved strong and highly consistent validation and test performance
- handled both extractive spans and no-answer cases well

The most important quantitative conclusion is:

- validation F1: `83.65`
- test F1: `83.62`

with near-identical EM and no-answer metrics as well.

That level of agreement strongly suggests the final setup is sound.

At the same time, the error analysis shows where improvement is still possible:

- null-vs-span calibration
- answer boundary precision
- occasional semantic confusion between nearby candidate spans

So the final result is best described as:

- a robust, well-matched first full SQuAD fine-tuning configuration
- with clear, practical avenues for further improvement
