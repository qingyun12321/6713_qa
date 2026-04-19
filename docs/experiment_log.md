# Experiment Log

This file records smoke-test results and the parameter decisions we lock afterward.

## Status

- `2026-04-18`: project scaffold created, grouped split and Longformer QA scripts added.
- `2026-04-18`: grouped split prepared at `outputs/prepared_data`.
- `2026-04-18`: window profile completed on a 128-example validation sample.
- `2026-04-18`: `3072 + 512` smoke training run completed successfully with `bf16` and gradient checkpointing.
- `2026-04-18`: evaluation and threshold search completed on 128 validation examples.
- `2026-04-19`: streamed preprocessing debug run completed successfully on `1000` train examples with `max_steps=5`.

## Current Baseline

- Data split:
  - train: `20150` rows, `367` titles
  - validation: `2300` rows, `41` titles
  - test: `4182` rows, `102` titles
- Training baseline:
  - `max_seq_length=3072`
  - `doc_stride=512`
  - `max_answer_length=256`
  - `learning_rate=2e-5`
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=8`
  - `gradient_checkpointing=true`
  - `bf16=true` on the local GPU

## Profiling Notes

- Sampled validation window expansion:
  - `2048 + 256`: `370` features from `128` examples
  - `3072 + 512`: `288` features from `128` examples
  - `4096 + 512`: `248` features from `128` examples
- The `3072 + 512` setting is the current default because it is cheaper than `4096` per feature while already fitting comfortably.
- Streamed debug run on real train data:
  - `1000` train examples expanded to `5377` train features
  - empirical train-side expansion in that slice: about `5.38x`
  - this is much higher than the earlier validation-sample expansion estimate, so full-epoch runtime may be materially longer than the original rough estimate

## Runtime Estimate

- Estimation method:
  - use grouped train size `20150`
  - use sampled feature expansion `2.25x` for `3072 + 512`
  - estimate about `45300` training features per epoch
  - divide by `gradient_accumulation_steps=8`
  - estimate about `5660` optimizer steps per epoch
- Based on smoke-test throughput, a realistic planning range is:
  - lower bound: about `3.7 to 6.6 hours` of pure training time per epoch
  - safer practical budget: about `5 to 7 hours` per epoch
- Planned safety rule:
  - no full preprocessing in parallel with training
  - no concurrent training jobs
  - start real runs with staged checks before committing to a full epoch

## Training Stop Rule

- For this project, `max_steps` should be treated as a safety cap for smoke tests.
- A real full training run should use:
  - `max_steps=-1`
  - `num_train_epochs=1`
- Only move to `num_train_epochs=2` after checking validation behavior and system stability.

## Smoke Run Notes

- Training smoke test:
  - examples: `8`
  - features: `80`
  - runtime: `4.63s`
  - steps per second: `0.432`
- Evaluation smoke test:
  - examples: `128`
  - features: `288`
  - best threshold: `-2.6809`
  - EM: `77.34`
  - F1: `77.34`
  - `has_answer_f1`: `3.33`
  - `no_answer_accuracy`: `100.0`

## Interpretation

- The system is currently biased toward no-answer because the checkpoint is barely trained.
- This is acceptable for now because the purpose of the smoke run was to validate preprocessing, training, checkpoint saving, evaluation, threshold search, and CLI prediction.
- Threshold tuning is confirmed to be necessary for final reporting.
- A later full-data attempt crashed during train-set preprocessing before training began.
- The evidence points to preprocessing memory pressure rather than GPU training:
  - `train.log` shows the run stopped after `data_loaded`
  - the crash happened during the `Map:` progress phase
  - the log still showed `cuda_allocated_mb = 0.0`, so no real GPU training step had started yet
- User observation: system memory usage was around `85% to 90%` at the time of the crash.
- Mitigations added:
  - training data is now streamed from JSON instead of being loaded fully into memory
  - train preprocessing now runs chunk-by-chunk to disk with default `--preprocess-chunk-size 100`
  - validation preprocessing was removed from the training script because it was unused when `eval_strategy=no`
- Streamed debug run evidence:
  - preprocessing completed across `10` chunks without memory runaway
  - process RSS stayed around `2.1 GB` to `2.9 GB` during preprocessing
  - GPU memory was still `0` during preprocessing, confirming the earlier crash happened before training proper
  - once training began, GPU allocated memory settled around `1.7 GB`, with peak allocated about `2.9 GB`
  - checkpoint saving at step `5` succeeded
- A later validation-evaluation attempt also pushed WSL too hard because the older evaluation script still kept too much validation content resident in memory.
- Evaluation mitigations added:
  - validation examples are now streamed from JSON in chunks
  - each chunk is tokenized, predicted, and released before moving to the next chunk
  - after chunk prediction, only minimal example metadata is retained for metric computation and prediction export
  - the same training-time window parameters should be reused for evaluation, especially `max_seq_length=3072` and `doc_stride=256` for the current checkpoint
- Early evidence from the new chunked evaluation path:
  - evaluation RSS stayed around `3 GB` to `4 GB` instead of pushing system memory near `90%`
  - the bottleneck shifted from memory pressure to slow window-by-window forward passes, which is acceptable

## Next Step

- Run a longer training job with the current `3072 + 512` baseline.
- Then run a controlled comparison against `4096 + 512`.
- If runtime needs to come down, test `3072 + 256` before shrinking `max_seq_length` aggressively.
- Only reduce `max_seq_length` after checking whether the runtime gain is worth the extra window fragmentation.
