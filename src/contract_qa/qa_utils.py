from __future__ import annotations

import collections
from typing import Any

import numpy as np


def sanitize_questions(questions: list[str]) -> list[str]:
    # Some upstream QA datasets contain malformed whitespace-only questions.
    # Stripping them avoids tokenizer truncation errors with only_second truncation.
    return [question.strip() for question in questions]


def build_global_attention_mask(sequence_ids: list[int | None], attention_mask: list[int]) -> list[int]:
    global_attention = []
    for seq_id, attn in zip(sequence_ids, attention_mask, strict=False):
        if not attn:
            global_attention.append(0)
        elif seq_id != 1:
            global_attention.append(1)
        else:
            global_attention.append(0)
    return global_attention


def prepare_train_features(
    examples: dict[str, list[Any]],
    tokenizer,
    max_seq_length: int,
    doc_stride: int,
) -> dict[str, list[Any]]:
    tokenized_examples = tokenizer(
        sanitize_questions(examples["question"]),
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["global_attention_mask"] = []

    for feature_index, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][feature_index]
        attention_mask = tokenized_examples["attention_mask"][feature_index]
        sequence_ids = tokenized_examples.sequence_ids(feature_index)
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sample_index = sample_mapping[feature_index]
        answers = examples["answers"][sample_index]

        tokenized_examples["global_attention_mask"].append(
            build_global_attention_mask(sequence_ids, attention_mask)
        )

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            continue

        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        tokenized_examples["start_positions"].append(token_start_index - 1)

        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(
    examples: dict[str, list[Any]],
    tokenizer,
    max_seq_length: int,
    doc_stride: int,
) -> dict[str, list[Any]]:
    tokenized_examples = tokenizer(
        sanitize_questions(examples["question"]),
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []
    tokenized_examples["global_attention_mask"] = []

    for feature_index in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(feature_index)
        sample_index = sample_mapping[feature_index]
        attention_mask = tokenized_examples["attention_mask"][feature_index]
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["global_attention_mask"].append(
            build_global_attention_mask(sequence_ids, attention_mask)
        )

        offsets = tokenized_examples["offset_mapping"][feature_index]
        tokenized_examples["offset_mapping"][feature_index] = [
            offset if sequence_ids[token_idx] == 1 else None
            for token_idx, offset in enumerate(offsets)
        ]

    return tokenized_examples


def postprocess_qa_predictions(
    examples: list[dict[str, Any]],
    features: list[dict[str, Any]],
    raw_predictions: tuple[np.ndarray, np.ndarray],
    n_best_size: int = 20,
    max_answer_length: int = 256,
) -> dict[str, dict[str, Any]]:
    start_logits, end_logits = raw_predictions
    example_id_to_index = {example["id"]: idx for idx, example in enumerate(examples)}
    features_per_example: dict[int, list[int]] = collections.defaultdict(list)
    for feature_index, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(feature_index)

    outputs: dict[str, dict[str, Any]] = {}
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        best_text = ""
        best_span_score = -float("inf")
        best_offsets = None

        context = example["context"]
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            input_ids = features[feature_index]["input_ids"]
            cls_index = input_ids.index(0)
            null_score = float(start_logit[cls_index] + end_logit[cls_index])
            if min_null_score is None or null_score < min_null_score:
                min_null_score = null_score

            start_indexes = np.argsort(start_logit)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        continue
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]
                    score = float(start_logit[start_index] + end_logit[end_index])
                    if score > best_span_score:
                        best_span_score = score
                        best_text = context[start_char:end_char]
                        best_offsets = [start_char, end_char]

        if min_null_score is None:
            min_null_score = 0.0

        outputs[example["id"]] = {
            "id": example["id"],
            "title": example["title"],
            "best_text": best_text,
            "best_span_score": best_span_score,
            "null_score": min_null_score,
            "score_diff": float(min_null_score - best_span_score),
            "best_offsets": best_offsets,
        }

    return outputs
