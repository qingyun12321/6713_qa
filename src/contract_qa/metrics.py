from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    return max(metric_fn(prediction, truth) for truth in ground_truths)


def apply_threshold(record: dict[str, Any], threshold: float) -> str:
    return "" if record["score_diff"] > threshold else record["best_text"]


def evaluate_prediction_records(
    examples: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    threshold: float,
) -> dict[str, float]:
    overall_exact = 0.0
    overall_f1 = 0.0
    has_ans_exact = 0.0
    has_ans_f1 = 0.0
    no_ans_exact = 0.0
    no_ans_f1 = 0.0
    has_ans_count = 0
    no_ans_count = 0

    no_answer_tp = 0
    no_answer_fp = 0
    no_answer_fn = 0
    no_answer_tn = 0

    for example in examples:
        record = records_by_id[example["id"]]
        prediction = apply_threshold(record, threshold)
        references = example["answers"]["text"] or [""]
        exact = metric_max_over_ground_truths(exact_match_score, prediction, references)
        f1 = metric_max_over_ground_truths(token_f1_score, prediction, references)

        overall_exact += exact
        overall_f1 += f1

        has_answer = bool(example["answers"]["text"])
        predicted_no_answer = prediction == ""
        if has_answer:
            has_ans_count += 1
            has_ans_exact += exact
            has_ans_f1 += f1
            if predicted_no_answer:
                no_answer_fp += 1
            else:
                no_answer_tn += 1
        else:
            no_ans_count += 1
            no_ans_exact += exact
            no_ans_f1 += f1
            if predicted_no_answer:
                no_answer_tp += 1
            else:
                no_answer_fn += 1

    total = max(1, len(examples))
    no_answer_precision = no_answer_tp / max(1, no_answer_tp + no_answer_fp)
    no_answer_recall = no_answer_tp / max(1, no_answer_tp + no_answer_fn)
    no_answer_f1 = (
        0.0
        if no_answer_precision + no_answer_recall == 0.0
        else 2 * no_answer_precision * no_answer_recall / (no_answer_precision + no_answer_recall)
    )

    return {
        "threshold": threshold,
        "exact": 100.0 * overall_exact / total,
        "f1": 100.0 * overall_f1 / total,
        "has_answer_exact": 100.0 * has_ans_exact / max(1, has_ans_count),
        "has_answer_f1": 100.0 * has_ans_f1 / max(1, has_ans_count),
        "no_answer_exact": 100.0 * no_ans_exact / max(1, no_ans_count),
        "no_answer_f1_text": 100.0 * no_ans_f1 / max(1, no_ans_count),
        "no_answer_accuracy": 100.0 * no_answer_tp / max(1, no_ans_count),
        "no_answer_precision": 100.0 * no_answer_precision,
        "no_answer_recall": 100.0 * no_answer_recall,
        "no_answer_f1": 100.0 * no_answer_f1,
        "has_answer_count": has_ans_count,
        "no_answer_count": no_ans_count,
    }


def search_best_threshold(
    examples: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    metric_name: str = "f1",
) -> tuple[float, dict[str, float]]:
    thresholds = sorted({record["score_diff"] for record in records_by_id.values()})
    candidate_thresholds = [thresholds[0] - 1.0, *thresholds, thresholds[-1] + 1.0] if thresholds else [0.0]

    best_threshold = candidate_thresholds[0]
    best_metrics = evaluate_prediction_records(examples, records_by_id, best_threshold)
    best_value = best_metrics[metric_name]

    for threshold in candidate_thresholds[1:]:
        metrics = evaluate_prediction_records(examples, records_by_id, threshold)
        value = metrics[metric_name]
        if value > best_value:
            best_value = value
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics

