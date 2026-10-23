"""Decision metrics for a mule-risk score.
You write this file. Keep the signatures exactly as they are: the grader calls them by name.

Rules for every function:
- y_true and scores are lists or 1-D arrays. Raise ValueError if the lengths differ, the
  input is empty, a label is not 0 or 1, or a score is NaN.
- A case is flagged when score >= threshold.
- Precision is 0.0 when nothing is flagged; recall is 0.0 when there are no positives.
"""
from __future__ import annotations
import numpy as np
import math

# a validation helper
def validate(y_true, scores):
    """Validate labels and scores and return lists."""
    y = np.asarray(y_true)
    s = np.asarray(scores)

    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")

    if len(y) == 0 or len(s) == 0:
        raise ValueError("Inputs must not be empty")

    if len(y) != len(s):
        raise ValueError("Labels and scores must have the same length")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")

    try:
        s = s.astype(float)
    except (TypeError, ValueError):
        raise ValueError("Scores must be numeric") from None

    if np.isnan(s).any():
        raise ValueError("Scores must not contain NaN")

    return y.astype(int).tolist(), s.tolist()


def confusion_at(y_true, scores, threshold: float) -> dict:
    """Calculate confusion counts at a threshold."""
    y, scores = validate(y_true, scores)
    tp = fp = tn = fn = 0

    for actual, score in zip(y, scores):
        flagged = score >= threshold

        if actual == 1 and flagged:
            tp += 1
        elif actual == 0 and flagged:
            fp += 1
        elif actual == 0 and not flagged:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_at(y_true, scores, threshold: float) -> tuple[float, float]:
    """Calculate precision and recall at a threshold."""
    counts = confusion_at(y_true, scores, threshold)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def precision_at_k(y_true, scores, k: int) -> float:
    """Calculate the share of positives among the top k scores."""
    y, scores = validate(y_true, scores)

    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, len(y))

    # Python's sort is stable, so tied scores keep their original order.
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    top_indices = ranked_indices[:k]
    positives = sum(y[i] == 1 for i in top_indices)
    return positives / k


def expected_cost(y_true, scores, threshold: float, *, cost_fp: float, cost_fn: float) -> float:
    """Calculate the cost of false positives and false negatives."""
    counts = confusion_at(y_true, scores, threshold)

    return counts["fp"] * cost_fp + counts["fn"] * cost_fn


def best_threshold(y_true, scores, *, cost_fp: float, cost_fn: float) -> float:
    """Find the lowest-cost threshold, preferring higher thresholds on ties."""
    y, scores = validate(y_true, scores)
    candidates = sorted(set(scores)) + [math.inf]

    best_score = math.inf
    best_value = math.inf

    for threshold in candidates:
        cost = expected_cost(
            y,
            scores,
            threshold,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )

        if cost < best_score or (
            cost == best_score and threshold > best_value
        ):
            best_score = cost
            best_value = threshold
    return float(best_value)


def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Find the lowest score threshold that meets the precision requirement."""
    y, scores = validate(y_true, scores)

    for threshold in sorted(set(scores)):
        precision, _ = precision_recall_at(y, scores, threshold)

        if precision >= min_precision:
            return float(threshold)
    return None
