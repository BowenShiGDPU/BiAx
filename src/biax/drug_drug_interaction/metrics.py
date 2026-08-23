from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRICS = ("precision", "recall", "f1_score", "auroc", "aupr", "mcc")


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    if y_true.ndim != 1 or probability.ndim != 1 or len(y_true) != len(probability):
        raise ValueError("y_true and probability must be aligned one-dimensional arrays")
    if np.unique(y_true).size != 2:
        raise ValueError("threshold selection requires both binary classes")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")

    # This is the evaluation rule in O(n log n): max MCC, then F1,
    # recall and lower threshold.  Each tied score is evaluated as one block.
    order = np.argsort(-probability, kind="mergesort")
    sorted_label = y_true[order]
    sorted_probability = probability[order]
    true_positive = np.cumsum(sorted_label, dtype=np.int64)
    false_positive = np.cumsum(1 - sorted_label, dtype=np.int64)
    total_positive = int(true_positive[-1])
    total_negative = int(false_positive[-1])
    false_negative = total_positive - true_positive
    true_negative = total_negative - false_positive
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(2 * true_positive, f1_denominator,
                   out=np.zeros_like(true_positive, dtype=float), where=f1_denominator > 0)
    recall = np.divide(true_positive, true_positive + false_negative,
                       out=np.zeros_like(true_positive, dtype=float),
                       where=(true_positive + false_negative) > 0)
    mcc_denominator = np.sqrt(
        (true_positive + false_positive).astype(float)
        * (true_positive + false_negative).astype(float)
        * (true_negative + false_positive).astype(float)
        * (true_negative + false_negative).astype(float)
    )
    mcc = np.divide(
        true_positive * true_negative - false_positive * false_negative,
        mcc_denominator,
        out=np.zeros_like(true_positive, dtype=float), where=mcc_denominator != 0,
    )
    tied_end = np.r_[sorted_probability[:-1] != sorted_probability[1:], True]
    included = np.flatnonzero(tied_end)
    thresholds = sorted_probability[included]
    candidate_mcc = mcc[included]
    candidate_f1 = f1[included]
    candidate_recall = recall[included]

    for threshold in (float(np.nextafter(probability.max(), math.inf)), 1.0):
        if threshold <= float(np.nextafter(1.0, math.inf)) and threshold > probability.max():
            thresholds = np.append(thresholds, threshold)
            candidate_mcc = np.append(candidate_mcc, 0.0)
            candidate_f1 = np.append(candidate_f1, 0.0)
            candidate_recall = np.append(candidate_recall, 0.0)
    if probability.min() > 0.0:
        thresholds = np.append(thresholds, 0.0)
        candidate_mcc = np.append(candidate_mcc, 0.0)
        candidate_f1 = np.append(
            candidate_f1, 2 * total_positive / (2 * total_positive + total_negative)
        )
        candidate_recall = np.append(candidate_recall, 1.0)
    best_index = np.lexsort(
        (-thresholds, candidate_recall, candidate_f1, candidate_mcc)
    )[-1]
    return float(thresholds[best_index])


def binary_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    prediction = probability >= threshold
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_score": float(f1_score(y_true, prediction, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, probability)),
        "aupr": float(average_precision_score(y_true, probability)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
    }


def multiclass_metrics(y_true: np.ndarray, probability: np.ndarray,
                       classes: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    probability = np.asarray(probability, dtype=float)
    prediction = classes[np.argmax(probability, axis=1)]
    indicator = np.equal.outer(y_true, classes).astype(np.int8)
    valid = indicator.sum(axis=0) > 0
    return {
        "n": int(len(y_true)),
        "classes_observed": int(np.unique(y_true).size),
        "precision": float(precision_score(
            y_true, prediction, labels=classes, average="macro", zero_division=0
        )),
        "recall": float(recall_score(
            y_true, prediction, labels=classes, average="macro", zero_division=0
        )),
        "f1_score": float(f1_score(
            y_true, prediction, labels=classes, average="macro", zero_division=0
        )),
        "auroc": float(roc_auc_score(indicator[:, valid], probability[:, valid], average="macro")),
        "aupr": float(average_precision_score(
            indicator[:, valid], probability[:, valid], average="macro"
        )),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
    }
