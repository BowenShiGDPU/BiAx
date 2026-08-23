from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)


def validation_threshold(y_true, y_prob) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(y_prob, dtype=np.float64)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys, dtype=np.int64)
    fp = np.cumsum(1 - ys, dtype=np.int64)
    fn = int(tp[-1]) - tp
    tn = int(fp[-1]) - fp
    f1_denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, f1_denom, out=np.zeros_like(tp, dtype=float), where=f1_denom > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    tp_f, fp_f, fn_f, tn_f = (value.astype(np.float64) for value in (tp, fp, fn, tn))
    denominator = np.sqrt((tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f))
    mcc = np.divide(
        tp_f * tn_f - fp_f * fn_f, denominator,
        out=np.zeros_like(tp_f), where=denominator > 0,
    )
    tied_end = np.r_[ps[:-1] != ps[1:], True]
    included = np.flatnonzero(tied_end)
    thresholds = ps[included]
    candidate_mcc, candidate_f1, candidate_recall = mcc[included], f1[included], recall[included]
    for threshold in (float(np.nextafter(p.max(), math.inf)), 1.0):
        if threshold <= float(np.nextafter(1.0, math.inf)) and threshold > p.max():
            thresholds = np.append(thresholds, threshold)
            candidate_mcc = np.append(candidate_mcc, 0.0)
            candidate_f1 = np.append(candidate_f1, 0.0)
            candidate_recall = np.append(candidate_recall, 0.0)
    if p.min() > 0.0:
        thresholds = np.append(thresholds, 0.0)
        candidate_mcc = np.append(candidate_mcc, 0.0)
        candidate_f1 = np.append(candidate_f1, 2 * int(tp[-1]) / (2 * int(tp[-1]) + int(fp[-1])))
        candidate_recall = np.append(candidate_recall, 1.0)
    ranking = np.lexsort((-thresholds, candidate_recall, candidate_f1, candidate_mcc))
    return float(thresholds[ranking[-1]])


def binary_metrics(y_true, y_prob, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(y_prob, dtype=np.float64)
    pred = (p >= threshold).astype(np.int8)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, p)),
        "aupr": float(average_precision_score(y, p)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "threshold": float(threshold), "n": int(y.size), "positives": int(y.sum()),
    }
