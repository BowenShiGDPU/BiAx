from __future__ import annotations
import math
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
METRICS = ('precision', 'recall', 'f1_score', 'auroc', 'aupr', 'mcc')

def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_prob) or len(y_true) == 0:
        raise ValueError('nonempty aligned metric arrays required')
    if not (np.isfinite(y_prob).all() and np.all((0.0 <= y_prob) & (y_prob <= 1.0))):
        raise ValueError('probability range violation')
    y_pred = (y_prob >= float(threshold)).astype(np.int8)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0
    return {'precision': float(precision_score(y_true, y_pred, zero_division=0)), 'recall': float(recall_score(y_true, y_pred, zero_division=0)), 'f1_score': float(f1_score(y_true, y_pred, zero_division=0)), 'auroc': float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size == 2 else float('nan'), 'aupr': float(average_precision_score(y_true, y_prob)) if np.any(y_true == 1) else float('nan'), 'mcc': float(mcc)}

def pooled_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int8).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    threshold = np.asarray(threshold, dtype=np.float64).reshape(-1)
    if not len(y_true) == len(y_prob) == len(threshold) or len(y_true) == 0:
        raise ValueError('nonempty aligned metric arrays required')
    if not (np.isfinite(y_prob).all() and np.all((0.0 <= y_prob) & (y_prob <= 1.0))):
        raise ValueError('probability range violation')
    y_pred = (y_prob >= threshold).astype(np.int8)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0
    return {'precision': float(precision_score(y_true, y_pred, zero_division=0)), 'recall': float(recall_score(y_true, y_pred, zero_division=0)), 'f1_score': float(f1_score(y_true, y_pred, zero_division=0)), 'auroc': float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size == 2 else float('nan'), 'aupr': float(average_precision_score(y_true, y_prob)) if np.any(y_true == 1) else float('nan'), 'mcc': float(mcc)}

def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.int8).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_prob) or len(y_true) == 0:
        raise ValueError('threshold arrays')
    if not np.isfinite(y_prob).all():
        raise ValueError('non-finite threshold score')
    candidates = np.unique(np.concatenate(([0.0], y_prob, [float(np.nextafter(y_prob.max(), math.inf))], [1.0])))
    candidates = candidates[(candidates >= 0.0) & (candidates <= float(np.nextafter(1.0, math.inf)))]
    best = None
    for threshold in candidates:
        metrics = binary_metrics(y_true, y_prob, float(threshold))
        key = (metrics['mcc'], metrics['f1_score'], metrics['recall'], -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    assert best is not None
    return (best[1], best[2])

def choose_threshold_robust(y_true: np.ndarray, y_prob: np.ndarray, tol: float=0.01) -> tuple[float, dict[str, float]]:
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    p = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError('threshold arrays')
    if not np.isfinite(p).all():
        raise ValueError('non-finite threshold score')
    order = np.argsort(-p, kind='mergesort')
    ys = y[order].astype(np.float64)
    ps = p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    pos, neg = (ys.sum(), len(ys) - ys.sum())
    fn, tn = (pos - tp, neg - fp)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)
    best = float(mcc.max())
    plateau = np.flatnonzero(mcc >= best - float(tol))
    threshold = float(np.median(ps[plateau]))
    return (threshold, binary_metrics(y, p, threshold))

def holm(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    m = len(values)
    order = np.argsort(values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted
