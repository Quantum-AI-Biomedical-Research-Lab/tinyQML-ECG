import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score, roc_curve
)

def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(min(thr[np.argmax(tpr - fpr)], 1.0))

def evaluate(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y, p),
        "Acc": (tp + tn) / len(y),
        "Se": tp / (tp + fn) if tp + fn else np.nan,
        "Sp": tn / (tn + fp) if tn + fp else np.nan,
        "F1": f1_score(y, yhat, zero_division=0),
        "BAcc": balanced_accuracy_score(y, yhat),
    }

def bootstrap_auc(y: np.ndarray, p: np.ndarray, n: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

def paired_bootstrap_delta_auc(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n: int = 2000,
    seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            deltas.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    d = np.asarray(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"dAUC": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "dAUC_lo": float(lo), "dAUC_hi": float(hi),
            "P_A_not_better": float(np.mean(d <= 0))}