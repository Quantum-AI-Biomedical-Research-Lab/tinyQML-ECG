import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef
)

def load_npz_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"baseline_core.py: cannot found {path}")
    with np.load(path, allow_pickle=True) as data:
        required_keys = {
            "X_baseline",
            "y",
            "record_ids",
            "beat_indices"
        }
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            raise KeyError(f"{path} do not have keys: {sorted(missing_keys)}")
        X = np.asarray(data["X_baseline"], dtype=np.float64)
        y = np.asarray(data["y"]).reshape(-1).astype(int)
        record_ids = np.asarray(data["record_ids"]).reshape(-1)
        beat_indices = np.asarray(data["beat_indices"]).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D matrix, shape={X.shape}")
    n_samples = len(y)
    if X.shape[0] != n_samples:
        raise ValueError("Number of sample of X and y are different")
    if len(record_ids) != n_samples:
        raise ValueError("Number of record_ids and samples are different")
    if len(beat_indices) != n_samples:
        raise ValueError("Number of beat_indices and samples are different")
    if not np.isfinite(X).all():
        raise ValueError(f"Dataset X contains NaN or Inf: {path}")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must be binary label 0/1")
    return X, y, record_ids, beat_indices

def compute_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob do not have the same samples")
    if len(y_true) == 0:
        raise ValueError("y_true is empty")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains NaN or Inf")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must be binary label 0/1")
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = float("nan")
    average_precision = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
            average_precision = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "sensitivity_se": float(sensitivity),
        "specificity_sp": float(specificity),
        "ppv_precision": float(precision),
        "npv": float(npv),
        "f1_score": float(f1),
        "auc_roc": float(auc),
        "average_precision": float(average_precision),
        "balanced_accuracy": float(balanced_acc),
        "mcc": float(mcc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, float]:
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if len(y_true) != len(y_prob) or len(y_true) == 0:
        raise ValueError("Validation labels/probabilities error")
    if not np.isfinite(y_prob).all():
        raise ValueError("Validation probabilities contain NaN or Inf")
    thresholds = np.unique(np.concatenate([np.array([0.0, 0.5, 1.0]), y_prob]))
    best_thresh = 0.5
    best_score = -1.0
    for threshold in thresholds:
        metrics = compute_clinical_metrics(y_true, y_prob, threshold=float(threshold))
        score = metrics["f1_score"] if metric == "f1" else metrics["sensitivity_se"]
        if score > best_score:
            best_score = score
            best_thresh = threshold
    return best_thresh, best_score

def export_baseline_artifacts(
    model: Any,
    model_name: str,
    optimal_threshold: float,
    test_metrics: Dict[str, float],
    output_dir: str = "results/classical",
    deploy_dir: str = "deploy/artifacts",
    extra_config: Dict[str, Any] = None
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(deploy_dir, exist_ok=True)
    model_pkl_path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(model, model_pkl_path)
    print(f"Saved {model_name}.pkl at {model_pkl_path}")
    deploy_config = {
        "model_name": model_name,
        "optimal_threshold": float(optimal_threshold),
        "positive_class": 1,
        "negative_class": 0,
        "test_metrics": test_metrics
    }
    if extra_config:
        deploy_config.update(extra_config)
    deploy_json_path = os.path.join(deploy_dir, f"{model_name}_config.json")
    def sanitize_json(value):
        if isinstance(value, dict):
            return {str(k): sanitize_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize_json(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            value = float(value)
            return value if np.isfinite(value) else None
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value
    with open(deploy_json_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_json(deploy_config), f, indent=4, ensure_ascii=False, allow_nan=False)
    print(f"Saved {model_name}_config.json at {deploy_json_path}")
    report_row = {
        "model": model_name,
        "optimal_threshold": float(optimal_threshold),
        **{
            f"test_{key}": value
            for key, value in test_metrics.items()
        },
    }
    df_report = pd.DataFrame([report_row])
    csv_path = os.path.join(output_dir, f"{model_name}_report.csv")
    df_report.to_csv(csv_path, index=False)
    print(f"Saved {model_name}_report.csv at {csv_path}")