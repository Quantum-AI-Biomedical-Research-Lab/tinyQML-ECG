import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

def load_npz_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"baseline_core.py: cannot found {path}")
    data = np.load(path, allow_pickle=True)
    return data["X_baseline"], data["y"], data["record_ids"], data["beat_indices"]

def compute_clinical_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Sensitivity
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # Specificity
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "sensitivity_se": float(se),
        "specificity_sp": float(sp),
        "ppv_precision": float(ppv),
        "npv": float(npv),
        "f1_score": float(f1),
        "auc_roc": float(auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 100)
    best_thresh = 0.5
    best_score = -1.0
    for t in thresholds:
        metrics = compute_clinical_metrics(y_true, y_prob, threshold=t)
        score = metrics["f1_score"] if metric == "f1" else metrics["sensitivity_se"]
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score

def export_baseline_artifacts(
    model: Any,
    model_name: str,
    optimal_threshold: float,
    test_metrics: Dict[str, float],
    output_dir: str = "results/classical",
    deploy_dir: str = "deploy/artifacts",
    extra_config: Dict[str, Any] = None
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(deploy_dir, exist_ok=True)
    model_pkl_path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(model, model_pkl_path)
    print(f"Saved {model_name}.pkl at {model_pkl_path}")
    deploy_config = {
        "model_name": model_name,
        "optimal_threshold": float(optimal_threshold),
        "test_metrics": test_metrics
    }
    if extra_config:
        deploy_config.update(extra_config)
    deploy_json_path = os.path.join(deploy_dir, f"{model_name}_config.json")
    with open(deploy_json_path, "w", encoding="utf-8") as f:
        json.dump(deploy_config, f, indent=4, ensure_ascii=False)
    print(f"Saved {model_name}_config.json at {deploy_json_path}")
    report_row = {
        "model": model_name,
        "optimal_threshold": optimal_threshold,
        "test_accuracy": test_metrics["accuracy"],
        "test_f1_score": test_metrics["f1_score"],
        "test_sensitivity_se": test_metrics["sensitivity_se"],
        "test_specificity_sp": test_metrics["specificity_sp"],
        "test_ppv": test_metrics["ppv_precision"],
        "test_auc": test_metrics["auc_roc"]
    }
    df_report = pd.DataFrame([report_row])
    csv_path = os.path.join(output_dir, f"{model_name}_report.csv")
    df_report.to_csv(csv_path, index=False)
    print(f"Saved {model_name}_report.csv at {csv_path}")