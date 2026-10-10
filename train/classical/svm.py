
import os
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import precision_recall_curve
from src.models.baseline_core import (
    load_npz_dataset,
    compute_clinical_metrics,
    find_optimal_threshold,
    export_baseline_artifacts
)

def find_best_svm_threshold(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(y_true)) != 2:
        raise ValueError("Validation set must contain both classes 0 and 1")
    if not np.all(np.isfinite(scores)):
        raise ValueError("SVM decision scores contain NaN or infinity")
    precision, recall, thresholds = precision_recall_curve(
        y_true, scores, pos_label=1
    )
    p = precision[:-1]
    r = recall[:-1]
    f1 = np.divide(
        2 * p * r,
        p + r,
        out=np.zeros_like(p),
        where=(p + r) > 0,
    )
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])

def check_dataset(name, X, y, record_ids):
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    record_ids = np.asarray(record_ids)
    if X.ndim != 2:
        raise ValueError(f"{name}: X must be a two-dimensional matrix")
    if len(X) != len(y) or len(y) != len(record_ids):
        raise ValueError(f"{name}: X, y, and record_ids must contain the same number of samples")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name}: X contains NaN or infinity")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{name}: Labels must be encoded as 0 and 1")
    if len(np.unique(y)) < 2:
        raise ValueError(f"{name}: Both Normal and Abnormal classes are required")
    print(
        f"{name:>5}: n={len(y):5d}, "
        f"features={X.shape[1]}, "
        f"positive_rate={np.mean(y):.2%}, "
        f"records={len(np.unique(record_ids))}"
    )

def verify_record_split(train_ids, val_ids, test_ids):
    train_set = set(np.asarray(train_ids).tolist())
    val_set = set(np.asarray(val_ids).tolist())
    test_set = set(np.asarray(test_ids).tolist())
    if train_set & val_set:
        raise ValueError("Data leakage detected: record IDs overlap between train and validation")
    if train_set & test_set:
        raise ValueError("Data leakage detected: record IDs overlap between train and test")
    if val_set & test_set:
        raise ValueError("Data leakage detected: record IDs overlap between validation and test")

def print_metrics(metrics):
    keys = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("f1_score", "F1-score"),
        ("sensitivity_se", "Sensitivity / Recall"),
        ("specificity_sp", "Specificity"),
        ("ppv_precision", "Precision / PPV"),
        ("npv", "NPV"),
        ("auc_roc", "ROC-AUC"),
        ("average_precision", "Average Precision"),
        ("mcc", "MCC"),
    ]
    for key, label in keys:
        if key in metrics:
            value = metrics[key]
            if value is not None and np.isfinite(value):
                print(f"  {label:<25}: {value:.4f}")
            else:
                print(f"  {label:<25}: N/A")
    print(
        "  Confusion matrix counts : "
        f"TP={metrics['tp']}, FP={metrics['fp']}, "
        f"TN={metrics['tn']}, FN={metrics['fn']}"
    )

def run_svm_pipeline(
    data_dir="dataqml/scaled",
    output_dir="results/classical",
    deploy_dir="deploy/artifacts",
    seed=42,
    n_splits=5,
):
    start_time = time.perf_counter()
    print("=" * 72)
    print("ECG SVM TRAINING PIPELINE")
    print("=" * 72)
    print("\n[1/6] Loading train / validation / test...")
    X_train, y_train, rec_train, _ = load_npz_dataset(os.path.join(data_dir, "train.npz"))
    X_val, y_val, rec_val, _ = load_npz_dataset(os.path.join(data_dir, "val.npz"))
    X_test, y_test, rec_test, _ = load_npz_dataset(os.path.join(data_dir, "test.npz"))
    check_dataset("Train", X_train, y_train, rec_train)
    check_dataset("Val", X_val, y_val, rec_val)
    check_dataset("Test", X_test, y_test, rec_test)
    if not (
        X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
    ):
        raise ValueError("The number of features differs between dataset splits")
    verify_record_split(rec_train, rec_val, rec_test)
    print("[2/6] Building the SVM pipeline...")
    pipeline = Pipeline([
        ("transform", "passthrough"),
        ("scaler", StandardScaler()),
        ("clf", SVC(
            class_weight="balanced",
            probability=False,
            cache_size=1000,
            random_state=seed,
        )),
    ])
    print("[3/6] Running grouped GridSearchCV...")
    power_transformer = PowerTransformer(
        method="yeo-johnson",
        standardize=False,
    )
    transforms = ["passthrough", power_transformer]
    param_grid = [
        {
            "transform": transforms,
            "clf__kernel": ["linear"],
            "clf__C": [0.1, 1.0, 10.0, 100.0],
        },
        {
            "transform": transforms,
            "clf__kernel": ["rbf"],
            "clf__C": [0.1, 1.0, 10.0, 100.0],
            "clf__gamma": ["scale", 0.01, 0.1, 1.0],
        },
        {
            "transform": transforms,
            "clf__kernel": ["poly"],
            "clf__degree": [2, 3],
            "clf__C": [0.1, 1.0, 10.0],
            "clf__gamma": ["scale", 0.1],
            "clf__coef0": [0.0, 1.0],
        },
    ]
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        error_score="raise",
        return_train_score=True,
    )
    grid_search.fit(
        X_train,
        y_train,
        groups=rec_train,
    )
    best_model = grid_search.best_estimator_
    print("\nBest parameters:")
    print(json.dumps(grid_search.best_params_, indent=2, default=str))
    print(f"Best grouped-CV F1: {grid_search.best_score_:.4f}")
    print("[4/6] Selecting decision threshold on validation...")
    val_scores = best_model.decision_function(X_val)
    threshold, val_f1 = find_best_svm_threshold(
        y_val,
        val_scores,
    )
    val_metrics = compute_clinical_metrics(
        y_val,
        val_scores,
        threshold=threshold,
    )
    print(f"Validation decision threshold: {threshold:.6f}")
    print(f"Validation F1 at threshold   : {val_f1:.4f}")
    print("\nValidation metrics:")
    print_metrics(val_metrics)
    print("\n[5/6] Evaluating independent test set...")
    test_scores = best_model.decision_function(X_test)
    test_metrics = compute_clinical_metrics(
        y_test,
        test_scores,
        threshold=threshold,
    )
    print("\nTEST METRICS")
    print("-" * 55)
    print_metrics(test_metrics)
    print("\n[6/6] Exporting model and artifacts...")
    svm_clf = best_model.named_steps["clf"]
    extra_config = {
        "model_type": "SVC",
        "score_type": "decision_function",
        "threshold_type": "decision_score_threshold",
        "optimal_threshold": float(threshold),
        "validation_f1_at_threshold": float(val_f1),
        "best_cv_f1": float(grid_search.best_score_),
        "best_params": {
            k: str(v) for k, v in grid_search.best_params_.items()
        },
        "kernel": str(svm_clf.kernel),
        "C": float(svm_clf.C),
        "gamma": (
            float(svm_clf.gamma)
            if isinstance(svm_clf.gamma, (int, float))
            else str(svm_clf.gamma)
        ),
        "n_support_vectors_per_class": svm_clf.n_support_.tolist(),
        "total_support_vectors": int(np.sum(svm_clf.n_support_)),
        "support_vectors_shape": list(svm_clf.support_vectors_.shape),
        "dual_coef_shape": list(svm_clf.dual_coef_.shape),
        "intercept": svm_clf.intercept_.tolist(),
        "input_feature_count": int(X_train.shape[1]),
        "threshold_selection_metric": "F1 on validation",
        "cv_strategy": "StratifiedGroupKFold",
        "cv_splits": int(n_splits),
        "seed": int(seed),
    }
    export_baseline_artifacts(
        model=best_model,
        model_name="svm",
        optimal_threshold=threshold,
        test_metrics=test_metrics,
        output_dir=output_dir,
        deploy_dir=deploy_dir,
        extra_config=extra_config,
    )
    os.makedirs(output_dir, exist_ok=True)
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(
        os.path.join(output_dir, "svm_report.csv"),
        index=False,
    )
    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 72)
    print("SVM TRAINING COMPLETED")
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print(f"Model: {output_dir}/svm.pkl")
    print(f"Config: {deploy_dir}/svm_config.json")
    print(f"Grid results: {output_dir}/svm_report.csv")
    print("=" * 72)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataqml/scaled")
    parser.add_argument("--output_dir", default="results/classical")
    parser.add_argument("--deploy_dir", default="deploy/artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    run_svm_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        deploy_dir=args.deploy_dir,
        seed=args.seed,
        n_splits=args.n_splits,
    )