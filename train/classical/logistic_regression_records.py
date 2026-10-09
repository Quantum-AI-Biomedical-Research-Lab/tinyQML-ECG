import os
import argparse
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from src.models.baseline_core import (
    load_npz_dataset,
    compute_clinical_metrics,
    find_optimal_threshold,
    export_baseline_artifacts
)

def run_logistic_regression_pipeline(
    data_dir: str = "dataqml/scaled",
    output_dir: str = "results/classical",
    deploy_dir: str = "deploy/artifacts",
    seed: int = 42,
    n_splits: int = 5,
    n_jobs: int = -1
):
    print("[1/5] Loading npz dataset...")
    X_train, y_train, train_record_ids, _ = load_npz_dataset(os.path.join(data_dir, "train.npz"))
    X_val, y_val, _, _ = load_npz_dataset(os.path.join(data_dir, "val.npz"))
    X_test, y_test, _, _ = load_npz_dataset(os.path.join(data_dir, "test.npz"))
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val  : X={X_val.shape}, y={y_val.shape}")
    print(f"  Test : X={X_test.shape}, y={y_test.shape}")
    print(f"  Train records: {len(np.unique(train_record_ids))}")
    print(
        f"  Train class distribution: "
        f"{dict(zip(*np.unique(y_train, return_counts=True)))}"
    )
    unique_records = np.unique(train_record_ids)
    if len(unique_records) < n_splits:
        raise ValueError(
            f"Just has {len(unique_records)} records in train, "
            f"not enough for {n_splits}-fold CV"
        )
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed,)
    print("[2/5] Building Pipeline and running GridSearch...")
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="saga", max_iter=3000, random_state=seed, class_weight="balanced"))
    ])
    C_values = np.logspace(-3, 2, 6)
    param_grid = [
        {
            "poly__degree": [1, 2],
            "clf__penalty": ["l1"],
            "clf__C": C_values,
        },
        {
            "poly__degree": [1, 2],
            "clf__penalty": ["l2"],
            "clf__C": C_values,
        },
        {
            "poly__degree": [1, 2],
            "clf__penalty": ["elasticnet"],
            "clf__C": C_values,
            "clf__l1_ratio": [0.2, 0.5, 0.8],
        },
    ]
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv, 
        scoring="f1",
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
        error_score="raise",
        return_train_score=True
    )
    grid_search.fit(X_train, y_train, groups=train_record_ids)
    best_model = grid_search.best_estimator_
    print(f"Best params:")
    for key, value in grid_search.best_params_.items():
        print(f"{key}: {value}")
    print(f"F1-Score (CV Train): {grid_search.best_score_:.4f}")
    print("[3/5] Optimizing threshold on validation set...")
    val_probs = best_model.predict_proba(X_val)[:, 1]
    opt_threshold, val_f1 = find_optimal_threshold(y_val, val_probs, metric="f1")
    val_metrics = compute_clinical_metrics(y_val, val_probs, threshold=opt_threshold)
    print(f"Optimal threshold: {opt_threshold:.4f}")
    print(f"Validation F1: {val_f1:.4f}")
    print(f"Validation sensitivity: {val_metrics['sensitivity_se']:.4f}")
    print(f"Validation specificity: {val_metrics['specificity_sp']:.4f}")
    print("[4/5] Evaluating on the test set...")
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_clinical_metrics(y_test, test_probs, threshold=opt_threshold)
    print("[SUMMARY - LOGISTIC REGRESSION]")
    print(f"Threshold: {opt_threshold:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"F1-score: {test_metrics['f1_score']:.4f}")
    print(f"Sensitivity / Recall: {test_metrics['sensitivity_se']:.4f}")
    print(f"Specificity: {test_metrics['specificity_sp']:.4f}")
    print(f"Precision / PPV: {test_metrics['ppv_precision']:.4f}")
    print(f"NPV: {test_metrics['npv']:.4f}")
    print(f"ROC-AUC: {test_metrics['auc_roc']:.4f}")
    print(f"Average Precision: {test_metrics['average_precision']:.4f}")
    print(f"MCC: {test_metrics['mcc']:.4f}")
    print(
        f"TP={test_metrics['tp']}  "
        f"FP={test_metrics['fp']}  "
        f"TN={test_metrics['tn']}  "
        f"FN={test_metrics['fn']}"
    )
    print("[5/5] Exporting model and reports...")
    poly = best_model.named_steps["poly"]
    scaler = best_model.named_steps["scaler"]
    clf = best_model.named_steps["clf"]
    extra_config = {
        "best_params": grid_search.best_params_,
        "best_group_cv_f1": float(grid_search.best_score_),
        "validation_metrics": val_metrics,
        "n_input_features": int(X_train.shape[1]),
        "n_polynomial_features": int(poly.n_output_features_),
        "polynomial_degree": int(poly.degree),
        "polynomial_feature_names": poly.get_feature_names_out().tolist(),
        "polynomial_scaler_mean": scaler.mean_.tolist(),
        "polynomial_scaler_scale": scaler.scale_.tolist(),
        "coefficients_after_polynomial_scaling": clf.coef_[0].tolist(),
        "intercept_after_polynomial_scaling": float(clf.intercept_[0]),
        "class_labels": clf.classes_.tolist(),
        "cv_strategy": "StratifiedGroupKFold",
        "cv_group": "record_id",
        "threshold_selection": "validation_f1",
    }
    export_baseline_artifacts(
        model=best_model,
        model_name="logistic_regression",
        optimal_threshold=opt_threshold,
        test_metrics=test_metrics,
        output_dir=output_dir,
        deploy_dir=deploy_dir,
        extra_config=extra_config,
    )
    print("\nTraining pipeline completed successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataqml/scaled")
    parser.add_argument("--output_dir", type=str, default="results/classical")
    parser.add_argument("--deploy_dir", type=str, default="deploy/artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()
    run_logistic_regression_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        deploy_dir=args.deploy_dir,
        seed=args.seed,
        n_splits=args.n_splits,
        n_jobs=args.n_jobs
    )

if __name__ == "__main__":
    main()