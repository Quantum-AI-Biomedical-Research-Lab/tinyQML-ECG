import os
import argparse
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
    seed: int = 42
):
    print("[1/4] Loading npz dataset...")
    X_train, y_train, _, _ = load_npz_dataset(os.path.join(data_dir, "train.npz"))
    X_val, y_val, _, _ = load_npz_dataset(os.path.join(data_dir, "val.npz"))
    X_test, y_test, _, _ = load_npz_dataset(os.path.join(data_dir, "test.npz"))
    print("[2/4] Building pipeline and conveying GridSearch...")
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="saga", max_iter=3000, random_state=seed, class_weight="balanced"))
    ])
    param_grid = [
        {
            "poly__degree": [1, 2],
            "clf__penalty": ["l1", "l2"],
            "clf__C": np.logspace(-3, 2, 6)
        },
        {
            "poly__degree": [1, 2],
            "clf__penalty": ["elasticnet"],
            "clf__C": np.logspace(-3, 2, 6),
            "clf__l1_ratio": [0.2, 0.5, 0.8]
        }
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best params: {grid_search.best_params_}")
    print(f"F1-Score (CV Train): {grid_search.best_score_:.4f}")
    print("[3/4] Optimizing threshold on validation set...")
    val_probs = best_model.predict_proba(X_val)[:, 1]
    opt_thresh, val_f1 = find_optimal_threshold(y_val, val_probs, metric="f1")
    print(f"Threshold optimization: {opt_thresh:.2f} (Val F1: {val_f1:.4f})")
    print("[4/4] Evaluating on the test set...")
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_clinical_metrics(y_test, test_probs, threshold=opt_thresh)
    print("[SUMMARY - LOGISTIC REGRESSION]")
    print(f"Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"Sensitivity: {test_metrics['sensitivity_se']*100:.2f}%")
    print(f"Specificity: {test_metrics['specificity_sp']*100:.2f}%")
    print(f"ROC-AUC: {test_metrics['auc_roc']:.4f}")
    lr_clf = best_model.named_steps["clf"]
    extra_lr_config = {
        "poly_degree": int(grid_search.best_params_["poly__degree"]),
        "coefficients": lr_clf.coef_.tolist()[0],
        "intercept": float(lr_clf.intercept_[0])
    }
    export_baseline_artifacts(
        model=best_model,
        model_name="logistic_regression",
        optimal_threshold=opt_thresh,
        test_metrics=test_metrics,
        output_dir=output_dir,
        deploy_dir=deploy_dir,
        extra_config=extra_lr_config
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataqml/scaled")
    parser.add_argument("--output_dir", type=str, default="results/classical")
    parser.add_argument("--deploy_dir", type=str, default="deploy/artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_logistic_regression_pipeline(data_dir=args.data_dir, output_dir=args.output_dir, deploy_dir=args.deploy_dir, seed=args.seed)

if __name__ == "__main__":
    main()