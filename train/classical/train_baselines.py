from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix, f1_score,
    roc_auc_score, roc_curve
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA = Path("dataqml/scaled")
OUT_TAB = Path("results/classical")
OUT_FIG = Path("results/classical/figures")
SEED = 42
N_BOOT = 1000
FEATURES = ["q_amplitude", "r_amplitude", "s_amplitude", "rr_interval_ms", "qrs_duration_ms"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "black",
})

def load_split(name: str) -> pd.DataFrame:
    d = np.load(DATA / f"{name}.npz", allow_pickle=True)
    df = pd.DataFrame(d["X_baseline"], columns=FEATURES)
    df["record_id"] = d["record_ids"].astype(str)
    df["beat_index"] = d["beat_indices"]
    df["label"] = d["y"]
    return df.sort_values(["record_id", "beat_index"]).reset_index(drop=True)

def rmssd(s: pd.Series) -> float:
    v = s.to_numpy()
    return float(np.sqrt(np.mean(np.diff(v) ** 2))) if len(v) > 1 else 0.0

def aggregate_records(df: pd.DataFrame):
    g = df.groupby("record_id")
    agg = g[FEATURES].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{f}_{s}" for f, s in agg.columns]
    agg["rr_rmssd"] = g["rr_interval_ms"].apply(rmssd)
    agg["n_beats"] = g.size()
    return agg.fillna(0.0), g["label"].first()

def make_models(seed: int) -> dict:
    return {
        "Majority (baseline)": DummyClassifier(strategy="prior"),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "RBF SVM": make_pipeline(
            StandardScaler(), SVC(C=1.0, gamma="scale", class_weight="balanced",
                                  probability=True, random_state=seed)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed),
    }

def run_beat_level(model, tr, va, te):
    model.fit(tr[FEATURES].to_numpy(), tr["label"].to_numpy())
    def record_prob(df):
        p = model.predict_proba(df[FEATURES].to_numpy())[:, 1]
        return pd.Series(p, index=df["record_id"].to_numpy()).groupby(level=0).mean()
    return record_prob(va), record_prob(te)

def run_record_level(model, tr, va, te):
    X_tr, y_tr = aggregate_records(tr)
    model.fit(X_tr.to_numpy(), y_tr.to_numpy())
    out = []
    for df in (va, te):
        X, _ = aggregate_records(df)
        out.append(pd.Series(model.predict_proba(X.to_numpy())[:, 1], index=X.index))
    return out

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

def bootstrap_auc(y: np.ndarray, p: np.ndarray, n: int = N_BOOT, seed: int = SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])

def plot_roc(roc_data: dict):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, linewidth=1.2, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "baselines_roc_test.png", dpi=300)

def main():
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    tr, va, te = (load_split(s) for s in ("train", "val", "test"))
    _, y_va = aggregate_records(va)
    _, y_te = aggregate_records(te)
    for name, df in (("train", tr), ("val", va), ("test", te)):
        _, y = aggregate_records(df)
        print(f"{name:5s}: {len(y):4d} records, {len(df):6d} beats, abnormality rate {y.mean():.1%}")

    rows, roc_data = [], {}
    settings = (("Beat-Level", run_beat_level), ("Record-Level", run_record_level))
    for setting, runner in settings:
        for name, model in make_models(SEED).items():
            p_va, p_te = runner(model, tr, va, te)
            p_va = p_va.loc[y_va.index].to_numpy()
            p_te = p_te.loc[y_te.index].to_numpy()
            thr = youden_threshold(y_va.to_numpy(), p_va)
            metrics = evaluate(y_te.to_numpy(), p_te, thr)
            lo, hi = bootstrap_auc(y_te.to_numpy(), p_te)
            rows.append({"Learning Strategy": setting, "Model": name, **metrics,
                         "AUC_lo": lo, "AUC_hi": hi, "Threshold": thr})
            if setting == "Record-Level":
                fpr, tpr, _ = roc_curve(y_te.to_numpy(), p_te)
                roc_data[name] = (fpr, tpr, metrics["AUC"])

    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    res.to_csv(OUT_TAB / "baselines_record_level.csv", index=False)
    plot_roc(roc_data)
    plt.show()

if __name__ == "__main__":
    main()