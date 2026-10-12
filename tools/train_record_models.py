import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.features.record_aggregator import META_COLS
from train.classical.train_baselines import bootstrap_auc, evaluate, make_models, youden_threshold

DATA = Path("dataqml/median/record_dataset.csv")
OUT_TAB = Path("results/classical")
DEPLOY = Path("deploy/artifacts")
SEED = 42
N_SELECT = 12
CORR_LIMIT = 0.9
VAL_FOLD, TEST_FOLD = 9, 10

def split(df: pd.DataFrame):
    tr = df[~df["strat_fold"].isin([VAL_FOLD, TEST_FOLD])]
    return tr, df[df["strat_fold"] == VAL_FOLD], df[df["strat_fold"] == TEST_FOLD]

def select_features(tr: pd.DataFrame, cols: list[str]) -> list[str]:
    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED
    ).fit(tr[cols], tr["label"])
    ranked = [c for _, c in sorted(zip(rf.feature_importances_, cols), reverse=True)]
    corr = tr[cols].corr().abs()
    chosen = []
    for c in ranked:
        if all(corr.loc[c, k] <= CORR_LIMIT for k in chosen):
            chosen.append(c)
        if len(chosen) == N_SELECT:
            break
    return chosen

def run(feature_set: str, cols: list[str], tr, va, te, preds: list) -> list[dict]:
    rows = []
    y_va, y_te = va["label"].to_numpy(), te["label"].to_numpy()
    for name, model in make_models(SEED).items():
        model.fit(tr[cols].to_numpy(), tr["label"].to_numpy())
        p_va = model.predict_proba(va[cols].to_numpy())[:, 1]
        p_te = model.predict_proba(te[cols].to_numpy())[:, 1]
        thr = youden_threshold(y_va, p_va)
        m = evaluate(y_te, p_te, thr)
        lo, hi = bootstrap_auc(y_te, p_te)
        rows.append({"Feature set": feature_set, "Model": name, **m,
                     "AUC_lo": lo, "AUC_hi": hi, "Threshold": thr})
        preds.extend({"feature_set": feature_set, "model": name, "record_id": r,
                      "label": y, "p": p, "threshold": thr}
                     for r, y, p in zip(te["record_id"], y_te, p_te))
    return rows

def main():
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, dtype={"record_id": str})
    all_cols = [c for c in df.columns if c not in ["record_id", *META_COLS]]
    tr, va, te = split(df)
    for name, part in (("train", tr), ("val", va), ("test", te)):
        print(f"{name:5s}: {len(part):5d} records, abnormality rate {part['label'].mean():.1%}")
    selected = select_features(tr, all_cols)
    print(f"\n{N_SELECT} features selected for Quantum Circuit: {selected}")
    DEPLOY.mkdir(parents=True, exist_ok=True)
    (DEPLOY / "selected_features.json").write_text(json.dumps(selected, indent=4), encoding="utf-8")
    preds: list = []
    rows = run(f"Full ({len(all_cols)})", all_cols, tr, va, te, preds)
    rows += run(f"Feature selection ({N_SELECT})", selected, tr, va, te, preds)
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))
    res.to_csv(OUT_TAB / "record_models.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT_TAB / "record_test_predictions.csv", index=False)

if __name__ == "__main__":
    main()