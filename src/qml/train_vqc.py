import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.qml.metrics import bootstrap_auc, evaluate, paired_bootstrap_delta_auc, youden_threshold
from src.qml.angle_scaler import AngleScaler
from src.qml.reuploading import ReuploadingVQC

DATA = Path("dataqml/median/record_dataset.csv")
SELECTED = Path("deploy/artifacts/selected_features.json")
CLASSICAL_PREDS = Path("results/classical/record_test_predictions.csv")
OUT_TAB = Path("results/quantum")
DEPLOY = Path("deploy/artifacts")
VAL_FOLD, TEST_FOLD = 9, 10

VARIANTS = {
    "Full": dict(trainable_scaling=True, entangle=True, readout="all_z"),
    "No encoding weights": dict(trainable_scaling=False, entangle=True, readout="all_z"),
    "No entanglement": dict(trainable_scaling=True, entangle=False, readout="all_z"),
    "Qubit-0 readout": dict(trainable_scaling=True, entangle=True, readout="z0"),
}

class FourierFeatures(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        n_freq: int = 80, 
        max_freq: int = 4, 
        seed: int = 0
    ):
        self.n_freq = n_freq
        self.max_freq = max_freq
        self.seed = seed

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.seed)
        self.W_ = rng.integers(-self.max_freq, self.max_freq + 1, size=(self.n_freq, X.shape[1]))
        return self

    def transform(self, X):
        Z = np.asarray(X) @ self.W_.T
        return np.hstack([np.cos(Z), np.sin(Z)])

def parse_args():
    ap = argparse.ArgumentParser(description="Train the data re-uploading quantum classifier")
    ap.add_argument("--layers", type=str, default="1,2,3,4,5,6")
    ap.add_argument("--ablation-layers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--skip-ablations", action="store_true")
    args = ap.parse_args()
    args.layers = [int(v) for v in args.layers.split(",")]
    return args

def load_data():
    df = pd.read_csv(DATA, dtype={"record_id": str})
    feats = json.loads(SELECTED.read_text(encoding="utf-8"))
    parts = {
        "train": df[~df["strat_fold"].isin([VAL_FOLD, TEST_FOLD])],
        "val": df[df["strat_fold"] == VAL_FOLD],
        "test": df[df["strat_fold"] == TEST_FOLD],
    }
    scaler = AngleScaler.fit(parts["train"][feats].to_numpy())
    data = {k: (scaler.transform(v[feats].to_numpy()), v["label"].to_numpy().astype(int),
                v["record_id"].to_numpy()) for k, v in parts.items()}
    return data, feats, scaler

def predict(model: ReuploadingVQC, X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).cpu().double().numpy()

def train_vqc(cfg: dict, n_layers: int, seed: int, data: dict, args):
    torch.manual_seed(seed)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    X_tr, y_tr, _ = data["train"]
    X_va, y_va, _ = data["val"]
    model = ReuploadingVQC(n_features=X_tr.shape[1], n_layers=n_layers, dtype=dtype, seed=seed, **cfg)
    Xt, yt = torch.tensor(X_tr, dtype=dtype), torch.tensor(y_tr, dtype=dtype)
    Xv = torch.tensor(X_va, dtype=dtype)
    pos_weight = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), dtype=dtype)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gen = torch.Generator().manual_seed(seed)
    best_auc, best_state, best_epoch, wait = -1.0, None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        auc = roc_auc_score(y_va, predict(model, Xv))
        if auc > best_auc + 1e-4:
            best_auc, best_epoch, wait = auc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, best_auc, best_epoch

def score_row(name: str, n_layers, seed: int, p_va, p_te, data: dict, extra: dict) -> dict:
    y_va, y_te = data["val"][1], data["test"][1]
    thr = youden_threshold(y_va, p_va)
    lo, hi = bootstrap_auc(y_te, p_te)
    return {"Model": name, "L": n_layers, "Seed": seed, **evaluate(y_te, p_te, thr),
            "AUC_lo": lo, "AUC_hi": hi, "Threshold": thr, **extra}

def robustness(model: ReuploadingVQC, data: dict) -> pd.DataFrame:
    X_te, y_te, _ = data["test"]
    z = model.expectations(torch.tensor(X_te, dtype=model.rdtype)).detach().double().numpy()
    hw = model.head_w.detach().double().numpy()
    hb = float(model.head_b.detach())
    depth = 2 * model.n_layers + 1
    rng = np.random.default_rng(0)
    rows = []
    for depol in [0.0, 0.001, 0.005, 0.01, 0.02]:
        for shots in [None, 100, 1000, 8000]:
            zn = z * (1 - depol) ** depth
            if shots:
                zn = 1 - 2 * rng.binomial(shots, np.clip((1 - zn) / 2, 0, 1)) / shots
            p = 1 / (1 + np.exp(-(zn @ hw + hb)))
            rows.append({"Depolarizing per layer": depol, "Shots": shots or "exact",
                         "AUC": roc_auc_score(y_te, p)})
    return pd.DataFrame(rows)

def matched_baselines(n_params: int, n_layers: int, data: dict, seeds: int):
    X_tr, y_tr, _ = data["train"]
    X_va, X_te = data["val"][0], data["test"][0]
    n_feat = X_tr.shape[1]
    hidden = max(1, round((n_params - 1) / (n_feat + 2)))
    n_freq = max(1, (n_params - 1) // 2)
    rows, preds = [], {}
    for seed in range(seeds):
        models = {
            f"Param-matched MLP ({hidden} neurons)": make_pipeline(
                StandardScaler(), 
                MLPClassifier(
                    hidden_layer_sizes=(hidden,), 
                    alpha=1e-3, 
                    max_iter=2000,
                    early_stopping=True, 
                    random_state=seed
                )
            ),
            f"Integer-frequency Fourier (L = {n_layers})": make_pipeline(
                FourierFeatures(n_freq, n_layers, seed),
                LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
            ),
        }
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            p_va, p_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
            rows.append(score_row(name, n_layers, seed, p_va, p_te, data, {"Params": n_params}))
            if seed == 0:
                preds[name] = p_te
    return rows, preds

def compare_with_classical(p_vqc: np.ndarray, data: dict, extra_preds: dict) -> pd.DataFrame:
    y_te, ids = data["test"][1], data["test"][2]
    others = dict(extra_preds)
    if CLASSICAL_PREDS.exists():
        cp = pd.read_csv(CLASSICAL_PREDS, dtype={"record_id": str})
        cp = cp[cp["feature_set"].astype(str).str.contains("12")]
        for name, g in cp.groupby("model"):
            aligned = g.set_index("record_id").reindex(ids)["p"]
            if aligned.notna().all():
                others[f"{name} (12 features)"] = aligned.to_numpy()
    rows = [{"Versus": name, **paired_bootstrap_delta_auc(y_te, p_vqc, p)} for name, p in others.items()]
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    DEPLOY.mkdir(parents=True, exist_ok=True)
    data, feats, scaler = load_data()
    for k in ("train", "val", "test"):
        print(f"{k:5s}: {len(data[k][1]):5d} records, abnormal rate {data[k][1].mean():.1%}")
    print(f"Selected features: {feats}")
    (DEPLOY / "angle_scaler.json").write_text(scaler.to_json(), encoding="utf-8")
    runs = [("Full", L) for L in args.layers]
    if not args.skip_ablations:
        runs += [(v, args.ablation_layers) for v in VARIANTS if v != "Full"]
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    X_va_t = torch.tensor(data["val"][0], dtype=dtype)
    X_te_t = torch.tensor(data["test"][0], dtype=dtype)
    rows, best = [], {}
    for variant, n_layers in runs:
        for seed in range(args.seeds):
            t0 = time.time()
            model, val_auc, epoch = train_vqc(VARIANTS[variant], n_layers, seed, data, args)
            p_va, p_te = predict(model, X_va_t), predict(model, X_te_t)
            row = score_row(f"VQC {variant}", n_layers, seed, p_va, p_te, data,
                            {"Params": model.n_params(), "Val AUC": val_auc,
                             "Best epoch": epoch, "Time (s)": round(time.time() - t0, 1)})
            rows.append(row)
            print(f"{variant:20s} L={n_layers} seed={seed}: val AUC {val_auc:.3f}, "
                  f"test AUC {row['AUC']:.3f}, {row['Params']} params, {row['Time (s)']} s")
            key = (variant, n_layers)
            if key not in best or val_auc > best[key]["val_auc"]:
                best[key] = {"val_auc": val_auc, "model": model, "p_te": p_te, "thr": row["Threshold"]}
    runs_df = pd.DataFrame(rows)
    full = runs_df[runs_df["Model"] == "VQC Full"]
    best_layers = int(full.groupby("L")["Val AUC"].mean().idxmax())
    champ = best[("Full", best_layers)]
    model = champ["model"]
    print(f"\nSelected by mean val AUC: L = {best_layers}, {model.n_params()} params")
    base_rows, base_preds = matched_baselines(model.n_params(), best_layers, data, args.seeds)
    all_df = pd.concat([runs_df, pd.DataFrame(base_rows)], ignore_index=True)
    all_df.to_csv(OUT_TAB / "qml_runs.csv", index=False)
    metrics = ["AUC", "Acc", "Se", "Sp", "F1", "BAcc"]
    summary = all_df.groupby(["Model", "L"])[metrics + ["Params"]].agg(["mean", "std"]).round(3)
    summary.to_csv(OUT_TAB / "qml_summary.csv")
    print(summary.to_string())
    comp = compare_with_classical(champ["p_te"], data, base_preds)
    comp.round(4).to_csv(OUT_TAB / "qml_vs_classical.csv", index=False)
    print("\nPaired comparison (VQC minus other model):")
    print(comp.round(4).to_string(index=False))
    rob = robustness(model, data)
    rob.round(4).to_csv(OUT_TAB / "qml_robustness.csv", index=False)
    print("\nRobustness to shots and noise:")
    print(rob.pivot(index="Depolarizing per layer", columns="Shots", values="AUC").round(3).to_string())
    pd.DataFrame({"record_id": data["test"][2], "label": data["test"][1], "p": champ["p_te"],
                  "threshold": champ["thr"]}).to_csv(OUT_TAB / "qml_test_predictions.csv", index=False)
    params = {**model.export(), "threshold": champ["thr"], "features": feats,
              "angle_scaler": "angle_scaler.json"}
    (DEPLOY / "vqc_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(f"\nSaved circuit parameters to {DEPLOY / 'vqc_params.json'}")

if __name__ == "__main__":
    main()