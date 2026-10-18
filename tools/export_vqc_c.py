from pathlib import Path
import numpy as np
import pandas as pd

from arduino_uno_q.linux.vqc_runtime import VQCRuntime

ARTIFACTS = Path("deploy/artifacts")
DATASET = Path("dataqml/median/record_dataset.csv")
OUT_DIR = Path("arduino_uno_q/mcu")
TEST_FOLD = 10
N_TESTS = 32

def c_float(a) -> str:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        s = f"{float(a):.9g}"
        if not any(ch in s for ch in ".eEn"):
            s += ".0"
        return s + "f"
    return "{" + ", ".join(c_float(x) for x in a) + "}"

def c_int(a) -> str:
    a = np.asarray(a, dtype=np.int64)
    if a.ndim == 0:
        return str(int(a))
    return "{" + ", ".join(c_int(x) for x in a) + "}"

def main():
    rt = VQCRuntime(ARTIFACTS)
    n_cz = len(rt.cz_pairs) if hasattr(rt, "cz_pairs") else 0
    import json
    p = json.loads((ARTIFACTS / "vqc_params.json").read_text(encoding="utf-8"))
    pairs = p["cz_pairs"]
    n_cz = len(pairs)
    readout = p["readout_qubits"]
    lines = [
        "#pragma once",
        f"// Generated from {ARTIFACTS / 'vqc_params.json'}",
        f"// Feature order: {', '.join(rt.features)}",
        f"#define VQC_N_QUBITS {rt.n}",
        f"#define VQC_N_LAYERS {rt.n_layers}",
        f"#define VQC_N_FEATURES {len(rt.features)}",
        f"#define VQC_N_OUT {len(readout)}",
        f"#define VQC_N_CZ {n_cz}",
        f"static const float VQC_ENC_W[VQC_N_LAYERS][VQC_N_QUBITS][3] = {c_float(rt.enc_w)};",
        f"static const float VQC_ENC_B[VQC_N_LAYERS][VQC_N_QUBITS][3] = {c_float(rt.enc_b)};",
        f"static const float VQC_ROT[VQC_N_LAYERS + 1][VQC_N_QUBITS][3] = {c_float(rt.rot)};",
        f"static const float VQC_HEAD_W[VQC_N_OUT] = {c_float(rt.head_w)};",
        f"static const float VQC_HEAD_B = {c_float(rt.head_b)};",
        f"static const int VQC_READOUT[VQC_N_OUT] = {c_int(readout)};",
        f"static const int VQC_CZ[VQC_N_CZ > 0 ? VQC_N_CZ : 1][2] = {c_int(pairs) if n_cz else '{{0, 0}}'};",
        f"static const float VQC_LO[VQC_N_FEATURES] = {c_float(rt.lo)};",
        f"static const float VQC_HI[VQC_N_FEATURES] = {c_float(rt.hi)};",
        f"static const float VQC_MAX_ANGLE = {c_float(rt.max_angle)};",
        f"static const float VQC_THRESHOLD = {c_float(rt.threshold)};",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "vqc_params.h").write_text("\n".join(lines) + "\n", encoding="utf-8")
    df = pd.read_csv(DATASET, dtype={"record_id": str})
    test = df[df["strat_fold"] == TEST_FOLD].head(N_TESTS)
    X = test[rt.features].to_numpy(dtype=np.float64)
    P = np.array([rt.predict(dict(zip(rt.features, x)))[0] for x in X])
    tv = [
        "#pragma once",
        '#include "vqc_params.h"',
        f"#define VQC_N_TESTS {len(X)}",
        f"static const float VQC_TEST_X[VQC_N_TESTS][VQC_N_FEATURES] = {c_float(X)};",
        f"static const float VQC_TEST_P[VQC_N_TESTS] = {c_float(P)};",
    ]
    (OUT_DIR / "test_vectors.h").write_text("\n".join(tv) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'vqc_params.h'} and {OUT_DIR / 'test_vectors.h'} ({len(X)} test vectors)")

if __name__ == "__main__":
    main()