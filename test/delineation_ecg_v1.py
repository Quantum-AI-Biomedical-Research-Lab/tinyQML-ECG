from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wfdb

from src.features.data_extractor import DataExtractor
from src.filters.pan_tompkins import PanTompkinsFilter

BASE = Path("data/physionet.org/files/ptb-xl/1.0.3")
OUT = Path("results/figures")
ECG_ID = 1
FS = 100
XLIM = (0.0, 10.0)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "black",
})


def load_lead(ecg_id: int, lead: str = "II") -> np.ndarray:
    path = BASE / "records100" / f"{(ecg_id // 1000) * 1000:05d}" / f"{ecg_id:05d}_lr"
    sig, meta = wfdb.rdsamp(str(path))
    return sig[:, meta["sig_name"].index(lead)].astype(np.float64)


def scatter(ax, t, idx, y, marker, color, label, size=40):
    ax.scatter(t[idx], y, marker=marker, color=color, s=size, zorder=3, label=label)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    de = DataExtractor(PanTompkinsFilter(fs=FS))
    _, xf, xs, beats = de.delineate_record(load_lead(ECG_ID))
    t = np.arange(len(xf)) / FS

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, xf, color="#0072BD", linewidth=0.8, label="Lọc 0,5–40 Hz")
    ax.plot(t, xs, color="#7E2F8E", linewidth=1.0, linestyle="--", label="Lọc 0,5–15 Hz (dò P, T)")

    for _, q, r, s, _, w in beats:
        lo = w.p_onset if w.p_onset is not None else q
        hi = w.t_end if w.t_end is not None else s
        ax.plot([t[lo], t[hi]], [w.baseline, w.baseline], color="gray", linewidth=0.6, linestyle=":")

    qs, rs, ss = ([b[k] for b in beats] for k in (1, 2, 3))
    scatter(ax, t, qs, xf[qs], "*", "black", "Q")
    scatter(ax, t, rs, xf[rs], "o", "#A2142F", "R")
    scatter(ax, t, ss, xf[ss], "^", "black", "S")

    def collect(attr, on_baseline):
        pts = [(getattr(b[5], attr), b[5].baseline) for b in beats if getattr(b[5], attr) is not None]
        idx = [p[0] for p in pts]
        y = [p[1] for p in pts] if on_baseline else xs[idx]
        return idx, y

    for attr, on_base, marker, color, label in [
        ("p_onset", True, "|", "#77AC30", "Bắt đầu P"),
        ("p_peak", False, "v", "#77AC30", "Đỉnh P"),
        ("t_onset", True, "|", "#D95319", "Bắt đầu T"),
        ("t_peak", False, "v", "#D95319", "Đỉnh T"),
        ("t_end", True, "x", "#D95319", "Kết thúc T"),
    ]:
        idx, y = collect(attr, on_base)
        scatter(ax, t, idx, y, marker, color, label, size=80 if marker == "|" else 40)

    ax.set_xlim(*XLIM)
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Biên độ (mV)")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"delineation_ecg{ECG_ID:05d}.png", dpi=300)

    for i, q, r, s, rr, w in beats[:4]:
        p_dur = (w.p_peak - w.p_onset) / FS * 1e3 if w.p_onset is not None else float("nan")
        print(f"Nhịp {i + 1}: P bắt đầu trước đỉnh {p_dur:.0f} ms, PQ {w.pq_interval_ms:.0f} ms, "
              f"QT {w.qt_interval_ms:.0f} ms, QTc {w.qtc_ms:.0f} ms, ST {w.st_level:+.3f} mV")
    plt.show()


if __name__ == "__main__":
    main()