import matplotlib.pyplot as plt
import numpy as np
import wfdb
from pathlib import Path
from src.features.data_extractor import DataExtractor
from src.filters.pan_tompkins import PanTompkinsFilter

BASE = Path("data/physionet.org/files/ptb-xl/1.0.3")
ECG_ID = 8
FS = 100
LEAD = "II"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": "black",
})

def load_lead(ecg_id: int, lead: str = LEAD) -> np.ndarray:
    path = BASE / "records100" / f"{(ecg_id // 1000) * 1000:05d}" / f"{ecg_id:05d}_lr"
    sig, meta = wfdb.rdsamp(str(path))
    return sig[:, meta["sig_name"].index(lead)].astype(np.float64)

def main():
    de = DataExtractor(PanTompkinsFilter(fs=FS))
    _, xf, beats = de.delineate_record(load_lead(ECG_ID))
    t = np.arange(len(xf)) / FS
    marks = {
        "P start": ("p_onset", "|", "#77AC30"),
        "P peak": ("p_peak", "v", "#77AC30"),
        "T start": ("t_onset", "|", "#D95319"),
        "T peak": ("t_peak", "v", "#D95319"),
        "T end": ("t_end", "x", "#D95319"),
    }
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, xf, color="#0072BD", linewidth=0.9, label=f"Lead {LEAD}")
    ax.plot(t[[b[1] for b in beats]], xf[[b[1] for b in beats]], "*", color="black", label="Q")
    ax.plot(t[[b[2] for b in beats]], xf[[b[2] for b in beats]], "o", mfc="none", color="#A2142F", label="R")
    ax.plot(t[[b[3] for b in beats]], xf[[b[3] for b in beats]], "^", mfc="none", color="black", label="S")
    for label, (attr, marker, color) in marks.items():
        idx = [getattr(b[5], attr) for b in beats if getattr(b[5], attr) is not None]
        ax.plot(t[idx], xf[idx], marker, color=color, markersize=8, label=label)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    for i, q, r, s, rr, w in beats[:3]:
        print(f"Beat {i + 1}: PQ {w.pq_interval_ms:.0f} ms, QT {w.qt_interval_ms:.0f} ms, "
              f"QTc {w.qtc_ms:.0f} ms, ST {w.st_level:+.3f} mV, T {w.t_amplitude:+.3f} mV")
    plt.show()

if __name__ == "__main__":
    main()