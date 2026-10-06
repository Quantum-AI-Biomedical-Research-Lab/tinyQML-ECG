import wfdb
import numpy as np
import matplotlib.pyplot as plt
from src.filters.pan_tompkins import PanTompkinsFilter, PanTompkinsFilterResult

FS = 100

def load_lead():
    path = "data/physionet.org/files/ptb-xl/1.0.3/records100/00000/00001_lr"
    signal, meta = wfdb.rdsamp(path)
    return signal[:, 0].astype(np.float64)

def plot_stages(x: np.ndarray, pt: PanTompkinsFilter, res: PanTompkinsFilterResult):
    t = np.arange(len(x)) / pt.fs
    delays = pt.stage_delays()
    names = {"lowpass": "Low pass", "highpass": "High pass", "derivative": "Derivative", "mwi": "Moving window integration"}
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(t, x, color="#7E2F8E", linewidth=0.9, label="Raw ECG")
    for ax, key in zip(axes[1:], names):
        ax.plot(t - delays[key] / pt.fs, res.stages[key], color="#0072BD", linewidth=0.9, label=names[key])
    ax_m, shift = axes[-1], delays["mwi"] / pt.fs
    if len(res.threshold_trace):
        idx = res.threshold_trace[:, 0].astype(int)
        ax_m.step(t[idx] - shift, res.threshold_trace[:, 1], where="post", color="#D95319", linewidth=0.9, label="Threshold 1")
    ax_m.plot(t[res.qrs_mwi] - shift, res.stages["mwi"][res.qrs_mwi], "o", mfc="none", color="#A2142F", label="QRS detected")
    fig.tight_layout()
    plt.show()

def plot_qrs_points(x: np.ndarray, pt: PanTompkinsFilter, res: PanTompkinsFilterResult):
    t = np.arange(len(x)) / pt.fs
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, x, color="#0072BD", linewidth=0.9, label="Lead I")
    ax.plot(t[res.q_points], x[res.q_points], "*", color="#77AC30", label="Q")
    ax.plot(t[res.r_peaks], x[res.r_peaks], "o", mfc="none", color="#A2142F", label="R")
    ax.plot(t[res.s_points], x[res.s_points], "^", mfc="none", color="black", label="S")
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()

def main():
    x = load_lead()
    pt = PanTompkinsFilter(fs=FS)
    pt.summary()
    res = pt.run(x)
    print(f"QRS: {len(res.r_peaks)}")
    print(f"QRS locations: {res.r_peaks}")
    print(f"Heart beats: {res.heart_rate_bpm} bpm")
    print(f"QRS duration: {res.qrs_duration_ms} ms")
    plot_stages(x, pt, res)

if __name__ == "__main__":
    main()