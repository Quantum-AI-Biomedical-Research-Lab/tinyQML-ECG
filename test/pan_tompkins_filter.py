import wfdb
import numpy as np
import matplotlib.pyplot as plt
from src.filters.pan_tompkins import PanTompkinsFilter

path = "data/physionet.org/files/ptb-xl/1.0.3/records100/00000/00001_lr"
signal, meta = wfdb.rdsamp(path)
ecg = signal[:, 0].astype(np.float64)
fs_original = float(meta["fs"])

print("Original ECG shape:", ecg.shape)
print("Original fs:", fs_original)

fs = 100.0
time = np.arange(len(ecg)) / fs
pt = PanTompkinsFilter(fs=fs)
lowpass = pt.low_pass_filter(ecg)
highpass = pt.high_pass_filter(lowpass)
derivative = pt.derivative(highpass)
squared = pt.squaring(derivative)
mwi = pt.moving_window_integration(squared)

fig, axes = plt.subplots(
    6,
    1,
    figsize=(14, 12),
    sharex=True
)
axes[0].plot(time, ecg)
axes[0].set_title("Original ECG — Lead I")
axes[1].plot(time, lowpass)
axes[1].set_title("Low-Pass")
axes[2].plot(time, highpass)
axes[2].set_title("High-Pass")
axes[3].plot(time, derivative)
axes[3].set_title("Derivative")
axes[4].plot(time, squared)
axes[4].set_title("Squaring")
axes[5].plot(time, mwi)
axes[5].set_title("Moving Window Integration")
axes[5].set_xlabel("Time (s)")
for ax in axes:
    ax.set_ylabel("Amplitude")
    ax.grid(True)
plt.tight_layout()
plt.show()