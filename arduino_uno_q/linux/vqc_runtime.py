import json
from pathlib import Path
import numpy as np

def rot_matrix(phi: float, theta: float, omega: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [np.exp(-0.5j * (phi + omega)) * c, -np.exp(0.5j * (phi - omega)) * s],
        [np.exp(-0.5j * (phi - omega)) * s, np.exp(0.5j * (phi + omega)) * c],
    ], dtype=np.complex128)

class VQCRuntime:
    def __init__(self, artifacts: Path):
        p = json.loads((artifacts / "vqc_params.json").read_text(encoding="utf-8"))
        sc = json.loads((artifacts / "angle_scaler.json").read_text(encoding="utf-8"))
        self.n, self.n_layers = p["n_qubits"], p["n_layers"]
        self.enc_w, self.enc_b = np.asarray(p["enc_w"]), np.asarray(p["enc_b"])
        self.rot = np.asarray(p["rot"])
        self.head_w, self.head_b = np.asarray(p["head_w"]), float(np.asarray(p["head_b"]).ravel()[0])
        self.readout = p["readout_qubits"]
        self.features = p["features"]
        self.threshold = float(p["threshold"])
        self.lo, self.hi, self.max_angle = np.asarray(sc["lo"]), np.asarray(sc["hi"]), float(sc["max_angle"])
        bits = np.indices((2,) * self.n).reshape(self.n, -1).T
        sign = np.ones(2 ** self.n)
        for a, b in p["cz_pairs"]:
            sign *= np.where((bits[:, a] == 1) & (bits[:, b] == 1), -1.0, 1.0)
        self.cz = sign.reshape((2,) * self.n)

    def angles(self, feats: dict) -> np.ndarray:
        x = np.array([feats[f] for f in self.features], dtype=np.float64)
        x = np.clip(x, self.lo, self.hi)
        theta = self.max_angle * (x - self.lo) / (self.hi - self.lo)
        return np.pad(theta, (0, 3 * self.n - len(theta)))

    def _apply(self, psi: np.ndarray, u: np.ndarray, q: int) -> np.ndarray:
        psi = np.moveaxis(psi, q, -1) @ u.T
        return np.moveaxis(psi, -1, q)

    def _block(self, psi: np.ndarray, k: int) -> np.ndarray:
        for q in range(self.n):
            psi = self._apply(psi, rot_matrix(*self.rot[k, q]), q)
        return psi * self.cz

    def expectations(self, theta: np.ndarray) -> np.ndarray:
        psi = np.zeros((2,) * self.n, dtype=np.complex128)
        psi[(0,) * self.n] = 1.0
        psi = self._block(psi, 0)
        x = theta.reshape(self.n, 3)
        for layer in range(self.n_layers):
            ang = self.enc_w[layer] * x + self.enc_b[layer]
            for q in range(self.n):
                psi = self._apply(psi, rot_matrix(*ang[q]), q)
            psi = self._block(psi, layer + 1)
        probs = np.abs(psi) ** 2
        return np.array([probs.take(0, axis=q).sum() - probs.take(1, axis=q).sum() for q in self.readout])

    def predict(self, feats: dict) -> tuple[float, bool]:
        z = self.expectations(self.angles(feats))
        p = 1.0 / (1.0 + np.exp(-(z @ self.head_w + self.head_b)))
        return float(p), bool(p >= self.threshold)