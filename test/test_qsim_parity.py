import numpy as np
import pytest
import torch
from src.qml.reuploading import ReuploadingVQC

qml = pytest.importorskip("pennylane")

def pennylane_expectations(model: ReuploadingVQC, x: np.ndarray) -> np.ndarray:
    n, n_layers = model.n_qubits, model.n_layers
    rot = model.rot.detach().numpy()
    w, b = model.enc_w.detach().numpy(), model.enc_b.detach().numpy()
    dev = qml.device("default.qubit", wires=n)
    @qml.qnode(dev)
    def circuit(xi):
        def block(k):
            for q in range(n):
                qml.Rot(*rot[k, q], wires=q)
            for a, c in model.pairs:
                qml.CZ(wires=[a, c])
        block(0)
        for layer in range(n_layers):
            ang = w[layer] * xi.reshape(n, 3) + b[layer]
            for q in range(n):
                qml.Rot(*ang[q], wires=q)
            block(layer + 1)
        return [qml.expval(qml.PauliZ(q)) for q in model.readout_qubits]
    return np.array([np.asarray(circuit(xi), dtype=np.float64) for xi in x])

def random_model(entangle: bool, readout: str) -> ReuploadingVQC:
    model = ReuploadingVQC(n_layers=3, entangle=entangle, readout=readout, dtype=torch.float64, seed=1)
    with torch.no_grad():
        model.enc_w.mul_(torch.rand_like(model.enc_w) + 0.5)
        model.enc_b.add_(torch.randn_like(model.enc_b))
        model.rot.copy_(torch.randn_like(model.rot))
    return model

@pytest.mark.parametrize("entangle", [True, False])
@pytest.mark.parametrize("readout", ["all_z", "z0"])
def test_matches_pennylane(entangle: bool, readout: str):
    model = random_model(entangle, readout)
    x = torch.rand(8, 12, dtype=torch.float64) * np.pi
    ours = model.expectations(x).detach().numpy()
    ref = pennylane_expectations(model, x.numpy())
    assert np.max(np.abs(ours - ref)) < 1e-10

def test_gradients_flow():
    model = random_model(True, "all_z")
    x = torch.rand(4, 12, dtype=torch.float64) * np.pi
    model(x).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name