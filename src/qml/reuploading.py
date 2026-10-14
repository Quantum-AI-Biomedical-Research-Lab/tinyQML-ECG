import torch
from torch import nn
from src.qml.quantum_simulator import (
    apply_1q, 
    cz_sign, 
    expect_z, 
    ring_pairs, 
    rot_matrix, 
    zero_state
)

class ReuploadingVQC(nn.Module):
    
    def __init__(
        self, 
        n_features: int = 12, 
        n_qubits: int = 4, 
        n_layers: int = 4,
        trainable_scaling: bool = True, 
        entangle: bool = True, 
        readout: str = "all_z",
        init_std: float = 0.1, 
        dtype: torch.dtype = torch.float64, 
        seed: int = 0
    ):
        super().__init__()
        if n_features > 3 * n_qubits:
            raise ValueError(f"At most {3 * n_qubits} features are allowed for {n_qubits} qubits")
        if readout not in ("all_z", "z0"):
            raise ValueError("readout must be 'all_z' or 'z0'")
        g = torch.Generator().manual_seed(seed)
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.trainable_scaling = trainable_scaling
        self.entangle = entangle
        self.readout = readout
        self.rdtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        shape = (n_layers, n_qubits, 3)
        w0, b0 = torch.ones(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)
        if trainable_scaling:
            self.enc_w = nn.Parameter(w0)
            self.enc_b = nn.Parameter(b0)
        else:
            self.register_buffer("enc_w", w0)
            self.register_buffer("enc_b", b0)
        self.rot = nn.Parameter(init_std * torch.randn((n_layers + 1, n_qubits, 3), generator=g, dtype=dtype))
        self.readout_qubits = list(range(n_qubits)) if readout == "all_z" else [0]
        n_out = len(self.readout_qubits)
        self.head_w = nn.Parameter(torch.full((n_out,), 2.0 / n_out, dtype=dtype))
        self.head_b = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.pairs = ring_pairs(n_qubits) if entangle else []
        self.register_buffer("cz", cz_sign(n_qubits, self.pairs, self.cdtype))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.rdtype)
        pad = 3 * self.n_qubits - x.shape[1]
        return nn.functional.pad(x, (0, pad)) if pad else x

    def _trainable_block(self, psi: torch.Tensor, k: int) -> torch.Tensor:
        for q in range(self.n_qubits):
            psi = apply_1q(psi, rot_matrix(*self.rot[k, q]), q)
        return psi * self.cz if self.pairs else psi

    def _encode(self, psi: torch.Tensor, x: torch.Tensor, layer: int) -> torch.Tensor:
        ang = self.enc_w[layer] * x.view(-1, self.n_qubits, 3) + self.enc_b[layer]
        for q in range(self.n_qubits):
            psi = apply_1q(psi, rot_matrix(ang[:, q, 0], ang[:, q, 1], ang[:, q, 2]), q)
        return psi

    def expectations(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        psi = zero_state(x.shape[0], self.n_qubits, self.cdtype, x.device)
        psi = self._trainable_block(psi, 0)
        for layer in range(self.n_layers):
            psi = self._encode(psi, x, layer)
            psi = self._trainable_block(psi, layer + 1)
        return expect_z(psi, self.readout_qubits)

    def head(self, z: torch.Tensor) -> torch.Tensor:
        return z.to(self.rdtype) @ self.head_w + self.head_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.expectations(x))

    def export(self) -> dict:
        t = lambda p: p.detach().cpu().double().numpy().tolist()
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers, 
            "n_features": self.n_features,
            "trainable_scaling": self.trainable_scaling, 
            "entangle": self.entangle,
            "cz_pairs": self.pairs, 
            "readout": self.readout, 
            "readout_qubits": self.readout_qubits,
            "enc_w": t(self.enc_w), 
            "enc_b": t(self.enc_b), 
            "rot": t(self.rot),
            "head_w": t(self.head_w), 
            "head_b": t(self.head_b),
        }