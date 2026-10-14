import numpy as np
import torch

def _phase(a: torch.Tensor) -> torch.Tensor:
    return torch.polar(torch.ones_like(a), a)

def rot_matrix(phi: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    m00 = _phase(-(phi + omega) / 2) * c
    m01 = -_phase((phi - omega) / 2) * s
    m10 = _phase(-(phi - omega) / 2) * s
    m11 = _phase((phi + omega) / 2) * c
    return torch.stack([torch.stack([m00, m01], -1), torch.stack([m10, m11], -1)], -2)

def zero_state(batch: int, n_qubits: int, dtype: torch.dtype, device=None) -> torch.Tensor:
    psi = torch.zeros((batch,) + (2,) * n_qubits, dtype=dtype, device=device)
    psi.view(batch, -1)[:, 0] = 1.0
    return psi

def apply_1q(psi: torch.Tensor, u: torch.Tensor, q: int) -> torch.Tensor:
    psi = psi.movedim(q + 1, -1)
    if u.dim() == 2:
        psi = torch.einsum("ij,...j->...i", u, psi)
    else:
        psi = torch.einsum("bij,b...j->b...i", u, psi)
    return psi.movedim(-1, q + 1)

def ring_pairs(n_qubits: int) -> list[tuple[int, int]]:
    if n_qubits < 2:
        return []
    if n_qubits == 2:
        return [(0, 1)]
    return [(q, (q + 1) % n_qubits) for q in range(n_qubits)]

def cz_sign(n_qubits: int, pairs: list[tuple[int, int]], dtype: torch.dtype) -> torch.Tensor:
    bits = np.indices((2,) * n_qubits).reshape(n_qubits, -1).T
    s = np.ones(2 ** n_qubits)
    for a, b in pairs:
        s *= np.where((bits[:, a] == 1) & (bits[:, b] == 1), -1.0, 1.0)
    return torch.tensor(s.reshape((2,) * n_qubits), dtype=dtype)

def expect_z(psi: torch.Tensor, qubits: list[int]) -> torch.Tensor:
    probs = psi.abs() ** 2
    out = []
    for q in qubits:
        pq = probs.movedim(q + 1, 1).reshape(probs.shape[0], 2, -1).sum(-1)
        out.append(pq[:, 0] - pq[:, 1])
    return torch.stack(out, -1)