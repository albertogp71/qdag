#!/usr/bin/env python3
"""
qldpc_depch_minsum_stim.py

Simulate a quantum LDPC (CSS) code defined by Hx and Hy under a depolarizing
channel using Stim, and decode using the Min-Sum algorithm.

Usage:
  python stim_qldpc_depo.py --Hx hx.npy --Hy hy.npy --p 0.02 --shots 1000

Optional:
  --lx lx.npy --lz lz.npy   logical operators to estimate logical error rate.
"""

from __future__ import annotations
import argparse
import numpy as np
import stim
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
def load_matrix(path: str) -> np.ndarray:
    """Load binary matrix from .npy or whitespace text file."""
    if path.endswith(".npy"):
        M = np.load(path)
    else:
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append([int(x) for x in line.split()])
        M = np.array(rows, dtype=int)
    return (M % 2).astype(np.int8)


def bitstrings_from_file(path: str) -> List[np.ndarray]:
    arr = load_matrix(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return [row.astype(int) for row in arr]


def mod2(x: np.ndarray) -> np.ndarray:
    return x % 2


# ---------------------------------------------------------------------
# Stim Circuit Builder
# ---------------------------------------------------------------------
def build_stim_circuit(Hx: np.ndarray, Hy: np.ndarray, p: float) -> Tuple[stim.Circuit, int, int, int]:
    """Construct a Stim circuit applying depolarizing noise and measuring stabilizers."""
    if Hx is None:
        Hx = np.zeros((0, 0), dtype=int)
    if Hy is None:
        Hy = np.zeros((0, 0), dtype=int)
    m_x, n_x = Hx.shape if Hx.size else (0, 0)
    m_z, n_z = Hy.shape if Hy.size else (0, 0)
    n = n_x if n_x != 0 else n_z
    if (n_x and n_z) and (n_x != n_z):
        raise ValueError("Hx and Hy must have the same number of columns")

    lines = []
    anc_start = n
    anc_z = list(range(anc_start, anc_start + m_z))
    anc_x = list(range(anc_start + m_z, anc_start + m_z + m_x))

    # Depolarizing noise on data qubits
    for q in range(n):
        lines.append(f"DEPOLARIZE1({p}) {q}")

    # Measure Z stabilizers (Hy)
    for i in range(m_z):
        a = anc_z[i]
        for q in np.where(Hy[i])[0]:
            lines.append(f"CNOT {q} {a}")
        lines.append(f"M {a}")

    # Measure X stabilizers (Hx)
    for i in range(m_x):
        a = anc_x[i]
        qs = np.where(Hx[i])[0]
        for q in qs:
            lines.append(f"H {q}")
        for q in qs:
            lines.append(f"CNOT {q} {a}")
        for q in qs:
            lines.append(f"H {q}")
        lines.append(f"M {a}")

    circ = stim.Circuit("\n".join(lines))
    total_meas = m_z + m_x
    return circ, n, total_meas, anc_start


# ---------------------------------------------------------------------
# Vectorized Min-Sum Decoder
# ---------------------------------------------------------------------
def min_sum_decode(H: np.ndarray, syndrome: np.ndarray, p_err: float, max_iter: int = 50, eps: float = 1e-9) -> np.ndarray:
    """
    Vectorized Min-Sum decoding for binary LDPC.

    Args:
        H : parity-check matrix (m x n)
        syndrome : length-m binary array
        p_err : prior error probability
    Returns:
        estimated error vector (0/1)
    """
    if H.size == 0 or syndrome.size == 0:
        return np.zeros(H.shape[1] if H.size else 0, dtype=np.int8)

    m, n = H.shape
    # Initialize messages
    L_ch = np.log((1 - p_err + eps) / (p_err + eps))
    msg_vc = np.zeros((m, n), dtype=np.float32)
    msg_vc[H == 1] = L_ch
    msg_cv = np.zeros_like(msg_vc)
    syn_sign = np.where(syndrome[:, None] == 1, -1.0, 1.0)

    for _ in range(max_iter):
        # Check node update
        abs_msg = np.abs(msg_vc)
        sign_msg = np.sign(msg_vc)
        sign_msg[sign_msg == 0] = 1.0
        prod_sign = np.prod(np.where(H == 1, sign_msg, 1.0), axis=1, keepdims=True)
        min_abs = np.min(np.where(H == 1, abs_msg, np.inf), axis=1, keepdims=True)
        min_abs[np.isinf(min_abs)] = 0.0
        msg_cv = np.where(H == 1, syn_sign * prod_sign * min_abs / (sign_msg + (1 - H)), 0.0)

        # Variable node update
        incoming_sum = np.sum(msg_cv, axis=0)
        posterior = L_ch + incoming_sum
        e_hat = (posterior < 0).astype(np.int8)
        if np.array_equal(syndrome, mod2(H.dot(e_hat))):
            return e_hat
        msg_vc = np.where(H == 1, L_ch + incoming_sum - msg_cv, 0.0)
    return e_hat


# ---------------------------------------------------------------------
# Simulation using Stim
# ---------------------------------------------------------------------
def simulate_with_stim(Hx: np.ndarray,
                       Hy: np.ndarray,
                       p: float,
                       shots: int = 1000,
                       logical_X_ops: Optional[List[np.ndarray]] = None,
                       logical_Z_ops: Optional[List[np.ndarray]] = None,
                       rng_seed: Optional[int] = None) -> dict:
    """Run Stim simulation, collect syndromes, and decode via Min-Sum."""
    if rng_seed is not None:
        np.random.seed(rng_seed)
    circ, n_data, n_meas, anc_start = build_stim_circuit(Hx, Hy, p)
    sampler = circ.compile_sampler()
    samples = sampler.sample(shots=shots)
    m_z = Hy.shape[0] if Hy.size else 0
    m_x = Hx.shape[0] if Hx.size else 0

    avg_weight = float(samples.sum(axis=1).mean())
    logical_error_rate = None

    if (logical_X_ops is not None) or (logical_Z_ops is not None):
        fails = 0
        for shot in samples:
            syn_z = shot[:m_z].astype(int) if m_z else np.array([], int)
            syn_x = shot[m_z:m_z + m_x].astype(int) if m_x else np.array([], int)
            eX_hat = min_sum_decode(Hy, syn_z, p_err=p / 3) if m_z else np.zeros(n_data, int)
            eZ_hat = min_sum_decode(Hx, syn_x, p_err=p / 3) if m_x else np.zeros(n_data, int)

            logical_fail = False
            if logical_X_ops is not None:
                for lx in logical_X_ops:
                    if np.dot(lx, eZ_hat) % 2:
                        logical_fail = True
                        break
            if (not logical_fail) and (logical_Z_ops is not None):
                for lz in logical_Z_ops:
                    if np.dot(lz, eX_hat) % 2:
                        logical_fail = True
                        break
            if logical_fail:
                fails += 1
        logical_error_rate = fails / shots

    return dict(
        shots=shots,
        avg_syndrome_weight=avg_weight,
        logical_error_rate=logical_error_rate,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stim QC-LDPC simulator with Min-Sum decoder.")
    parser.add_argument("--Hx", required=True, help="Path to Hx matrix (.npy or text).")
    parser.add_argument("--Hy", required=True, help="Path to Hy matrix (.npy or text).")
    parser.add_argument("--p", type=float, required=True, help="Depolarizing error probability.")
    parser.add_argument("--shots", type=int, default=1000, help="Number of Monte Carlo shots.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--lx", help="Optional logical-X operators file.")
    parser.add_argument("--lz", help="Optional logical-Z operators file.")
    args = parser.parse_args()

    Hx = load_matrix(args.Hx)
    Hy = load_matrix(args.Hy)
    logical_X_ops = bitstrings_from_file(args.lx) if args.lx else None
    logical_Z_ops = bitstrings_from_file(args.lz) if args.lz else None

    res = simulate_with_stim(Hx, Hy, p=args.p, shots=args.shots,
                             logical_X_ops=logical_X_ops,
                             logical_Z_ops=logical_Z_ops,
                             rng_seed=args.seed)
    print(f"Depolarizing p = {args.p}")
    print(f"Shots = {res['shots']}")
    print(f"Average syndrome weight = {res['avg_syndrome_weight']:.3f}")
    if res['logical_error_rate'] is not None:
        print(f"Estimated logical error rate = {res['logical_error_rate']:.6f}")
    else:
        print("Logical operators not provided (only syndrome statistics computed).")


if __name__ == "__main__":
    main()
