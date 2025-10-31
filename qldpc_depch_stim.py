#!/usr/bin/env python3
"""
qldpc_depch_stim.py

Simulate a quantum LDPC (CSS-like) code defined by Hx and Hy under a depolarizing
channel using Stim. Constructs ancilla-based stabilizer measurements and applies
DEPOLARIZE1(p) to each data qubit, then samples measurement outcomes.

Usage examples:
  python stim_qldpc_depo.py --Hx hx.npy --Hy hy.npy --p 0.01 --shots 1000
  python stim_qldpc_depo.py --Hx hx.txt --Hy hy.txt --p 0.02 --shots 2000 --lx lx.npy --lz lz.npy

Input formats:
 - .npy (numpy) files saving binary matrices (dtype 0/1)
 - plain whitespace text files with 0/1 entries (rows lines, columns separated by spaces)

Outputs:
 - prints basic statistics (average syndrome weight) and (if logical ops supplied) an estimate
   of the logical error rate using a trivial naive decoder (syndrome -> greedy correction).
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import stim
from typing import Optional, List, Tuple


# -----------------------------
# I/O helpers
# -----------------------------
def load_matrix(path: str) -> np.ndarray:
    """Load a binary matrix from .npy or whitespace text."""
    if path.endswith(".npy"):
        mat = np.load(path)
    else:
        # attempt to parse whitespace-separated 0/1 text
        mat = []
        with open(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = [int(x) for x in line.split()]
                mat.append(row)
        mat = np.array(mat, dtype=int)
    return (mat % 2).astype(np.int8)


def bitstrings_from_file(path: str) -> List[np.ndarray]:
    """Load a list of binary bitstrings (rows) from file (.npy or whitespace text)."""
    arr = load_matrix(path)
    # if a single vector, ensure shape (k, n)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return [row.astype(int) for row in arr]


# -----------------------------
# Stim circuit construction
# -----------------------------
def build_stim_circuit(Hx: np.ndarray, Hy: np.ndarray, p: float) -> Tuple[stim.Circuit, int, int, int]:
    """
    Construct a stim.Circuit that:
      - has n data qubits (indices 0..n-1)
      - has ancilla qubits for Hy (Z-checks) and Hx (X-checks)
      - applies DEPOLARIZE1(p) to each data qubit
      - measures each stabilizer once, producing one measurement bit per stabilizer

    Returns (circuit, n_data, n_meas, ancilla_offset)
      - n_data: number of data qubits
      - n_meas: total number of measurement bits (rows(Hy) + rows(Hx))
      - ancilla_offset: first ancilla qubit index (useful for debugging / extension)
    """
    if Hx is None:
        Hx = np.zeros((0, 0), dtype=int)
    if Hy is None:
        Hy = np.zeros((0, 0), dtype=int)

    m_x, n_x = Hx.shape if Hx.size else (0, 0)
    m_z, n_z = Hy.shape if Hy.size else (0, 0)
    # require same number of data qubits
    if (n_x != 0 and n_z != 0) and (n_x != n_z):
        raise ValueError("Hx and Hy must have the same number of columns (data qubits).")
    n = n_x if n_x != 0 else n_z

    # Build a circuit as a list of lines -> feed to stim.Circuit(...)
    lines = []
    # allocate qubits implicitly by referencing indices in operations. Stim
    # will automatically size the circuit qubit register to accommodate the largest index used.
    # We'll use data qubits 0..n-1, then ancillas n..n+m_z+m_x-1
    ancilla_start = n
    ancilla_for_Z = list(range(ancilla_start, ancilla_start + m_z))
    ancilla_for_X = list(range(ancilla_start + m_z, ancilla_start + m_z + m_x))

    # 1) Apply depolarizing channel to each data qubit
    #    Use the operation text form: DEPOLARIZE1(p) q
    for q in range(n):
        lines.append(f"DEPOLARIZE1({p}) {q}")

    # 2) For each Z-check (row of Hy): measure product of Zs using an ancilla
    #    Circuit pattern:
    #      # ancilla starts in |0>
    #      CNOT data_q ancilla    for each q in support
    #      M ancilla
    #
    #    We'll append measurements in the same order as Hy rows.
    for row_idx in range(m_z):
        anc = ancilla_for_Z[row_idx]
        # apply CNOT from each data qubit in the stabilizer to ancilla
        cols = np.where(Hy[row_idx] % 2 == 1)[0]
        for q in cols:
            lines.append(f"CNOT {q} {anc}")
        # measure ancilla in Z
        lines.append(f"M {anc}")

    # 3) For each X-check (row of Hx): measure product of Xs by rotating data qubits with H
    #    Pattern:
    #      H q        for q in support
    #      CNOT q anc  for each q in support
    #      H q        for q in support    # undo
    #      M anc
    for row_idx in range(m_x):
        anc = ancilla_for_X[row_idx]
        cols = np.where(Hx[row_idx] % 2 == 1)[0]
        for q in cols:
            lines.append(f"H {q}")
        for q in cols:
            lines.append(f"CNOT {q} {anc}")
        for q in cols:
            lines.append(f"H {q}")
        lines.append(f"M {anc}")

    # Build circuit
    circ_text = "\n".join(lines)
    circ = stim.Circuit(circ_text)
    total_meas = m_z + m_x
    return circ, n, total_meas, ancilla_start


# -----------------------------
# Simple naive syndrome->correction heuristic (example only)
# -----------------------------
def naive_greedy_decoder(H: np.ndarray, syndrome_bits: np.ndarray) -> np.ndarray:
    """
    A very simple greedy decoder for demonstration:
    - repeatedly pick a variable node connected to the currently nonzero syndrome with highest degree,
      flip that variable in the candidate error, update the syndrome, and repeat up to some limit.
    This is NOT a production-grade decoder, but serves as a placeholder.
    """
    m, n = H.shape
    residual = syndrome_bits.copy()
    est = np.zeros(n, dtype=np.int8)
    # build adjacency lists
    check_to_vars = [list(np.where(H[i, :] == 1)[0]) for i in range(m)]
    var_to_checks = [list(np.where(H[:, j] == 1)[0]) for j in range(n)]

    max_steps = n * 2
    steps = 0
    while (residual.sum() > 0) and (steps < max_steps):
        steps += 1
        # compute scores for variables: number of failing checks it touches
        scores = np.zeros(n, dtype=int)
        failing_checks = np.where(residual == 1)[0]
        for c in failing_checks:
            for v in check_to_vars[c]:
                scores[v] += 1
        if scores.max() == 0:
            break
        v = int(np.argmax(scores))
        # flip v
        est[v] ^= 1
        # update residual
        for c in var_to_checks[v]:
            residual[c] ^= 1
    return est


# -----------------------------
# Main simulation function
# -----------------------------
def simulate_with_stim(Hx: np.ndarray,
                       Hy: np.ndarray,
                       p: float,
                       shots: int = 1000,
                       logical_X_ops: Optional[List[np.ndarray]] = None,
                       logical_Z_ops: Optional[List[np.ndarray]] = None,
                       rng_seed: Optional[int] = None) -> dict:
    """
    Build the stim circuit, run shots, and (optionally) do naive decoding to estimate
    logical error rate.

    Returns a dictionary with:
      - 'shots' : number of measurement shots
      - 'avg_syndrome_weight' : average number of 1 bits in stabilizer measurement
      - 'syndrome_counts' : histogram dict mapping syndrome bitstrings to counts (only if shots small)
      - 'logical_error_rate' : estimated if logical ops provided (else None)
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    circ, n_data, n_meas, anc_start = build_stim_circuit(Hx, Hy, p)

    # compile sampler and sample many shots
    sampler = circ.compile_sampler()
    # sample returns dtype uint8 array of shape (shots, n_meas) with measurement bits in the order we appended
    samples = sampler.sample(shots=shots)

    # compute average syndrome weight
    weights = samples.sum(axis=1)
    avg_weight = float(weights.mean())

    # syndromes histogram if small
    syndrome_counts = None
    if shots <= 2000:
        # convert rows to bitstring keys
        keys, counts = np.unique(samples.astype(np.uint8).view([('b', 'u1') * samples.shape[1]]), return_counts=True)
        # simpler: use string keys
        sc = {}
        for row in samples:
            s = "".join(str(int(b)) for b in row.tolist())
            sc[s] = sc.get(s, 0) + 1
        syndrome_counts = sc

    logical_error_rate = None
    if (logical_X_ops is not None) or (logical_Z_ops is not None):
        # attempt simple decoding: for each shot, try to produce estimated errors for X and Z
        # From CSS: Hy (Z checks) detect X errors; Hx (X checks) detect Z errors.
        # We'll use naive_greedy_decoder on each set independently.
        m_z = Hy.shape[0] if Hy.size else 0
        m_x = Hx.shape[0] if Hx.size else 0

        failures = 0
        for shot_idx in range(shots):
            row = samples[shot_idx]
            # row structure: [Hy measurements...] followed by [Hx measurements...]
            sy_z = row[:m_z].astype(int) if m_z else np.array([], dtype=int)
            sy_x = row[m_z:m_z+m_x].astype(int) if m_x else np.array([], dtype=int)

            # decode X-errors from sy_z using Hy (Hy * eX = sy_z)
            eX_hat = naive_greedy_decoder(Hy if Hy.size else np.zeros((0, n_data), dtype=int), sy_z)
            # decode Z-errors from sy_x using Hx
            eZ_hat = naive_greedy_decoder(Hx if Hx.size else np.zeros((0, n_data), dtype=int), sy_x)

            # residuals are simply e_hat (since circuit applies error then we decode based on syndrome only)
            # Determine whether residuals anticommute with logical operators
            logical_failure = False
            if logical_X_ops is not None:
                # logical X ops detect Z residuals: dot(lx, eZ_hat) mod 2 != 0 indicates flip
                for lx in logical_X_ops:
                    if int(np.dot(lx % 2, eZ_hat % 2) % 2) == 1:
                        logical_failure = True
                        break
            if (not logical_failure) and (logical_Z_ops is not None):
                # logical Z ops detect X residuals
                for lz in logical_Z_ops:
                    if int(np.dot(lz % 2, eX_hat % 2) % 2) == 1:
                        logical_failure = True
                        break
            if logical_failure:
                failures += 1

        logical_error_rate = failures / shots

    return {
        "shots": shots,
        "avg_syndrome_weight": avg_weight,
        "syndrome_counts": syndrome_counts,
        "logical_error_rate": logical_error_rate
    }


# -----------------------------
# CLI
# -----------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Stim-based QC-LDPC depolarizing-channel simulator.")
    parser.add_argument("--Hx", required=True, help="Path to Hx parity-check matrix (.npy or whitespace text).")
    parser.add_argument("--Hy", required=True, help="Path to Hy parity-check matrix (.npy or whitespace text).")
    parser.add_argument("--p", type=float, required=True, help="Depolarizing probability per qubit (0..1).")
    parser.add_argument("--shots", type=int, default=1000, help="Number of Monte Carlo shots.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument("--lx", required=False, help="Optional path to logical-X operators (rows of binary vectors).")
    parser.add_argument("--lz", required=False, help="Optional path to logical-Z operators (rows of binary vectors).")
    args = parser.parse_args(argv)

    Hx = load_matrix(args.Hx)
    Hy = load_matrix(args.Hy)
    logical_X_ops = None
    logical_Z_ops = None
    if args.lx:
        logical_X_ops = bitstrings_from_file(args.lx)
    if args.lz:
        logical_Z_ops = bitstrings_from_file(args.lz)

    res = simulate_with_stim(Hx, Hy, p=args.p, shots=args.shots,
                             logical_X_ops=logical_X_ops, logical_Z_ops=logical_Z_ops,
                             rng_seed=args.seed)

    print(f"shots: {res['shots']}")
    print(f"avg_syndrome_weight: {res['avg_syndrome_weight']:.4f}")
    if res['syndrome_counts'] is not None:
        print(f"unique syndrome patterns: {len(res['syndrome_counts'])}")
    if res['logical_error_rate'] is not None:
        print(f"estimated logical error rate: {res['logical_error_rate']:.6e}")
    else:
        print("logical operators not provided; only syndrome stats computed.")


if __name__ == "__main__":
    main()
