#!/usr/bin/env python3
"""
stim_qcldpc_bp_ft.py

Stim-based repeated-round QC-LDPC simulator with a space-time Min-Sum (BP) decoder.

- Builds rounds of noisy stabilizer measurements for Hx (X-checks) and Hy (Z-checks).
- Samples raw ancilla measurement bits using stim.Circuit.compile_sampler().
- Builds space-time parity-check matrix for each Pauli type and decodes with Min-Sum.
- Example toy Hx/Hy is included in `example()`.

Author: ChatGPT (2025)
"""

from __future__ import annotations
import argparse
from typing import Optional, List, Tuple
import numpy as np
import stim


# -----------------------------
# Utilities
# -----------------------------
def load_matrix_txt_or_npy(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path) % 2
    rows = []
    with open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([int(x) for x in line.split()])
    return np.array(rows, dtype=int) % 2


def mod2(x):
    return x % 2


# -----------------------------
# Build repeated-round sampling circuit (raw measurements)
# -----------------------------
def build_rounds_sampling_circuit(Hx: np.ndarray,
                                  Hy: np.ndarray,
                                  p_data: float,
                                  p_meas: float,
                                  rounds: int,
                                  measure_logicals: bool = True) -> Tuple[stim.Circuit, int, int, int]:
    """
    Build a stim.Circuit that:
      - has data qubits 0..n-1
      - ancillas: one per stabilizer (m_z + m_x) reused each round
      - each round:
          DEPOLARIZE1(p_data) on every data qubit
          for each Z-check row: CNOT data->anc, DEPOLARIZE1(p_meas) anc, M anc  (collect measurement)
          for each X-check row: H on data in support, CNOT data->anc, H undo, DEPOLARIZE1(p_meas) anc, M anc
      - finally optionally measure logical ops (Z in Z-basis, X via H+M)
    Returns (circuit, n_data, m_per_round, n_measurements_total_per_shot)
    """
    if Hx is None:
        Hx = np.zeros((0, 0), dtype=int)
    if Hy is None:
        Hy = np.zeros((0, 0), dtype=int)

    m_x, n_x = Hx.shape if Hx.size else (0, 0)
    m_z, n_z = Hy.shape if Hy.size else (0, 0)
    if (n_x and n_z) and (n_x != n_z):
        raise ValueError("Hx and Hy must have same number of columns")
    n = n_x if n_x else n_z

    lines = []
    m_per_round = m_z + m_x
    anc_start = n
    ancillas = list(range(anc_start, anc_start + m_per_round))

    # Round loop
    for r in range(rounds):
        # data depolarize
        for q in range(n):
            lines.append(f"DEPOLARIZE1({p_data}) {q}")

        # Z checks
        for i in range(m_z):
            a = ancillas[i]
            cols = np.where(Hy[i] == 1)[0]
            for q in cols:
                lines.append(f"CNOT {q} {a}")
            lines.append(f"DEPOLARIZE1({p_meas}) {a}")
            lines.append(f"M {a}")

        # X checks
        for i in range(m_x):
            a = ancillas[m_z + i]
            cols = np.where(Hx[i] == 1)[0]
            for q in cols:
                lines.append(f"H {q}")
            for q in cols:
                lines.append(f"CNOT {q} {a}")
            for q in cols:
                lines.append(f"H {q}")
            lines.append(f"DEPOLARIZE1({p_meas}) {a}")
            lines.append(f"M {a}")

    # measure logicals at end if requested (we will append these measurements after all rounds)
    # We'll return their measurements to the caller.
    if measure_logicals:
        # placeholder: caller will append specific measurement commands if they supply logical ops
        pass

    circ = stim.Circuit("\n".join(lines))
    # number of measured ancilla bits per shot (per round)
    meas_per_shot = m_per_round * rounds
    return circ, n, m_per_round, meas_per_shot


# -----------------------------
# Build space-time parity-check matrix
# -----------------------------
def build_space_time_matrix(H: np.ndarray, rounds: int) -> Tuple[np.ndarray, List[Tuple[str, int, int]]]:
    """
    Create H_space_time which maps variables -> measurement rows.

    Variables ordering:
      - data variables: d_{q,t} for t in 0..rounds-1, q in 0..n-1   (total n * rounds)
      - measurement error variables: m_{s,t} for stabilizer s in 0..m-1, t in 0..rounds-1  (total m * rounds)

    For each measurement row (s,t) the equation is:
      measured_bit[s,t] = sum_{q in support(s)} d_{q,t} + m_{s,t}   (mod 2)

    Returns (H_space_time, variable_index_info) where variable_index_info is a list of tuples
    describing each column variable: ("d", q, t) or ("m", s, t)
    """
    if H.size == 0:
        return np.zeros((0, 0), dtype=int), []

    m, n = H.shape
    rows = m * rounds
    cols = n * rounds + m * rounds
    Hst = np.zeros((rows, cols), dtype=int)
    var_info = []
    # index mapping
    def d_idx(q, t):
        return t * n + q
    def m_idx(s, t):
        return n * rounds + t * m + s

    # fill matrix
    for t in range(rounds):
        for s in range(m):
            row = t * m + s
            # data contributions
            for q in range(n):
                if H[s, q]:
                    Hst[row, d_idx(q, t)] = 1
            # measurement error variable
            Hst[row, m_idx(s, t)] = 1

    # build var_info list (columns)
    for t in range(rounds):
        for q in range(n):
            var_info.append(("d", q, t))
    for t in range(rounds):
        for s in range(m):
            var_info.append(("m", s, t))
    return Hst, var_info


# -----------------------------
# Min-Sum decoder (vectorized)
# -----------------------------
def min_sum_decode(H: np.ndarray, syndrome: np.ndarray, p_prior_cols: np.ndarray,
                   max_iter: int = 50, eps: float = 1e-12) -> np.ndarray:
    """
    Min-Sum decoding for binary linear system H x = syndrome (mod 2),
    where each variable column j has prior probability p_prior_cols[j] of being 1.
    Returns binary estimate x_hat.
    """
    if H.size == 0:
        return np.zeros(0, dtype=int)
    m, n = H.shape
    # adjacency masks
    Hbool = (H != 0)
    # initial LLRs per variable
    Lch = np.log((1.0 - p_prior_cols + eps) / (p_prior_cols + eps)).astype(np.float32)  # shape (n,)

    # messages: msg_vc (m x n) from variable to check; msg_cv from check to var
    msg_vc = np.zeros((m, n), dtype=np.float32)
    msg_vc[Hbool] = Lch[np.newaxis, :][:, Hbool[0]] if False else 0.0  # we'll initialize below instead of using this odd expression
    # simpler initialization: each edge gets channel LLR
    for j in range(n):
        rows = np.where(Hbool[:, j])[0]
        msg_vc[rows, j] = Lch[j]

    msg_cv = np.zeros_like(msg_vc)

    syn_sign = np.where(syndrome[:, None] == 1, -1.0, 1.0)

    for _ in range(max_iter):
        # check node update (vectorized)
        abs_msg = np.abs(msg_vc)
        sign_msg = np.sign(msg_vc)
        sign_msg[sign_msg == 0] = 1.0
        # product of signs across each row over edges
        prod_sign = np.prod(np.where(Hbool, sign_msg, 1.0), axis=1, keepdims=True)
        # min magnitude across each row
        min_abs = np.min(np.where(Hbool, abs_msg, np.inf), axis=1, keepdims=True)
        min_abs[np.isinf(min_abs)] = 0.0
        # message to each variable: sign = prod_sign / sign_msg, magnitude = min_abs
        # handle zeros in sign_msg where Hbool==0 by masking
        msg_cv = np.where(Hbool,
                          syn_sign * prod_sign * min_abs / sign_msg,
                          0.0)

        # variable update
        incoming = np.sum(msg_cv, axis=0)  # sum over checks for each variable
        posterior = Lch + incoming
        x_hat = (posterior < 0).astype(int)
        if np.array_equal(mod2(H.dot(x_hat)), syndrome):
            return x_hat
        # prepare messages back to checks
        # msg_vc = Lch + incoming - msg_cv[c, j] for each edge
        msg_vc = np.where(Hbool, Lch[np.newaxis, :] + incoming[np.newaxis, :] - msg_cv, 0.0)

    return x_hat


# -----------------------------
# Full pipeline: sample with stim and decode with space-time BP
# -----------------------------
def run_trial_decode(Hx: np.ndarray, Hy: np.ndarray,
                     p_data: float, p_meas: float,
                     rounds: int, shots: int,
                     logical_X_ops: Optional[List[np.ndarray]] = None,
                     logical_Z_ops: Optional[List[np.ndarray]] = None,
                     seed: Optional[int] = None) -> dict:
    """
    Run Stim sampling (raw ancilla measurements per round) and decode with Min-Sum BP in space-time.
    Returns stats including logical error rate if logical ops provided.
    """
    if seed is not None:
        np.random.seed(seed)

    # Build sampling circuit (raw M results)
    circ, n_data, m_per_round, meas_per_shot = build_rounds_sampling_circuit(Hx, Hy, p_data, p_meas, rounds)
    sampler = circ.compile_sampler()
    # sample shape: (shots, meas_per_shot)
    samples = sampler.sample(shots=shots).astype(int)

    # split samples into per-round blocks: rows are shots
    # For CSS: Hy rows (m_z) come first each round, then Hx rows (m_x)
    m_z = Hy.shape[0] if Hy.size else 0
    m_x = Hx.shape[0] if Hx.size else 0
    m_total = m_z + m_x

    # Build space-time matrices for Hy (decoding X errors) and Hx (decoding Z errors)
    Hst_Z = None
    var_info_Z = None
    if m_z > 0:
        Hst_Z, var_info_Z = build_space_time_matrix(Hy, rounds)
    Hst_X = None
    var_info_X = None
    if m_x > 0:
        Hst_X, var_info_X = build_space_time_matrix(Hx, rounds)

    # Priors for variables: data vars have p_data, measurement vars p_meas
    # Build priors array for each Hst
    results = {"shots": shots, "logical_error_rate": None}
    logical_failures = 0

    for shot_i in range(shots):
        row = samples[shot_i]
        # build measurement RHS for Hy-space-time and Hx-space-time
        # measured bits ordering per round: [Hy rows..., Hx rows...] repeated
        meas_bits_Z = []
        meas_bits_X = []
        for t in range(rounds):
            start = t * m_total
            meas_bits_Z.extend(row[start:start + m_z].tolist() if m_z else [])
            meas_bits_X.extend(row[start + m_z:start + m_total].tolist() if m_x else [])

        # decode X-errors from Hy measurements (meas_bits_Z)
        if m_z > 0:
            syndrome_Z = np.array(meas_bits_Z, dtype=int)  # length m_z * rounds
            # priors: columns = n_data*rounds + m_z*rounds
            n_vars = Hst_Z.shape[1]
            # build p_prior_cols for Hst_Z
            n_data_vars = n_data * rounds
            n_meas_vars = m_z * rounds
            p_prior = np.zeros(n_vars, dtype=float)
            # data var columns first
            p_prior[:n_data_vars] = p_data
            p_prior[n_data_vars:] = p_meas
            x_hat_Z = min_sum_decode(Hst_Z, syndrome_Z, p_prior, max_iter=80)
            # extract data-error estimates (first n_data*rounds), then collapse across rounds
            data_err_vec = x_hat_Z[:n_data * rounds].reshape((rounds, n_data))
            # total data-X error per qubit across rounds (sum mod 2)
            total_data_X = np.sum(data_err_vec, axis=0) % 2
        else:
            total_data_X = np.zeros(n_data, dtype=int)

        # decode Z-errors from Hx measurements (meas_bits_X)
        if m_x > 0:
            syndrome_X = np.array(meas_bits_X, dtype=int)
            n_vars = Hst_X.shape[1]
            n_data_vars = n_data * rounds
            n_meas_vars = m_x * rounds
            p_prior = np.zeros(n_vars, dtype=float)
            p_prior[:n_data_vars] = p_data
            p_prior[n_data_vars:] = p_meas
            x_hat_X = min_sum_decode(Hst_X, syndrome_X, p_prior, max_iter=80)
            data_err_vec_z = x_hat_X[:n_data * rounds].reshape((rounds, n_data))
            total_data_Z = np.sum(data_err_vec_z, axis=0) % 2
        else:
            total_data_Z = np.zeros(n_data, dtype=int)

        # Now check logical operators if provided. For CSS:
        # logical X ops detect Z residuals (total_data_Z), logical Z ops detect X residuals (total_data_X)
        logical_fail = False
        if logical_X_ops is not None:
            for lx in logical_X_ops:
                if int(np.dot(lx % 2, total_data_Z % 2) % 2) == 1:
                    logical_fail = True
                    break
        if (not logical_fail) and (logical_Z_ops is not None):
            for lz in logical_Z_ops:
                if int(np.dot(lz % 2, total_data_X % 2) % 2) == 1:
                    logical_fail = True
                    break

        if logical_fail:
            logical_failures += 1

    if (logical_X_ops is not None) or (logical_Z_ops is not None):
        results["logical_error_rate"] = logical_failures / shots

    return results


# -----------------------------
# Example Hx / Hy and CLI
# -----------------------------
def example():
    """
    Example small CSS code:
    - 6 data qubits
    - Hy (Z-checks): 3 checks
    - Hx (X-checks): 3 checks
    This is a toy example (not fault-tolerant optimal) but useful to demo the pipeline.
    """
    # toy Hx, Hy (each row corresponds to a stabilizer, columns to data qubits)
    Hy = np.array([
        [1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0]
    ], dtype=int)

    Hx = np.array([
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 0]
    ], dtype=int)

    return Hx % 2, Hy % 2


def main():
    parser = argparse.ArgumentParser(description="Stim FT sampler + space-time Min-Sum BP decoder demo.")
    parser.add_argument("--p_data", type=float, default=0.01, help="Depolarizing prob on data qubits per round.")
    parser.add_argument("--p_meas", type=float, default=0.02, help="Depolarizing prob on ancilla per measurement.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds.")
    parser.add_argument("--shots", type=int, default=200, help="Number of shots.")
    parser.add_argument("--example", action="store_true", help="Run built-in toy example Hx/Hy.")
    parser.add_argument("--Hx", help="Path to Hx (.npy or text).")
    parser.add_argument("--Hy", help="Path to Hy (.npy or text).")
    args = parser.parse_args()

    if args.example:
        Hx, Hy = example()
    else:
        if not args.Hx or not args.Hy:
            print("Provide --example or both --Hx and --Hy", flush=True)
            return
        Hx = load_matrix_txt_or_npy(args.Hx)
        Hy = load_matrix_txt_or_npy(args.Hy)

    # Create trivial logical ops for demo (single logical Z = parity of all qubits, single logical X same)
    logical_Z_ops = [np.ones(Hx.shape[1], dtype=int)]
    logical_X_ops = [np.ones(Hx.shape[1], dtype=int)]

    res = run_trial_decode(Hx, Hy,
                           p_data=args.p_data,
                           p_meas=args.p_meas,
                           rounds=args.rounds,
                           shots=args.shots,
                           logical_X_ops=logical_X_ops,
                           logical_Z_ops=logical_Z_ops,
                           seed=12345)
    print("Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
