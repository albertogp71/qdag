#!/usr/bin/env python3
"""
stim_qcldpc_ft.py

Fault-tolerant-style Stim simulation for a CSS QC-LDPC code with repeated
rounds of noisy stabilizer measurements. Uses DetectorSampler to sample detection
events and logical observables.

Usage:
  python stim_qcldpc_ft.py --Hx Hx.npy --Hy Hy.npy --p_data 0.01 --p_meas 0.02 \
      --rounds 5 --shots 1000 --lx lx.npy --lz lz.npy --seed 1

Notes:
 - Hx (X-checks) and Hy (Z-checks) must have the same number of columns (data qubits).
 - Each round:
     * Apply DEPOLARIZE1(p_data) to all data qubits
     * For each Z-check: CNOT data -> anc, DEPOLARIZE1(p_meas) anc, M anc, CREATE a DETECTOR linking to previous-round measurement
     * For each X-check: H on data in support, CNOT data -> anc, H undo, DEPOLARIZE1(p_meas) anc, M anc, DETECTOR
 - Logical operators (optional) are measured at the end and included with OBSERVABLE_INCLUDE.
"""

from __future__ import annotations
import argparse
from typing import Optional, List, Tuple
import numpy as np
import stim


# ----------------------------
# I/O helpers
# ----------------------------
def load_matrix(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        M = np.load(path)
    else:
        rows = []
        with open(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append([int(x) for x in line.split()])
        M = np.array(rows, dtype=int)
    return (M % 2).astype(np.int8)


def bitstrings_from_file(path: str) -> List[np.ndarray]:
    arr = load_matrix(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return [row.astype(int) for row in arr]


# ----------------------------
# Circuit builder (multi-round)
# ----------------------------
def build_ft_circuit(Hx: np.ndarray,
                     Hy: np.ndarray,
                     p_data: float,
                     p_meas: float,
                     rounds: int,
                     logical_X_ops: Optional[List[np.ndarray]] = None,
                     logical_Z_ops: Optional[List[np.ndarray]] = None) -> Tuple[stim.Circuit, int, int, int, int]:
    """
    Build a Stim circuit for `rounds` rounds of noisy stabilizer measurement.

    Returns: (circuit, n_data, n_detectors_per_round, total_detectors, total_observables_count_start_index)
      - n_data: number of data qubits
      - n_detectors_per_round: number of stabilizers measured per round (m_z + m_x)
      - total_detectors: total detectors created across all rounds
      - obs_start: number of measurements before logical observable measurements (useful for indexing)
    """
    if Hx is None:
        Hx = np.zeros((0, 0), dtype=int)
    if Hy is None:
        Hy = np.zeros((0, 0), dtype=int)
    m_x, n_x = Hx.shape if Hx.size else (0, 0)
    m_z, n_z = Hy.shape if Hy.size else (0, 0)
    if (n_x and n_z) and (n_x != n_z):
        raise ValueError("Hx and Hy must have same number of columns")
    n_data = n_x if n_x != 0 else n_z
    m_per_round = m_z + m_x

    circ_lines = []
    # We'll use data qubits 0..n_data-1
    # ancillas: allocate per-round ancilla indices? For simplicity re-use ancillas each round: allocate m_per_round ancillas
    # Stim circuits are static; measurement outputs are appended each time we M anc.
    ancilla_start = n_data
    ancillas_round = list(range(ancilla_start, ancilla_start + m_per_round))

    # Keep track: order of measurements per round: first m_z (Hy), then m_x (Hx)
    # We'll generate detectors referencing previous-round same-index measurement via rec[-1] and rec[-(1 + m_per_round)]
    for r in range(rounds):
        # 1) data depolarization (error during round)
        for q in range(n_data):
            circ_lines.append(f"DEPOLARIZE1({p_data}) {q}")

        # 2) Z-check measurements (Hy): CNOT data -> anc, anc depolarize (measurement error), M anc
        for i in range(m_z):
            anc = ancillas_round[i]
            cols = np.where(Hy[i] == 1)[0]
            for q in cols:
                circ_lines.append(f"CNOT {q} {anc}")
            # model ancilla/measurement error
            circ_lines.append(f"DEPOLARIZE1({p_meas}) {anc}")
            circ_lines.append(f"M {anc}")
            # DETECTOR: if first round, boundary detector is rec[-1], else compare to previous round's same measurement rec[-1] and rec[-(1 + m_per_round)]
            if r == 0:
                circ_lines.append("DETECTOR rec[-1]")
            else:
                circ_lines.append(f"DETECTOR rec[-1] rec[-{1 + m_per_round}]")

        # 3) X-check measurements (Hx): measure product of X via H, CNOTs, H, ancilla depolarize, M anc
        for i in range(m_x):
            anc = ancillas_round[m_z + i]
            cols = np.where(Hx[i] == 1)[0]
            # rotate data qubits into Z basis
            for q in cols:
                circ_lines.append(f"H {q}")
            for q in cols:
                circ_lines.append(f"CNOT {q} {anc}")
            for q in cols:
                circ_lines.append(f"H {q}")
            circ_lines.append(f"DEPOLARIZE1({p_meas}) {anc}")
            circ_lines.append(f"M {anc}")
            if r == 0:
                circ_lines.append("DETECTOR rec[-1]")
            else:
                circ_lines.append(f"DETECTOR rec[-1] rec[-{1 + m_per_round}]")

    # After rounds, we can optionally measure logical operators and include them as observables
    # Observables are created by measuring specified data qubits (in Z or X basis) and adding OBSERVABLE_INCLUDE
    # We'll collect observables in order and use them to estimate logical error rates.
    obs_count = 0
    # Measure logical Z operators: product of Zs -> measure data qubits in Z; include XOR of their measurements as observable
    if logical_Z_ops is not None:
        for lz in logical_Z_ops:
            # measure each qubit in support
            idxs = np.where(np.array(lz) % 2 == 1)[0].tolist()
            if len(idxs) == 0:
                # trivial operator
                circ_lines.append("OBSERVABLE_INCLUDE()")  # no measurement, zero observable
                obs_count += 1
                continue
            for q in idxs:
                circ_lines.append(f"M {q}")
            # build OBSERVABLE_INCLUDE referencing the last len(idxs) measurement records: rec[-1], rec[-2], ...
            tokens = " ".join(f"rec[-{i}]" for i in range(1, len(idxs) + 1))
            circ_lines.append(f"OBSERVABLE_INCLUDE {tokens}")
            obs_count += 1

    # Measure logical X operators: measure in X basis (H then M), include observable
    if logical_X_ops is not None:
        for lx in logical_X_ops:
            idxs = np.where(np.array(lx) % 2 == 1)[0].tolist()
            if len(idxs) == 0:
                circ_lines.append("OBSERVABLE_INCLUDE()")
                obs_count += 1
                continue
            for q in idxs:
                circ_lines.append(f"H {q}")
            for q in idxs:
                circ_lines.append(f"M {q}")
            for q in idxs:
                circ_lines.append(f"H {q}")  # optional undo: not necessary but keep clean
            tokens = " ".join(f"rec[-{i}]" for i in range(1, len(idxs) + 1))
            circ_lines.append(f"OBSERVABLE_INCLUDE {tokens}")
            obs_count += 1

    circuit_text = "\n".join(circ_lines)
    circ = stim.Circuit(circuit_text)
    total_detectors = circ.num_detectors
    return circ, n_data, m_per_round, total_detectors, obs_count


# ----------------------------
# Run simulation with DetectorSampler
# ----------------------------
def simulate_ft(Hx: np.ndarray,
                Hy: np.ndarray,
                p_data: float,
                p_meas: float,
                rounds: int,
                shots: int,
                logical_X_ops: Optional[List[np.ndarray]] = None,
                logical_Z_ops: Optional[List[np.ndarray]] = None,
                seed: Optional[int] = None) -> dict:
    """
    Build circuit and sample detection events + observables via DetectorSampler.

    Returns:
      - 'shots'
      - 'detector_event_rate' : fraction of detectors that fired on average
      - 'logical_error_rate' : if logical ops provided (else None)
      - 'det_samples' : None or first few detection samples (for debugging)
      - 'obs_samples' : None or first few observable bits
    """
    if seed is not None:
        np.random.seed(seed)

    circ, n_data, per_round, total_detectors, obs_count = build_ft_circuit(
        Hx, Hy, p_data, p_meas, rounds, logical_X_ops, logical_Z_ops
    )

    sampler = circ.compile_detector_sampler()
    # sample returns two numpy arrays: detection_events (shots x num_detectors) and observables (shots x num_observables)
    dets, obs = sampler.sample(shots=shots)

    # average detector event rate (per detector)
    avg_det_rate = float(dets.sum() / dets.size) if dets.size else 0.0

    logical_error_rate = None
    if obs_count > 0:
        # obs is shape (shots, obs_count), each entry 0/1
        # by convention, logical error corresponds to observable value 1 (depends on how logical ops defined)
        # If multiple logicals provided, consider any observable flip as logical failure (OR), or report per-logical rates.
        per_shot_fail = (obs.sum(axis=1) % 2) != 0 if obs_count > 1 else (obs[:, 0] == 1)
        logical_error_rate = float(np.mean(per_shot_fail))

    # keep a small sample preview for debugging
    preview = min(10, shots)
    det_preview = dets[:preview].astype(int).tolist()
    obs_preview = obs[:preview].astype(int).tolist() if obs.size else None

    return {
        "shots": shots,
        "avg_det_event_rate": avg_det_rate,
        "logical_error_rate": logical_error_rate,
        "det_preview": det_preview,
        "obs_preview": obs_preview,
        "num_detectors": total_detectors,
        "num_observables": obs_count
    }


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Fault-tolerant Stim QC-LDPC simulator with DetectorSampler.")
    parser.add_argument("--Hx", required=True, help="Path to Hx matrix (.npy or text).")
    parser.add_argument("--Hy", required=True, help="Path to Hy matrix (.npy or text).")
    parser.add_argument("--p_data", type=float, required=True, help="Depolarizing prob on data qubits per round.")
    parser.add_argument("--p_meas", type=float, required=True, help="Depolarizing prob on ancilla (measurement) qubits.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of repeated measurement rounds.")
    parser.add_argument("--shots", type=int, default=1000, help="Number of Monte Carlo shots.")
    parser.add_argument("--lx", help="Optional path to logical-X operators (.npy or text rows).")
    parser.add_argument("--lz", help="Optional path to logical-Z operators (.npy or text rows).")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    args = parser.parse_args()

    Hx = load_matrix(args.Hx)
    Hy = load_matrix(args.Hy)
    lxs = bitstrings_from_file(args.lx) if args.lx else None
    lzs = bitstrings_from_file(args.lz) if args.lz else None

    res = simulate_ft(Hx, Hy,
                      p_data=args.p_data,
                      p_meas=args.p_meas,
                      rounds=args.rounds,
                      shots=args.shots,
                      logical_X_ops=lxs,
                      logical_Z_ops=lzs,
                      seed=args.seed)

    print("Simulation summary:")
    print(f"  shots: {res['shots']}")
    print(f"  num_detectors: {res['num_detectors']}")
    print(f"  num_observables: {res['num_observables']}")
    print(f"  avg_det_event_rate: {res['avg_det_event_rate']:.6f}")
    if res['logical_error_rate'] is not None:
        print(f"  logical_error_rate (simple parity across observables): {res['logical_error_rate']:.6e}")
    else:
        print("  no logical operators provided -> logical_error_rate = None")
    print("  detector preview (first shots):")
    for row in res['det_preview']:
        print("   ", "".join(str(int(b)) for b in row))
    if res['obs_preview'] is not None:
        print("  observable preview (first shots):", res['obs_preview'])


if __name__ == "__main__":
    main()
