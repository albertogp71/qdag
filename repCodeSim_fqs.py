# repCodeSim.py
"""
Full quantum-state simulation (statevector) of the 9-qubit repetition (Shor) code.
- Avoids density matrices (works with 512-dim statevectors).
- Encodes a single-qubit state into 9 physical qubits (Shor encoding),
  applies independent bit-flip (X) and phase-flip (Z) errors to each physical
  qubit with specified probabilities, performs syndrome measurements and
  correction (projective collapse simulated on the statevector), and checks
  whether decoding recovers the original logical qubit (up to global phase).

Usage:
    from repCodeSim import simulate_shor_code
    result = simulate_shor_code(p_bit=0.05, p_phase=0.02, n_codewords=1000, seed=1)
    print(result)

Author: Expert Python Developer (2025)
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Sequence

# single-qubit matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = 1.0 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)

NUM_QUBITS = 9
DIM = 2 ** NUM_QUBITS


def kron_n(ops: Sequence[np.ndarray]) -> np.ndarray:
    """Kronecker product of a sequence of operators (left-to-right)."""
    res = ops[0]
    for op in ops[1:]:
        res = np.kron(res, op)
    return res


def single_qubit_operator(op: np.ndarray, target: int) -> np.ndarray:
    """Build full 2^9 x 2^9 operator that applies `op` on qubit `target` (0-indexed)."""
    ops = []
    for q in range(NUM_QUBITS):
        ops.append(op if q == target else I)
    return kron_n(ops)


def pauli_string_operator(paulis: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Build operator for a product of single-qubit Pauli operators.
    `paulis` maps qubit index -> single-qubit matrix (X,Z,H,...). Qubits not in map get I.
    """
    ops = []
    for q in range(NUM_QUBITS):
        ops.append(paulis[q] if q in paulis else I)
    return kron_n(ops)


def normalize_state(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0:
        raise ValueError("Zero vector encountered when normalizing.")
    return vec / n


def random_qubit_state() -> np.ndarray:
    """Return a random normalized single-qubit state (2-vector complex)."""
    v = (np.random.randn(2) + 1j * np.random.randn(2)).astype(complex)
    return v / np.linalg.norm(v)


def build_logical_basis() -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct logical |0_L> and |1_L> states for Shor 9-qubit code using the
    (|000> +/- |111>)/sqrt(2) blocks.
    Returns (zero_L, one_L) as 512-dim normalized statevectors.
    """
    # single-block states (3 qubits): |0_block> = (|000> + |111>) / sqrt(2)
    zero_3 = np.zeros(2 ** 3, dtype=complex)
    one_3 = np.zeros(2 ** 3, dtype=complex)
    # basis ordering: |000>=index 0, |111>=index 7
    zero_3[0] = 1.0
    zero_3[7] = 1.0
    zero_3 /= np.sqrt(2)
    one_3[0] = 1.0
    one_3[7] = -1.0
    one_3 /= np.sqrt(2)

    zero_L = np.kron(np.kron(zero_3, zero_3), zero_3)
    one_L = np.kron(np.kron(one_3, one_3), one_3)
    # ensure normalization
    zero_L = normalize_state(zero_L)
    one_L = normalize_state(one_L)
    return zero_L, one_L


ZERO_L, ONE_L = build_logical_basis()


def encode_state(psi: np.ndarray) -> np.ndarray:
    """
    Encode a single-qubit state psi = [alpha, beta] into 9-qubit logical state:
        |psi_L> = alpha |0_L> + beta |1_L>
    """
    if psi.shape != (2,):
        raise ValueError("psi must be shape (2,)")
    return psi[0] * ZERO_L + psi[1] * ONE_L


def apply_errors(state: np.ndarray, p_bit: float, p_phase: float) -> np.ndarray:
    """
    Apply independent bit-flip and phase-flip errors to each qubit.
    For each physical qubit i:
      - with probability p_bit apply X
      - with probability p_phase apply Z
    Order: apply X (if any) then Z (if any). Both may be applied.
    """
    s = state.copy()
    for q in range(NUM_QUBITS):
        if np.random.rand() < p_bit:
            s = single_qubit_operator(X, q).dot(s)
        if np.random.rand() < p_phase:
            s = single_qubit_operator(Z, q).dot(s)
    return s


def measure_pauli_observable(state: np.ndarray, pauli_map: Dict[int, np.ndarray]) -> Tuple[int, np.ndarray]:
    """
    Measure an observable S = ⊗_{q} P_q where P_q ∈ {I,X,Y,Z} provided in pauli_map.
    This returns an eigenvalue outcome (+1 or -1) and the post-measurement collapsed statevector.
    The measurement is projective with projectors P_{±} = (I ± S)/2.
    """
    S = pauli_string_operator(pauli_map)
    # projector for +1 and -1
    P_plus = 0.5 * (np.eye(DIM, dtype=complex) + S)
    P_minus = 0.5 * (np.eye(DIM, dtype=complex) - S)

    p_plus = np.real_if_close(np.vdot(state, P_plus.dot(state))).item()
    # numerical guard
    p_plus = max(0.0, min(1.0, float(np.real(p_plus))))
    if p_plus == 0.0:
        outcome = -1
        post = P_minus.dot(state)
        post = normalize_state(post)
        return outcome, post
    if p_plus == 1.0:
        outcome = +1
        post = P_plus.dot(state)
        post = normalize_state(post)
        return outcome, post

    if np.random.rand() < p_plus:
        outcome = +1
        post = P_plus.dot(state)
    else:
        outcome = -1
        post = P_minus.dot(state)
    # normalize post state
    post = normalize_state(post)
    return outcome, post


def measure_block_bit_syndromes(state: np.ndarray, block_start: int) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    For a 3-qubit block starting at qubit index block_start (0,3,6),
    measure stabilizers S1 = Z_{i} Z_{i+1} and S2 = Z_{i+1} Z_{i+2}.
    Returns ((s1, s2), post_state), where s1,s2 ∈ {+1,-1}.
    Measurements are performed sequentially: S1 then S2 (each collapses).
    """
    i = block_start
    # S1 = Z_i Z_{i+1}
    s1_map = {i: Z, i + 1: Z}
    s1, state_after_s1 = measure_pauli_observable(state, s1_map)
    # S2 = Z_{i+1} Z_{i+2}
    s2_map = {i + 1: Z, i + 2: Z}
    s2, state_after_s2 = measure_pauli_observable(state_after_s1, s2_map)
    return (s1, s2), state_after_s2


def deduce_bit_error_from_syndromes(s1: int, s2: int) -> int:
    """
    Given stabilizer outcomes for a block (s1, s2), deduce the index within the block
    (0,1,2) of the qubit that likely suffered a bit flip.
    Mapping (s1,s2):
      (+1,+1) -> no error (return -1)
      (-1,+1) -> error on qubit 0
      (-1,-1) -> error on qubit 1
      (+1,-1) -> error on qubit 2
    Return -1 for no correction needed.
    """
    if s1 == +1 and s2 == +1:
        return -1
    if s1 == -1 and s2 == +1:
        return 0
    if s1 == -1 and s2 == -1:
        return 1
    if s1 == +1 and s2 == -1:
        return 2
    # fallback
    return -1


def correct_bit_flips(state: np.ndarray) -> np.ndarray:
    """
    Correct bit-flip errors block-wise (three blocks: starts at 0,3,6).
    For each block measure S1,S2, deduce which qubit to flip and apply X to correct.
    Returns the post-correction statevector.
    """
    s = state
    for b in (0, 3, 6):
        (s1, s2), s = measure_block_bit_syndromes(s, b)
        idx_in_block = deduce_bit_error_from_syndromes(s1, s2)
        if idx_in_block >= 0:
            qubit_to_flip = b + idx_in_block
            s = single_qubit_operator(X, qubit_to_flip).dot(s)
    return s


def correct_phase_flips(state: np.ndarray) -> np.ndarray:
    """
    Correct phase flips by switching to X-basis (Hadamard on all qubits),
    performing the same block-wise bit-flip correction on the 3 blocks formed by
    qubits [0,3,6], [1,4,7], [2,5,8], then undoing the Hadamards.
    This mirrors the standard Shor-code procedure.
    """
    # Apply Hadamard to all qubits
    H_all = kron_n([H] * NUM_QUBITS)
    s = H_all.dot(state)

    # Now treat three 'blocks' each collecting qubits at positions (0,3,6), (1,4,7), (2,5,8)
    # For each such logical-block, perform bit-flip correction using the same stabilizer measurement
    # pattern but on these triplets.
    triplets = [(0, 3, 6), (1, 4, 7), (2, 5, 8)]
    for trip in triplets:
        # For measurement functions we need contiguous indices; to reuse measurement code,
        # we create a mapping that permutes qubits so that the triplet maps to positions [0,1,2],
        # perform measurements, then un-permute. Simpler (although less efficient) approach:
        # build stabilizers S1 = Z_{a} Z_{b}, S2 = Z_{b} Z_{c} and measure them directly.
        a, b, c = trip
        s1_map = {a: Z, b: Z}
        s1, s = measure_pauli_observable(s, s1_map)
        s2_map = {b: Z, c: Z}
        s2, s = measure_pauli_observable(s, s2_map)
        idx = deduce_bit_error_from_syndromes(s1, s2)
        if idx >= 0:
            # map idx 0->a, 1->b, 2->c
            qubit_to_flip = (a, b, c)[idx]
            # flip in H-basis corresponds to X in H-basis; since we are in H-basis we use X
            s = single_qubit_operator(X, qubit_to_flip).dot(s)

    # Undo Hadamards
    s = H_all.dot(s)
    return s


def logical_state_amplitudes(state: np.ndarray) -> np.ndarray:
    """
    Compute amplitudes (gamma0, gamma1) of the 9-qubit state in the logical basis {|0_L>, |1_L>}.
    Returns a length-2 complex numpy array.
    """
    g0 = np.vdot(ZERO_L, state)  # <0_L | psi>
    g1 = np.vdot(ONE_L, state)   # <1_L | psi>
    return np.array([g0, g1], dtype=complex)


def logical_fidelity(original_psi: np.ndarray, decoded_amps: np.ndarray) -> float:
    """
    Given original single-qubit psi (alpha,beta) and decoded logical amplitudes
    decoded_amps = (gamma0,gamma1), compute fidelity up to global phase.
    We compute |<original|decoded>| where original and decoded are 2-vectors.
    """
    # normalize decoded_amps (should be normalized if encoding/decoding ideal)
    if np.linalg.norm(decoded_amps) == 0:
        return 0.0
    decoded = decoded_amps / np.linalg.norm(decoded_amps)
    original = original_psi / np.linalg.norm(original_psi)
    inner = np.vdot(np.conj(original), decoded)  # <orig|dec>
    return float(np.abs(inner))


def simulate_shor_code(p_bit: float, p_phase: float, n_codewords: int, seed: int = None) -> Dict[str, Any]:
    """
    Monte Carlo simulation of the full-state 9-qubit Shor code.

    Parameters:
      p_bit : probability of single-qubit bit-flip (X) error per physical qubit
      p_phase: probability of single-qubit phase-flip (Z) error per physical qubit
      n_codewords: how many random codewords to simulate
      seed: optional RNG seed for reproducibility

    Returns a dictionary with keys:
      - n_codewords
      - logical_errors : number of codewords that failed (fidelity < 1 - tol)
      - logical_error_rate
      - avg_fidelity : average fidelity (|<orig|decoded>|) across trials
    """
    if seed is not None:
        np.random.seed(seed)

    tol = 1e-8  # fidelity tolerance to consider success (we use 1.0 ideally)
    failures = 0
    fidelity_sum = 0.0

    for _ in range(n_codewords):
        # 1) random logical qubit
        psi = random_qubit_state()  # shape (2,)
        # 2) encode
        logical = encode_state(psi)
        # 3) apply errors
        errored = apply_errors(logical, p_bit, p_phase)
        # 4) bit-flip correction (measure S1,S2 per block and apply X corrections)
        after_bit = correct_bit_flips(errored)
        # 5) phase-flip correction (via H basis)
        after_phase = correct_phase_flips(after_bit)
        # 6) measure logical amplitudes
        decoded_amps = logical_state_amplitudes(after_phase)
        fid = logical_fidelity(psi, decoded_amps)
        fidelity_sum += fid
        if fid < 1.0 - 1e-6:  # allow tiny numeric slack
            failures += 1

    avg_fid = fidelity_sum / n_codewords
    return {
        "n_codewords": n_codewords,
        "logical_errors": failures,
        "logical_error_rate": failures / n_codewords,
        "avg_fidelity": avg_fid
    }


if __name__ == "__main__":
    # quick example run
    import time
    p_bit = 0.05
    p_phase = 0.02
    n = 200  # keep small for demonstration; full runs may be slower
    t0 = time.time()
    res = simulate_shor_code(p_bit, p_phase, n_codewords=n, seed=42)
    t1 = time.time()
    print("Shor 9-qubit simulation (full statevector):")
    for k, v in res.items():
        print(f"{k}: {v}")
    print(f"Elapsed: {t1 - t0:.2f} s")
