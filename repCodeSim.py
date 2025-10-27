# repCodeSim.py
"""
Quantum 9-qubit repetition (Shor) code simulator.

Simulates the effect of bit-flip and phase-flip noise on logical qubits encoded
using the 9-qubit repetition code.

Author: Expert Python Developer (2025)
"""

import numpy as np

# Pauli matrices for convenience
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def random_qubit():
    """Generate a random single-qubit state |ψ⟩ = α|0⟩ + β|1⟩, normalized."""
    a, b = np.random.randn(2) + 1j * np.random.randn(2)
    vec = np.array([a, b], dtype=complex)
    vec /= np.linalg.norm(vec)
    return vec


def apply_error(qubit_state, p_bitflip, p_phaseflip):
    """Apply random bit flip (X) and phase flip (Z) to a single qubit state."""
    state = qubit_state.copy()
    if np.random.rand() < p_bitflip:
        state = X @ state
    if np.random.rand() < p_phaseflip:
        state = Z @ state
    return state


def encode_9qubit(psi):
    """
    Encode 1 qubit |ψ⟩ = α|0⟩ + β|1⟩ into the 9-qubit repetition code.
    Encoding:
        |0⟩ -> |0_L⟩ = (|000⟩ + |111⟩)/√2 ⊗ (|000⟩ + |111⟩)/√2 ⊗ (|000⟩ + |111⟩)/√2
        |1⟩ -> |1_L⟩ = (|000⟩ - |111⟩)/√2 ⊗ (|000⟩ - |111⟩)/√2 ⊗ (|000⟩ - |111⟩)/√2
    """
    # basis states for one triple
    zero_block = (np.kron(np.kron([1,0],[1,0]),[1,0]) + np.kron(np.kron([0,1],[0,1]),[0,1])) / np.sqrt(2)
    one_block  = (np.kron(np.kron([1,0],[1,0]),[1,0]) - np.kron(np.kron([0,1],[0,1]),[0,1])) / np.sqrt(2)

    # 9-qubit logical basis states
    zero_L = np.kron(np.kron(zero_block, zero_block), zero_block)
    one_L  = np.kron(np.kron(one_block, one_block), one_block)

    # logical encoding of |ψ>
    psi_L = psi[0] * zero_L + psi[1] * one_L
    return psi_L


def apply_errors_9qubit(state, p_bitflip, p_phaseflip):
    """Apply independent random bit/phase flips to each of 9 qubits."""
    out = state.copy()
    for i in range(9):
        # single-qubit operators extended to 9-qubit space
        if np.random.rand() < p_bitflip:
            out = apply_single_qubit_gate(out, X, i)
        if np.random.rand() < p_phaseflip:
            out = apply_single_qubit_gate(out, Z, i)
    return out


def apply_single_qubit_gate(state, gate, qubit_index):
    """Apply a single-qubit gate to one qubit in a 9-qubit state."""
    # Tensor structure: gate on qubit i, identity elsewhere
    op = 1
    for j in range(9):
        op = np.kron(op, gate if j == qubit_index else np.eye(2))
    return op @ state


def decode_9qubit(state):
    """
    Decode the 9-qubit repetition code by majority voting on bit and phase errors.
    Simplified: For simulation, we determine logical state fidelity by measuring
    in computational basis and in Hadamard basis to detect bit and phase errors.
    """
    # For this simulation, we extract reduced density for logical qubit.
    # We'll treat the code as correcting up to one X and one Z per block of 3.
    # To simulate decoding, we apply block majority vote in computational basis
    # and phase correction via Hadamard transform and majority vote.
    # Simplified non-unitary correction process:
    corrected_blocks = []
    psi_collapsed = state.reshape([2]*9)
    for block in range(3):
        # indices for this block
        start = block * 3
        sub = slice(start, start + 3)
        # extract probabilities of |000>, |111>
        # approximate by computing expectation of Z⊗Z on each block
        # Here we assume perfect classical correction restoring block to |000> or |111>
        corrected_blocks.append(0)  # placeholder; we only need logical-level check
    return state  # placeholder: decoding simulated classically below


def logical_decode(physical_states):
    """
    Perform classical majority vote decoding:
      - For bit flips: take sign of expectation of Z per block.
      - For phase flips: similar idea in X basis.
    Here we simulate that decoding succeeds if <=1 X and <=1 Z per block.
    """
    pass  # implemented in simulate loop analytically


def simulate_repetition_code(p_bitflip=0.1, p_phaseflip=0.1, n_codewords=1000, seed=None):
    """
    Run Monte Carlo simulation of 9-qubit repetition code.

    Returns:
        total: number of codewords simulated
        errors: number of logical errors after decoding
        logical_error_rate: errors / total
    """
    if seed is not None:
        np.random.seed(seed)

    total = 0
    errors = 0

    for _ in range(n_codewords):
        psi = random_qubit()  # random qubit to encode
        # Represent logical info as α,β
        alpha, beta = psi

        # Shor code protects both X and Z using 3x repetition in two bases
        # For Monte Carlo simplicity:
        #   - We simulate bit/phase flips as random booleans on 9 qubits.
        #   - Decoding succeeds if each block of 3 has <=1 X and <=1 Z error.
        #   - Logical qubit fails otherwise.
        bit_errors = np.random.rand(9) < p_bitflip
        phase_errors = np.random.rand(9) < p_phaseflip

        # majority vote per block (3 qubits per block)
        bit_fail = False
        phase_fail = False
        for b in range(3):
            block_b = bit_errors[b*3:(b+1)*3]
            block_p = phase_errors[b*3:(b+1)*3]
            # If >1 errors in block, block correction fails
            if np.sum(block_b) > 1:
                bit_fail = True
            if np.sum(block_p) > 1:
                phase_fail = True

        # Logical error occurs if majority fails in *any* layer
        logical_error = bit_fail or phase_fail
        if logical_error:
            errors += 1
        total += 1

    return {
        "n_codewords": total,
        "logical_errors": errors,
        "logical_error_rate": errors / total
    }


if __name__ == "__main__":
    # Example run
    result = simulate_repetition_code(p_bitflip=0.1, p_phaseflip=0.1, n_codewords=10000, seed=42)
    print("Simulation results:")
    for k, v in result.items():
        print(f"{k}: {v}")
