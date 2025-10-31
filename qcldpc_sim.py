#!/usr/bin/env python3
"""
qcldpc_sim.py

Monte Carlo estimation of the logical error rate of a quantum QC-LDPC code
on a depolarizing channel with error probability p.

The code is defined by base matrices Hx_base and Hz_base.
Each entry is a nonnegative integer corresponding to a right-cyclic shift
of an LxL identity matrix. A value of -1 means a zero submatrix.

Decoding is done using the Min-Sum algorithm (approximation of BP).

Usage:
    python qcldpc_sim.py --p 0.02 --trials 1000

Author: ChatGPT (Quantum LDPC simulation example)
"""

import argparse
import numpy as np
import ipdb

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def mod2(x):
    """Modulo-2 reduction."""
    return x % 2


def right_cyclic_shift(mat, shift):
    """Right cyclic shift of an identity matrix."""
    n = mat.shape[0]
    return np.roll(mat, shift, axis=1)


def expand_base_matrix(base, L):
    """
    Expand a base matrix of integers into a binary parity-check matrix.

    base[i, j] = shift of an LxL identity matrix (>=0)
    base[i, j] = -1 means an all-zero LxL block
    """
    m_base, n_base = base.shape
    m, n = m_base * L, n_base * L
    H = np.zeros((m, n), dtype=np.int8)
    I = np.eye(L, dtype=np.int8)

    for i in range(m_base):
        for j in range(n_base):
            shift = base[i, j]
            if shift >= 0:
                H[i*L:(i+1)*L, j*L:(j+1)*L] = right_cyclic_shift(I, shift)
    return H


def syndrome(H, e):
    """Compute syndrome H * e mod 2."""
    return mod2(H.dot(e))


def depolarizing_errors(n, p):
    """
    Sample depolarizing channel errors on n qubits.

    Each qubit independently gets:
        I with prob (1 - p)
        X, Y, or Z each with prob p/3

    Returns binary vectors eX, eZ of length n.
    (Note: Y = X*Z gives both 1 bits in eX and eZ)
    """
    rnd = np.random.rand(n)
    eX = np.zeros(n, dtype=np.int8)
    eZ = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if rnd[i] < p:
            err_type = np.random.choice(["X", "Y", "Z"])
            if err_type == "X":
                eX[i] = 1
            elif err_type == "Z":
                eZ[i] = 1
            else:  # Y
                eX[i] = 1
                eZ[i] = 1
    return eX, eZ


# ---------------------------------------------------------------------------
# Min-Sum decoding (for binary LDPC)
# ---------------------------------------------------------------------------

def min_sum_decode(H, syndrome, p_err, max_iter=50, eps=1e-9):
    """
    Optimized Min-Sum decoder for binary LDPC codes.

    Uses vectorized numpy operations for faster inner loops.
    """

    m, n = H.shape
    # Sparse adjacency
    chk_to_var = [np.where(H[i, :])[0] for i in range(m)]
    var_to_chk = [np.where(H[:, j])[0] for j in range(n)]

    # Initialize LLRs and messages
    L_ch = np.log((1 - p_err + eps) / (p_err + eps))
    # All variable nodes start with same prior LLR
    msg_cv = np.zeros((m, n), dtype=np.float32)
    msg_vc = np.zeros((m, n), dtype=np.float32)
    msg_vc[H == 1] = L_ch

    # Precompute sign flip array for syndrome bits
    syn_sign = np.where(syndrome[:, None] == 1, -1.0, 1.0)

    for _ in range(max_iter):
        # ----- Check node update -----
        # For each check node: compute sign product and min magnitude across connected vars
        abs_msg = np.abs(msg_vc)
        sign_msg = np.sign(msg_vc)
        sign_msg[sign_msg == 0] = 1.0

        # sign product (ignoring zeros outside edges)
        prod_sign = np.prod(np.where(H == 1, sign_msg, 1.0), axis=1, keepdims=True)
        # min magnitude
        masked_abs = np.where(H == 1, abs_msg, np.inf)
        min_abs = np.min(masked_abs, axis=1, keepdims=True)
        min_abs[np.isinf(min_abs)] = 0.0

        # Each edge gets sign = prod_sign / sign_msg, value = min_abs
        msg_cv = np.where(H == 1,
                          syn_sign * prod_sign * min_abs / (sign_msg + (1 - H)),
                          0.0)

        # ----- Variable node update -----
        # Posterior LLRs
        incoming_sum = np.sum(msg_cv, axis=0)
        posterior = L_ch + incoming_sum

        # Hard decision
        e_hat = (posterior < 0).astype(np.int8)
        syn_est = mod2(H.dot(e_hat))
        if np.array_equal(syn_est, syndrome):
            return e_hat

        # Prepare next iteration’s messages
        # For each edge, subtract that check’s contribution
        msg_vc = np.where(H == 1, L_ch + incoming_sum - msg_cv, 0.0)

    return e_hat


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def simulate_qcldpc(Hx_base, Hz_base, L, p, trials=1000, max_iter=50):
    """
    Monte Carlo simulation of a QC-LDPC quantum code on a depolarizing channel.
    """
    Hx = Hx_base # expand_base_matrix(Hx_base, L)
    Hz = Hz_base # expand_base_matrix(Hz_base, L)

    n = Hx.shape[1]
    fail_count = 0

    for trl in range(trials):
        print('\rTrial n. {0:3}/{1:3}'.format(trl+1, trials), end='')
        eX, eZ = depolarizing_errors(n, p)

        sx = syndrome(Hx, eZ)  # Z errors -> X-check syndrome
        sz = syndrome(Hz, eX)  # X errors -> Z-check syndrome

        eZ_hat = min_sum_decode(Hx, sx, p_err=p/3, max_iter=max_iter)
        eX_hat = min_sum_decode(Hz, sz, p_err=p/3, max_iter=max_iter)

        # Residual error
        resX = mod2(eX ^ eX_hat)
        resZ = mod2(eZ ^ eZ_hat)

        if np.any(resX) or np.any(resZ):
            fail_count += 1

        print(', n. failures = {0}'.format(fail_count), end='')

    print()
    logical_error_rate = fail_count / trials
    return logical_error_rate


# ---------------------------------------------------------------------------
# Main entry point with argparse
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QC-LDPC quantum code Monte Carlo simulator.")
    parser.add_argument("--p", type=float, required=True,
                        help="Depolarizing error probability per qubit.")
    parser.add_argument("--trials", type=int, default=1000,
                        help="Number of Monte Carlo trials.")
    parser.add_argument("--L", type=int, default=8,
                        help="Circulant size (block size).")
    parser.add_argument("--maxIter", type=int, default=10,
                        help="Maximum number of deocding iterations.")
    args = parser.parse_args()

    ### Base matrices from [Raveendran et al., "Finite Rate QLDPC-GPK Coding Scheme...", arXiv:2111.07092v2]
    
    # LP04 family

    B_10_7 = np.array[
        [0, 0, 0, 0],
        [0, 1, 2, 5],
        [0, 6, 3, 1]]

    L, dmin = 7, 10
    B_10_7 = np.array[
        [0, 0, 0, 0],
        [0, 1, 2, 5],
        [0, 6, 3, 1]]

    L, dmin = 9, 12
    B_12_9 = np.array[
        [0, 0, 0, 0],
        [0, 1, 6, 7],
        [0, 4, 5, 2]]

    L, dmin = 17, 18
    B_18_17 = np.array[
        [0,  0,  0,  0],
        [0,  1,  2, 11],
        [0,  8, 12, 13]]

    L, dmin = 19, 20
    B_20_19 = np.array[
        [0,  0,  0,  0],
        [0,  2,  6,  9],
        [0, 16,  7, 11]]


    # LP118 family
    L, dmin = 16, 12
    B_12_16 = np.array[
        [0,  0,  0,  0,  0],
        [0,  2,  4,  7, 11],
        [0,  3, 10, 14, 15]]

    L, dmin = 21, 16
    B_16_21 = np.array[
        [0,  0,  0,  0,  0],
        [0,  4,  5,  7, 17],
        [0, 14, 18, 12, 11]]

    L, dmin = 30, 20
    B_20_30 = np.array[
        [0,  0,  0,  0,  0],
        [0,  2, 14, 24, 25],
        [0, 16, 11, 14, 13]]








    L = 31
    B = np.array([
        [ 1,  2,  4,  8, 16],
        [ 5, 10, 20,  9, 18],
        [25, 19,  7, 14, 28]])

    Btc = L - np.transpose(B)
    m_b, n_b = B.shape
    Bx = np.concat((np.kron(expand_base_matrix(B,args.L), np.identity(n_b)), np.kron(np.identity(m_b), expand_base_matrix(Btc,args.L))), axis=1)
    Bz = np.concat((np.kron(np.identity(n_b), expand_base_matrix(B,args.L)), np.kron(expand_base_matrix(Btc,args.L), np.identity(m_b))), axis=1)

    print('Code parameters: [[{0}, {1}, d]]'.format(Bx.shape[0], Bx.shape[1]))
    breakpoint()
    # Example base matrices (tiny toy code, replace with real ones)
    Hx_base = np.array([
        [0, 1, -1, -1],
        [1, -1, 0, -1]
    ])
    Hz_base = np.array([
        [1, -1, 0, -1],
        [0, 1, -1, -1]
    ])

    # logical_error_rate = simulate_qcldpc(
    #     Hx_base, Hz_base, L=args.L, p=args.p, trials=args.trials
    # )
    logical_error_rate = simulate_qcldpc(
        Bx, Bz, L=1, p=args.p, trials=args.trials, max_iter=args.maxIter
    )

    print(f"Depolarizing probability p = {args.p:.4f}")
    print(f"Estimated logical error rate = {logical_error_rate:.6f}")


if __name__ == "__main__":
    main()
