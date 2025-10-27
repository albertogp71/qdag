# qlldpc_bp.py
"""
Simple quantum LDPC simulation with belief-propagation decoding for CSS codes.

- Hx, Hz: binary parity-check numpy arrays of shape (m_x, n) and (m_z, n).
- Logical operators: optional lists of binary vectors (lx_list, lz_list) of length n
  giving X- and Z-logical operators (so you can detect logical failure).
- BP decoder: log-domain sum-product (syndrome-aware).
- Post-processing: small local search flipping least-likely bits until syndrome matches.

This is a *practical starter*. For production use, add:
 - damping, scheduling or normalized/min-sum approximations
 - GF(4) BP for non-CSS codes or to capture degeneracy better
 - BP+OSD as post-processing (higher complexity but better performance)
"""
import numpy as np

# ---------- small helpers ----------
def mod2(x):
    return x % 2

def syndrome(H, e):
    return mod2(H.dot(e))

# ---------- Tanner graph structure ----------
def build_tanner(H):
    # H shape (m,n)
    m, n = H.shape
    var_to_checks = [list(np.where(H[:, j])[0]) for j in range(n)]
    check_to_vars = [list(np.where(H[i, :])[0]) for i in range(m)]
    return var_to_checks, check_to_vars

# ---------- log-domain sum-product for binary LDPC, syndrome-aware ----------
def bp_decode(H, syn, p_err, max_iters=50, eps=1e-12, damping=0.0):
    """
    H: (m,n) parity-check matrix (numpy 0/1)
    syn: syndrome vector length m (0/1)
    p_err: channel error prob for variable nodes (prior prob of bit=1)
    returns: estimated error vector e_hat (0/1)
    """
    m, n = H.shape
    var_to_checks, check_to_vars = build_tanner(H)

    # initial LLR from channel: L = log(P(0)/P(1)) = log((1-p)/p)
    L_ch = np.log((1.0 - p_err + eps) / (p_err + eps))
    # variable-to-check messages: initialize to channel LLR
    # shape (n, ), but we store msg_vc as dict keyed by (v,c) -> value
    msg_vc = {}
    msg_cv = {}

    for v in range(n):
        for c in var_to_checks[v]:
            msg_vc[(v, c)] = L_ch

    # iterative updates
    for it in range(max_iters):
        # check -> variable (use log-tanh identity)
        for c in range(m):
            for v in check_to_vars[c]:
                # product over other variables connected to check c
                prod = 1.0
                sign_prod = 1.0
                # in log domain we use tanh halves:
                # message = 2 * atanh( prod_{u != v} tanh( msg_vc[(u,c)]/2 ) )
                t_vals = []
                for u in check_to_vars[c]:
                    if u == v:
                        continue
                    val = msg_vc[(u, c)]
                    # cap for numerical stability
                    val = np.clip(val, -40.0, 40.0)
                    t_vals.append(np.tanh(val / 2.0))
                if len(t_vals) == 0:
                    # degree-1 check: message is simply sign from syndrome
                    msg = 0.0
                else:
                    prod_t = np.prod(t_vals)
                    # numerical guard
                    prod_t = np.clip(prod_t, -1 + 1e-12, 1 - 1e-12)
                    msg = 2.0 * np.arctanh(prod_t)
                # if check syndrome is 1, flip sign of message
                if syn[c] == 1:
                    msg = -msg
                # damping with previous msg if exists
                old = msg_cv.get((c, v), 0.0)
                msg_cv[(c, v)] = damping * old + (1.0 - damping) * msg

        # variable -> check
        posterior_L = np.zeros(n)
        for v in range(n):
            for c in var_to_checks[v]:
                # sum incoming check->var messages except from c plus channel LLR
                s = L_ch
                for c2 in var_to_checks[v]:
                    if c2 == c:
                        continue
                    s += msg_cv[(c2, v)]
                old = msg_vc.get((v, c), 0.0)
                new = damping * old + (1.0 - damping) * s
                msg_vc[(v, c)] = new
            # compute posterior LLR for variable v (sum of all incoming messages + channel)
            s_all = L_ch + sum(msg_cv[(c2, v)] for c2 in var_to_checks[v])
            posterior_L[v] = s_all

        # tentative decision
        e_hat = (posterior_L < 0).astype(int)  # negative LLR -> prefer 1
        # check syndrome
        syn_est = mod2(H.dot(e_hat))
        if np.array_equal(syn_est, syn):
            return e_hat  # success

    # if BP didn't converge to correct syndrome, return current hard decision
    return e_hat

# ---------- cheap post-processing: flip least-likely bits until syndrome solved ----------
def local_search_fix(H, syn, posterior_L, max_flips=4):
    """
    posterior_L: LLR array length n (higher -> more likely 0)
    Try flipping small sets of least-likely bits (highest probability of being 1)
    up to size max_flips until syndrome matches. This is a tiny OSD-like fallback.
    """
    n = len(posterior_L)
    # sort by probability of being 1: p1 = 1/(1+exp(LLR))
    p1 = 1.0 / (1.0 + np.exp(posterior_L))
    order = np.argsort(-p1)  # descending p1
    # try flipping increasing sets
    base = (posterior_L < 0).astype(int)
    from itertools import combinations
    for k in range(1, max_flips + 1):
        for comb in combinations(order[:12], k):  # only consider top-12 least-likely bits
            cand = base.copy()
            cand[list(comb)] ^= 1
            if np.array_equal(mod2(H.dot(cand)), syn):
                return cand
    return base

# ---------- full quantum simulation loop ----------
def simulate_css_bp(Hx, Hz, lx_list=None, lz_list=None,
                    p_X=0.01, p_Z=0.01, n_trials=1000, max_iters=50, seed=None):
    """
    Simulate CSS quantum code:
    - Hx: checks for X-logical (detect Z errors) -> decode e_Z via Hx
    - Hz: checks for Z-logical (detect X errors) -> decode e_X via Hz
    - lx_list/lz_list: optional lists of logical operator binary vectors length n (for final logical error check)
    Returns dict with stats and average fidelities (if logical ops provided).
    """
    if seed is not None:
        np.random.seed(seed)
    m_x, n = Hx.shape
    m_z, _ = Hz.shape

    total = 0
    logical_failures = 0

    for _ in range(n_trials):
        # sample true errors
        e_x = (np.random.rand(n) < p_X).astype(int)
        e_z = (np.random.rand(n) < p_Z).astype(int)

        # syndromes
        s_x = mod2(Hx.dot(e_z))  # Z errors create X-check syndrome
        s_z = mod2(Hz.dot(e_x))  # X errors create Z-check syndrome

        # decode e_z from s_x using BP on Hx
        e_z_hat = bp_decode(Hx, s_x, p_err=p_Z, max_iters=max_iters)
        # compute posterior LLRs for optional local search (recompute quickly):
        # posterior_L_z: channel LLR for Z decoding
        # For simplicity reuse bp_decode internal result: recompute naive LLRs
        # We just recompute a posterior LLR via running BP one more iteration without syndrome check:
        # (skipped here) - instead apply local_search fallback using bp hard decision
        # If mismatch in syndrome, attempt local search
        syn_check = mod2(Hx.dot(e_z_hat))
        if not np.array_equal(syn_check, s_x):
            # optional small local search: (we cannot get posterior easily here; do a trivial try)
            # fallback simply leaves e_z_hat as is
            pass

        # decode e_x from s_z using BP on Hz
        e_x_hat = bp_decode(Hz, s_z, p_err=p_X, max_iters=max_iters)

        # evaluate residual errors
        res_x = mod2(e_x ^ e_x_hat)
        res_z = mod2(e_z ^ e_z_hat)

        # check logical operators: if any logical operator anticommutes with residual -> failure
        # For CSS with provided logicals:
        failed = False
        if lz_list is not None:  # Z logical operators detect X residuals
            for lz in lz_list:
                # logical flips if dot(lz, res_x) mod2 == 1
                if mod2(np.dot(lz, res_x)) == 1:
                    failed = True
                    break
        if not failed and lx_list is not None:  # X logical operators detect Z residuals
            for lx in lx_list:
                if mod2(np.dot(lx, res_z)) == 1:
                    failed = True
                    break

        if failed:
            logical_failures += 1
        total += 1

    return {
        "n_trials": total,
        "logical_failures": logical_failures,
        "logical_error_rate": logical_failures / total if total else None
    }

# Example usage:
if __name__ == "__main__":
    # toy example:  [replace with real Hx, Hz and logicals]
    n = 8
    # small repetition code (not LDPC) example H matrices:
    Hx = np.array([[1,1,0,0,0,0,0,0],
                   [0,1,1,0,0,0,0,0]], dtype=int)
    Hz = Hx.copy()
    # logical operators (toy)
    lx = [np.array([1,0,0,0,0,0,0,0], dtype=int)]
    lz = [np.array([0,0,0,0,0,0,0,1], dtype=int)]

    res = simulate_css_bp(Hx, Hz, lx_list=lx, lz_list=lz,
                          p_X=0.05, p_Z=0.02, n_trials=200, max_iters=50, seed=1)
    print(res)
