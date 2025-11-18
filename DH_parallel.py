import numpy as np
from qutip import *
from scipy.special import eval_genlaguerre, factorial
import numba
from joblib import Parallel, delayed
from get_Hamiltonians import H_DH_MN, H_EFF_1N, H_EFF_2N, H_EFF_3N, build_UP, H_CDH_MN
from scipy.linalg import expm
import os
from functools import partial
from tqdm import tqdm

def order_params(Delta, lam, gammas, H_func, M, N, periodic, CDH, model):

    Omega = 2.0
    gamma_x, gamma_y, gamma_z = gammas

    HMN = H_func(M, N)
    H_qobj = HMN(Delta, Omega, lam, gamma_x, gamma_y, gamma_z, periodic, CDH=CDH)
    GS = H_qobj.groundstate()[1]

    if model != 0 or CDH == True:

        #build S as sum sigma_x across the N spins
        S = np.zeros((2**N, 2**N), dtype=complex)
        for i in range(N):
            ops_i = [qeye(2)] * N
            ops_i[i] = sigmax()
            S += tensor(ops_i).full()
        S = S / np.sqrt(N) # ALSO GIVES GOOD RESULT WITH TRUNCATION IN EXPONENT
        obsS = Qobj(S, dims=[[2]*N,[2]*N])

        UP = build_UP(S, lam / Omega, M, N)
        #UP = ( tensor( create(M)-destroy(M) , lam/Omega * obsS ) ).expm()
        #UP = Qobj(UP, dims=[[M] + [2]*N,[M] + [2]*N])
    else:
        UP = qeye(M*2**N)
        UP = Qobj(UP, dims=[[M] + [2]*N,[M] + [2]*N])

    # prepare operators for expectation calculations
    ops_Mz = []
    ops_Mx = []
    for i in range(N):
        if M > 1:
            ops = [qeye(M)] + [qeye(2)] * N
            opsX = [qeye(M)] + [qeye(2)] * N
            opsX[i + 1] = sigmax() / N
            ops[i + 1] = sigmaz() / N
            op_mz_q = UP @ tensor(ops) @ UP.dag()
            op_mx_q = tensor(opsX)
        else:
            ops = [qeye(2)] * N
            opsX = [qeye(2)] * N
            opsX[i] = sigmax() / N
            ops[i] = sigmaz() / N
            op_mz_q = UP @ tensor(ops) @ UP.dag()
            op_mx_q = tensor(opsX)

        ops_Mz.append(op_mz_q)
        ops_Mx.append(op_mx_q)

    # Convert GS to dense state-vector (numpy array)
    GS_vec = GS.full().ravel()
    # compute expectations without building stacked dense arrays
    def expectation_list_from_qobjs(qobj_list, state_vec):
        vals = []
        for op in qobj_list:
            # use sparse matrix * dense vector: op.data is scipy sparse
            tmp = op.data.to_array() @ state_vec
            val = np.vdot(state_vec, tmp)           # <state|op|state>
            vals.append(np.real(val))
        return np.array(vals)

    Mz_list = expectation_list_from_qobjs(ops_Mz, GS_vec)
    Mx_list = expectation_list_from_qobjs(ops_Mx, GS_vec)

    Mz = Mz_list.sum()
    Mx = Mx_list.sum()

    if N >= 2 and M > 1:
        rho_12 = GS.ptrace([1, 2])
    elif N >= 2 and M == 1:
        rho_12 = GS.ptrace([0, 1])
    else:
        raise ValueError("Need at least 2 spins to compute concurrence")

    spin_indices = list(range(1, N + 1))  # exclude bosonic mode
    half = len(spin_indices) // 2
    left_half = spin_indices[:half]
    rho_half = GS.ptrace(left_half)
    S_entropy = entropy_vn(rho_half)

    return np.array([Mz, S_entropy, Mx], dtype=np.complex128)


# ---- Worker computing whole grid for a particular (model, N, igamma) ---------------
def compute_grid_for_case(model, N, igamma, lam_max=4.0, delta_max=2.0, res=50, n_jobs=-1, out_dir="data_new"):
    # Select M and H_func and gammas
    H_list = np.array([H_DH_MN, H_EFF_1N, H_EFF_2N, H_EFF_3N, H_CDH_MN], dtype=object)
    M = [20, 1, 2, 3, 3][model]
    H_func = H_list[model]

    if igamma == 0:
        gammas = np.array([0.33, 0.33, 0.33])
    elif igamma == 1:
        gammas = np.array([0.25, 0.25, 0.0])
    elif igamma == 2:
        gammas = np.array([0.3, 0.2, 0.0])
    elif igamma == 3:
        gammas = np.array([0.1, 0.2, 0.3])
    else:
        raise ValueError("invalid igamma")

    lam_values = np.linspace(0.0, lam_max, res)
    delta_values = [0.1, 1.0] #np.linspace(0.0, delta_max, res)

    # Prepare flattened grid
    grid = [(Delta, lam) for lam in lam_values for Delta in delta_values]

    # Partial function for joblib
    periodic = True
    worker = partial(order_params, gammas=gammas, H_func=H_func, M=M, N=N, periodic=periodic, CDH=False, model=model)

    # Run in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(worker)(Delta, lam) for (Delta, lam) in tqdm(grid, desc=f"model{model}_N{N}_ig{igamma}", leave=False)
    )

    # Reconstruct Z as (res x res x 3)
    Z = np.zeros((len(lam_values), len(delta_values), 3), dtype=np.complex128)
    idx = 0
    for i_lam in range(len(lam_values)):
        for j_delta in range(len(delta_values)):
            Z[i_lam, j_delta, :] = results[idx]
            idx += 1

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"DH_model{model}_N{N}_M{M}_gam{igamma}_cut.npy")
    np.save(fname, Z)
    return fname


# ---- Main: loop over Ns, models, igammas, but dispatch each case in parallel if you want ----
if __name__ == "__main__":
    Ns = [4,6,8]
    models = [0,1,3]
    igammas = [0,1]
    
    for N in Ns:
        for model in models:
            for igamma in igammas:
                print(f"Starting model={model}, N={N}, igamma={igamma}")
                out_file = compute_grid_for_case(model, N, igamma, lam_max=4.0, delta_max=2.0, res=50, n_jobs=-1)
                print(f"Saved {out_file}")