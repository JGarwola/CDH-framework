import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
from qutip import *
from joblib import Parallel, delayed
from get_Hamiltonians import H_DH_MN, H_EFF_1N, H_EFF_2N, H_EFF_3N, build_UP


def order_params(Delta, lam, gamma_z, H, M, N, periodic, model):
    Omega = 2
    gamma_x = 0
    gamma_y = 0

    HMN = H(M, N)
    H_full = HMN(Delta, Omega, lam, gamma_x, gamma_y, gamma_z, periodic)
    eigvals, eigvecs = H_full.eigenstates()
    ground_energy = eigvals[0]
    tol = 1e-10

    for val, state in zip(eigvals, eigvecs):
        if np.abs(val - ground_energy) < tol:
            GS = state
            break

    if model != 0 and M > 1:
        S = np.zeros((2**N, 2**N), dtype=complex)
        for i in range(N):
            ops_i = [qeye(2)] * N
            ops_i[i] = sigmax()
            S += tensor(ops_i).full()
        S = S / np.sqrt(N)
        UP = build_UP(S, lam / Omega, M, N)
    else:
        ops = [qeye(M)] + [qeye(2)] * N
        UP = tensor(ops)

    Mz = 0
    for i in range(N):
        if M > 1:
            ops = [qeye(M)] + [qeye(2)] * N
            ops[i + 1] = sigmaz() / N
        else:
            ops = [qeye(2)] * N
            ops[i] = sigmaz() / N
        UP.dims = tensor(ops).dims
        Mz += expect(UP @ tensor(ops) @ UP.dag(), GS)

    return np.real(Mz)


def compute_phase_diagram(N, model):
    M = [20, 1, 2, 3][model]
    H_list = [H_DH_MN, H_EFF_1N, H_EFF_2N, H_EFF_3N]
    H = H_list[model]

    lam_max = 2
    gamma_z_max = 2
    Delta = 1
    res = 50
    lam_vals = np.linspace(0, lam_max, res)
    gamma_vals = np.linspace(-gamma_z_max, gamma_z_max/2, res)

    # Use limited number of workers, process-based backend
    results = Parallel(n_jobs=4, backend="loky", verbose=5)(
        delayed(order_params)(Delta, lam, gamma_z, H, M, N, True, model)
        for lam in lam_vals
        for gamma_z in gamma_vals
    )

    Z = np.array(results).reshape((res, res))
    np.save(f"data_new/Jachym_DH_model{model}_N{N}_M{M}.npy", Z)
    return model, N


if __name__ == "__main__":
    Ns = [4]
    models = [0, 1, 3]
    # Run outer loop sequentially to avoid nested parallelism
    for N in Ns:
        for model in models:
            compute_phase_diagram(N, model)