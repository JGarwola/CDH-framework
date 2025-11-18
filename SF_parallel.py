import numpy as np
from qutip import *
from joblib import Parallel, delayed
from get_Hamiltonians import H_DH_MN, H_EFF_1N, H_EFF_2N, H_EFF_3N, build_UP

def compute_structure_factor(N):
    gammas = np.array([0.25, 0.25, 0.0])
    ops = [sigmax(), sigmay(), sigmaz()]
    Hs = [H_DH_MN,] #, H_EFF_1N, H_EFF_2N, H_EFF_3N]
    lams = np.linspace(0, 4, 100)
    Structure_factors = np.zeros([100, 3, 4])

    for k, op in enumerate(ops):
        for iH, H in enumerate(Hs):
            M = [50, 1, 2, 3][iH]
            HMN = H(M=M, N=N)

            # Parallelize over lambda
            def compute_for_lambda(iL, lam):
                GS = HMN(Delta=1, Omega=2, lam=lam,
                         gamma_x=0.25, gamma_y=0.25, gamma_z=0.0,
                         periodic=True).groundstate()[1]

                Sx = 0
                if iH != 0:
                    for i in range(N):
                        ops_tmp = [qeye(2)] * N
                        ops_tmp[i] = sigmax() / np.sqrt(N)
                        Sx += tensor(ops_tmp)
                    UP = build_UP(Sx.full(), lam/2, M, N)
                else:
                    if M > 1:
                        ops_tmp = [qeye(M)] + [qeye(2)] * N
                        UP = tensor(ops_tmp)
                    else:
                        UP = qeye(2**N)  # pure spin system, no boson

                S_sum = 0
                for i in range(N):
                    for j in range(N):
                        if M > 1:
                            ops_tmp = [qeye(M)] + [qeye(2)] * N
                            ops_tmp[i+1] = op
                            ops_tmp[j+1] = op
                        else:
                            ops_tmp = [qeye(2)] * N
                            ops_tmp[i] = op
                            ops_tmp[j] = op

                        UP.dims = tensor(ops_tmp).dims
                        S_sum += np.real(expect(UP @ tensor(ops_tmp) @ UP.dag(), GS) / N**2)

                return iL, S_sum

            # Compute for all λ in parallel
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(compute_for_lambda)(iL, lam) for iL, lam in enumerate(lams)
            )
            for iL, S_val in results:
                Structure_factors[iL, k, iH] = S_val

    np.save(f'data_new/structure_factors_N{N}.npy', Structure_factors)
    return N


if __name__ == "__main__":
    Ns = [2,4,6,8]
    Parallel(n_jobs=len(Ns))(
        delayed(compute_structure_factor)(N) for N in Ns
    )