import numpy as np
from qutip import *
from scipy.special import eval_genlaguerre, factorial
    
def build_UP(S_mat, g, dim_bos, n_spins):

    if dim_bos <= 1:
        n_sys = S_mat.shape[0]
        return Qobj(qeye(n_sys), dims=[[2]*n_spins,[2]*n_spins])

    eigvals, V = np.linalg.eigh(S_mat)
    Vdag = V.conj().T
    n_sys = S_mat.shape[0]
    I_b = qeye(dim_bos)

    UP0 = 0
    for k, lam in enumerate(eigvals):
        alpha_k = g * lam
        Dk = displace(dim_bos, alpha_k)
        e_k = np.zeros((n_sys,), dtype=complex); e_k[k] = 1.0
        proj_k = Qobj(np.outer(e_k, e_k.conj()), dims=[[2]*n_spins, [2]*n_spins])
        UP0 += tensor(Dk, proj_k)

    L = tensor(I_b, Qobj(V, dims=[[2]*n_spins, [2]*n_spins]))
    R = tensor(I_b, Qobj(Vdag, dims=[[2]*n_spins, [2]*n_spins]))

    UP = L * UP0 * R
    if dim_bos > 1:
        UP.dims = [[dim_bos] + [2]*n_spins, [dim_bos] + [2]*n_spins]
    else:
        UP.dims = [[2]*n_spins, [2]*n_spins]
    return UP

def H_DH_MN(M, N):

    def H_DH(Delta, Omega, lam, gamma_x, gamma_y, gamma_z, periodic=False, CDH=False):
        
        # Identity operators
        id_boson = qeye(M)
        id_spin = qeye(2)

        # Pauli matrices
        sx = sigmax()
        sy = sigmay()
        sz = sigmaz()
        
        # Initialize Hamiltonian
        H = 0

        if M > 1:
            # Bosonic creation and annihilation
            a = destroy(M)
            adag = create(M)
            # Zeeman term (Delta * sum of sigma_z)
            for i in range(N):
                ops = [id_boson] + [id_spin] * N
                ops[i + 1] = sz  # spin i (shifted by 1 due to boson at position 0)
                H += Delta * tensor(ops)

            #print(Qobj( adag.full() @ a.full()) )

            # Bosonic energy term: Omega * a†a
            ops = [ Qobj( adag.full() @ a.full())] + [id_spin] * N
            H += Omega * tensor(ops)

            # Coupling term: lam * (a + a†) * sum sigma_x
            for i in range(N):
                ops = [a + adag] + [id_spin] * N
                ops[i + 1] = sx
                H += lam/np.sqrt(N) * tensor(ops)

            # Spin-spin interaction terms
            for i in range(N - 1):
                for gamma, op in zip([gamma_x, gamma_y, gamma_z], [sx, sy, sz]):
                    ops = [id_boson] + [id_spin] * N
                    ops[i + 1] = op
                    ops[i + 2] = op
                    H -= gamma * tensor(ops)

            # Add periodic boundary condition terms
            if periodic and N > 2:
                for gamma, op in zip([gamma_x, gamma_y, gamma_z], [sx, sy, sz]):
                    ops = [id_boson] + [id_spin] * N
                    ops[1] = op          # spin 0
                    ops[-1] = op         # spin N-1
                    H -= gamma * tensor(ops)
                    
        elif M == 1:
            # Zeeman term (Delta * sum of sigma_z)
            for i in range(N):
                ops = [id_spin] * N
                ops[i] = sz  # spin i (shifted by 1 due to boson at position 0)
                H += Delta * tensor(ops)

            # Spin-spin interaction terms
            for i in range(N - 1):
                for gamma, op in zip([gamma_x, gamma_y, gamma_z], [sx, sy, sz]):
                    ops = [id_spin] * N
                    ops[i ] = op
                    ops[i + 1] = op
                    H -= gamma * tensor(ops)

            # Add periodic boundary condition terms
            if periodic and N > 2:
                for gamma, op in zip([gamma_x, gamma_y, gamma_z], [sx, sy, sz]):
                    ops = [id_spin] * N
                    ops[0] = op          # spin 0
                    ops[-1] = op         # spin N-1
                    H -= gamma * tensor(ops)

        return H
    return H_DH

def H_CDH_MN(M, N):
    def H_CDH(Delta, Omega, lam, gamma_x, gamma_y, gamma_z, periodic=False, CDH=True):
        
        M_big = 3

        # Identity operators
        id_boson = qeye(M_big)
        id_spin = qeye(2)

        # Pauli matrices
        sx = sigmax()
        sy = sigmay()
        sz = sigmaz()
        
        H = 0
        a = destroy(M_big)
        adag = create(M_big)
        # Zeeman term (Delta * sum of sigma_z)
        for i in range(N):
            ops = [id_boson] + [id_spin] * N
            ops[i + 1] = sz  # spin i (shifted by 1 due to boson at position 0)
            H += Delta * tensor(ops)

        #print(Qobj( adag.full() @ a.full()) )

        # Bosonic energy term: Omega * a†a
        ops = [ Qobj( adag.full() @ a.full())] + [id_spin] * N
        H += Omega * tensor(ops)

        # Coupling term: lam * (a + a†) * sum sigma_x
        for i in range(N):
            ops = [a + adag] + [id_spin] * N
            ops[i + 1] = sx
            H += lam/np.sqrt(N) * tensor(ops)

        # Spin-spin interaction terms
        for i in range(N - 1):
            for gamma, op in zip([gamma_x, gamma_y, gamma_z], [sx, sy, sz]):
                ops = [id_boson] + [id_spin] * N
                ops[i + 1] = op
                ops[i + 2] = op
                H -= gamma * tensor(ops)

        # Add periodic boundary condition terms
        if periodic and N > 2:
            for gamma, op in zip([gamma_x, gamma_y, gamma_z], [sx, sy, sz]):
                ops = [id_boson] + [id_spin] * N
                ops[1] = op          # spin 0
                ops[-1] = op         # spin N-1
                H -= gamma * tensor(ops)

        S = 0
        for i in range(N):
            ops_i = [qeye(2)] * N
            ops_i[i] = sigmax()
            S += tensor(ops_i)
        S = S / np.sqrt(N)
        UP = build_UP(S.full(), lam / Omega, M_big, N)
        #UP = ( tensor( create(M_big)-destroy(M_big), lam/Omega * S ) ).expm()
        H = UP @ H @ UP.dag()

        H = H.full().reshape(M_big,2**N,M_big,2**N)[:M,:,:M,:]
        H = Qobj( H.reshape(M*2**N,M*2**N) , dims=[[M] + [2]*N,[M] + [2]*N] )

        return H
    return H_CDH

# Effective Hamiltonian for M = 1

def f0(eps):
    return ( 1 + np.exp( -8*eps**2 ) )/2

def g0(eps):
    return ( 1 - np.exp( -8*eps**2 ) )/2

def H_EFF_1N(M, N):
    def H_EFF_1(Delta, Omega, lam, gamma_x, gamma_y, gamma_z, periodic=False, CDH=True):

        #lam = lam/np.sqrt(N)
        eps = lam / ( Omega * np.sqrt(N) )

        # Identity and Pauli operators
        id2 = qeye(2)
        sx = sigmax()
        sy = sigmay()
        sz = sigmaz()

        H = 0

        # Zeeman-like term with exponential factor
        for i in range(N):
            ops = [id2] * N
            ops[i] = sz
            H += Delta * np.exp(-2 * eps**2) * tensor(ops)

        # s^2 term all-to-all interaction
        for i in range(N):
            for j in range(N):
                ops = [id2] * N
                ops[i] *= sx
                ops[j] *= sx
                H -= (lam**2/ ( Omega * N ) ) * tensor(ops)

        # Nearest-neighbor spin-spin interactions
        for i in range(N - 1):
            # σ^x σ^x term
            ops_xx = [id2] * N
            ops_xx[i] = sx
            ops_xx[i + 1] = sx
            H -= gamma_x * tensor(ops_xx)

            # σ^y σ^y term
            ops_yy = [id2] * N
            ops_yy[i] = sy
            ops_yy[i + 1] = sy
            H -= gamma_y * f0(eps) * tensor(ops_yy)
            H -= gamma_z * g0(eps) * tensor(ops_yy)

            # σ^z σ^z term
            ops_zz = [id2] * N
            ops_zz[i] = sz
            ops_zz[i + 1] = sz
            H -= gamma_y * g0(eps) * tensor(ops_zz)
            H -= gamma_z * f0(eps) * tensor(ops_zz)

        # Add periodic boundary condition terms
        if periodic and N > 2:
            # σ^x σ^x
            ops_xx = [id2] * N
            ops_xx[0] = sx
            ops_xx[-1] = sx
            H -= gamma_x * tensor(ops_xx)

            # σ^y σ^y
            ops_yy = [id2] * N
            ops_yy[0] = sy
            ops_yy[-1] = sy
            H -= gamma_y * f0(eps) * tensor(ops_yy)
            H -= gamma_z * g0(eps) * tensor(ops_yy)

            # σ^z σ^z
            ops_zz = [id2] * N
            ops_zz[0] = sz
            ops_zz[-1] = sz
            H -= gamma_y * g0(eps) * tensor(ops_zz)
            H -= gamma_z * f0(eps) * tensor(ops_zz)

        return H
    return H_EFF_1

# Effective Hamiltonin for M = 2

def f1(eps):
    return ( 1 + np.exp(-8*eps**2) * ( 1 - 16*eps**2 ) )/2

def g1(eps):
    return ( 1 + np.exp(-8*eps**2) * ( -1 + 16*eps**2 ) )/2

def h(eps):
    return 2*eps*np.exp(-8*eps**2)

def ket(n,M):
    tmp = np.zeros(M)
    tmp[n] = 1
    return Qobj(tmp)

def KetBra(s, r, M):
    return ket(s,M)*ket(r,M).dag() * (1j)**(s-r) #RECENT FIX HAVE TO CONFIRM WITH ANAYTICS!

def H_EFF_2N(M, N):
    def H_EFF_2(Delta, Omega, lam, gamma_x, gamma_y, gamma_z, periodic=False, CDH=True):

        eps = lam / (Omega * np.sqrt(N) )
        M = 2  # fix bosonic subspace dimension

        id2 = qeye(2)
        idM = qeye(M)
        sx = sigmax()
        sy = sigmay()
        sz = sigmaz()

        # Basis state projectors in boson subspace
        P00 = KetBra(0, 0, M)
        P11 = KetBra(1, 1, M)
        P01 = KetBra(0, 1, M) + KetBra(1, 0, M)

        H = 0

        # Energy shift: Omega * |1><1|
        ops = [P11] + [id2] * N
        H += Omega * tensor(ops)

        # s^2 term all-to-all interaction
        for i in range(N):
            for j in range(N):
                ops = [idM] + [id2] * N
                ops[i+1] *= sx
                ops[j+1] *= sx
                H -= (lam**2/ ( Omega * N ) ) * tensor(ops)

        # Delta * exp(-2*eps^2) terms: acting on each spin
        for i in range(N):

            # sigmay coupling to P01
            ops = [P01] + [id2] * N
            ops[i + 1] = -2 * eps * sy
            H += Delta * np.exp(-2 * eps**2) * tensor(ops)

            # sigmaz * P00
            ops = [P00] + [id2] * N
            ops[i + 1] = sz
            H += Delta * np.exp(-2 * eps**2) * tensor(ops)

            # sigmaz * P11 with (1 - 4eps²)
            ops = [P11] + [id2] * N
            ops[i + 1] = (1 - 4 * eps**2) * sz
            H += Delta * np.exp(-2 * eps**2) * tensor(ops)

        # Heisenberg interaction terms
        for i in range(N - 1):

            # sx sx with P00 and P11
            ops = [idM] + [id2] * N
            ops[i + 1] = sx
            ops[i + 2] = sx
            H += -gamma_x * tensor(ops)

            # sy sy and sz sz with P00
            for op, fy, fz in [(sy, f0(eps), g0(eps)), (sz, g0(eps), f0(eps))]:
                ops = [P00] + [id2] * N
                ops[i + 1] = op
                ops[i + 2] = op
                H += -gamma_y * fy * tensor(ops)
                H += -gamma_z * fz * tensor(ops)

            # h(eps) * sy sz + sz sy with P01
            for A, B in [(sy, sz), (sz, sy)]:
                ops = [P01] + [id2] * N
                ops[i + 1] = A
                ops[i + 2] = B
                H += -(gamma_y - gamma_z) * h(eps) * tensor(ops)

            # sy sy and sz sz with P11
            for op, fy, fz in [(sy, f1(eps), g1(eps)), (sz, g1(eps), f1(eps))]:
                ops = [P11] + [id2] * N
                ops[i + 1] = op
                ops[i + 2] = op
                H += -gamma_y * fy * tensor(ops)
                H += -gamma_z * fz * tensor(ops)

        # Add periodic boundary condition terms
        if periodic and N > 2:
            i, j = N - 1, 0  # indices for periodic pair

            # h(eps) * sy sz + sz sy with P01
            for A, B in [(sy, sz), (sz, sy)]:
                ops = [P01] + [id2] * N
                ops[i + 1] = A
                ops[j + 1] = B
                H += -(gamma_y - gamma_z) * h(eps) * tensor(ops)

            # sx sx with P00 and P11
            ops = [idM] + [id2] * N
            ops[i + 1] = sx
            ops[j + 1] = sx
            H += -gamma_x * tensor(ops)

            # sy sy and sz sz with P00
            for op, fy, fz in [(sy, f0(eps), g0(eps)), (sz, g0(eps), f0(eps))]:
                ops = [P00] + [id2] * N
                ops[i + 1] = op
                ops[j + 1] = op
                H += -gamma_y * fy * tensor(ops)
                H += -gamma_z * fz * tensor(ops)

            # sy sy and sz sz with P11
            for op, fy, fz in [(sy, f1(eps), g1(eps)), (sz, g1(eps), f1(eps))]:
                ops = [P11] + [id2] * N
                ops[i + 1] = op
                ops[j + 1] = op
                H += -gamma_y * fy * tensor(ops)
                H += -gamma_z * fz * tensor(ops)

        return H
    return H_EFF_2

# DH effective for M = 3 

def f2(eps): # yy term from yy
    return 0.5 * ( 1 + np.exp(-8*eps**2)*( 1 - 32*eps**2 + 128*eps**4 ) )

def g2(eps): # zz term from yy 
    return 0.5 * ( 1 + np.exp(-8*eps**2)*( -1 + 32*eps**2 - 128*eps**4 ) )

def v(eps): # P21 term
    return 2*np.sqrt(2)*eps*(1-8*eps**2)*np.exp(-8*eps**2)

def w(eps): # P20 term for yy, zz is opposite sign
    return -4*np.sqrt(2)*eps**2*np.exp(-8*eps**2)

def H_EFF_3N(M, N):
    def H_EFF_3(Delta, Omega, lam, gamma_x, gamma_y, gamma_z, periodic=False, CDH=True):

        eps = lam / ( Omega * np.sqrt(N) )
        M = 3  # fix bosonic subspace dimension

        id2 = qeye(2)
        idM = qeye(M)
        sx = sigmax()
        sy = sigmay()
        sz = sigmaz()

        # Basis state projectors in boson subspace
        P00 = KetBra(0, 0, M)
        P11 = KetBra(1, 1, M)
        P22 = KetBra(2, 2, M)
        P01 = KetBra(0, 1, M) + KetBra(1, 0, M)
        P02 = KetBra(0, 2, M) + KetBra(2, 0, M)
        P12 = KetBra(2, 1, M) + KetBra(1, 2, M)

        H = 0

        # Energy shift: Omega * n
        ops = [P11 + 2*P22] + [id2] * N
        H += Omega * tensor(ops)

        # s^2 term all-to-all interaction
        for i in range(N):
            for j in range(N):
                ops = [idM] + [id2] * N
                ops[i+1] *= sx
                ops[j+1] *= sx
                H -= (lam**2/ ( Omega * N ) ) * tensor(ops)

        # Delta * exp(-2*eps^2) terms: acting on each spin
        for i in range(N):

            ops = [P00] + [id2] * N
            ops[i + 1] = sz
            H += Delta * np.exp(-2 * eps**2) * tensor(ops)

            ops = [P01] + [id2] * N
            ops[i + 1] = -2 * eps * sy
            H += Delta * np.exp(-2 * eps**2) * tensor(ops)

            ops = [P11] + [id2] * N
            ops[i + 1] = (1 - 4 * eps**2) * sz
            H += Delta * np.exp(-2 * eps**2) * tensor(ops)

            ops = [P22] + [id2] * N
            ops[i + 1] = (8*eps**4 - 8*eps**2 + 1) * sz
            H += Delta * np.exp(-2 * eps**2) * tensor(ops)

            ops = [P12] + [id2] * N
            ops[i + 1] = (2*np.sqrt(2)*eps*( 2*eps**2 - 1 )) * sy
            H += Delta * np.exp(-2 * eps**2) * tensor(ops)

            ops = [P02] + [id2] * N
            ops[i + 1] = -(2*np.sqrt(2)*eps**2) * sz
            H += Delta * np.exp(-2 * eps**2) * tensor(ops)

        # Heisenberg interaction terms
        for i in range(N - 1):

            # sx sx on the diagonal blocks
            ops = [idM] + [id2] * N
            ops[i + 1] = sx
            ops[i + 2] = sx
            H += -gamma_x * tensor(ops)

            # sy sy and sz sz on the diagonal
            for op, fy, fz in [(sy, f0(eps), g0(eps)), (sz, g0(eps), f0(eps))]:
                ops = [P00] + [id2] * N
                ops[i + 1] = op
                ops[i + 2] = op
                H += -gamma_y * fy * tensor(ops)
                H += -gamma_z * fz * tensor(ops)
            for op, fy, fz in [(sy, f1(eps), g1(eps)), (sz, g1(eps), f1(eps))]:
                ops = [P11] + [id2] * N
                ops[i + 1] = op
                ops[i + 2] = op
                H += -gamma_y * fy * tensor(ops)
                H += -gamma_z * fz * tensor(ops)
            for op, fy, fz in [(sy, f2(eps), g2(eps)), (sz, g2(eps), f2(eps))]:
                ops = [P22] + [id2] * N
                ops[i + 1] = op
                ops[i + 2] = op
                H += -gamma_y * fy * tensor(ops)
                H += -gamma_z * fz * tensor(ops)

            # off-diagnal blocks
            for A, B in [(sy, sz), (sz, sy)]:
                ops = [P01] + [id2] * N
                ops[i + 1] = A
                ops[i + 2] = B
                H += -(gamma_y - gamma_z) * h(eps) * tensor(ops)
            for A, B in [(sy, sz), (sz, sy)]:
                ops = [P12] + [id2] * N
                ops[i + 1] = A
                ops[i + 2] = B
                H += -(gamma_y - gamma_z) * v(eps) * tensor(ops)
            #for A, B in [(sy, sy), (-sz, sz)]:
            #    ops = [P02] + [id2] * N
            #    ops[i + 1] = A
            #    ops[i + 2] = B
            #    H += -(gamma_y - gamma_z) * w(eps) * tensor(ops)
            for op, fy, fz in [(sy, w(eps), -w(eps)), (sz, -w(eps), w(eps))]:
                ops = [P02] + [id2] * N
                ops[i + 1] = op
                ops[i + 2] = op
                H += -gamma_y * fy * tensor(ops)
                H += -gamma_z * fz * tensor(ops)

        # Add periodic boundary condition terms
        if periodic and N > 2:
            i, j = N - 1, 0  # indices for periodic pair

            # sx sx terms
            ops = [idM] + [id2] * N
            ops[i + 1] = sx
            ops[j + 1] = sx
            H += -gamma_x * tensor(ops)

            # off-diagonals
            for A, B in [(sy, sz), (sz, sy)]:
                ops = [P01] + [id2] * N
                ops[i + 1] = A
                ops[j + 1] = B
                H += -(gamma_y - gamma_z) * h(eps) * tensor(ops)
            for A, B in [(sy, sz), (sz, sy)]:
                ops = [P12] + [id2] * N
                ops[i + 1] = A
                ops[j + 1] = B
                H += -(gamma_y - gamma_z) * v(eps) * tensor(ops)
            #for A, B in [(sy, sy), (-sz, sz)]:
            #    ops = [P02] + [id2] * N
            #    ops[i + 1] = A
            #    ops[j + 1] = B
            #    H += -(gamma_y - gamma_z) * w(eps) * tensor(ops)
            for op, fy, fz in [(sy, w(eps), -w(eps)), (sz, -w(eps), w(eps))]:
                ops = [P02] + [id2] * N
                ops[i + 1] = op
                ops[j + 1] = op
                H += -gamma_y * fy * tensor(ops)
                H += -gamma_z * fz * tensor(ops)

            # sy sy and sz sz on the diagonals
            for op, fy, fz in [(sy, f0(eps), g0(eps)), (sz, g0(eps), f0(eps))]:
                ops = [P00] + [id2] * N
                ops[i + 1] = op
                ops[j + 1] = op
                H += -gamma_y * fy * tensor(ops)
                H += -gamma_z * fz * tensor(ops)
            for op, fy, fz in [(sy, f1(eps), g1(eps)), (sz, g1(eps), f1(eps))]:
                ops = [P11] + [id2] * N
                ops[i + 1] = op
                ops[j + 1] = op
                H += -gamma_y * fy * tensor(ops)
                H += -gamma_z * fz * tensor(ops)
            for op, fy, fz in [(sy, f2(eps), g2(eps)), (sz, g2(eps), f2(eps))]:
                ops = [P22] + [id2] * N
                ops[i + 1] = op
                ops[j + 1] = op
                H += -gamma_y * fy * tensor(ops)
                H += -gamma_z * fz * tensor(ops)

        return H
    return H_EFF_3