import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import functools
import numpy as np
import scipy.sparse as sparse
import scipy.optimize as optimize
import qutip as qt
import multiprocessing

# small, quick funtions to easily generate spin operators
opstr2fun = {'x': lambda dim: qt.spin_Jx((dim-1)/2),
             'y': lambda dim: qt.spin_Jy((dim-1)/2),
             'z': lambda dim: qt.spin_Jz((dim-1)/2),
             'p': lambda dim: qt.spin_Jp((dim-1)/2),
             'm': lambda dim: qt.spin_Jm((dim-1)/2),
             'i': qt.identity}

def mkSpinOp(dims, specs): # creates spin projection operator for the system
    ops = [qt.identity(d) for d in dims] # one matrix per particle
    for ind, opstr in specs:
        ops[ind] = ops[ind] * opstr2fun[opstr](dims[ind]) # get projection operator for each particle
    return qt.tensor(ops) # tensor product of all operators, then return the resulting tensor

def mkH1(dims, ind, parvec): # hamilitonian for interactions with magnetic field
    axes = ['x', 'y', 'z']
    return functools.reduce(lambda a, b: a + b, # dot product of magnetic field with spin operators
               [v * mkSpinOp(dims, [(ind,ax)]) for v, ax in zip(parvec, axes) if v!=0])

def mkH12(dims, ind1, ind2, parmat): # hamiltonian for interaction between particles
    axes = ['x', 'y', 'z']
    ops = []
    for i in range(3): # for each axis for particle 1
        for j in range(3): # for each axis for particle 2
            if parmat[i,j] != 0:
                ops.append(parmat[i,j] * mkSpinOp(dims, [(ind1,axes[i]), (ind2,axes[j])]))
                # gives each spin projection matrix, multiplied by the appropriate coupling coefficient
    return functools.reduce(lambda a, b: a + b, ops) # sum all operators and return the result


def fibonacci_sphere(samples=1000):

    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians
    
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append((x, y, z))

    return np.array(points)


def signalling_yield_laplace(H1eff, H2eff, rho0, kf, k12, k21):
    dim = H1eff.shape[0]
    dim2 = dim*dim
    
    I = sparse.eye(dim, format="csr")
    II = sparse.eye(dim2, format="csr")
    L1 = 1j * (sparse.kron(H1eff,I) - sparse.kron(I,H1eff.conj())) + k12 * II
    L2 = 1j * (sparse.kron(H2eff,I) - sparse.kron(I,H2eff.conj())) + k21 * II
    L = sparse.vstack([sparse.hstack([L1, -k21*II]), sparse.hstack([-k12*II, L2])],format="csr")
    
    rho = sparse.linalg.spsolve(L,rho0)
    rho1 = np.reshape(rho[:dim2], (dim, dim))
    rho2 = np.reshape(rho[dim2:], (dim, dim))
    y1 = kf * np.real(np.trace(rho1))
    y2 = kf * np.real(np.trace(rho2))

    return y1+y2, y1, y2


def calc_yields(sys, oris):

    dims = sys['dims']
    dim = np.prod(dims)
    dim2 = dim**2

    kf = sys['kf']
    kr1 = sys['kr1']
    kr2 = sys['kr2']
    k12 = sys['k12']
    k21 = sys['k21']
    b0 = sys['b0']
    
    Ps = 1/4 * mkSpinOp(dims,[]) - mkH12(dims, 0, 1, np.identity(3))
    # Pt = mkSpinOp(dims,[]) - Ps
    
    Hhfc1 = sum(mkH12(dims, i, j, A)  for i, j, A in sys['As'][0])
    Hhfc2 = sum(mkH12(dims, i, j, A)  for i, j, A in sys['As'][1])
    Heed1 = mkH12(dims, 0, 1, sys['eed'][0])
    Heed2 = mkH12(dims, 0, 1, sys['eed'][1])
    K1 = kr1/2 * Ps + kf/2 * mkSpinOp(dims,[])
    K2 = kr2/2 * Ps + kf/2 * mkSpinOp(dims,[])
    H01eff = (Hhfc1 + Heed1 - 1j*K1)
    H02eff = (Hhfc2 + Heed2 - 1j*K2)

    rho10 = (Ps / Ps.tr()).full()
    rho20 = np.zeros(dim2)
    rho0 = np.hstack([np.reshape(rho10, -1), rho20])
 
    yields = []
    for ori in oris:
        B0 = ori * b0
        Hzee = mkH1(dims, 0, B0) + mkH1(dims, 1, B0)
        H1eff = H01eff + Hzee
        H2eff = H02eff + Hzee
        qy12, qy1, qy2 = signalling_yield_laplace(H1eff.data.as_scipy(), H2eff.data.as_scipy(), rho0, kf, k12, k21)
        yields.append((qy12, qy1, qy2))

    return yields

def calc_yields_parallel(sys, oris, nprocs=1):

    if nprocs == 1:
        return calc_yields(sys, oris)
    
    f = functools.partial(calc_yields, sys)
    with multiprocessing.Pool(processes = nprocs) as pool:
        y = pool.map(f, np.array_split(oris, nprocs)) 
    yields = np.vstack(y)

    return yields


def calc_mfe(sys, oris, nprocs=1):
    yields = calc_yields_parallel(sys, oris, nprocs=nprocs)
    y12 = np.array([y for y, _, _ in yields])
    return (np.min(y12), np.mean(y12), np.max(y12))


def point_dipole_dipole_coupling(r):
 
    dr3 = -4*np.pi*1e-7 * (2.0023193043617 * 9.27400968e-24)**2 / (4*np.pi*1e-30)/6.62606957e-34/1e6 # MHz * A^3
 
    if np.isscalar(r):
        # assume r is aligned with z
        d = dr3 / r**3
        A = np.diag([-d, -d, 2*d])
    else:
        norm_r = np.linalg.norm(r)
        d = dr3 / norm_r**3
        e = r / norm_r
        A = d * (3 * e[:,np.newaxis] * e[np.newaxis,:] - np.eye(3))
 
    return A


def rotation_matrix_zxz(psi, theta, phi):
    def Rx(gamma):
        return np.array([[ 1, 0           , 0           ],
                        [ 0, np.cos(gamma),-np.sin(gamma)],
                        [ 0, np.sin(gamma), np.cos(gamma)]])
    
    def Rz(gamma):
        return np.array([[ np.cos(gamma), -np.sin(gamma), 0 ],
                        [ np.sin(gamma), np.cos(gamma) , 0 ],
                        [ 0           , 0            , 1 ]])

    R = Rz(psi) @ Rx(theta) @ Rz(phi)
    return R


###################################################################################################


def mfe(p):

    log_kr1, log_kr2, log_k12, log_k21 = p

    N5 = np.array([[-2.84803, 0.0739994, -1.75741],
            [0.0739994, -2.5667, 0.326813],
            [-1.75741, 0.326813, 53.686]]) * 2*np.pi

    N1 = np.array([[-1.94218, -0.0549954, -0.21326],
                [-0.0549954, -2.29723, -0.441875],
                [-0.21326, -0.441875, 19.156]]) * 2*np.pi

    oriC = np.array([119.982, 129.967, 353.895]) / 180*np.pi
    dC = [10.1746,-13.3164,5.18675]

    oriD = np.array([70.9499, 81.5553, 1.12128]) / 180*np.pi
    dD = [9.21606,-18.14,3.32885]

    rotC = rotation_matrix_zxz(*oriC)
    rotD = rotation_matrix_zxz(*oriD)

    N1C = rotC.T @ N1 @ rotC
    N1D = rotD.T @ N1 @ rotD

    eedC = point_dipole_dipole_coupling(dC) * 2*np.pi
    eedD = point_dipole_dipole_coupling(dD) * 2*np.pi

    sys = {
        'b0': 2*np.pi * 1.4,
        'kr1': 10**log_kr1,
        'kr2': 10**log_kr2,
        'kf': 1,
        'k12': 10**log_k12,
        'k21': 10**log_k21,
        'dims': [2,2, 2,2,2],
        'As': [[(0,2,N5), (1,3,N1C)], [(0,2,N5), (1,4,N1D)]], # 0:e 1:e 2:N5 3:N1C 4:N1D
        'eed': [eedC, eedD]
    }

    oris = fibonacci_sphere(144) # !!!
    ys = calc_mfe(sys, oris, nprocs=16) # !!!
    delta = ys[2] - ys[0]
    # print(p, delta)

    return delta


def fun(p, info):
    delta = mfe(p)

    if info['Nfeval']%5 == 0:
        print(("{:4d}   " + "   ".join(["{:3.6f}" for _ in range(len(p)+1)])).format(info['Nfeval'], *p, delta))
    info['Nfeval'] += 1

    return -delta


result_file = "run_opt01c_results.txt"

beta = 1.4 # 1/A
R = 3.77 # A
dG = 0
lam = 0.3
kBT = 0.0270584264 # eV, 314 K 
log_k12max = 13 - beta/np.log(10) * (R - 3.6) - 3.1 * (dG + lam)**2/lam - 6 # -> 1/us
log_k21max = log_k12max + dG/kBT
bounds = [(-2, 2), (-5, 2), (-2, log_k12max), (-2, log_k21max)]

repeats = 25
for n in range(repeats):
    # x0 = np.array([lb + np.random.rand() * (ub - lb) for ub, lb in bounds])
    x0 = np.array([lb + np.random.rand() * (ub - lb) for ub, lb in [(0,2), (-3,0), (2,5), (2,5)]])

    res = optimize.minimize(
        fun,
        x0=x0,
        bounds=bounds,
        args=({'Nfeval':0},), 
        method='nelder-mead',
        options={'xatol': 1e-5, 'fatol': 1e-7, 'disp': True, 'maxiter': 400} 
    )

    print(res.x)

    with open(result_file, "a") as f:
        f.write(" ".join([str(v) for v in res.x] + [str(v) for v in (-res.fun, res.status, res.nit, res.nfev)]) + "\n")


