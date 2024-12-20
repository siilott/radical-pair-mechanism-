import math
import functools
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import qutip as qt
import time
from scipy.integrate import ode
from scipy.spatial import distance
from itertools import product
import multiprocessing 

start_time = time.time()
# This dictionary maps string keys ('x', 'y', 'z', 'p', 'm', 'i') to functions that generate spin operators for a given dimension dim.
opstr2fun = {'x': lambda dim: qt.spin_Jx((dim-1)/2),
             'y': lambda dim: qt.spin_Jy((dim-1)/2),
             'z': lambda dim: qt.spin_Jz((dim-1)/2),
             'p': lambda dim: qt.spin_Jp((dim-1)/2),
             'm': lambda dim: qt.spin_Jm((dim-1)/2),
             'i': qt.identity}

# Initializes ops as a list of identity matrices for each dimension in dims. Iterates over specs to replace the identity matrix at the specified index with the corresponding spin operator. Returns the tensor product of the operators in ops using qt.tensor.
def mkSpinOp(dims, specs):
    ops = [qt.identity(d) for d in dims]
    for ind, opstr in specs:
        ops[ind] = ops[ind] * opstr2fun[opstr](dims[ind])
    return qt.tensor(ops)
    
# Constructs a Hamiltonian for a single spin system with interactions along the x, y, and z axes.
def mkH1(dims, ind, parvec):
    axes = ['x', 'y', 'z']
    # Creates a list of spin operators weighted by the corresponding parameters in parvec (ignores zero parameters). Uses functools.reduce to sum these weighted spin operators.
    return functools.reduce(lambda a, b: a + b, 
               [v * mkSpinOp(dims, [(ind,ax)]) for v, ax in zip(parvec, axes) if v!=0])

# Constructs a Hamiltonian for the interaction between two spin systems with interaction terms along all combinations of x, y, and z axes.
def mkH12(dims, ind1, ind2, parmat):
    axes = ['x', 'y', 'z']
    ops = []
    parmat  = np.array(parmat)
    # Iterates over all combinations of the x, y, and z axes for the two spins. For each non-zero element in parmat, adds the corresponding spin-spin interaction term to the empty list ops.
    for i in range(3):
        for j in range(3):
            if parmat[i,j] != 0:
                ops.append(parmat[i,j] * mkSpinOp(dims, [(ind1,axes[i]), (ind2,axes[j])]))
    return functools.reduce(lambda a, b: a + b, ops) # Uses functools.reduce to sum these interaction terms.


# N5_C =  2*np.pi* np.array([[-0.36082693, -0.0702137 , -1.41518116],
#       [-0.0702137 , -0.60153649,  0.32312139],
#       [-1.41518116,  0.32312139, 50.80213093]]) # in MHz
	  
# N1_C = 2*np.pi*np.array([[  2.13814981,   3.19255832,  -2.48895215],
#       [  3.19255832,  15.45032887, -12.44778343],
#       [ -2.48895215, -12.44778343,  12.49532827]]) # in MHz

# N5_D =  2*np.pi*np.array([[-2.94412424e-01, -5.68059200e-02, -1.02860888e+00],
#       [-5.68059200e-02, -5.40578469e-01, -2.67686240e-02],
#       [-1.02860888e+00, -2.67686240e-02,  5.05815320e+01]]) # in MHz
	  
# N1_D = 2*np.pi* np.array([[ 0.98491908,  3.28010265, -0.53784491],
#       [ 3.28010265, 25.88547678, -1.6335986 ],
#       [-0.53784491, -1.6335986 ,  1.41368001]]) # in MHz

# ErC_Dee =  np.array([[ 26.47042689, -55.90357828,  50.1679204 ],
#                             [-55.90357828, -20.86385225,  76.13493805],
#                              [ 50.1679204,  76.13493805,  -5.60657464]]) # in Mrad/s

# ErD_Dee = np.array([[ 11.08087889, -34.6687169,   12.14623706],
#                             [-34.6687169,  -33.09039672,  22.36229081],
#                             [ 12.14623706,  22.36229081,  22.00951783]]) #  in Mrad/s

#For FAD
N5 = [[-2.84803, 0.0739994, -1.75741],
[0.0739994, -2.5667, 0.326813],
[-1.75741, 0.326813, 53.686]]

N10 = [[-0.0979402, 0.00195169, 1.80443],
[0.00195169, -0.513124, -0.508695],
[1.80443, -0.508695, 19.109]]

#For Trp
N1 = [[-1.94218, -0.0549954, -0.21326],
[-0.0549954, -2.29723, -0.441875],
[-0.21326, -0.441875, 19.156]]

H1 = [[-2.14056, 6.31534, 0.17339],
[6.31534, -18.9038, -0.0420204],
[0.17339, -0.0420204, -14.746]]

# Function to sample points on a Fibonacci sphere
def fibonacci_sphere(samples):
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle in radians
    xyz = []
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # Radius at y
        theta = phi * i  # Golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        xyz.append([x, y, z])
    return np.array(xyz)

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

def compute_zxz_rotation_tensor(orientation):
    psi = orientation[0]
    theta = orientation[1]
    phi = orientation[2]
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

def moser_dutton_rate(r):
    delta_G = 0
    lam = 0.2
    A = 13.0
    B = 0.4
    C=3.1
    R0=3.6

    """
    Calculate the electron transfer rate using the Moser-Dutton ruler.
    
    Parameters:
    - delta_G : float : Free energy difference (ΔG°, in eV)
    - r       : float : Distance between donor and acceptor (Å)
    - lam     : float : Reorganization energy (λ, in eV) [default = 0.2-1.5]
    - A       : float : Distance of optimal electron transfer (Å) [default = 13-15]
    - B       : float : Decay constant (Å^-1) [default = beta/ 2.303] 
    - C       : float : Quantized nuclear term (eV^-1) [default = 3.1] (incorrect terminology, check literature)
    - R0      : float : van der Waals contact distance (Å) [default = 3.6]

    Returns:
    - k_ET    : float : Electron transfer rate (s^-1)
    """

    # Ensure inputs are within reasonable physical limits
    # if r <= R0:
    #     raise ValueError("Distance(r) must be greater than van der Waals contact distance (R0).")

    # Calculate distance-dependent term
    distance_term = A - B * (r - R0)
    
    # Calculate energy-dependent term
    energy_term = -C * (delta_G + lam) ** 2 /lam 
    
    # Combine terms to calculate the rate
    log_k_ET = distance_term + energy_term
    k_ET = 10 ** log_k_ET  # Convert from log10 to actual rate

    return k_ET

# Function to calculate the electron transfer rate for the different position vectors 
def electron_transfer_rates(parameters):
    FAD_rs = parameters['FAD_rs'] 
    Trp_rs = parameters['Trp_rs']
    TrpC_orientation = parameters['TrpC_orientation']
    TrpD_orientation = parameters['TrpD_orientation']
    TrpC_d= parameters['TrpC_d']
    TrpD_d = parameters['TrpD_d']

    # Compute the rotation tensor for TrpC and TrpD
    TrpD_R = compute_zxz_rotation_tensor(TrpD_orientation)
    TrpC_R = compute_zxz_rotation_tensor(TrpC_orientation)

    TrpC_rs = []
    TrpD_rs = []
    # Compute the position vectors for each contribution 
    for Trp_r  in Trp_rs:
        TrpC_r = TrpC_d + TrpC_R.T @ Trp_r
        TrpC_rs.append(TrpC_r)
        TrpD_r = TrpD_d + TrpD_R.T @ Trp_r
        TrpD_rs.append(TrpD_r)

    # Compute the minimum distances 
    FWc_rmins = distance.cdist(FAD_rs, TrpC_rs).min(axis=1)
    WcWd_rmins = distance.cdist(TrpC_rs, TrpD_rs).min(axis=1)
    # closest_indices_FWc = FWc_rmins.argmin()
    # closest_indices_WcWd  = WcWd_rmins.argmin()
    # print(FWc_rmins)
    # print(WcWd_rmins)

    # Calculate the transfer rates using Moser-Dutton function 
    kCfs = []
    closest_indices_FWc = []
    for FWc_rmin in FWc_rmins:
        kCf = moser_dutton_rate(FWc_rmin)
        kCfs.append(kCf)
        closest_indice_FWc = FWc_rmins.argmin()
        closest_indices_FWc.append(closest_indice_FWc)

    kDfs = []
    closest_indices_WcWd = []
    for WcWd_rmin in  WcWd_rmins:
        kDf = moser_dutton_rate(WcWd_rmin)
        kDfs.append(kDf)
        closest_indice_WcWd = WcWd_rmins.argmin()
        closest_indices_WcWd.append(closest_indice_WcWd)

    # Prepare results for return
    closest_FWc_data = [
        {"FAD_r": FAD_rs[i], "TrpC_r": TrpC_rs[idx], "FWc_rmin": dist,  "kCf": kCfs[i]}
        for i, (dist, idx) in enumerate(zip(FWc_rmins, closest_indices_FWc))
    ]

    closest_WcWd_data = [
        {"TrpC_r": TrpC_rs[i], "TrpD_r": TrpD_rs[idx], "WcWd_rmin": dist, "kDf": kDfs[i]}
        for i, (dist, idx) in enumerate(zip(WcWd_rmins, closest_indices_WcWd))
    ]

    # Return all calculated data
    return {
        "closest_FWc_data": closest_FWc_data,
        "closest_WcWd_data": closest_WcWd_data,
    }

# Function to perform the simulation
def run_simulation(parameters):
    b0 = parameters['b0']
    krC = parameters['krC']
    krD = parameters['krD']
    kf = parameters['kf']
    kCD = parameters['kCD']
    kDC = parameters['kDC']
    num_orientation_samples = parameters['num_orientation_samples']
    dims = parameters['dims'] # Dimensions of system components (2 qubits, 1 spin-1 nucleus) 
    TrpC_orientation = parameters['TrpC_orientation']
    TrpD_orientation = parameters['TrpD_orientation']
    FWc_rmin = parameters['FWc_rmin'] 
    WcWd_rmin = parameters['WcWd_rmin']
    kCf = parameters['kCf']
    kDf = parameters['kDf']

    # Generate orientations on a Fibonacci sphere
    oris = fibonacci_sphere(num_orientation_samples)
    
    # Convert Cartesian coordinates to latitude and longitude
    num_points = len(oris)
    lat = np.zeros(num_points)
    lon = np.zeros(num_points)

    for i in range(num_points):
        x, y, z = oris[i]
        lat[i] = np.arcsin(z) * (180 / np.pi)
        lon[i] = np.arctan2(y, x) * (180 / np.pi)
    
    dim = np.prod(dims)  # Total dimension of the composite system

    # # Define the magnetic field vectors for each direction
    # Bx = [b0, 0, 0]  # Magnetic field in the x direction
    # By = [0, b0, 0]  # Magnetic field in the y direction
    # Bz = [0, 0, b0]  # Magnetic field in the z direction

    # # Store them in a list
    # B_fields = [Bx, By, Bz]

    B_fields = []
    
    for orientation in oris:
        B0 = b0 * orientation  # Magnetic field vector along orientation
        B_fields.append(B0)  

    Ps = 1/4 * mkSpinOp(dims,[]) - mkH12(dims, 0, 1, np.identity(3))  # Singlet projection operator

    rho0_C = (Ps / Ps.tr()).full().flatten()  # Initial density matrix for singlet state
    rho0_D = np.zeros_like(rho0_C)
    initial_state = np.concatenate((rho0_C, rho0_D)).flatten()
    Ps = Ps.data.as_scipy()

    # Compute the rotation tensor for TrpC and TrpD
    TrpD_R = compute_zxz_rotation_tensor(TrpD_orientation)
    TrpC_R = compute_zxz_rotation_tensor(TrpC_orientation)

    def mesolve(t, combined_rho, P_s, HA, HB, dimA, dimB):
        # Reshape rho back to a matrix
        lenA = dimA * dimA
        lenB = dimB * dimB
        rhoA = combined_rho[:lenA].reshape((dimA, dimA))
        rhoB = combined_rho[lenB:].reshape((dimB, dimB))
        
        # Compute the derivative of rho
        drhoA_dt = -1j * (HA @ rhoA - rhoA @ HA) - krC/2*(P_s @ rhoA + rhoA @ P_s) - (kCD+kCf)*rhoA + kDC*rhoB
        drhoB_dt = -1j * (HB @ rhoB - rhoB @ HB) - krD/2*(P_s @ rhoB + rhoB @ P_s) - (kDC+kDf)*rhoB + kCD*rhoA
        
        # Flatten the derivative to a vector
        return np.concatenate((drhoA_dt.flatten(), drhoB_dt.flatten()))

    yr_c_list = []  # List to store singlet yield for component C
    yr_d_list = []  # List to store singlet yield for component D
    
    N1_rotated_C = TrpC_R.T @ N1 @ TrpC_R
    N1_rotated_D = TrpD_R.T @ N1 @ TrpD_R

    ErTrpC_Dee = point_dipole_dipole_coupling(FWc_rmin)
    ErTrpD_Dee = point_dipole_dipole_coupling(WcWd_rmin)
   
    for field in B_fields:
        #Compute Hamiltonians for each orientation
        Hzee = mkH1(dims, 0, field) + mkH1(dims, 1, field)  # Zeeman Hamiltonian for two spins
        Hhfc_C = mkH12(dims, 0, 2, N5) + mkH12(dims, 1, 3, N1_rotated_C)
        Hhfc_D = mkH12(dims, 0, 2, N5) + mkH12(dims, 1, 4, N1_rotated_D)
        Hdee_C = mkH12(dims, 0, 1, ErTrpC_Dee)
        Hdee_D = mkH12(dims, 0, 1, ErTrpD_Dee)
        H0_C = Hzee + Hhfc_C + Hdee_C  # Total Hamiltonian for component C
        H0_D = Hzee + Hhfc_D + Hdee_D  # Total Hamiltonian for component D
        H_C = H0_C.data.as_scipy()
        H_D = H0_D.data.as_scipy()

        # Create the solver instance
        solver = ode(mesolve).set_integrator('zvode', atol=1e-7, rtol=1e-6, method='adams', order=12)
        solver.set_initial_value(initial_state, 0).set_f_params(Ps, H_C, H_D, dim, dim)
        
        t = [(0., 1., 0.)]
        dt = 0.001
        tmax = 12. / kf  # Maximum time in microseconds

        while solver.successful() and solver.t < tmax:
            rho = solver.integrate(solver.t + dt)
            rho_c = rho[:dim**2].reshape((dim, dim))
            rho_d = rho[dim**2:].reshape((dim, dim))
            t.append((solver.t, np.trace(Ps @ rho_c), np.trace(Ps @ rho_d)))

        # Convert lists to arrays
        tlist = np.array([x for x, y, z in t])
        ps_c = np.array([np.real(y) for x, y, z in t])
        ps_d = np.array([np.real(z) for x, y, z in t])
        
        # Compute yields
        yr_c = krC * sci.integrate.simpson(ps_c, x=tlist)
        yr_d = krD * sci.integrate.simpson(ps_d, x=tlist)
        
        yr_c_list.append(yr_c)
        yr_d_list.append(yr_d)

    # Plot results
    # plotlat = lat.tolist()
    # plotlon = lon.tolist()
    plotyc = yr_c_list
    plotyd = yr_d_list

    max_yield = max(plotyc)+max(plotyd)
    min_yield = min(plotyc)+min(plotyd)
    # total_yield_x = plotyc[0]+plotyd[0]
    # total_yield_y = plotyc[1] + plotyd[1]
    # total_yield_z = plotyc[2]+ plotyd[2]
    avg_yield = sum(plotyc+plotyd)/len(plotyc)
    compass_sensitivity = max_yield - min_yield 
    chi = compass_sensitivity / avg_yield 
    # print('total yield x=', total_yield_x)
    # print('total yeild y=', total_yield_y)
    # print('total yeild z=', total_yield_z)
    # print('compass sensitivity = ', compass_sensitivity) 
    # print('chi = ', chi)
    print([min_yield, max_yield, avg_yield])

if __name__ == "__main__":
    # Base parameter set
    params = {
        'b0': 1.4 * 2 * np.pi,  # Zeeman field strength in radians per microsecond
        'krC': 5.0,             # Default values, will be overridden
        'krD': 0,
        'kf': 1.0,
        'kCD': 60255.95860743581, # change the number of points so that there a slightly more points for one of the rate constants than the other 
        'kDC': 10e5,
        'dims': [2, 2, 2, 2, 2],  # Dimensions of system components (2 qubits, 1 spin-1 nucleus)
        'num_orientation_samples': 10,      # Number of samples (unused here, just an example)
        'FAD_rs':[[1.05272,	0.474844,	9.61309*10**-17],
        [0.349471,	-0.685166,	-0.0232927],
        [1.05867,	-1.91279,	-0.0166991],
        [0.429269,	-3.15141,	-0.0697218],
        [1.24726,	-4.42354,	-0.0596015],
        [-0.982216,	-3.19343,	-0.149678],
        [-1.72019,	-4.50846,	-0.253066],
        [-1.70664,	-1.99109,	-0.150443],
        [-1.07843,	-0.744145,	-0.0631673],
        [-1.79304,	0.474844,	0.],
        [-1.09023,	1.68232,	-0.0234944],
        [-1.81808,	2.81296,	0.0106907],
        [-1.78693,	5.10798,	0.0711242],
        [0.210804,	4.01406,	0.021478],
        [1.05059,	2.89559,	-0.0125263],
        [2.27577,	3.0514,	-0.019355],
        [0.329372,	1.62462,	-0.0232927]],  # Coordinates of core (i.e. ring) atoms of FAD cantered on centre of spin density and aligned to molecular axes
        'Trp_rs':[[1.77314,	0.978905,	0.0187266],
        [0.700867,	1.77992,	0.0377116],
        [-0.493628,	0.999096,	0.010858],
        [-0.0625267,	-0.363978,	0.],
        [-0.743141,	-1.60399,	0.0289536],
        [0.0136988,	-2.78355,	0.0446603],
        [1.40985,	-2.74558,	0.030763],
        [2.11628,	-1.51743,	0.012728],
        [1.36023,	-0.363978,	0.]], # Coordinates of core (i.e. ring) atoms of Trp cantered on centre of spin density and aligned to molecular axes (first column: atomic number; columns 2 to 4: x, y, and z-coordinates in Angstroms)
        'TrpC_orientation': np.array([119.982, 129.967, 353.895]), # Euler angles to be used in rotation tensors 
        'TrpD_orientation': np.array([70.,	81.5553, 1.12128]),
        'TrpC_d': np.array([10.1746,-13.3164,5.18675]), # dislacement vector for TrpD
        'TrpD_d': np.array([9.21606,-18.14,3.32885]) # displacemet vector for TrpC
    }
    transfer_rates = electron_transfer_rates(params)
    print(transfer_rates)

    # Get all combinations of kCfs and kDfs
    closest_FWc_data = transfer_rates['closest_FWc_data']
    closest_WcWd_data = transfer_rates['closest_WcWd_data']

    combinations = list(product(closest_FWc_data, closest_WcWd_data))

    #Prepare the yields array (len(kCDs) x len(kDCs) x 3)
    yields = np.zeros((len(closest_FWc_data['kCf']), len(closest_WcWd_data), 3))

    #Create a list of parameter combinations for multiprocessing
    parameter_combinations = [{**{'b0': params['b0'], 'krC': params['krC'], 'krD': params['krD'], 'kf': params['kf'],
                        'dims': params['dims'], 'num_orientation_samples': params['num_orientation_samples'],
                            'kCD': params['kCD'], 'kDC': params['kDC'], 'TrpC_orientation':params['TrpC_orientation'], 'TrpD_orientation': params['TrpD_orientation'], 'TrpC_d': params['TrpC_d'], 'TrpD_d': params['TrpD_d']}, **fwc, **wcwd} for fwc, wcwd in combinations]
    
    # Run simulations in parallel using multiprocessing
    with multiprocessing.Pool() as pool:
        results = pool.map(run_simulation, parameter_combinations)

    # Store the results in the transfer rates array
    for idx, combination in enumerate(parameter_combinations):
        i = combination.get("kCf")
        j = combination.get("kDf")
        k = combination.get("FWc_rmin")
        l = combination.get("WcWd_rmin")
        yields[i, j, k, l, :] = results[idx]


    # #Save the results to a .npz file
    np.savez('output.npz', kCFs == kCfs, kDfs == kDfs, yields=yields)
    print("--- %s seconds ---" % (time.time() - start_time))