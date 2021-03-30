"""
This script reconstructs the covariance matrix given a list of transfer matrix
elements and corresponding measured beam moments.

The following files are assumed to be in './output/':
    'transfer_mat_elems_i.dat'
        Transfer matrix at each wire-scanner for scan index i. There will be
        one row per wire-scanner in the order [ws02, ws20, ws21, ws23, ws24].
        Each row lists the 16 elements of the transfer matrix in the order [00,
        01, 02, 03, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33].
    'moments_i.dat'
        Real space beam moments at each wire-scanner for scan index i. There 
        will be one row per wire-scanner in the order [ws02, ws20, ws21, ws23,
        ws24]. Each row lists [<xx>, <yy>, <xy>].
        
It is theoretically possible to perform the least-squares fit using the 
OpenXAL Solver class, but so far I haven't gotten it to work. Unlike 
scipy.optimize.lsq_linear, it requires an initial guess of the solution. 
"""
import numpy as np
import scipy.optimize as opt


# Collect data
nscans = 12

def read_file(filename):
    """Read file and return data in list of shape (nlines, ncols)."""
    return [[float(token) for token in line.rstrip().split()] 
            for line in open(filename, 'r')]

transfer_mat_elems_list, moments_list = [], []
for i in range(1, nscans + 1):
    filename = 'output/transfer_mat_elems_{}.dat'.format(i)
    for transfer_mat_elems in read_file(filename):
        transfer_mat_elems_list.append(transfer_mat_elems)
    for moments in read_file('output/moments_{}.dat'.format(i)):
        moments_list.append(moments)

# Form coefficient and observation arrays
A, b = [], []
for transfer_mat_elems, moments in zip(transfer_mat_elems_list, moments_list):
    (M11, M12, M13, M14, M21, M22, M23, M24, 
     M31, M32, M33, M34, M41, M42, M43, M44) = transfer_mat_elems
    sig_xx, sig_yy, sig_xy = moments
    A.append([M11**2, M12**2, 2*M11*M12, 0, 0, 0, 0, 0, 0, 0])
    A.append([0, 0, 0, M33**2, M34**2, 2*M33*M34, 0, 0, 0, 0])
    A.append([0, 0, 0, 0, 0, 0, M11*M33,  M12*M33,  M11*M34,  M12*M23])
    b.append(sig_xx)
    b.append(sig_yy)
    b.append(sig_xy)
A, b = np.array(A), np.array(b)
print 'A.shape =', A.shape
print 'b.shape =', b.shape

# Fit moments
A, b = np.array(A), np.array(b)
lb = 10 * [-np.inf]
ub = 10 * [+np.inf]
lb[0] = lb[1] = lb[3] = lb[4] = 0.0 # squared moments can't be negative
result = opt.lsq_linear(A, b, bounds=(lb, ub), verbose=2)
s11, s22, s12, s33, s44, s34, s13, s23, s14, s24 = 1e6 * result.x
Sigma = np.array([[s11, s12, s13, s14], 
                  [s12, s22, s23, s24],
                  [s13, s23, s33, s34],
                  [s14, s24, s34, s44]])

# Print results
ex = np.sqrt(np.linalg.det(Sigma[:2, :2]))
ey = np.sqrt(np.linalg.det(Sigma[2:, 2:]))
bx = Sigma[0, 0] / ex
by = Sigma[2, 2] / ey
ax = -Sigma[0, 1] / ex
ay = -Sigma[2, 3] / ey

calc_twiss = {'ax':ax, 'bx':bx, 'ex':ex, 'ay':ay, 'by':by, 'ey':ey}
true_twiss = {'ax':-1.378, 'ay':0.645, 'ex':20.0, 'bx': 6.243, 'by':10.354, 'ey':20.0}
for key in true_twiss.keys():
    print('{} true, calc = {:.3f}, {:.3f}'.format(key, true_twiss[key], calc_twiss[key]))
    
    
## Attempt with OpenXAL solver
# from lib.utils import least_squares
# lb = 10 * [-1000.]
# ub = 10 * [+1000.]
# lb[0] = lb[1] = lb[3] = lb[4] = 0.0 # squared moments can't be negative
# x0 = [200, 20, 0, 200, 20, 0, 0, 0, 0, 0]
# sigma = least_squares(A, b, x0, bounds=(lb, ub))