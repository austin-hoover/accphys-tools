import sys
import numpy as np
import pandas as pd

from bunch import Bunch
from orbit.lattice import AccLattice
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.twiss import twiss
from orbit.utils import helper_funcs as hf


def unpack(tracked_twiss):
    data = [[s, nu, alpha, beta] for (s, nu), (s, alpha), (s, beta) in zip(*tracked_twiss)]
    return np.array(data).T

    
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV]

lattice = teapot.TEAPOT_Lattice()
lattice.readMADX('optics/rtbt.lat', 'rtbt_rtbt')
lattice.set_fringe(False)
init_twiss = {'alpha_x': -1.3168000000000002,
              'alpha_y': 0.6830999999999999,
              'beta_x': 5.847100000000001,
              'beta_y': 9.260699999999998}

# Save default optics
bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
lattice.split(0.1)
matlat = TEAPOT_MATRIX_Lattice(lattice, bunch)
tracked_twiss_x = matlat.trackTwissData(init_twiss['alpha_x'], init_twiss['beta_x'], direction='x')
tracked_twiss_y = matlat.trackTwissData(init_twiss['alpha_y'], init_twiss['beta_y'], direction='y')
pos, nu_x, alpha_x, beta_x = unpack(tracked_twiss_x)
pos, nu_y, alpha_y, beta_y = unpack(tracked_twiss_y)
tracked_twiss = np.vstack([pos, nu_x, nu_y, alpha_x, alpha_y, beta_x, beta_y]).T
twiss_df = pd.DataFrame(tracked_twiss, columns=['s','nux','nuy','ax','ay','bx','by'])
twiss_df[['nux','nuy']] %= 1
twiss_df.to_csv('twiss.dat', index=False)


1.6960377354292564