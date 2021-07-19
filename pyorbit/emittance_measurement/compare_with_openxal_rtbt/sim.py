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

    
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV]
Brho = hf.get_Brho(mass, kin_energy) # magnetic rigidity

lattice = teapot.TEAPOT_Lattice()
lattice.readMADX('optics/rtbt.lat', 'whole1')
lattice.set_fringe(False)

# Get OpenXAL node names and magnet strengths [T/m]. Included in this list are
# all quadrupoles, quadrupole correctors, and dipole correctors. 
file = open('./optics/rtbt_live.dat', 'r')
xal_node_names, xal_vals = [], []
for line in file:
    node_name, val = line.rstrip().split(',')
    node_name = node_name.lower().split(':')[1]
    xal_node_names.append(node_name)
    xal_vals.append(float(val))
file.close()

# Store PyORBIT node names


# Modify lattice according to XAL optics. 
node_names = [node.getName() for node in lattice.getNodes()]
for xal_node_name, xal_val in zip(xal_node_names, xal_vals):
    # Dipole correctors (kickers)
    if xal_node_name.startswith('dc') or xal_node_name.startswith('ekick'):
        node = lattice.getNodeForName(xal_node_name)
        field_times_length = xal_val
        kick_angle = field_times_length / Brho
        if xal_node_name.startswith('dch'): 
            node.setParam('kx', kick_angle)
        elif xal_node_name.startswith('dcv'):
            node.setParam('ky', kick_angle)
    # Change 'qh' and 'qv' to 'q'
    if xal_node_name.startswith('qh') or xal_node_name.startswith('qv'):
        xal_node_name = xal_node_name[0] + xal_node_name[2:]
    # The first quad is split into two nodes in PyORBIT.
    if xal_node_name == 'q01': 
        for node_name in ['q01s1', 'q01s2']:
            node = lattice.getNodeForName(node_name)
            if node.getType() == 'quad teapot':
                node.setParam('kq', xal_val / Brho)    
    else:
        node = lattice.getNodeForName(xal_node_name)
        if node.getType() == 'quad teapot':
            Brho = hf.get_Brho(mass, kin_energy)
            node.setParam('kq', xal_val / Brho)
            

# Expected Twiss at RTBT entrance 
alpha_x, alpha_y, beta_x, beta_y = [-0.25897, 0.9749, 2.2991, 14.2583]

# Save default optics
bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
lattice.split(0.1)
matlat = TEAPOT_MATRIX_Lattice(lattice, bunch)
tracked_twiss_x = matlat.trackTwissData(alpha_x, beta_x, direction='x')
tracked_twiss_y = matlat.trackTwissData(alpha_y, beta_y, direction='y')

def unpack(tracked_twiss):
    data = [[s, nu, alpha, beta] for (s, nu), (s, alpha), (s, beta) in zip(*tracked_twiss)]
    return np.array(data).T

pos, nu_x, alpha_x, beta_x = unpack(tracked_twiss_x)
pos, nu_y, alpha_y, beta_y = unpack(tracked_twiss_y)
tracked_twiss = np.vstack([pos, nu_x, nu_y, alpha_x, alpha_y, beta_x, beta_y]).T

twiss_df = pd.DataFrame(tracked_twiss, columns=['s','nux','nuy','ax','ay','bx','by'])
twiss_df[['nux','nuy']] %= 1
twiss_df.to_csv('twiss.dat', index=False)