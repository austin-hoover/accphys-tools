import sys
import numpy as np
from tqdm import tqdm, trange

from bunch import Bunch
from orbit.lattice import AccLattice
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.twiss import twiss
from orbit.utils import helper_funcs as hf

    
mass = 0.93827231 # [GeV/c^2]
kin_energy = 1.0 # [GeV]


lattice = teapot.TEAPOT_Lattice()
lattice.readMADX('SNSring_nux6.23_nuy6.20.lat', 'rnginj')
lattice.set_fringe(False)



# Get OpenXAL node names and magnet strengths [T/m]. Included in this list are
# all quadrupoles, quadrupole correctors, and dipole correctors. 
file = open('./optics/ring_quads_model.dat', 'r')
xal_node_names, xal_fields = [], []
for line in file:
    node_name, field = line.rstrip().split(',')
    node_name = node_name.lower().split(':')[1]
    xal_node_names.append(node_name)
    xal_fields.append(float(field))
file.close()

# Store the PyORBIT lattice node names.
node_names = [node.getName() for node in lattice.getNodes()]


for i, xal_node_name in enumerate(xal_node_names):
    if xal_node_name not in node_names:
        
        # In the PyORBIT lattice we have `dchv` which is a combined 
        # horizontal/vertical dipole corrector. In OpenXAL, these are two 
        # separate nodes (`dch` and `dcv`). We need to store both of these
        # and edit the single PyORBIT node.
        if xal_node_name.startswith('dc'):
            if xal_node_name.startswith('dch'): # horizontal
                pass
            elif xal_node_name.startswith('dcv'): # vertical
                pass
      
        print 'Could not find', xal_node_name
        break        
        
        
    print xal_node_name

        
# for node in lattice.getNodes():
#     print node.getName(), node.getType()
# print hf.twiss_at_injection(lattice, mass, kin_energy)
# print hf.get_tunes(lattice, mass, kin_energy)
