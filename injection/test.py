from spacecharge import LSpaceChargeCalc
from orbit.space_charge.sc1d import SC1D_AccNode

b_a = 10.0 / 3.0
length = 248.0
use_spacecharge = 1
min_n_macros = 1000
n_long_slices = 64 
position = 124.0
Z = 32 * [complex(1., 0.)]
sc_node_long = SC1D_AccNode(b_a, length, min_n_macros, use_spacecharge, n_long_slices)
sc_node_long.assignImpedance(Z)

print('Done.')