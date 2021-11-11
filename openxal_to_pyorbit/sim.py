from __future__ import print_function
from orbit.teapot import teapot
from orbit.utils import helper_funcs as hf
from orbit.utils.consts import mass_proton


lattice = teapot.TEAPOT_Lattice()
lattice.readMADX('lattice.lat', 'ring_ring')

mass = mass_proton # [GeV/c^2]
kin_energy = 1.0 # [GeV]
print(hf.twiss_at_entrance(lattice, mass, kin_energy))
print(hf.get_tunes(lattice, mass, kin_energy))
