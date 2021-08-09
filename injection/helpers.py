import numpy as np

from bunch import Bunch
from orbit.diagnostics import BunchMonitorNode
from orbit.diagnostics import add_analysis_nodes
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from orbit.teapot import teapot, TEAPOT_Lattice
from orbit.utils import helper_funcs as hf


def get_traj(lattice, init_coords, mass, kin_energy):
    """Return single particle trajectory through lattice."""
    bunch_, params_dict_ = hf.initialize_bunch(mass, kin_energy)
    x, xp, y, yp = init_coords
    bunch_.addParticle(x, xp, y, yp, 0.0, 0.0)
    monitor_nodes = add_analysis_nodes(BunchMonitorNode, lattice, dense=True, 
                                       transverse_only=True, mm_mrad=False)
    lattice.trackBunch(bunch_, params_dict_)
    coords, positions = [], []
    for monitor_node in monitor_nodes:
        coords.append(monitor_node.get_data(turn=0)[0])
        positions.append(monitor_node.position)
        monitor_node.clear_data()
    return np.array(coords), np.array(positions)


def track_part(lattice, init_coords, mass, kin_energy):
    """Return coords after tracking single particle through lattice."""
    bunch_, params_dict = hf.initialize_bunch(mass, kin_energy)
    x, xp, y, yp = init_coords
    bunch_.addParticle(x, xp, y, yp, 0.0, 0.0)
    lattice.trackBunch(bunch_, params_dict)
    return get_part_coords(bunch_)


def get_part_coords(bunch):
    """Return list of transverse particle coordinates from bunch."""
    return [bunch.x(0), bunch.xp(0), bunch.y(0), bunch.yp(0)]