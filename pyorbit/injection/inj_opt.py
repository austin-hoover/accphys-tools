kicker_names = ['ikickh_a10', 'ikickv_a10', 'ikickh_a11', 'ikickv_a11',
                'ikickh_a12', 'ikickv_a12', 'ikickh_a13', 'ikickv_a13']


def get_kicker_nodes(lattice):
    return [lattice.getNodeForName(name) for name in kicker_names]


def set_kicker_strengths(kicker_nodes, kicker_strengths):
    kicker_nodes[0].setParam('kx', kicker_strengths[0])
    kicker_nodes[1].setParam('ky', kicker_strengths[1])
    kicker_nodes[2].setParam('kx', kicker_strengths[2])
    kicker_nodes[3].setParam('ky', kicker_strengths[3])
    kicker_nodes[4].setParam('kx', kicker_strengths[4])
    kicker_nodes[5].setParam('ky', kicker_strengths[5])
    kicker_nodes[6].setParam('kx', kicker_strengths[6])
    kicker_nodes[7].setParam('ky', kicker_strengths[7])
    
    
def penalty(kicker_strengths, kicker_nodes, monitor_nodes, desired_foil_coords):
    """Penalty function for injection region closed orbit.
    
    1. Must have zero slope at the start, foil, and end of the
       injection region.
    2. Must pierce the center of the foil and return to x = y = 0 
       at the start/end of the injection region.
           
    Note: in PyORBIT the closed orbit is fixed -- it is the trajectory of a
    particle with zero initial displacement and zero slope at the entrance of 
    the first kicker when the kickers are turned off.
           
    Parameters
    ----------
    kicker_strengths : ndarray, shape (8,)
        Strengths of kickers [k_hkicker10, k_vkicker10, k_hkicker11,
        k_vkicker11, k_hkicker13, k_vkicker13, k_hkicker12, k_vkicker12] in
        mrad.
    kicker_nodes : list[KickTEAPOT]
        List of injection kicker nodes.
    monitor_nodes : list[AnalysisNode]
        List of AnalysisNodes which monitor the bunch coordinates. The first 
        element gives the coordinates at the foil, and the second element
        gives the coordinates at the end of the injection region.
    desired_foil_coords : ndarray, shape (4,)
        The desired [x, x', y, y'] coordinates at the foil.
        
    Returns
    -------
    float
        Sum of squared differences between desired and calculated coordinates 
        at the foil.
    """
    set_kicker_strengths(kicker_nodes, kicker_strengths)
    for node in monitor_nodes:
        node.clear_data()
    bunch, params_dict = hf.initialize_bunch(mass, kin_energy)
    bunch.addParticle(0., 0., 0., 0., 0., 0.)
    inj_region.trackBunch(bunch, params_dict)
    foil_coords, end_coords = [node.get_data('bunch_coords')[0] 
                               for node in monitor_nodes]
    return 1e6 * sum(end_coords**2 + (foil_coords - desired_foil_coords)**2)


def set_coords_at_foil(coords, guess=None, max_nfev=int(1e4), verbose=2):
    """Set (x, x', y, y') at the foil using the mini-lattice."""
    if guess is None:
        guess = 8 * [0.0]
    lb = -np.inf
    ub = +np.inf
    result = opt.least_squares(penalty, guess, bounds=(lb, ub), 
                               args=(inj_kicker_nodes, inj_monitor_nodes, coords), 
                               max_nfev=max_nfev, verbose=verbose)
    inj_kicker_strengths = result.x
    return inj_kicker_strengths