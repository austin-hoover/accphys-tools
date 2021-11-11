# Toggle skew quads. These are skew quad correctors, technically. 
def get_skew_quad_nodes(ring):
    skew_quad_nodes = []
    for node in ring.getNodes():
        name = node.getName()
        if name.startswith('qsc'):
            node.setParam('skews', [0, 1])
            skew_quad_nodes.append(node)
    return skew_quad_nodes
        
def set_skew_quad_strengths(skew_quad_nodes, skew_quad_strengths):
    for node, strength in zip(skew_quad_nodes, skew_quad_strengths):
        node.setParam('kls', [0.0, strength])
        
if switches['skew quads']:
    skew_quad_nodes = get_skew_quad_nodes(ring)
    env_skew_quad_nodes = get_skew_quad_nodes(env_ring)
    skew_quad_strengths = np.zeros(len(skew_quad_nodes))
    skew_quad_strengths[12] = 0.1
    set_skew_quad_strengths(skew_quad_nodes, skew_quad_strengths)
    set_skew_quad_strengths(env_skew_quad_nodes, skew_quad_strengths)