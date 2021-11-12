def get_skew_quad_nodes(lattice):
    """MADX model of SNS ring as nodes that start with 'qsc'. 
    These combined function skew quad correctors. I dont' know the power supply limits.
    """
    skew_quad_nodes = []
    for node in lattice.getNodes():
        if node.getName().startswith('qsc'):
            node.setParam('skews', [0, 1])
            skew_quad_nodes.append(node)
    return skew_quad_nodes
        
def set_skew_quad_strengths(skew_quad_nodes, skew_quad_strengths):
    for node, strength in zip(skew_quad_nodes, skew_quad_strengths):
        node.setParam('kls', [0.0, strength])