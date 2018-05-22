import numpy as np

# Constants for defining the node map...
node_types = {
    'FLUID_NODE': np.int32(0),
    'WALL_NODE': np.int32(1),
    'NOT_IN_DOMAIN': np.int32(2),
    'FIXED_DENSITY': np.int32(3)
}

# Alleles are always defined as node types < 0.