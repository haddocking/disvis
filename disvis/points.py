from __future__ import division, absolute_import

import numpy as np
from .libdisvis import diluate_points as _dilate_points, dilate_points_add as _dilate_points_add

def dilate_points(points, radii, volume):
        
    ijkpoints = ((points - volume.origin)/volume.voxelspacing).astype(np.float64)
    ijkradii = (radii/volume.voxelspacing).astype(np.float64)

    _dilate_points(ijkpoints,
                   ijkradii,
                   volume.data.astype(np.float64))
    return volume

def dilate_points_add(points, radii, volume):
    
    ijkpoints = ((points - volume.origin)/volume.voxelspacing).astype(np.float64)
    ijkradii = (radii/volume.voxelspacing).astype(np.float64)

    _dilate_points_add(ijkpoints,
                       ijkradii,
                       volume.data.astype(np.float64))
    return volume
