from __future__ import division, absolute_import, print_function

import numpy as np
from .libdisvis import dilate_points as _dilate_points, dilate_points_add as _dilate_points_add

def dilate_points(points, radii, volume):
        
    ijkpoints = (float64_array(points) - float64_array(volume.origin))/volume.voxelspacing
    ijkradii = float64_array(radii)/volume.voxelspacing

    _dilate_points(ijkpoints,
                   ijkradii,
                   volume.array)
    return volume

def dilate_points_add(points, radii, volume):
    
    ijkpoints = (float64_array(points) - float64_array(volume.origin))/volume.voxelspacing
    ijkradii = float64_array(radii)/volume.voxelspacing

    _dilate_points_add(ijkpoints,
                       ijkradii,
                       volume.array)
    return volume

def float64_array(array_like):
    return np.asarray(array_like, dtype=np.float64)
