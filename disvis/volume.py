from __future__ import division
import numpy as np
from scipy.ndimage import binary_erosion
from .libdisvis import rotate_image3d

class Volume(object):

    def __init__(self, data, voxelspacing=1.0, origin=(0, 0, 0)):

        self._data = data
        self._voxelspacing = voxelspacing
        self._origin = origin

    @property
    def data(self):
        return self._data

    @property
    def voxelspacing(self):
        return self._voxelspacing
    @voxelspacing.setter
    def voxelspacing(self, voxelspacing):
        self._voxelspacing = voxelspacing

    @property
    def origin(self):
        return self._origin
    @origin.setter
    def origin(self, origin):
        self._origin = origin

    @property
    def shape(self):
        return self.data.shape[::-1]

    @property
    def dimensions(self):
        return [x*self.voxelspacing for x in self.shape]

    @property
    def start(self):
        return [x/self.voxelspacing for x in self.origin]
    @start.setter
    def start(self, start):
        self._origin = [x*self.voxelspacing for x in start]

    def duplicate(self):
        return Volume(self.data.copy(), voxelspacing=self.voxelspacing,
                      origin=self.origin)

# builders
def zeros_like(volume):
    return Volume(np.zeros_like(volume.data), volume.voxelspacing, volume.origin)

# functions
def erode(volume, iterations, out=None):
    if out is None:
        out = zeros_like(volume)
    binary_erosion(volume, iterations=iterations, output=out.data)
    return out

def radix235(ninit):
    while True:
        n = ninit
        divided = True
        while divided:
            divided = False
            for radix in (2, 3, 5):
                quot, rem = divmod(n, radix)
                if not rem:
                    n = quot
                    divided = True
        if n != 1:
            ninit += 1
        else:
            return ninit

def rotate_volume(volume, vlength, rotmat, out=None):
    if out is None:
        out = zeros_like(volume)

    rotate_image3d(volume, int(vlength/self.voxelspacing), rotmat, out)

    return out
