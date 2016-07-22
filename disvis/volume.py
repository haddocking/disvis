from __future__ import division

import numpy as np

from .libdisvis import binary_erosion
from .IO.mrc import to_mrc, parse_mrc

class Volume(object):

    @classmethod
    def fromfile(cls, fid):
        array, voxelspacing, origin = parse_mrc(fid)
        return cls(array, voxelspacing, origin)

    def __init__(self, array, voxelspacing=1.0, origin=(0, 0, 0)):

        self._array = array
        self._voxelspacing = voxelspacing
        self._origin = origin

    @property
    def array(self):
        return self._array

    @property
    def voxelspacing(self):
        return self._voxelspacing

    @voxelspacing.setter
    def voxelspacing(self, voxelspacing):
        self._voxelspacing = voxelspacing

    @property
    def origin(self):
        return np.asarray(self._origin, dtype=np.float64)

    @origin.setter
    def origin(self, origin):
        self._origin = origin

    @property
    def shape(self):
        return self.array.shape

    @property
    def dimensions(self):
        return [x*self.voxelspacing for x in self.shape][::-1]

    @property
    def start(self):
        return [x/self.voxelspacing for x in self.origin]

    @start.setter
    def start(self, start):
        self._origin = [x*self.voxelspacing for x in start]

    def duplicate(self):
        return Volume(self.array.copy(), voxelspacing=self.voxelspacing,
                      origin=self.origin)
    def tofile(self, fid):
        to_mrc(fid, self)

# builders
def zeros(shape, voxelspacing, origin):
    return Volume(np.zeros(shape, dtype=np.float64), voxelspacing, origin)

def zeros_like(volume):
    return Volume(np.zeros_like(volume.array), volume.voxelspacing, volume.origin)

# functions
def erode(volume, iterations, out=None):
    if out is None:
        out = zeros_like(volume)
        tmp = volume.array.copy()
        for i in range(iterations):
            binary_erosion(tmp, out.array)
            tmp[:] = out.array[:]

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
