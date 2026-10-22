"""A 3D density on a regular grid, with its MRC input and output."""
import numpy as np

from .libdisvis import binary_erosion
from .IO.mrc import to_mrc, parse_mrc


class Volume:
    """A 3D array with a voxel spacing and an origin (in Angstrom).

    The array is indexed as [z, y, x].
    """

    @classmethod
    def fromfile(cls, fid):
        """Read a volume from an MRC file (a file name or binary file)."""
        array, voxelspacing, origin = parse_mrc(fid)
        return cls(array, voxelspacing, origin)

    def __init__(self, array, voxelspacing=1.0, origin=(0, 0, 0)):

        self._array = array
        self._voxelspacing = voxelspacing
        self._origin = origin

    @property
    def array(self):
        """The 3D array of the volume, indexed as [z, y, x]."""
        return self._array

    @property
    def voxelspacing(self):
        """The distance between grid points in Angstrom."""
        return self._voxelspacing

    @voxelspacing.setter
    def voxelspacing(self, voxelspacing):
        self._voxelspacing = voxelspacing

    @property
    def origin(self):
        """The (x, y, z) coordinates of the first grid point in Angstrom."""
        return np.asarray(self._origin, dtype=np.float64)

    @origin.setter
    def origin(self, origin):
        self._origin = origin

    @property
    def shape(self):
        """The shape of the array, as (nz, ny, nx)."""
        return self.array.shape

    @property
    def dimensions(self):
        """The (x, y, z) size of the volume in Angstrom."""
        return [x * self.voxelspacing for x in self.shape][::-1]

    @property
    def start(self):
        """The origin in grid units."""
        return [x / self.voxelspacing for x in self.origin]

    @start.setter
    def start(self, start):
        self._origin = [x * self.voxelspacing for x in start]

    def duplicate(self):
        """Return a copy of the volume."""
        return Volume(self.array.copy(), voxelspacing=self.voxelspacing,
                      origin=self.origin)

    def tofile(self, fid):
        """Write the volume to an MRC file (a file name or binary file)."""
        to_mrc(fid, self)

# builders


def zeros(shape, voxelspacing, origin):
    """Return a volume of zeros with the given shape, spacing and origin."""
    return Volume(np.zeros(shape, dtype=np.float64), voxelspacing, origin)


def zeros_like(volume):
    """Return a volume of zeros with the same grid as volume."""
    return Volume(np.zeros_like(volume.array), volume.voxelspacing,
                  volume.origin)

# functions


def erode(volume, iterations, out=None):
    """Return the binary erosion of volume, repeated iterations times."""
    if out is None:
        out = zeros_like(volume)
        tmp = volume.array.copy()
        for _ in range(iterations):
            binary_erosion(tmp, out.array)
            tmp[:] = out.array[:]

    return out


def radix235(ninit):
    """Return the smallest integer >= ninit with only 2, 3 and 5 as factors.

    FFTs are fastest for such sizes.
    """
    if ninit < 1:
        raise ValueError(
            "Grid size should be at least 1, got {}.".format(ninit))
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
