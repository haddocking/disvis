from __future__ import division

import numpy as np

from .IO.mrc import to_mrc, parse_mrc
from ._extensions import dilate_points


class Volume(object):
    @classmethod
    def fromfile(cls, fid):
        array, voxelspacing, origin = parse_mrc(fid)
        return cls(array, voxelspacing, origin)

    @classmethod
    def zeros(cls, shape, voxelspacing=1, origin=(0, 0, 0), dtype=np.float64):
        return cls(np.zeros(shape, dtype=dtype), voxelspacing, origin)

    @classmethod
    def zeros_like(cls, volume, dtype=None):
        if dtype is None:
            dtype = volume.array.dtype
        return cls(np.zeros_like(volume.array, dtype=dtype), volume.voxelspacing,
                   volume.origin)

    def __init__(self, array, voxelspacing=1.0, origin=(0, 0, 0)):
        self.array = array
        self.voxelspacing = voxelspacing
        self.origin = np.asarray(origin, dtype=np.float64)
        self.shape = array.shape

    @property
    def dimensions(self):
        return [x * self.voxelspacing for x in self.shape][::-1]

    @property
    def start(self):
        return [x / self.voxelspacing for x in self.origin]

    @start.setter
    def start(self, start):
        self.origin = [x * self.voxelspacing for x in start]

    def duplicate(self):
        return Volume(self.array.copy(), voxelspacing=self.voxelspacing,
                      origin=self.origin.copy())

    def tofile(self, fid):
        to_mrc(fid, self)


class Volumizer(object):
    """Create volumes or shapes from a receptor and ligand."""

    def __init__(self, receptor, ligand, voxelspacing=1,
                 interaction_radius=3):
        self.receptor = receptor
        self.ligand = ligand
        self.voxelspacing = voxelspacing
        self.interaction_radius = interaction_radius
        longest_distance = np.linalg.norm(
            ligand.coor - ligand.center, axis=1).max()
        bottom_left = (receptor.coor.min(axis=0) -
                       longest_distance - interaction_radius)
        top_right = (receptor.coor.max(axis=0) +
                     longest_distance + interaction_radius)
        self.shape = [closest_multiple(int(np.ceil(x)))
                      for x in (top_right - bottom_left)[::-1] / self.voxelspacing]
        self.origin = bottom_left

        self.rcore = Volume(np.zeros(self.shape, dtype=np.float64),
                            self.voxelspacing, self.origin)
        self.rsurface = Volume.zeros_like(self.rcore)
        self.lcore = Volume.zeros_like(self.rcore)

        receptor_coor_grid = np.ascontiguousarray((self.receptor.coor - self.origin).T / self.voxelspacing)
        receptor_radii = self.receptor.vdw_radius / self.voxelspacing
        dilate_points(receptor_coor_grid, receptor_radii, 1, self.rcore.array)
        receptor_radii += interaction_radius / self.voxelspacing
        dilate_points(receptor_coor_grid, receptor_radii, 1, self.rsurface.array)

        self._ligand_coor_grid = (self.ligand.coor - self.ligand.center).T / self.voxelspacing
        self._ligand_coor_grid_rot = np.ascontiguousarray(self._ligand_coor_grid.copy())
        self._ligand_radii = self.ligand.vdw_radius / self.voxelspacing

    def generate_lcore(self, rotmat):
        self.lcore.array.fill(0)
        np.dot(rotmat, self._ligand_coor_grid, out=self._ligand_coor_grid_rot)
        dilate_points(self._ligand_coor_grid_rot, self._ligand_radii, 1,
                      self.lcore.array)


def closest_multiple(ninit, multiples=(2, 3, 5, 7)):
    while True:
        n = ninit
        divided = True
        while divided:
            divided = False
            for radix in multiples:
                quot, rem = divmod(n, radix)
                if not rem:
                    n = quot
                    divided = True
        if n != 1:
            ninit += 1
        else:
            return ninit
