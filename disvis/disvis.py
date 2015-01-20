from __future__ import print_function, absolute_import, division

import numpy as np
from numpy.fft import rfftn, irfftn
from pyfftw.interfaces.numpy_fft import rfftn, irfftn

from math import floor

from disvis import volume
from .volume import radix235
from .points import dilate_points
from .libdisvis import rotate_image3d, dilate_points_add


class DisVis(object):

    def __init__(self):
        # parameters to be defined
        self._receptor = None
        self._ligand = None

        # parameters with standard values
        self.rotations = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
        self.voxelspacing = 1.0
        self.erosion_iterations = 2
        self.surface_radius = 2.5
        self.max_clash = 25
        self.min_interaction = 100
        self.distance_restraints = []

        # unchangeable
        self._data = {}

    @property
    def receptor(self):
        return self._receptor

    @receptor.setter
    def receptor(self, receptor):
        self._receptor = receptor.duplicate()

    @property
    def ligand(self):
        return self._ligand

    @ligand.setter
    def ligand(self, ligand):
        self._ligand = ligand.duplicate()

    @property
    def rotations(self):
        return self._rotations
    @rotations.setter
    def rotations(self, rotations):
        self._rotations = rotations

    @property
    def surface_radius(self):
        return self._surface_radius
    @surface_radius.setter
    def surface_radius(self, radius):
        self._surface_radius = radius

    @property
    def voxelspacing(self):
        return self._voxelspacing

    @voxelspacing.setter
    def voxelspacing(self, voxelspacing):
        self._voxelspacing = voxelspacing

    @property
    def max_clash(self):
        return self._max_clash
    @max_clash.setter
    def max_clash(self, max_clash):
        self._max_clash = max_clash

    @property
    def min_interaction(self):
        return self._min_interaction
    @min_interaction.setter
    def min_interaction(self, min_interaction):
        self._min_interaction = min_interaction
        
    @property
    def erosion_iterations(self):
        return self._erosion_iterations
    @erosion_iterations.setter
    def erosion_iterations(self, erosion_iterations):
        self._erosion_iterations = erosion_iterations

    @property
    def data(self):
        return self._data

    def add_distance_restraint(self, receptor_selection, ligand_selection, distance):
        distance_restraint = [receptor_selection, ligand_selection, distance]
        self.distance_restraints.append(distance_restraint)

    def _initialize(self):

        # check if requirements are set
        if any(x is None for x in (self.receptor, self.ligand)):
            raise ValueError("Not all requirements are met for a search")

        d = self.data

        # determine size for grid
        shape = grid_shape(self.receptor.coor, self.ligand.coor, self.voxelspacing)
        # calculate the interaction surface and core of the receptor
        radii = np.zeros(self.receptor.coor.shape[0], dtype=np.float64)
        radii.fill(self.surface_radius)
        d['rsurf'] = rsurface(self.receptor.coor, radii, shape, self.voxelspacing)
        d['rsurf'].tofile('rsurf.mrc')
        d['rcore'] = volume.erode(d['rsurf'], self.erosion_iterations)

        # keep track of some data for later calculations
        d['origin'] = d['rcore'].origin
        d['shape'] = d['rcore'].shape
        d['start'] = d['rcore'].start

        # ligand center is needed for distance calculations during search
        d['lcenter'] = self.ligand.center

        # set ligand center to the origin of the receptor map
        # and make a grid of the ligand
        radii = np.zeros(self.ligand.coor.shape[0], dtype=np.float64)
        radii.fill(self.surface_radius)
        d['lsurf'] = dilate_points((self.ligand.coor - self.ligand.center + d['rcore'].origin), radii, volume.zeros_like(d['rcore']))

        # setup the distance restraints
        d['nrestraints'] = len(self.distance_restraints)
        if self.distance_restraints:
            d['restraints'] = grid_restraints(self.distance_restraints, self.voxelspacing, d['origin'], d['lcenter'])

    def search(self):
        self._initialize()
        self._cpu_init()
        self._cpu_search()

        accessible_interaction_space = volume.Volume(self.data['accessible_interaction_space'], self.voxelspacing, self.data['origin'])

        return accessible_interaction_space, self.data['accessible_complexes']

    def _cpu_init(self):

        self.cpu_data = {}
        c = self.cpu_data
        d = self.data

        c['rcore'] = d['rcore'].array
        c['rsurf'] = d['rsurf'].array
        c['im_lsurf'] = d['lsurf'].array

        c['lsurf'] = np.zeros_like(c['rcore'])
        c['clashvol'] = np.zeros_like(c['rcore'])
        c['intervol'] = np.zeros_like(c['rcore'])
        c['interspace'] = np.zeros_like(c['rcore'])
        c['access_interspace'] = np.zeros_like(c['rcore'])
        c['restspace'] = np.zeros_like(c['rcore'])

        # complex arrays
        c['ft_shape'] = list(d['shape'])
        c['ft_shape'][-1] = d['shape'][-1]//2 + 1
        c['ft_lsurf'] = np.zeros(c['ft_shape'], dtype=np.complex128)
        c['ft_rcore'] = np.zeros(c['ft_shape'], dtype=np.complex128)
        c['ft_rsurf'] = np.zeros(c['ft_shape'], dtype=np.complex128)

        # initial calculations
        c['ft_rcore'][:] = rfftn(c['rcore'])
        c['ft_rsurf'][:] = rfftn(c['rsurf'])
        c['rotmat'] = np.asarray(self.rotations, dtype=np.float64)
        c['weights'] = np.asarray(self.weights, dtype=np.float64)

    def _cpu_search(self):

        d = self.data
        c = self.cpu_data

        c['vlength'] = int(np.linalg.norm(self.ligand.coor - self.ligand.center, axis=1).max() + self.surface_radius + 1)/self.voxelspacing
        tot_complex = 0
        list_total_allowed = np.zeros(max(2, d['nrestraints'] + 1), dtype=np.float64)

        for n in xrange(c['rotmat'].shape[0]):
            # rotate ligand image
            print(n)
            rotate_image3d(c['im_lsurf'], c['vlength'], c['rotmat'][n], c['lsurf'])

            c['ft_lsurf'][:] = rfftn(c['lsurf']).conj()
            c['clashvol'][:] = irfftn(c['ft_lsurf'] * c['ft_rcore'])
            c['intervol'][:] = irfftn(c['ft_lsurf'] * c['ft_rsurf'])

            c['interspace'][:] = np.logical_and(c['clashvol'] < self.max_clash,
                                        c['intervol'] > self.min_interaction)

            tot_complex += c['weights'][n] * c['interspace'].sum()

            if self.distance_restraints:
                c['restspace'].fill(0)

                rest_center = d['restraints'][:, :3] - (np.mat(c['rotmat'][n]) * np.mat(d['restraints'][:,3:6]).T).T
                radii = d['restraints'][:,6]
                dilate_points_add(rest_center, radii, c['restspace'])

                c['interspace'] *= c['restspace']

            np.maximum(c['interspace'], c['access_interspace'], \
                       c['access_interspace'])

            list_total_allowed += c['weights'][n] *\
                        np.bincount(c['interspace'].astype(np.int32).flatten(),
                        minlength=(max(2, d['nrestraints']+1)))

        d['accessible_interaction_space'] = c['access_interspace']
        d['accessible_complexes'] = [tot_complex] + list_total_allowed[1:].tolist()
        d['accessible_complexes'] = [int(x) for x in d['accessible_complexes']]
        for i in range(1, len(d['accessible_complexes'])):
            d['accessible_complexes'][i] = sum(d['accessible_complexes'][i:])
        

def rsurface(points, radius, shape, voxelspacing):

    dimensions = [x*voxelspacing for x in shape]
    origin = volume_origin(points, dimensions)
    rsurf = volume.zeros(shape, voxelspacing, origin)

    rsurf = dilate_points(points, radius, rsurf)

    return rsurf

def volume_origin(points, dimensions):
    
    center = points.mean(axis=0)
    origin = [(c - d/2.0) for c, d in zip(center, dimensions)]

    return origin
    
def grid_restraints(restraints, voxelspacing, origin, lcenter):

    nrestraints = len(restraints)
    g_restraints = np.zeros((nrestraints, 8), dtype=np.float64)

    for n in range(nrestraints):
        r_sel, l_sel, distance = restraints[n]

        r_pos = (r_sel.center - origin)/voxelspacing
        l_pos = (l_sel.center - lcenter)/voxelspacing

        g_restraints[n, 0:3] = r_pos
        g_restraints[n, 3:6] = l_pos
        g_restraints[n, 6] = distance/voxelspacing

    return g_restraints


def grid_shape(points1, points2, voxelspacing):
    shape = min_grid_shape(points1, points2, voxelspacing)
    shape = [radix235(x) for x in shape]
    return shape

def min_grid_shape(points1, points2, voxelspacing):
    maxdist1 = np.linalg.norm(points1 - points1.mean(axis=0), axis=1).max()
    maxdist2 = np.linalg.norm(points2 - points2.mean(axis=0), axis=1).max()

    grid_length = int(2*(maxdist1 + maxdist2)/voxelspacing)

    grid_shape = [grid_length]*3
    return grid_shape
