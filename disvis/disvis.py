from __future__ import absolute_import, division
from sys import stdout as _stdout
from time import time as _time
from os.path import join as _join

import numpy as np
from numpy.fft import rfftn as np_rfftn, irfftn as np_irfftn

try:
    from pyfftw import zeros_aligned
    from pyfftw.builders import rfftn as rfftn_builder, irfftn \
        as irfftn_builder

    PYFFTW = True
except ImportError:
    PYFFTW = False
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    from . import pyclfft
    from .kernels import Kernels

    PYOPENCL = True
except ImportError:
    PYOPENCL = False

from disvis import volume
from .pdb import PDB
from .libdisvis import (
    dilate_points, distance_restraint, count_violations, count_interactions
)
from ._extensions import rotate_grid3d


class DisVis(object):
    def __init__(self, fftw=True, print_callback=True):
        # parameters to be defined
        self.receptor = None
        self.ligand = None
        self.distance_restraints = []

        # parameters with standard values that can be set by the user
        self.rotations = np.asarray([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                                    dtype=np.float64)
        self.weights = None
        self.voxelspacing = 1.0
        self.interaction_radius = 2.5
        self.max_clash = 100
        self.min_interaction = 300
        self.receptor_interaction_selection = None
        self.ligand_interaction_selection = None
        self.interaction_distance = 10
        self.interaction_restraints_cutoff = None
        self.occupancy_analysis = False
        self.queue = None
        self._fftw = fftw and PYFFTW
        self.print_callback = print_callback
        self.save = False
        self.save_job = 0
        self.save_directory = '.'

        # Output attributes
        # Array containing number of complexes consistent with EXACTLY n
        # restraints, no more, no less.
        self.accessable_complexes = None
        self.accessable_interaction_space = None
        self.violations = None
        self.occupancy_grids = None
        self.interaction_matrix = None

    def add_distance_restraint(self, receptor_selection, ligand_selection,
                               mindis, maxdis):
        """

        :param receptor_selection:
        :param ligand_selection:
        :param mindis:
        :param maxdis:
        :return:
        """
        distance_restraint = [receptor_selection, ligand_selection,
                              mindis, maxdis]
        self.distance_restraints.append(distance_restraint)

    def search(self):
        self._initialize()
        if self.queue is None:
            self._cpu_init()
            self._cpu_search()
        else:
            self._gpu_init()
            self._gpu_search()

        # Set the results
        self.accessible_interaction_space = volume.Volume(
            self._access_interspace, self.voxelspacing,
            self._origin)
        self.accessible_complexes = self._accessible_complexes
        self.violations = self._violations

        if self.occupancy_analysis:
            self.occupancy_grids = {}
            for i in xrange(self._nrestraints + 1):
                try:
                    occ_grid = self._occ_grid[i]
                except KeyError:
                    occ_grid = None
                if occ_grid is not None:
                    self.occupancy_grids[i] = volume.Volume(occ_grid,
                                                            self.voxelspacing,
                                                            self._origin)

        if self._interaction_analysis:
            self.interaction_matrix = self._interaction_matrix

    @staticmethod
    def _minimal_volume_parameters(fixed_coor, scanning_coor, offset,
                                   voxelspacing):
        """

        :param fixed_coor:
        :param scanning_coor:
        :param offset:
        :param voxelspacing:
        :return:
        """
        # the minimal grid shape is the size of the fixed protein in
        # each dimension and the longest diameter of the scanning chain

        offset += np.linalg.norm(scanning_coor - scanning_coor.mean(axis=0),
                                 axis=1).max()
        mindist = fixed_coor.min(axis=0) - offset
        maxdist = fixed_coor.max(axis=0) + offset
        shape = [volume.radix235(int(np.ceil(x)))
                 for x in (maxdist - mindist) / voxelspacing][::-1]
        origin = mindist
        return shape, origin

    def _initialize(self):

        # check if requirements are set
        if any(x is None for x in (self.receptor, self.ligand)) \
                or not self.distance_restraints:
            raise ValueError("Not all requirements are met for a search")

        if self.weights is None:
            self.weights = np.ones(self.rotations.shape[0], dtype=np.float64)

        if self.weights.size != self.rotations.shape[0]:
            raise ValueError("Weight array has incorrect size.")

        # Determine size for volume to hold the recepter and ligand densities
        vdw_radii = self.receptor.vdw_radius
        self._shape, self._origin = \
            self._minimal_volume_parameters(self.receptor.coor,
                                            self.ligand.coor,
                                            self.interaction_radius + vdw_radii.max(),
                                            self.voxelspacing)

        # Calculate the interaction surface and core of the receptor
        # Move the coordinates to the grid-frame
        self._rgridcoor = (self.receptor.coor - self._origin) / \
                          self.voxelspacing
        self._rcore = np.zeros(self._shape, dtype=np.float64)
        self._rsurf = np.zeros(self._shape, dtype=np.float64)
        radii = vdw_radii / self.voxelspacing
        dilate_points(self._rgridcoor, radii, self._rcore)
        radii += self.interaction_radius / self.voxelspacing
        dilate_points(self._rgridcoor, radii, self._rsurf)

        # Set ligand center to the origin of the grid and calculate the core
        # shape. The coordinates are wrapped around in the density.
        self._lgridcoor = (self.ligand.coor - self.ligand.center) / \
                          self.voxelspacing
        radii = self.ligand.vdw_radius
        self._lcore = np.zeros(self._shape, dtype=np.float64)
        dilate_points(self._lgridcoor, radii, self._lcore)

        # Normalize the requirements for a complex in grid quantities
        self._grid_max_clash = self.max_clash / self.voxelspacing ** 3
        self._grid_min_interaction = self.min_interaction / \
                                     self.voxelspacing ** 3

        # Setup the distance restraints
        self._nrestraints = len(self.distance_restraints)
        self._grid_restraints = grid_restraints(self.distance_restraints,
                                                self.voxelspacing,
                                                self._origin,
                                                self.ligand.center)
        self._rrestraints = self._grid_restraints[:, 0:3]
        self._lrestraints = self._grid_restraints[:, 3:6]
        self._mindis = self._grid_restraints[:, 6]
        self._maxdis = self._grid_restraints[:, 7]

        self._accessible_complexes = np.zeros(self._nrestraints + 1,
                                              dtype=np.float64)
        self._access_interspace = np.zeros(self._shape, dtype=np.int32)
        self._violations = np.zeros((self._nrestraints, self._nrestraints),
                                    dtype=np.float64)

        # Calculate the average occupancy grid only for complexes consistent
        # with interaction_restraints_cutoff and more. By default, only
        # analyze solutions that max violate 3 restraints
        if self.interaction_restraints_cutoff is None:
            # Do not calculate the interactions for complexes consistent
            # with 0 restraints
            cutoff = min(3, self._nrestraints - 1)
            self.interaction_restraints_cutoff = self._nrestraints - cutoff

        # Allocate an occupancy grid for all restraints that are being
        # investigated
        self._occ_grid = {}
        if self.occupancy_analysis:
            for i in xrange(self.interaction_restraints_cutoff,
                            self._nrestraints + 1):
                self._occ_grid[i] = np.zeros(self._shape, np.float64)

        # Check if we want to do an interaction analysis, i.e. whether
        # interface residues are given for both the ligand and receptor.
        selection = (self.ligand_interaction_selection,
                     self.receptor_interaction_selection)
        self._interaction_analysis = any(x is not None for x in selection)
        if self._interaction_analysis:
            # Since calculating all interactions is costly, only analyze
            # solutions that are consistent with more than N restraints.
            self._lselect = (self.ligand_interaction_selection.coor -
                             self.ligand_interaction_selection.center) / self.voxelspacing
            self._rselect = (self.receptor_interaction_selection.coor -
                             self._origin) / self.voxelspacing
            shape = (
                self._nrestraints + 1 - self.interaction_restraints_cutoff,
                self._lselect.shape[0], self._rselect.shape[0])
            self._interaction_matrix = np.zeros(shape, dtype=np.float64)
            self._sub_interaction_matrix = np.zeros(shape, dtype=np.int64)

        # Calculate the longest distance in the lcore. This helps in making the
        # grid rotation faster, as less points need to be considered for
        # rotation
        self._llength = int(np.ceil(
            np.linalg.norm(self._lgridcoor, axis=1).max() +
            self.ligand.vdw_radius.max() / self.voxelspacing
        )) + 1

    @staticmethod
    def _allocate_array(shape, dtype, fftw):
        """

        :param shape:
        :param dtype:
        :param fftw:
        :return:
        """
        if fftw:
            arr = zeros_aligned(shape, dtype)
        else:
            arr = np.zeros(shape, dtype)
        return arr

    def rfftn(self, in_arr, out_arr):
        """Provide a similar interface to PyFFTW and numpy interface"""
        if self._fftw:
            out_arr = self._rfftn(in_arr, out_arr)
        else:
            out_arr = self._rfftn(in_arr)
        return out_arr

    def irfftn(self, in_arr, out_arr):
        """Provide a similar interface to PyFFTW and numpy interface"""
        if self._fftw:
            out_arr = self._irfftn(in_arr, out_arr)
        else:
            out_arr = self._irfftn(in_arr, s=self._shape)
        return out_arr

    def _cpu_init(self):

        # Allocate arrays for FFT's
        # Real arrays
        for arr in 'rot_lcore clashvol intervol tmp'.split():
            setattr(self, '_' + arr, self._allocate_array(
                self._shape, np.float64, self._fftw))

        # Complex arrays
        self._ft_shape = list(self._shape)[:-1] + [self._shape[-1] // 2 + 1]
        for arr in 'lcore lcore_conj rcore rsurf tmp'.split():
            setattr(self, '_ft_' + arr,
                    self._allocate_array(self._ft_shape, np.complex128,
                                         self._fftw))

        # Integer arrays
        for arr in 'interspace red_interspace access_interspace restspace'.split():
            setattr(self, "_" + arr, np.zeros(self._shape, np.int32))

        # Boolean arrays
        for arr in 'not_clashing interacting'.split():
            setattr(self, "_" + arr, np.zeros(self._shape, np.bool))

        # Array for rotating points and restraint coordinates
        self._restraints_center = np.zeros_like(self._grid_restraints[:, 3:6])
        self._rot_lrestraints = np.zeros_like(self._lrestraints)
        if self._interaction_analysis:
            self._rot_lselect = np.zeros_like(self._lselect)

        # Build the FFT's if we are using pyfftw
        if self._fftw:
            self._rfftn = rfftn_builder(self._rcore)
            self._irfftn = irfftn_builder(self._ft_rcore, s=self._shape)
        else:
            self._rfftn = np_rfftn
            self._irfftn = np_irfftn

        # initial calculations
        self._ft_rcore = self.rfftn(self._rcore, self._ft_rcore)
        self._ft_rsurf = self.rfftn(self._rsurf, self._ft_rsurf)

        # Keep track of number of consistent complexes
        self._tot_complex = 0
        self._consistent_complexes = np.zeros(self._nrestraints + 1,
                                              dtype=np.float64)

    def _rotate_lcore(self, rotmat):
        """

        :param rotmat:
        :return:
        """
        rotate_grid3d(self._lcore, rotmat, self._llength,
                      self._rot_lcore, True)

    def _get_interaction_space(self):
        # Calculate the clashing and interaction volume
        self._ft_lcore = self.rfftn(self._rot_lcore, self._ft_lcore)
        np.conjugate(self._ft_lcore, self._ft_lcore_conj)
        np.multiply(self._ft_lcore_conj, self._ft_rcore, self._ft_tmp)
        self._clashvol = self.irfftn(self._ft_tmp, self._clashvol)
        # Round up values, as they should be integers
        np.round(self._clashvol, out=self._clashvol)
        np.multiply(self._ft_lcore_conj, self._ft_rsurf, self._ft_tmp)
        self._intervol = self.irfftn(self._ft_tmp, self._intervol)
        np.round(self._intervol, out=self._intervol)

        # Determine the interaction space, i.e. all the translations where
        # the receptor and ligand are interacting and not clashing
        np.less_equal(self._clashvol, self._grid_max_clash, self._not_clashing)
        np.greater_equal(self._intervol, self._grid_min_interaction,
                         self._interacting)
        np.logical_and(self._not_clashing, self._interacting, self._interspace)

    def _get_restraints_center(self, rotmat):
        """Rotate the restraints and determine the restraints center point"""
        np.dot(self._lrestraints, rotmat.T, self._rot_lrestraints)
        np.subtract(self._rrestraints, self._rot_lrestraints,
                    self._restraints_center)

    def _get_restraint_space(self):
        # Determine the consistent restraint space
        self._restspace.fill(0)
        distance_restraint(self._restraints_center, self._mindis, self._maxdis,
                           self._restspace)

    def _get_reduced_interspace(self):
        np.multiply(self._interspace, self._restspace, self._red_interspace)

    def _count_complexes(self, weight):
        self._tot_complex += weight * self._interspace.sum()
        self._consistent_complexes += weight * \
                                      np.bincount(self._red_interspace.ravel(),
                                                  minlength=(max(2,
                                                                 self._nrestraints + 1))
                                                  )

    def _count_violations(self, weight):
        # Count all sampled complexes
        count_violations(self._restraints_center, self._mindis,
                         self._maxdis, self._red_interspace, weight,
                         self._violations)

    def _get_access_interspace(self):
        np.maximum(self._red_interspace, self._access_interspace,
                   self._access_interspace)

    def _get_occupancy_grids(self, weight):
        for i in xrange(self.interaction_restraints_cutoff,
                        self._nrestraints + 1):
            np.greater_equal(self._red_interspace, np.int32(i), self._tmp)
            self._ft_tmp = self.rfftn(self._tmp, self._ft_tmp)
            np.multiply(self._ft_tmp, self._ft_lcore, self._ft_tmp)
            self._tmp = self.irfftn(self._ft_tmp, self._tmp)
            np.round(self._tmp, out=self._tmp)
            self._occ_grid[i] += weight * self._tmp

    def _get_interaction_matrix(self, rotmat, weight):
        # Rotate the ligand coordinates
        self._rot_lselect = np.dot(self._lselect, rotmat.T)
        count_interactions(self._red_interspace, self._rselect,
                           self._rot_lselect,
                           np.float64(self.interaction_distance / self.voxelspacing),
                           weight,
                           np.int32(self.interaction_restraints_cutoff),
                           self._interaction_matrix)

    def _cpu_search(self):

        time0 = _time()
        for n in xrange(self.rotations.shape[0]):
            rotmat = self.rotations[n]
            weight = self.weights[n]
            # Rotate the ligand grid
            self._rotate_lcore(rotmat)

            self._get_interaction_space()

            # Rotate the restraints
            self._get_restraints_center(rotmat)

            # Determine the consistent restraint space
            self._get_restraint_space()

            # Calculate the reduced accessible interaction
            self._get_reduced_interspace()

            # Perform some statistics, such as the number of restraint
            # violations and the number of accessible complexes consistent with
            # exactly N restraints.
            self._count_complexes(weight)
            self._count_violations(weight)

            # Calculate shapes for visual information, such as the highest
            # number of consisistent restraints found at each grid point
            self._get_access_interspace()

            # Calculate an occupancy grid for complexes consistent with at
            # least i restraints
            if self.occupancy_analysis:
                self._get_occupancy_grids(weight)

            # Perform interaction analysis if required
            if self._interaction_analysis:
                self._get_interaction_matrix(rotmat, weight)

            # Write reduced accessible interaction space to file, if required
            if self.save:
                fname = _join(self.save_directory,
                              'red_interspace_{:d}_{:d}.mrc'.format(
                                  self.save_job, n))
                volume.Volume(self._red_interspace, self.voxelspacing,
                              self._origin).tofile(fname)

            if self.print_callback is not None:
                # self.print_callback(n, total, time0)
                self._print_progress(n, self.rotations.shape[0], time0)

        # Get the number of accessible complexes consistent with exactly N
        # restraints. We need to correct the total number of complexes sampled
        # for this.
        self._accessible_complexes[:] = self._consistent_complexes
        self._accessible_complexes[0] = self._tot_complex - \
                                        self._accessible_complexes[1:].sum()

    @staticmethod
    def _print_progress(n, total, time0):
        m = n + 1
        pdone = m / total
        t = _time() - time0
        _stdout.write('\r{:d}/{:d} ({:.2%}, ETA: {:d}s)'
                      '    '.format(m, total, pdone, int(t / pdone - t)))
        _stdout.flush()

    def _gpu_init(self):

        q = self.queue

        # Move arrays to GPU
        self._cl_rcore = cl_array.to_device(q, self._rcore.astype(np.float32))
        self._cl_rsurf = cl_array.to_device(q, self._rsurf.astype(np.float32))
        self._cl_lcore = cl_array.to_device(q, self._lcore.astype(np.float32))

        # Make the rotations float16 arrays
        self._cl_rotations = np.zeros((self.rotations.shape[0], 16),
                                      dtype=np.float32)
        self._cl_rotations[:, :9] = self.rotations.reshape(-1, 9)

        # Allocate arrays
        # Float32
        self._cl_shape = tuple(self._shape)
        arr_names = 'rot_lcore clashvol intervol tmp'.split()
        for arr_name in arr_names:
            setattr(self, '_cl_' + arr_name,
                    cl_array.zeros(q, self._cl_shape, dtype=np.float32)
                    )

        # Int32
        arr_names = 'interspace red_interspace restspace access_interspace'.split()
        for arr_name in arr_names:
            setattr(self, '_cl_' + arr_name,
                    cl_array.zeros(q, self._cl_shape, dtype=np.int32)
                    )

        # Boolean
        arr_names = 'not_clashing interacting'.split()
        for arr_name in arr_names:
            setattr(self, '_cl_' + arr_name,
                    cl_array.zeros(q, self._cl_shape, dtype=np.int32)
                    )

        # Complex64
        self._ft_shape = tuple(
            [self._shape[0] // 2 + 1] + list(self._shape)[1:])
        arr_names = 'lcore lcore_conj rcore rsurf tmp'.split()
        for arr_name in arr_names:
            setattr(self, '_cl_ft_' + arr_name,
                    cl_array.empty(q, self._ft_shape, dtype=np.complex64)
                    )

        # Restraints arrays
        self._cl_rrestraints = np.zeros((self._nrestraints, 4),
                                        dtype=np.float32)
        self._cl_rrestraints[:, :3] = self._rrestraints
        self._cl_rrestraints = cl_array.to_device(q, self._cl_rrestraints)
        self._cl_lrestraints = np.zeros((self._nrestraints, 4),
                                        dtype=np.float32)
        self._cl_lrestraints[:, :3] = self._lrestraints
        self._cl_lrestraints = cl_array.to_device(q, self._cl_lrestraints)
        self._cl_mindis = cl_array.to_device(q,
                                             self._mindis.astype(np.float32))
        self._cl_maxdis = cl_array.to_device(q,
                                             self._maxdis.astype(np.float32))
        self._cl_mindis2 = cl_array.to_device(q, self._mindis.astype(
            np.float32) ** 2)
        self._cl_maxdis2 = cl_array.to_device(q, self._maxdis.astype(
            np.float32) ** 2)
        self._cl_rot_lrestraints = cl_array.zeros_like(self._cl_rrestraints)
        self._cl_restraints_center = cl_array.zeros_like(self._cl_rrestraints)

        # kernels
        self._kernel_constants = {'interaction_cutoff': 10,
                                  'nrestraints': self._nrestraints,
                                  'shape_x': self._shape[2],
                                  'shape_y': self._shape[1],
                                  'shape_z': self._shape[0],
                                  'llength': self._llength,
                                  'nreceptor_coor': 0,
                                  'nligand_coor': 0,
                                  }

        # Counting arrays
        self._cl_hist = cl_array.zeros(self.queue, self._nrestraints,
                                       dtype=np.int32)
        self._cl_consistent_complexes = cl_array.zeros(self.queue,
                                                       self._nrestraints,
                                                       dtype=np.float32)
        self._cl_viol_hist = cl_array.zeros(self.queue, (self._nrestraints,
                                                         self._nrestraints),
                                            dtype=np.int32)
        self._cl_violations = cl_array.zeros(self.queue, (self._nrestraints,
                                                          self._nrestraints),
                                             dtype=np.float32)

        # Conversions
        self._cl_grid_max_clash = np.float32(self._grid_max_clash)
        self._cl_grid_min_interaction = np.float32(self._grid_min_interaction)
        self._CL_ZERO = np.int32(0)

        # Occupancy analysis
        self._cl_occ_grid = {}
        if self.occupancy_analysis:
            for i in xrange(self.interaction_restraints_cutoff,
                            self._nrestraints + 1):
                self._cl_occ_grid[i] = cl_array.zeros(self.queue,
                                                      self._cl_shape,
                                                      dtype=np.float32)

        # Interaction analysis
        if self._interaction_analysis:
            shape = (self._lselect.shape[0], self._rselect.shape[0])
            self._cl_interaction_hist = cl_array.zeros(self.queue, shape,
                                                       dtype=np.int32)
            self._cl_interaction_matrix = {}
            for i in xrange(self._nrestraints + 1 -
                                    self.interaction_restraints_cutoff):
                self._cl_interaction_matrix[i] = cl_array.zeros(self.queue,
                                                                shape,
                                                                dtype=np.float32)
            # Coordinate arrays
            self._cl_rselect = np.zeros((self._rselect.shape[0], 4),
                                        dtype=np.float32)
            self._cl_rselect[:, :3] = self._rselect
            self._cl_rselect = cl_array.to_device(q, self._cl_rselect)
            self._cl_lselect = np.zeros((self._lselect.shape[0], 4),
                                        dtype=np.float32)
            self._cl_lselect[:, :3] = self._lselect
            self._cl_lselect = cl_array.to_device(q, self._cl_lselect)
            self._cl_rot_lselect = cl_array.zeros_like(self._cl_lselect)

            # Update kernel constants
            self._kernel_constants['nreceptor_coor'] = self._cl_rselect.shape[
                0]
            self._kernel_constants['nligand_coor'] = self._cl_lselect.shape[0]

        self._cl_kernels = Kernels(q.context, self._kernel_constants)
        self._cl_rfftn = pyclfft.RFFTn(q.context, self._shape)
        self._cl_irfftn = pyclfft.iRFFTn(q.context, self._shape)

        # Initial calculations
        self._cl_rfftn(q, self._cl_rcore, self._cl_ft_rcore)
        self._cl_rfftn(q, self._cl_rsurf, self._cl_ft_rsurf)
        self._cl_tot_complex = cl_array.sum(self._cl_interspace,
                                            dtype=np.dtype(np.float32))

    def _cl_rotate_lcore(self, rotmat):
        self._cl_kernels.rotate_grid3d(self.queue, self._cl_lcore, rotmat,
                                       self._cl_rot_lcore)
        self.queue.finish()

    def _cl_get_interaction_space(self):
        k = self._cl_kernels
        self._cl_rfftn(self.queue, self._cl_rot_lcore, self._cl_ft_lcore)
        k.conj(self._cl_ft_lcore, self._cl_ft_lcore_conj)
        k.cmultiply(self._cl_ft_lcore_conj, self._cl_ft_rcore, self._cl_ft_tmp)
        self._cl_irfftn(self.queue, self._cl_ft_tmp, self._cl_clashvol)
        k.round(self._cl_clashvol, self._cl_clashvol)
        k.cmultiply(self._cl_ft_lcore_conj, self._cl_ft_rsurf, self._cl_ft_tmp)
        self._cl_irfftn(self.queue, self._cl_ft_tmp, self._cl_intervol)
        k.round(self._cl_intervol, self._cl_intervol)

        k.less_equal(self._cl_clashvol, self._cl_grid_max_clash,
                     self._cl_not_clashing)
        k.greater_equal(self._cl_intervol, self._cl_grid_min_interaction,
                        self._cl_interacting)
        k.logical_and(self._cl_not_clashing, self._cl_interacting,
                      self._cl_interspace)
        self.queue.finish()

    def _cl_get_restraints_center(self, rotmat):
        k = self._cl_kernels
        k.rotate_points3d(self.queue, self._cl_lrestraints, rotmat,
                          self._cl_rot_lrestraints)
        k.subtract(self._cl_rrestraints, self._cl_rot_lrestraints,
                   self._cl_restraints_center)
        self.queue.finish()

    def _cl_get_restraint_space(self):
        k = self._cl_kernels
        k.set_to_i32(np.int32(0), self._cl_restspace)
        k.dilate_point_add(self.queue, self._cl_restraints_center,
                           self._cl_mindis, self._cl_maxdis,
                           self._cl_restspace)
        self.queue.finish()

    def _cl_get_reduced_interspace(self):
        self._cl_kernels.multiply_int32(self._cl_restspace,
                                        self._cl_interspace,
                                        self._cl_red_interspace)
        self.queue.finish()

    def _cl_count_complexes(self, weight):
        # Count all sampled complexes
        self._cl_tot_complex += cl_array.sum(self._cl_interspace,
                                             dtype=np.dtype(
                                                 np.float32)) * weight
        self._cl_kernels.set_to_i32(np.int32(0), self._cl_hist)

        self._cl_kernels.histogram(self.queue, self._cl_red_interspace,
                                   self._cl_hist)
        self._cl_kernels.multiply_add(self._cl_hist, weight,
                                      self._cl_consistent_complexes)
        self.queue.finish()

    def _cl_count_violations(self, weight):
        self._cl_kernels.set_to_i32(np.int32(0), self._cl_viol_hist)
        self._cl_kernels.count_violations(self.queue,
                                          self._cl_restraints_center,
                                          self._cl_mindis2, self._cl_maxdis2,
                                          self._cl_red_interspace,
                                          self._cl_viol_hist)
        self._cl_kernels.multiply_add(self._cl_viol_hist, weight,
                                      self._cl_violations)
        self.queue.finish()

    def _cl_get_access_interspace(self):
        cl_array.maximum(self._cl_red_interspace, self._cl_access_interspace,
                         self._cl_access_interspace)
        self.queue.finish()

    def _cl_get_interaction_matrix(self, rotmat, weight):
        self._cl_kernels.rotate_points3d(self.queue, self._cl_lselect, rotmat,
                                         self._cl_rot_lselect)

        for nconsistent in np.arange(self.interaction_restraints_cutoff,
                                     self._nrestraints + 1, dtype=np.int32):
            self._cl_kernels.set_to_i32(np.int32(0), self._cl_interaction_hist)
            self._cl_kernels.count_interactions(self.queue, self._cl_rselect,
                                                self._cl_rot_lselect,
                                                self._cl_red_interspace,
                                                nconsistent,
                                                self._cl_interaction_hist)
            self._cl_kernels.multiply_add(self._cl_interaction_hist, weight,
                                          self._cl_interaction_matrix[
                                              nconsistent - self.interaction_restraints_cutoff])
        self.queue.finish()

    def _cl_get_occupancy_grids(self, weight):
        # Calculate an average occupancy grid to provide an average shape
        # for several number of consistent restraints
        k = self._cl_kernels
        for i in xrange(self.interaction_restraints_cutoff,
                        self._nrestraints + 1):
            # Get a grid for all translations that are consistent with at least
            # N restraints
            k.greater_equal_iif(self._cl_red_interspace, np.int32(i),
                                self._cl_tmp)
            self._cl_rfftn(self.queue, self._cl_tmp, self._cl_ft_tmp)
            k.cmultiply(self._cl_ft_tmp, self._cl_ft_lcore, self._cl_ft_tmp)
            self._cl_irfftn(self.queue, self._cl_ft_tmp, self._cl_tmp)
            k.round(self._cl_tmp, self._cl_tmp)
            k.multiply_add2(self._cl_tmp, weight, self._cl_occ_grid[i])
        self.queue.finish()

    def _gpu_search(self):

        time0 = _time()
        for n in xrange(self.rotations.shape[0]):

            rotmat = self._cl_rotations[n]
            weight = np.float32(self.weights[n])

            # Rotate the ligand
            self._cl_rotate_lcore(rotmat)

            # Calculate the clashing and interaction volume for each
            # translation
            self._cl_get_interaction_space()

            # Rotate the restraints and determine the center point
            self._cl_get_restraints_center(rotmat)

            self._cl_get_restraint_space()

            # Check for each complex how many restraints are consistent
            self._cl_get_reduced_interspace()

            # Do some analysis such as counting complexes and violations
            self._cl_count_complexes(weight)
            self._cl_count_violations(weight)
            self._cl_get_access_interspace()

            # Optional analyses
            if self.occupancy_analysis:
                self._cl_get_occupancy_grids(weight)

            if self._interaction_analysis:
                self._cl_get_interaction_matrix(rotmat, weight)

            # Print progress
            if _stdout.isatty():
                self._print_progress(n, self.rotations.shape[0], time0)

        self.queue.finish()
        # Get the data from GPU

        self._accessible_complexes = self._cl_consistent_complexes.get()
        self._accessible_complexes = np.asarray([self._cl_tot_complex.get()] +
                                                self._accessible_complexes.tolist(),
                                                dtype=np.float64)
        self._accessible_complexes[0] -= self._accessible_complexes[1:].sum()

        self._violations = self._cl_violations.get().astype(np.float64)
        self._cl_access_interspace.get(ary=self._access_interspace)

        if self.occupancy_analysis:
            for i in xrange(self.interaction_restraints_cutoff,
                            self._nrestraints + 1):
                self._occ_grid[i] = \
                    self._cl_occ_grid[i].get().astype(np.float64)

        if self._interaction_analysis:
            for i in xrange(self._nrestraints + 1 -
                                    self.interaction_restraints_cutoff):
                self._interaction_matrix[i] = \
                    self._cl_interaction_matrix[i].get().astype(np.float64)


def grid_restraints(restraints, voxelspacing, origin, lcenter):
    nrestraints = len(restraints)
    g_restraints = np.zeros((nrestraints, 8), dtype=np.float64)
    for n in range(nrestraints):
        r_sel, l_sel, mindis, maxdis = restraints[n]

        r_pos = (r_sel.center - origin) / voxelspacing
        l_pos = (l_sel.center - lcenter) / voxelspacing

        g_restraints[n, 0:3] = r_pos
        g_restraints[n, 3:6] = l_pos
        g_restraints[n, 6] = mindis / voxelspacing
        g_restraints[n, 7] = maxdis / voxelspacing

    return g_restraints
