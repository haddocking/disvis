from __future__ import print_function, division, absolute_import
from os import remove
from os.path import join, abspath
from sys import exit
from sys import stdout as _stdout
from time import time as _time

import numpy as np
from numpy.fft import rfftn as np_rfftn, irfftn as np_irfftn
try:
    from pyfftw import zeros_aligned
    from pyfftw.builders import rfftn as rfftn_builder, irfftn as irfftn_builder
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

from .volume import Volume, radix235
from .pdb import PDB
from .libdisvis import (
        dilate_points, distance_restraint, count_violations, count_interactions
        )
from ._extensions import rotate_grid3d





import multiprocessing as mp
from argparse import ArgumentParser
import logging


from .rotations import proportional_orientations, quat_to_rotmat
from .helpers import mkdir_p


class DisVis(object):

    def __init__(self, fftw=True, print_callback=True):
        # parameters to be defined
        self.receptor = None
        self.ligand = None
        self.distance_restraints = []

        # parameters with standard values that can be set by the user
        self.rotations = np.asarray([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float64)
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

        # Output attributes
        # Array containing number of complexes consistent with EXACTLY n
        # restraints, no more, no less.
        self.accessable_complexes = None
        self.accessable_interaction_space = None
        self.violations = None
        self.occupancy_grids = None
        self.interaction_matrix = None

    def add_distance_restraint(self, receptor_selection, ligand_selection, mindis, maxdis):
        distance_restraint = [receptor_selection, ligand_selection, mindis, maxdis]
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
        self.accessible_interaction_space = Volume(
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
                    self.occupancy_grids[i] = Volume(occ_grid,
                            self.voxelspacing, self._origin)

        if self._interaction_analysis:
            self.interaction_matrix = self._interaction_matrix

    @staticmethod
    def _minimal_volume_parameters(fixed_coor, scanning_coor, offset, voxelspacing):
        # the minimal grid shape is the size of the fixed protein in 
        # each dimension and the longest diameter of the scanning chain

        offset += np.linalg.norm(scanning_coor - scanning_coor.mean(axis=0), axis=1).max()
        mindist = fixed_coor.min(axis=0) - offset
        maxdist = fixed_coor.max(axis=0) + offset
        shape = [radix235(int(np.ceil(x)))
                     for x in (maxdist - mindist) / voxelspacing][::-1]
        origin = mindist
        return shape, origin

    def _initialize(self):

        # check if requirements are set
        if any(x is None for x in (self.receptor, self.ligand)) or not self.distance_restraints:
            raise ValueError("Not all requirements are met for a search")

        if self.weights is None:
            self.weights = np.ones(self.rotations.shape[0], dtype=np.float64)

        if self.weights.size != self.rotations.shape[0]:
            raise ValueError("Weight array has incorrect size.")

        # Determine size for volume to hold the recepter and ligand densities
        vdw_radii = self.receptor.vdw_radius
        self._shape, self._origin = self._minimal_volume_parameters(self.receptor.coor, 
                self.ligand.coor, self.interaction_radius + vdw_radii.max(), self.voxelspacing)

        # Calculate the interaction surface and core of the receptor
        # Move the coordinates to the grid-frame
        self._rgridcoor = (self.receptor.coor - self._origin) / self.voxelspacing
        self._rcore = np.zeros(self._shape, dtype=np.float64)
        self._rsurf = np.zeros(self._shape, dtype=np.float64)
        radii = vdw_radii / self.voxelspacing
        dilate_points(self._rgridcoor, radii, self._rcore)
        radii += self.interaction_radius / self.voxelspacing
        dilate_points(self._rgridcoor, radii, self._rsurf)

        # Set ligand center to the origin of the grid and calculate the core
        # shape. The coordinates are wrapped around in the density.
        self._lgridcoor = (self.ligand.coor - self.ligand.center) / self.voxelspacing
        radii = self.ligand.vdw_radius
        self._lcore = np.zeros(self._shape, dtype=np.float64)
        dilate_points(self._lgridcoor, radii, self._lcore)

        # Normalize the requirements for a complex in grid quantities
        self._grid_max_clash = self.max_clash / self.voxelspacing**3
        self._grid_min_interaction = self.min_interaction / self.voxelspacing**3

        # Setup the distance restraints
        self._nrestraints = len(self.distance_restraints)
        self._grid_restraints = grid_restraints(self.distance_restraints, 
                self.voxelspacing, self._origin, self.ligand.center)
        self._rrestraints = self._grid_restraints[:, 0:3]
        self._lrestraints = self._grid_restraints[:, 3:6]
        self._mindis = self._grid_restraints[:,6]
        self._maxdis = self._grid_restraints[:,7]

        self._accessible_complexes = np.zeros(self._nrestraints + 1, dtype=np.float64)
        self._access_interspace = np.zeros(self._shape, dtype=np.int32)
        self._violations = np.zeros((self._nrestraints, self._nrestraints), dtype=np.float64)

        # Calculate the average occupancy grid only for complexes consistent
        # with interaction_restraints_cutoff and more. By default, only
        # analyze solutions that max violate 3 restraints
        if self.interaction_restraints_cutoff is None:
            # Do not calculate the interactions for complexes consistent
            # with 0 restraints
            cutoff = min(3, self._nrestraints - 1)
            self.interaction_restraints_cutoff = self._nrestraints - cutoff

        # Allocate an occupancy grid for all restraints that are being investigated
        self._occ_grid = {}
        if self.occupancy_analysis:
            for i in xrange(self.interaction_restraints_cutoff, self._nrestraints + 1):
                self._occ_grid[i] = np.zeros(self._shape, np.float64)

        # Check if we want to do an interaction analysis, i.e. whether
        # interface residues are given for both the ligand and receptor.
        selection = (self.ligand_interaction_selection, self.receptor_interaction_selection)
        self._interaction_analysis = any(x is not None for x in selection)
        if self._interaction_analysis:
            # Since calculating all interactions is costly, only analyze
            # solutions that are consistent with more than N restraints.
            self._lselect = (self.ligand_interaction_selection.coor -
                    self.ligand_interaction_selection.center) / self.voxelspacing
            self._rselect = (self.receptor_interaction_selection.coor -
                    self._origin) / self.voxelspacing
            shape = (self._nrestraints + 1 - self.interaction_restraints_cutoff, 
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
            setattr(self, '_' + arr, self._allocate_array(self._shape, np.float64, self._fftw))

        # Complex arrays
        self._ft_shape = list(self._shape)[:-1] + [self._shape[-1] // 2 + 1]
        for arr in 'lcore lcore_conj rcore rsurf tmp'.split():
            setattr(self, '_ft_' + arr,
                    self._allocate_array(self._ft_shape, np.complex128, self._fftw))

        # Integer arrays
        for arr in 'interspace red_interspace access_interspace restspace'.split():
            setattr(self, "_" + arr, np.zeros(self._shape, np.int32))

        # Boolean arrays
        for arr in 'not_clashing interacting'.split():
            setattr(self, "_" + arr, np.zeros(self._shape, np.bool))

        # Array for rotating points and restraint coordinates
        self._restraints_center = np.zeros_like(self._grid_restraints[:,3:6])
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
        self._consistent_complexes = np.zeros(self._nrestraints + 1, dtype=np.float64)

    def _rotate_lcore(self, rotmat):
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
        np.greater_equal(self._intervol, self._grid_min_interaction, self._interacting)
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
        self._consistent_complexes += weight *\
                    np.bincount(self._red_interspace.ravel(),
                            minlength=(max(2, self._nrestraints + 1))
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
        for i in xrange(self.interaction_restraints_cutoff, self._nrestraints + 1):
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
                self._rot_lselect, np.float64(self.interaction_distance / self.voxelspacing),
                weight, np.int32(self.interaction_restraints_cutoff),
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

            if self.print_callback is not None:
                #self.print_callback(n, total, time0)
                self._print_progress(n, self.rotations.shape[0], time0)

        # Get the number of accessible complexes consistent with exactly N
        # restraints. We need to correct the total number of complexes sampled
        # for this.
        self._accessible_complexes[:] = self._consistent_complexes
        self._accessible_complexes[0] = self._tot_complex - self._accessible_complexes[1:].sum()
        
    @staticmethod
    def _print_progress(n, total, time0):
        m = n + 1
        pdone = m/total
        t = _time() - time0
        _stdout.write('\r{:d}/{:d} ({:.2%}, ETA: {:d}s)    '\
                .format(m, total, pdone, 
                        int(t/pdone - t)))
        _stdout.flush()

    def _gpu_init(self):

        q = self.queue

        # Move arrays to GPU
        self._cl_rcore = cl_array.to_device(q, self._rcore.astype(np.float32))
        self._cl_rsurf = cl_array.to_device(q, self._rsurf.astype(np.float32))
        self._cl_lcore = cl_array.to_device(q, self._lcore.astype(np.float32))

        # Make the rotations float16 arrays
        self._cl_rotations = np.zeros((self.rotations.shape[0], 16), dtype=np.float32)
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
        self._ft_shape = tuple([self._shape[0] // 2 + 1] + list(self._shape)[1:])
        arr_names = 'lcore lcore_conj rcore rsurf tmp'.split()
        for arr_name in arr_names:
            setattr(self, '_cl_ft_' + arr_name, 
                    cl_array.empty(q, self._ft_shape, dtype=np.complex64)
                    )

        # Restraints arrays
        self._cl_rrestraints = np.zeros((self._nrestraints, 4), dtype=np.float32)
        self._cl_rrestraints[:, :3] = self._rrestraints
        self._cl_rrestraints = cl_array.to_device(q, self._cl_rrestraints)
        self._cl_lrestraints = np.zeros((self._nrestraints, 4), dtype=np.float32)
        self._cl_lrestraints[:, :3] = self._lrestraints
        self._cl_lrestraints = cl_array.to_device(q, self._cl_lrestraints)
        self._cl_mindis = cl_array.to_device(q, self._mindis.astype(np.float32))
        self._cl_maxdis = cl_array.to_device(q, self._maxdis.astype(np.float32))
        self._cl_mindis2 = cl_array.to_device(q, self._mindis.astype(np.float32) ** 2)
        self._cl_maxdis2 = cl_array.to_device(q, self._maxdis.astype(np.float32) ** 2)
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
        self._cl_hist = cl_array.zeros(self.queue, self._nrestraints, dtype=np.int32)
        self._cl_consistent_complexes = cl_array.zeros(self.queue,
                self._nrestraints, dtype=np.float32)
        self._cl_viol_hist = cl_array.zeros(self.queue, (self._nrestraints,
            self._nrestraints), dtype=np.int32)
        self._cl_violations = cl_array.zeros(self.queue, (self._nrestraints,
            self._nrestraints), dtype=np.float32)

        # Conversions
        self._cl_grid_max_clash = np.float32(self._grid_max_clash)
        self._cl_grid_min_interaction = np.float32(self._grid_min_interaction)
        self._CL_ZERO = np.int32(0)

        # Occupancy analysis
        self._cl_occ_grid = {}
        if self.occupancy_analysis:
            for i in xrange(self.interaction_restraints_cutoff, self._nrestraints + 1):
                self._cl_occ_grid[i] = cl_array.zeros(self.queue,
                        self._cl_shape, dtype=np.float32)

        # Interaction analysis
        if self._interaction_analysis:
            shape = (self._lselect.shape[0], self._rselect.shape[0])
            self._cl_interaction_hist = cl_array.zeros(self.queue, shape,
                    dtype=np.int32)
            self._cl_interaction_matrix = {}
            for i in xrange(self._nrestraints + 1 - self.interaction_restraints_cutoff):
                self._cl_interaction_matrix[i] = cl_array.zeros(self.queue, shape,
                        dtype=np.float32)
            # Coordinate arrays
            self._cl_rselect = np.zeros((self._rselect.shape[0], 4), dtype=np.float32)
            self._cl_rselect[:, :3] = self._rselect
            self._cl_rselect = cl_array.to_device(q, self._cl_rselect)
            self._cl_lselect = np.zeros((self._lselect.shape[0], 4), dtype=np.float32)
            self._cl_lselect[:, :3] = self._lselect
            self._cl_lselect = cl_array.to_device(q, self._cl_lselect)
            self._cl_rot_lselect = cl_array.zeros_like(self._cl_lselect)

            # Update kernel constants
            self._kernel_constants['nreceptor_coor'] = self._cl_rselect.shape[0]
            self._kernel_constants['nligand_coor'] = self._cl_lselect.shape[0]

        self._cl_kernels = Kernels(q.context, self._kernel_constants)
        self._cl_rfftn = pyclfft.RFFTn(q.context, self._shape)
        self._cl_irfftn = pyclfft.iRFFTn(q.context, self._shape)

        # Initial calculations
        self._cl_rfftn(q, self._cl_rcore, self._cl_ft_rcore)
        self._cl_rfftn(q, self._cl_rsurf, self._cl_ft_rsurf)
        self._cl_tot_complex = cl_array.sum(self._cl_interspace, dtype=np.dtype(np.float32))

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

        k.less_equal(self._cl_clashvol, self._cl_grid_max_clash, self._cl_not_clashing)
        k.greater_equal(self._cl_intervol, self._cl_grid_min_interaction, self._cl_interacting)
        k.logical_and(self._cl_not_clashing, self._cl_interacting, self._cl_interspace)
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
        k.dilate_point_add(self.queue, self._cl_restraints_center, self._cl_mindis,
                self._cl_maxdis, self._cl_restspace)
        self.queue.finish()

    def _cl_get_reduced_interspace(self):
        self._cl_kernels.multiply_int32(self._cl_restspace,
                self._cl_interspace, self._cl_red_interspace)
        self.queue.finish()

    def _cl_count_complexes(self, weight):
        # Count all sampled complexes
        self._cl_tot_complex += cl_array.sum(self._cl_interspace,
                dtype=np.dtype(np.float32)) * weight
        self._cl_kernels.set_to_i32(np.int32(0), self._cl_hist)

        self._cl_kernels.histogram(self.queue, self._cl_red_interspace, self._cl_hist)
        self._cl_kernels.multiply_add(self._cl_hist, weight,
                self._cl_consistent_complexes)
        self.queue.finish()

    def _cl_count_violations(self, weight):
        self._cl_kernels.set_to_i32(np.int32(0), self._cl_viol_hist)
        self._cl_kernels.count_violations(self.queue,
                self._cl_restraints_center, self._cl_mindis2, self._cl_maxdis2,
                self._cl_red_interspace, self._cl_viol_hist)
        self._cl_kernels.multiply_add(self._cl_viol_hist, weight, self._cl_violations)
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
                    self._cl_rot_lselect, self._cl_red_interspace, nconsistent,
                    self._cl_interaction_hist)
            self._cl_kernels.multiply_add(self._cl_interaction_hist, weight,
                    self._cl_interaction_matrix[nconsistent - self.interaction_restraints_cutoff])
        self.queue.finish()

    def _cl_get_occupancy_grids(self, weight):
        # Calculate an average occupancy grid to provide an average shape
        # for several number of consistent restraints
        k = self._cl_kernels
        for i in xrange(self.interaction_restraints_cutoff, self._nrestraints + 1):
            # Get a grid for all translations that are consistent with at least
            # N restraints
            k.greater_equal_iif(self._cl_red_interspace, np.int32(i), self._cl_tmp)
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

            # Calculate the clashing and interaction volume for each translation
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
            self._accessible_complexes.tolist(), dtype=np.float64)
        self._accessible_complexes[0] -= self._accessible_complexes[1:].sum()

        self._violations = self._cl_violations.get().astype(np.float64)
        self._cl_access_interspace.get(ary=self._access_interspace)

        if self.occupancy_analysis:
            for i in xrange(self.interaction_restraints_cutoff, self._nrestraints + 1):
                self._occ_grid[i] = self._cl_occ_grid[i].get().astype(np.float64)

        if self._interaction_analysis:
            for i in xrange(self._nrestraints + 1 - self.interaction_restraints_cutoff):
                self._interaction_matrix[i] = self._cl_interaction_matrix[i].get().astype(np.float64)

def grid_restraints(restraints, voxelspacing, origin, lcenter):

    nrestraints = len(restraints)
    g_restraints = np.zeros((nrestraints, 8), dtype=np.float64)
    for n in range(nrestraints):
        r_sel, l_sel, mindis, maxdis = restraints[n]

        r_pos = (r_sel.center - origin)/voxelspacing
        l_pos = (l_sel.center - lcenter)/voxelspacing

        g_restraints[n, 0:3] = r_pos
        g_restraints[n, 3:6] = l_pos
        g_restraints[n, 6] = mindis/voxelspacing
        g_restraints[n, 7] = maxdis/voxelspacing

    return g_restraints

def parse_args():
    """Parse the command-line arguments."""

    p = ArgumentParser()

    p.add_argument('receptor', type=file,
            help='PDB-file containing fixed chain.')

    p.add_argument('ligand', type=file,
            help='PDB-file containing scanning chain.')

    p.add_argument('restraints', type=file,
            help='File containing the distance restraints')

    p.add_argument('-a', '--angle', dest='angle', type=float, default=15, metavar='<float>',
            help='Rotational sampling density in degrees. Default is 15 degrees.')

    p.add_argument('-vs', '--voxelspacing', dest='voxelspacing', metavar='<float>',
            type=float, default=1,
            help='Voxel spacing of search grid in angstrom. Default is 1A.')

    p.add_argument('-ir', '--interaction-radius',
            dest='interaction_radius', type=float, default=3.0, metavar='<float>',
            help='Radius of the interaction space for each atom in angstrom. '
                 'Atoms are thus considered interacting if the distance is '
                 'larger than the vdW radius and shorther than or equal to '
                 'vdW + interaction_radius. Default is 3A.')

    p.add_argument('-cv', '--max-clash',
            dest='max_clash', type=float, default=200, metavar='<float>',
            help='Maximum allowed volume of clashes. Increasing this '
                 'number results in more allowed complexes. '
                 'Default is 200 A^3.')

    p.add_argument('-iv', '--min-interaction',
            dest='min_interaction', type=float, default=300, metavar='<float>',
            help='Minimal required interaction volume for a '
                 'conformation to be considered a '
                 'complex. Increasing this number results in a '
                 'stricter counting of complexes. '
                 'Default is 300 A^3.')

    p.add_argument('-d', '--directory', dest='directory', metavar='<dir>',
            type=abspath, default='.',
            help='Directory where results are written to. '
                 'Default is current directory.')

    p.add_argument('-p', '--nproc', dest='nproc', type=int, default=1, metavar='<int>',
            help='Number of processors used during search.')

    p.add_argument('-g', '--gpu', dest='gpu', action='store_true',
            help='Use GPU-acceleration for search. If not available '
                 'the CPU-version will be used with the given number '
                 'of processors.')

    help_msg = ("File containing residue number for which interactions will be counted. "
                "The first line holds the receptor residue, "
                "and the second line the ligand residue numbers.")
    p.add_argument('-is', '--interaction-selection', metavar='<file>',
            dest='interaction_selection', type=file, default=None,
            help=help_msg)

    help_msg = ("Number of minimal consistent restraints for which an interaction "
                "or occupancy analysis will be performed. "
                "Default is number of restraints minus 1.")
    p.add_argument('-ic', '--interaction-restraints-cutoff', metavar='<int>',
            dest='interaction_restraints_cutoff', type=int, default=None,
            help=help_msg)

    p.add_argument('-oa', '--occupancy-analysis', dest='occupancy_analysis',
            action='store_true',
            help=("Perform an occupancy analysis, ultimately providing "
                  "a volume where each grid point represents the "
                  "normalized probability of that spot being occupied by the ligand."
                  )
            )

    return p.parse_args()


def parse_interaction_selection(fid, pdb1, pdb2):
    """Parse the interaction selection file, i.e. all residues for which an
    interaction analysis is performed."""

    resi1 = [int(x) for x in fid.readline().split()]
    resi2 = [int(x) for x in fid.readline().split()]

    pdb1_sel = pdb1.select('name', ('CA', "O3'")).select('resi', resi1)
    pdb2_sel = pdb2.select('name', ('CA', "O3'")).select('resi', resi2)

    if (len(resi1) != pdb1_sel.natoms) or (len(resi2) != pdb2_sel.natoms):
        msg = ("Some selected residues where not found in the PDB file. Please "
               "check your input residues.")
        raise ValueError(msg)

    return pdb1_sel, pdb2_sel


def parse_restraints(fid, pdb1, pdb2):
    """Parse the restraints file."""

    dist_restraints = []

    for line in fid:
        # ignore comments and empty lines
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        chain1, resi1, name1, chain2, resi2, name2, mindis, maxdis = line.split()
        pdb1_sel = pdb1.select('chain', chain1).select('resi',
                int(resi1)).select('name', name1).duplicate()
        pdb2_sel = pdb2.select('chain', chain2).select('resi',
                int(resi2)).select('name', name2).duplicate()

        if pdb1_sel.natoms == 0 or pdb2_sel.natoms == 0:
            raise ValueError("A restraint selection was not found in line:\n{:s}".format(str(line)))

        dist_restraints.append([pdb1_sel, pdb2_sel, float(mindis), float(maxdis)])

    fid.close()
    return dist_restraints


class Joiner(object):
    def __init__(self, directory):
        self.directory = directory

    def __call__(self, fname):
        """Join fname with set directory."""
        return join(self.directory, fname)


class Results(object):
    """Simple container"""
    pass


def run_disvis_instance(queue, receptor, ligand, distance_restraints, rotmat,
        weights, n, pdb1_sel, pdb2_sel, args):
    """Run a single DisVis instance."""

    dv = DisVis()

    dv.receptor = receptor
    dv.ligand = ligand
    dv.distance_restraints = distance_restraints
    dv.rotations = rotmat
    dv.weights = weights

    dv.voxelspacing = args.voxelspacing
    dv.interaction_radius = args.interaction_radius
    dv.max_clash = args.max_clash
    dv.min_interaction = args.min_interaction
    dv.interaction_restraints_cutoff = args.interaction_restraints_cutoff

    if args.interaction_selection is not None:
        dv.receptor_interaction_selection = pdb1_sel
        dv.ligand_interaction_selection = pdb2_sel
    dv.occupancy_analysis = args.occupancy_analysis

    dv.search()

    # Save results to file, to be combined later
    joiner = Joiner(args.directory)
    fname = joiner('accessible_interaction_space_{:d}.mrc').format(n)
    dv.accessible_interaction_space.tofile(fname)

    fname = joiner('violations_{:d}.npy').format(n)
    np.save(fname, dv.violations)

    if dv.interaction_matrix is not None:
        fname = joiner('interaction_matrix_{:d}.npy'.format(n))
        np.save(fname, dv.interaction_matrix)

    if dv.occupancy_analysis:
        for key, value in dv.occupancy_grids.iteritems():
            fname = joiner('occupancy_{:d}_{:d}.mrc'.format(key, n))
            value.tofile(fname)

    queue.put(dv.accessible_complexes)


def mp_cpu_disvis(receptor, ligand, rotmat, weights, distance_restraints,
        pdb1_sel, pdb2_sel, args):
    """Run several DisVis instances, each with a subset of all rotations."""

    # multi-threaded CPU version
    try:
        max_cpu  = mp.cpu_count()
        jobs = min(max_cpu, args.nproc)
    except NotImplementedError:
        jobs = args.nproc
    # in case more processes are requested than the number
    # of rotations sampled
    nrot = rotmat.shape[0]
    if jobs > nrot:
        jobs = nrot
    nrot_per_job = nrot//jobs
    write('Number of processors used: {:d}'.format(jobs))
    write('Number of rotations per job: {:d}'.format(nrot_per_job))

    write('Creating jobs')

    queue = mp.Queue()
    processes = []
    for n in xrange(jobs):
        # Determine the rotations that each job needs to sample
        init_rot = n * nrot_per_job
        end_rot = (n + 1) * nrot_per_job
        if n == (jobs - 1):
            end_rot = None

        sub_rotmat = rotmat[init_rot: end_rot]
        sub_weights = weights[init_rot: end_rot]

        disvis_args = (queue, receptor, ligand, distance_restraints,
                sub_rotmat, sub_weights, n, pdb1_sel, pdb2_sel, args)
        process = mp.Process(target=run_disvis_instance, args=disvis_args)
        processes.append(process)

    write('Starting jobs')
    for p in processes:
        p.start()
    write('Waiting for jobs to finish')

    for p in processes:
        p.join()

    # Check whether the queue is empty, this indicates failure to run on
    # multi-processor runs.
    if queue.empty():
        raise mp.Queue.Empty

    write('Searching done. Combining results')

    # Create dummy class with similar results attributes as DisVis class
    results = Results()
    joiner = Joiner(args.directory)

    fname_interspace = joiner('accessible_interaction_space_{:d}.mrc')
    fname_violations = joiner('violations_{:d}.npy')
    fname_intermat = joiner('interaction_matrix_{:d}.npy')

    accessible_complexes = np.asarray(queue.get(), dtype=np.float64)
    accessible_interaction_space = Volume.fromfile(fname_interspace.format(0))
    violations = np.load(fname_violations.format(0))
    for n in xrange(1, jobs):
        accessible_complexes += np.asarray(queue.get(), dtype=np.float64)
        np.maximum(accessible_interaction_space.array,
                Volume.fromfile(fname_interspace.format(n)).array,
                accessible_interaction_space.array)
        violations += np.load(fname_violations.format(n))

    # Combine the occupancy grids
    occupancy = None
    if args.occupancy_analysis:
        fname_occupancy = joiner('occupancy_{:d}_{:d}.mrc')
        occupancy = {}
        for consistent_restraints in xrange(args.interaction_restraints_cutoff,
                len(distance_restraints) + 1):
            occupancy[consistent_restraints] = Volume.fromfile(
                    fname_occupancy.format(consistent_restraints, 0))
            for n in range(1, jobs):
                occupancy[consistent_restraints]._array += (
                        Volume.fromfile(fname_occupancy.format(consistent_restraints, n))._array
                        )

    # Combine the interaction analysis
    results.interaction_matrix = None
    if args.interaction_selection is not None:
        interaction_matrix = np.load(fname_intermat.format(0))
        for n in range(1, jobs):
            interaction_matrix += np.load(fname_intermat.format(n))
        results.interaction_matrix = interaction_matrix

    # Remove the intermediate files
    write('Cleaning')
    for n in xrange(jobs):
        remove(fname_interspace.format(n))
        remove(fname_violations.format(n))
        if args.interaction_selection is not None:
            remove(fname_intermat.format(n))
        if args.occupancy_analysis:
            for consistent_restraints in xrange(
                    args.interaction_restraints_cutoff, len(distance_restraints) + 1):
                remove(fname_occupancy.format(consistent_restraints, n))

    results.accessible_interaction_space = accessible_interaction_space
    results.accessible_complexes = accessible_complexes
    results.violations = violations
    results.occupancy_grids = occupancy

    return results


def run_disvis(queue, receptor, ligand, rotmat, weights, distance_restraints,
        pdb1_sel, pdb2_sel, args):

    dv = DisVis()

    dv.receptor = receptor
    dv.ligand = ligand
    dv.distance_restraints = distance_restraints
    dv.rotations = rotmat
    dv.weights = weights

    dv.voxelspacing = args.voxelspacing
    dv.interaction_radius = args.interaction_radius
    dv.max_clash = args.max_clash
    dv.min_interaction = args.min_interaction
    dv.queue = queue
    dv.occupancy_analysis = args.occupancy_analysis
    dv.interaction_restraints_cutoff = args.interaction_restraints_cutoff

    if not any([x is None for x in (pdb1_sel, pdb2_sel)]):
        dv.receptor_interaction_selection = pdb1_sel
        dv.ligand_interaction_selection = pdb2_sel
    dv.search()

    return dv


def write(line):
    if _stdout.isatty():
        print(line)
    logging.info(line)


def main():

    args = parse_args()

    mkdir_p(args.directory)
    joiner = Joiner(args.directory)

    logging.basicConfig(filename=joiner('disvis.log'),
            level=logging.INFO, format='%(asctime)s %(message)s')

    time0 = _time()

    queue = None
    if args.gpu:
        from disvis.helpers import get_queue
        queue = get_queue()
        if queue is None:
            raise ValueError("No GPU queue was found.")

    write('Reading fixed model from: {:s}'.format(args.receptor.name))
    receptor = PDB.fromfile(args.receptor)
    write('Reading scanning model from: {:s}'.format(args.ligand.name))
    ligand = PDB.fromfile(args.ligand)

    write('Reading in rotations.')
    q, weights, a = proportional_orientations(args.angle)
    rotmat = quat_to_rotmat(q)
    write('Requested rotational sampling density: {:.2f}'.format(args.angle))
    write('Real rotational sampling density: {:.2f}'.format(a))
    write('Number of rotations: {:d}'.format(rotmat.shape[0]))

    write('Reading in restraints from file: {:s}'.format(args.restraints.name))
    distance_restraints = parse_restraints(args.restraints, receptor, ligand)
    write('Number of distance restraints: {:d}'.format(len(distance_restraints)))

    # If the interaction restraints cutoff is not specified, only calculate the
    # interactions and occupancy grids for complexes consistent with at least 1
    # restraints or more, with a limit of three.
    if args.interaction_restraints_cutoff is None:
        args.interaction_restraints_cutoff = max(len(distance_restraints) - 3, 1)

    pdb1_sel = pdb2_sel = None
    if args.interaction_selection is not None:
        write('Reading in interaction selection from file: {:s}'
                .format(args.interaction_selection.name))
        pdb1_sel, pdb2_sel = parse_interaction_selection(
                args.interaction_selection, receptor, ligand)

        write('Number of receptor residues: {:d}'.format(pdb1_sel.natoms))
        write('Number of ligand residues: {:d}'.format(pdb2_sel.natoms))

    write('Voxel spacing set to: {:.2f}'.format(args.voxelspacing))
    write('Interaction radius set to: {:.2f}'.format(args.interaction_radius))
    write('Minimum required interaction volume: {:.2f}'.format(args.min_interaction))
    write('Maximum allowed volume of clashes: {:.2f}'.format(args.max_clash))
    if args.occupancy_analysis:
        write('Performing occupancy analysis')

    if queue is None:
        # CPU-version
        if args.nproc > 1:
            try:
                dv = mp_cpu_disvis(receptor, ligand, rotmat, weights,
                        distance_restraints, pdb1_sel, pdb2_sel, args)
            except Queue.Empty:
                msg = ('ERROR: Queue.Empty exception raised while processing job, '
                       'stopping execution ...')
                write(msg)
                exit(-1)
        else:
            dv = run_disvis(queue, receptor, ligand, rotmat, weights,
                    distance_restraints, pdb1_sel, pdb2_sel, args)
    else:
        # GPU-version
        write('Using GPU accelerated search.')
        dv = run_disvis(queue, receptor, ligand, rotmat, weights,
                         distance_restraints, pdb1_sel, pdb2_sel, args)

    # write out accessible interaction space
    fname = joiner('accessible_interaction_space.mrc')
    write('Writing accessible interaction space to: {:s}'.format(fname))
    dv.accessible_interaction_space.tofile(fname)

    # write out accessible complexes
    accessible_complexes = dv.accessible_complexes
    norm = sum(accessible_complexes)
    digits = len(str(int(norm))) + 1
    cum_complexes = np.cumsum(np.asarray(accessible_complexes)[::-1])[::-1]
    with open(joiner('accessible_complexes.out'), 'w') as f_accessible_complexes:
        write('Writing number of accessible complexes to: {:s}'.format(f_accessible_complexes.name))
        header = '# consistent restraints | accessible complexes |' +\
                 'relative | cumulative accessible complexes | relative\n'
        f_accessible_complexes.write(header)
        for n, acc in enumerate(accessible_complexes):
            f_accessible_complexes.write('{0:3d} {2:{1}d} {3:8.6f} {4:{1}d} {5:8.6f}\n'\
                    .format(n, digits, int(acc), acc/norm,
                    int(cum_complexes[n]), cum_complexes[n]/norm))

    # writing out violations
    violations = dv.violations
    cum_violations = violations[::-1].cumsum(axis=0)[::-1]
    with open(joiner('violations.out'), 'w') as f_viol:
        write('Writing violations to file: {:s}'.format(f_viol.name))
        num_violations = violations.sum(axis=1)
        nrestraints = num_violations.shape[0]
        header = ('# row represents the number of consistent restraints\n'
                  '# column represents how often that restraint is violated\n')
        f_viol.write(header)
        header = ('   ' + '{:8d}'*nrestraints + '\n').format(*range(1, nrestraints + 1))
        f_viol.write(header)
        for n, line in enumerate(cum_violations):
            f_viol.write('{:<2d} '.format(n+1))
            for word in line:
                if num_violations[n] > 0:
                    percentage_violated = word/cum_complexes[n+1]
                else:
                    percentage_violated = 0
                f_viol.write('{:8.4f}'.format(percentage_violated))
            f_viol.write('\n')

    # Give user indication for false positives.
    # Determine minimum number of false positives.
    nrestraints = len(distance_restraints)
    n = 1
    while accessible_complexes[-n] == 0:
        n += 1
    if n > 1:
        msg = ('Not all restraints are consistent. '
               'Number of false-positive restraints present '
               'is at least: {:d}'.format(n - 1))
        write(msg)

    # next give possible false-positives based on the percentage of violations
    # and their associated Z-score
    if n == 1:
        n = None
    else:
        n = -n + 1
    percentage_violated = cum_violations[:n]/np.asarray(cum_complexes[1:n]).reshape(-1, 1)
    average_restraint_violation = percentage_violated.mean(axis=0)
    std_restraint_violation = percentage_violated.std(axis=0)
    zscore_violations = ((average_restraint_violation - average_restraint_violation.mean())
            / average_restraint_violation.std())
    ind_false_positives = np.flatnonzero(zscore_violations >= 1.0)
    nfalse_positives = ind_false_positives.size
    if nfalse_positives > 0:
        ind_false_positives += 1
        write(('Possible false-positive restraints (z-score > 1.0):' +\
                ' {:d}'*nfalse_positives).format(*tuple(ind_false_positives)))

    with open(joiner('z-score.out'), 'w') as f:
        write('Writing z-score of each restraint to {:s}'.format(f.name))
        for n in xrange(zscore_violations.shape[0]):
            f.write('{:2d} {:6.3f} {:6.3f} {:6.3f}\n'.format(n+1,
                    average_restraint_violation[n], std_restraint_violation[n],
                    zscore_violations[n]))


    # Write all occupancy grids to MRC-files if requested
    if args.occupancy_analysis:
        for n, vol in dv.occupancy_grids.iteritems():
            # Normalize the occupancy grid
            if cum_complexes[n] > 0:
                vol._array /= cum_complexes[n]
            vol.tofile(joiner('occupancy_{:d}.mrc'.format(n)))

    # Write out interaction analysis
    # the interaction_matrix gives the number of interactions between each
    # residue of the receptor and ligand for complexes consistent with exactly
    # N restraints.
    interaction_matrix = dv.interaction_matrix
    if interaction_matrix is not None:

        ## Save interaction matrix
        #f = joiner('interaction_matrix.npy')
        #write('Writing interaction-matrix to: {:s}'.format(f))
        #np.save(f, interaction_matrix)

        # Save contacted receptor and ligand residue interaction for each analyzed number
        # of consistent restraints
        write('Writing contacted receptor residue interactions to files.')
        # Take the cumsum in order to give the number of interactions for complexes
        # with at least N restraints.
        receptor_cum_interactions = interaction_matrix.sum(axis=1)[::-1].cumsum(axis=0)[::-1]
        ligand_cum_interactions = interaction_matrix.sum(axis=2)[::-1].cumsum(axis=0)[::-1]
        fname = joiner('receptor_interactions.txt')
        with open(fname, 'w') as f:
            # Write header
            f.write('#resi')
            for consistent_restraints in xrange(args.interaction_restraints_cutoff, nrestraints + 1):
                f.write(' {:>6d}'.format(consistent_restraints))
            f.write('\n')

            for n, resi in enumerate(pdb1_sel.data['resi']):
                f.write('{:<5d}'.format(resi))
                for consistent_restraints in xrange(args.interaction_restraints_cutoff, nrestraints + 1):
                    index = consistent_restraints - args.interaction_restraints_cutoff
                    interactions = receptor_cum_interactions[index, n]
                    cum_complex = cum_complexes[consistent_restraints]
                    if cum_complex > 0:
                        relative_interactions = interactions / cum_complex
                    else:
                        relative_interactions = 0
                    f.write(' {:6.3f}'.format(relative_interactions))
                f.write('\n')

        fname = joiner('ligand_interactions.txt')
        with open(fname, 'w') as f:
            # Write header
            f.write('#resi')
            for consistent_restraints in xrange(args.interaction_restraints_cutoff, nrestraints + 1):
                f.write(' {:>6d}'.format(consistent_restraints))
            f.write('\n')

            for n, resi in enumerate(pdb2_sel.data['resi']):
                f.write('{:<5d}'.format(resi))
                for consistent_restraints in xrange(args.interaction_restraints_cutoff, nrestraints + 1):
                    index = consistent_restraints - args.interaction_restraints_cutoff
                    interactions = ligand_cum_interactions[index, n]
                    cum_complex = cum_complexes[consistent_restraints]
                    if cum_complex > 0:
                        relative_interactions = interactions / cum_complex
                    else:
                        relative_interactions = 0
                    f.write(' {:6.3f}'.format(relative_interactions))
                f.write('\n')

    # time indication
    seconds = int(round(_time() - time0))
    m, s = divmod(seconds, 60)
    write('Total time passed: {:d}m {:2d}s'.format(m, s))


if __name__=='__main__':
    main()
