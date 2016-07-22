from __future__ import print_function, absolute_import, division
from sys import stdout as _stdout
from time import clock as _clock
from collections import defaultdict

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
    import disvis.pyclfft
    from .kernels import Kernels
    from disvis import pyclfft
    import pyopencl.array as cl_array
    PYOPENCL = True
except ImportError:
    PYOPENCL = False

from disvis import volume
from .pdb import PDB
from .libdisvis import (
        rotate_grid_nearest, dilate_points, distance_restraint,
        count_violations, count_interactions
        )

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
        #else:
        #    self._gpu_init()
        #    self._gpu_search()

        # Set the results
        self.accessible_interaction_space = volume.Volume(
                self._access_interspace, self.voxelspacing,
                self._origin)
        self.accessible_complexes = self._accessible_complexes
        self.violations = self._violations

        if self.occupancy_analysis:
            for i in xrange(self._nrestraints + 1):
                self.occupancy_grids = defaultdict(None)
                occ_grid = self._occ_grid[i]
                if occ_grid is not None:
                    self.occupancy_grids[i] = Volume(occ_grid, self.voxelspacing, self._origin)

        if self._interaction_analysis:
            self.interaction_matrix = self._interaction_matrix

    @staticmethod
    def _minimal_volume_parameters(fixed_coor, scanning_coor, offset, voxelspacing):
        # the minimal grid shape is the size of the fixed protein in 
        # each dimension and the longest diameter of the scanning chain

        offset += np.linalg.norm(scanning_coor - scanning_coor.mean(axis=0), axis=1).max()
        mindist = fixed_coor.min(axis=0) - offset
        maxdist = fixed_coor.max(axis=0) + offset
        shape = [volume.radix235(int(np.ceil(x))) 
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

        self._accessible_complexes = np.zeros(self._nrestraints + 1, dtype=np.float64)
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
        self._occ_grid = defaultdict(None)
        if self.occupancy_analysis:
            for i in xrange(self.interaction_restraints_cutoff, self._nrestraints + 1):
                self._occ_grid[i] = np.zeros(self._shape, np.float64)

        # Check if we want to do an interaction analysis, i.e. whether
        # interface residues are given for both the ligand and receptor.
        selection = (self.ligand_interaction_selection, self.receptor_interaction_selection)
        self._interaction_analysis = any(x is not None for x in selection)
        if self._interaction_analysis:
            # Since calculating all interactions is costly, only analyze
            # solutions that are consistent with more than N restraints. By
            shape = (self.interaction_restraints_cutoff, 
                    self._lgridcoor.shape[0], self._rgridcoor.shape[0])
            self._interaction_matrix = np.zeros(shape, dtype=np.float64)
            self._sub_interaction_matrix = np.zeros(shape, dtype=np.int64)

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
            out_arr = self._irfftn(in_arr)
        return out_arr

    def _cpu_init(self):

        # Allocate arrays for FFT's
        # Real arrays
        for arr in 'rot_lcore clashvol intervol tmp'.split():
            setattr(self, '_' + arr, self._allocate_array(self._shape, np.float64, self._fftw))
        # Complex arrays
        self._ft_shape = list(self._shape)[:-1] + [self._shape[-1] // 2 + 1]
        for arr in 'lcore rcore rsurf tmp'.split():
            setattr(self, '_ft_' + arr,
                    self._allocate_array(self._ft_shape, np.complex128, self._fftw))

        # Integer arrays
        for arr in 'interspace access_interspace restspace'.split():
            setattr(self, "_" + arr, np.zeros(self._shape, np.int32))

        # Boolean arrays
        for arr in 'not_clashing interacting'.split():
            setattr(self, "_" + arr, np.zeros(self._shape, np.bool))

        # Array for rotating points and restraint coordinates
        self._rot_lgridcoor = np.zeros_like(self._lgridcoor)
        self._rrestraints = self._grid_restraints[:, 0:3]
        self._lrestraints = self._grid_restraints[:, 3:6]
        self._mindis = self._grid_restraints[:,6]
        self._maxdis = self._grid_restraints[:,7]
        self._rot_lrestraints = np.zeros_like(self._lrestraints)
        self._restraints_center = np.zeros_like(self._grid_restraints[:,3:6])

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

        # Calculate the longest distance in the lcore. This helps in making the
        # grid rotation faster, as less points need to be considered for
        # rotation
        self._llength = int(np.ceil(
            np.linalg.norm(self._lgridcoor, axis=1).max() + 
            self.ligand.vdw_radius.max() / self.voxelspacing
            ))

        # Keep track of number of consistent complexes
        self._tot_complex = 0
        self._consistent_complexes = np.zeros(self._nrestraints + 1, dtype=np.float64)

    def _rotate_ligand(self, rotmat):
        rotate_grid_nearest(self._lcore, self._llength, rotmat,
                self._rot_lcore)
        
    def _get_interaction_space(self):
        # Calculate the clashing and interaction volume
        self._ft_lcore = self.rfftn(self._rot_lcore, self._ft_lcore)
        np.conjugate(self._ft_lcore, self._ft_lcore)
        np.multiply(self._ft_lcore, self._ft_rcore, self._ft_tmp)
        self._clashvol = self.irfftn(self._ft_tmp, self._clashvol)
        np.multiply(self._ft_lcore, self._ft_rsurf, self._ft_tmp)
        self._intervol = self.irfftn(self._ft_tmp, self._intervol)

        # Determine the interaction space, i.e. all the translations where
        # the receptor and ligand are interacting and not clashing
        np.less_equal(self._clashvol, self._grid_max_clash, self._not_clashing)
        np.greater_equal(self._intervol, self._grid_min_interaction, self._interacting)
        np.logical_and(self._not_clashing, self._interacting, self._interspace)

    def _rotate_restraints(self, rotmat):
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
        np.multiply(self._interspace, self._restspace, self._interspace)

    def _count_complexes(self, weight):
        self._tot_complex += weight * self._interspace.sum()
        self._consistent_complexes += weight *\
                    np.bincount(self._interspace.ravel(),
                            minlength=(max(2, self._nrestraints + 1))
                            )

    def _count_violations(self, weight):
        count_violations(self._restraints_center, self._mindis,
                self._maxdis, self._interspace, weight,
                self._violations)

    def _get_occupancy_grids(self, weight):
        for i in xrange(self.interaction_restraints_cutoff, self._nrestraints + 1):
            np.equal(self._interspace, np.int32(i), self._tmp)
            self._ft_tmp = self.rfftn(self._tmp, self._ft_tmp)
            np.multiply(self._ft_tmp, self._ft_lcore, self._ft_tmp)
            self._tmp = self.irfftn(self._ft_tmp, self._tmp)
            self._occ_grid[i] += weight * self._tmp

    def _get_interaction_matrix(self, rotmat, weight):
        # Rotate the ligand coordinates
        #np.dot(self._lgridcoor, rotmat.T, self._rot_lgridcoor)
        self._rot_lgridcoor = np.dot(self._lgridcoor, rotmat.T)
        #print(self._lgridcoor.shape)
        #print(rotmat.shape)
        #print(self._rot_lgridcoor.shape)
        count_interactions(self._interspace, self._rgridcoor,
                self._rot_lgridcoor, np.float64(self.interaction_distance / self.voxelspacing),
                weight, np.int32(self.interaction_restraints_cutoff),
                self._interaction_matrix)

    def _cpu_search(self):

        time0 = _clock()
        for n in xrange(self.rotations.shape[0]):
            rotmat = self.rotations[n]
            weight = self.weights[n]
            # Rotate the ligand grid
            self._rotate_ligand(rotmat)

            self._get_interaction_space()

            # Rotate the restraints
            self._rotate_restraints(rotmat)

            # Determine the consistent restraint space
            self._get_restraint_space()

            # Calculate the reduced accessible interaction
            self._get_reduced_interspace()

            # Perform some statistics, such as the number of restraint
            # violations and the number of accessible complexes consistent with
            # exactly N restraints.
            self._count_complexes(weight)
            self._count_violations(weight)

            # Calculate an occupancy grid for complexes consistent with at
            # least i restraints
            if self.occupancy_analysis:
                self._get_occupancy_grids(weight)

            # Calculate shapes for visual information, such as the highest
            # number of consisistent restraints found at each grid point
            np.maximum(self._interspace, self._access_interspace,
                    self._access_interspace)

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
        # Normalize the occupancy grids
        if self.occupancy_analysis:
            for i in xrange(self.interaction_restraints_cutoff, self._nrestraints + 1):
                self._occ_grid[i] /= self._consistent_complexes[i:].sum()
        
    @staticmethod
    def _print_progress(n, total, time0):
        m = n + 1
        pdone = m/total
        t = _clock() - time0
        _stdout.write('\r{:d}/{:d} ({:.2%}, ETA: {:d}s)    '\
                .format(m, total, pdone, 
                        int(t/pdone - t)))
        _stdout.flush()

#    def _gpu_init(self):
#
#        self.gpu_data = {}
#        g = self.gpu_data
#        d = self.data
#        q = self.queue
#
#        # Move arrays to GPU
#        g['rcore'] = cl_array.to_device(q, float32array(d['rcore'].array))
#        g['rsurf'] = cl_array.to_device(q, float32array(d['rsurf'].array))
#        g['im_lsurf'] = cl.image_from_array(q.context, float32array(d['lsurf'].array))
#        g['sampler'] = cl.Sampler(q.context, False, cl.addressing_mode.CLAMP,
#                                  cl.filter_mode.LINEAR)
#
#        if self.distance_restraints:
#            g['restraints'] = cl_array.to_device(q, float32array(d['restraints']))
#
#        # Allocate real arrays
#        arr_names = ('lsurf clashvol intervol interspace restspace ' +
#                'access_interspace best_access_interspace').split()
#        arr_types = [np.float32] * 3 + [np.int32] * 4
#        for arr_name, arr_type in zip(arr_names, arr_types):
#            g[arr_name] = cl_array.zeros(q, d['shape'], dtype=arr_type)
#
#        # arrays for counting
#        WORKGROUPSIZE = 32
#        g['subhists'] = cl_array.zeros(q, 
#                (g['rcore'].size, d['nrestraints'] + 1), 
#                dtype=np.float32)
#        g['viol_counter'] = cl_array.zeros(q, 
#                (g['rcore'].size, d['nrestraints'], d['nrestraints']), 
#                dtype=np.float32)
#
#        # Allocate complex arrays
#        g['ft_shape'] = [d['shape'][0] // 2 + 1] + list(d['shape'][1:])
#        g['ft_rcore'] = cl_array.zeros(q, g['ft_shape'], dtype=np.complex64)
#        for arr_name in 'rsurf lsurf clashvol intervol'.split():
#            g['ft_' + arr_name] = cl_array.zeros_like(q, g['ft_rcore'])
#
#        # kernels
#        g['k'] = Kernels(q.context)
#        g['k'].rfftn = pyclfft.RFFTn(q.context, d['shape'])
#        g['k'].irfftn = pyclfft.iRFFTn(q.context, d['shape'])
#
#        g['k'].rfftn(q, g['rcore'], g['ft_rcore'])
#        g['k'].rfftn(q, g['rsurf'], g['ft_rsurf'])
#
#        g['nrot'] = d['nrot']
#        g['max_clash'] = d['max_clash']
#        g['min_interaction'] = d['min_interaction']
#
#
#    def _gpu_search(self):
#        d = self.data
#        g = self.gpu_data
#        q = self.queue
#        k = g['k']
#
#        tot_complexes = cl_array.sum(g['interspace'], dtype=np.float32)
#
#        time0 = _clock()
#        for n in xrange(g['nrot']):
#
#            # Rotate the ligand
#            k.rotate_image3d(q, g['sampler'], g['im_lsurf'],
#                    self.rotations[n], g['lsurf'], d['im_center'])
#
#            # Calculate the clashing and interaction volume for each translation
#            k.rfftn(q, g['lsurf'], g['ft_lsurf'])
#            k.c_conj_multiply(q, g['ft_lsurf'], g['ft_rcore'], g['ft_clashvol'])
#            k.irfftn(q, g['ft_clashvol'], g['clashvol'])
#
#            k.c_conj_multiply(q, g['ft_lsurf'], g['ft_rsurf'], g['ft_intervol'])
#            k.irfftn(q, g['ft_intervol'], g['intervol'])
#
#            # Determine the translations where the subunits form a complex
#            k.touch(q, g['clashvol'], g['max_clash'],
#                    g['intervol'], g['min_interaction'],
#                    g['interspace'])
#
#            # Determine for each translation how many restraints are consistent
#            k.fill(q, g['restspace'], 0)
#            k.distance_restraint(q, g['restraints'],
#                    self.rotations[n], g['restspace'])
#
#            # Check for each complex how many restraints are consistent
#            k.multiply(q, g['restspace'], g['interspace'], g['access_interspace'])
#
#            # TODO
#            # Calculate an average occupancy grid to provide an average shape
#            # for several number of consistent restraints
#            #for i, cl_occ_grid in enumerate(self._cl_occ_grids):
#            #    # Get a grid for all translations that are consistent with at least
#            #    # N restraints
#            #    k.equal_to(q, g['access_interspace'], np.int32(self.nrestraints - i),
#            #            g['sub_access_interspace'])
#            #    k.rfftn(q, g['sub_access_interspace'], g['ft_sub_access_interspace'])
#            #    k.multiply(q, g['ft_lsurf'], g['ft_sub_access_interspace'], g['ft_occ_grid'])
#            #    k.irfftn(q, g['ft_occ_grid'], g['occ_grid'])
#            #    k.add(q, cl_occ_grid, self.weights[n], self._cl_occ_grid[i])
#
#
#            # Perform some statistics on the data
#            tot_complexes += (cl_array.sum(g['interspace'], dtype=np.float32) *
#                             np.float32(self.weights[n]))
#            cl_array.maximum(g['best_access_interspace'],
#                    g['access_interspace'], g['best_access_interspace'])
#
#            # Count the number of complexes consistent with exactly N restraints
#            k.histogram(q, g['access_interspace'], g['subhists'],
#                    self.weights[n], d['nrestraints'])
#
#            # Count how often a restraint is violated for all complexes
#            # consistent with exactly N restraints
#            k.count_violations(q, g['restraints'], self.rotations[n], 
#                    g['access_interspace'], g['viol_counter'], self.weights[n])
#
#            # TODO
#            # Count the number of interactions each residue has for all
#            # complexes consistent with exactly N restraints
#
#            if _stdout.isatty():
#                self._print_progress(n, g['nrot'], time0)
#
#        self.queue.finish()
#
#        # Combine the subhistograms
#        access_complexes = g['subhists'].get().sum(axis=0)
#        access_complexes[0] = tot_complexes.get() - sum(access_complexes[1:])
#        access_interaction_space = g['best_access_interspace'].get()
#
#        d['accessible_interaction_space'] = access_interaction_space 
#        d['accessible_complexes'] = access_complexes
#        d['violations'] = g['viol_counter'].get().sum(axis=0)


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


def float32array(array_like):
    return np.asarray(array_like, dtype=np.float32)
