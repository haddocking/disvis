from __future__ import print_function, absolute_import, division
from sys import stdout as _stdout
from time import time as _time
from math import ceil

import numpy as np

# try to import pyfftw for faster FFT, else fall back on standard NumPy
# implementation
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(10)
    rfftn = pyfftw.interfaces.numpy_fft.rfftn
    irfftn = pyfftw.interfaces.numpy_fft.irfftn

except ImportError:
    from numpy.fft import rfftn, irfftn


from disvis import volume
from .points import dilate_points
from .libdisvis import (rotate_image3d, dilate_points_add, longest_distance, 
        distance_restraint, count_violations)
# try to import the pyopencl package for accessing the GPU. If it is not
# accessable just pass
try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    import disvis.pyclfft
    from .kernels import Kernels
    from disvis import pyclfft
except ImportError:
    pass

class DisVis(object):
    """Main object to perform the full-exhaustive search"""

    def __init__(self):
        # parameters to be defined
        self._receptor = None
        self._ligand = None

        # parameters with standard values
        self.rotations = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]
        self.weights = None
        self.voxelspacing = 1.0
        self.interaction_radius = 2.5
        self.max_clash = 100
        self.min_interaction = 300
        self.distance_restraints = []

        # CPU or GPU
        self._queue = None

        # unchangeable
        self._data = {}


    @property
    def receptor(self):
        """Get the fixed chain/receptor"""
        return self._receptor


    @receptor.setter
    def receptor(self, receptor):
        """Set the fixed chain/receptor"""
        self._receptor = receptor.duplicate()


    @property
    def ligand(self):
        """Get the scanning chain/ligand"""
        return self._ligand


    @ligand.setter
    def ligand(self, ligand):
        """Set the scanning chain/ligand"""
        self._ligand = ligand.duplicate()


    @property
    def rotations(self):
        """Get the rotations"""
        return self._rotations


    @rotations.setter
    def rotations(self, rotations):
        """Set the rotations."""
        rotmat = np.asarray(rotations, dtype=np.float64)

        if rotmat.ndim != 3:
            raise ValueError("Input should be a list of rotation matrices.")

        self._rotations = rotmat


    @property
    def weights(self):
        """Get the weights of each rotation sampled."""
        return self._weights


    @weights.setter
    def weights(self, weights):
        """Set the weights of each rotation sampled."""
        self._weights = weights


    @property
    def interaction_radius(self):
        """Get the interaction-radius"""
        return self._interaction_radius


    @interaction_radius.setter
    def interaction_radius(self, radius):
        """Set the interaction radius"""
        if radius <= 0:
            raise ValueError("Interaction radius should be bigger than zero")
        self._interaction_radius = radius


    @property
    def voxelspacing(self):
        """Get the voxel spacing."""
        return self._voxelspacing


    @voxelspacing.setter
    def voxelspacing(self, voxelspacing):
        """Set the voxel spacing."""
        self._voxelspacing = voxelspacing


    @property
    def max_clash(self):
        """Get the maximum allowed volume of clashes"""
        return self._max_clash


    @max_clash.setter
    def max_clash(self, max_clash):
        """Set the maximum allowed volume of clashes"""
        if max_clash < 0:
            raise ValueError("Maximum allowed clashing volume cannot be negative")
        self._max_clash = max_clash + 0.9


    @property
    def min_interaction(self):
        """Get the minimum required volume of interaction"""
        return self._min_interaction


    @min_interaction.setter
    def min_interaction(self, min_interaction):
        """Set the minimum required volume of interaction"""
        if min_interaction < 1:
            raise ValueError("Minimum required interaction volume cannot be smaller than 1")
        self._min_interaction = min_interaction + 0.9
        

    @property
    def queue(self):
        """Get the queue, required for GPU-accelerated search"""
        return self._queue


    @queue.setter
    def queue(self, queue):
        """Set the queue, required for GPU-accelerated search"""
        self._queue = queue


    @property
    def data(self):
        """Get dictionary that contains all the arrays and info for search"""
        return self._data


    def add_distance_restraint(self, receptor_selection, ligand_selection, mindis, maxdis):
        """Add a distance restraint"""
        # a distance restraint is internally represented by a PDB-object of the
        # receptor-selection and ligand-selection together with a mindistance
        # and maxdistance
        distance_restraint = [receptor_selection, ligand_selection, mindis, maxdis]
        self.distance_restraints.append(distance_restraint)


    def _initialize(self):
        """Initialize all the arrays and data required to perfrom the search.
        
        This method transforms the atomic structures into the search objects
        and projects them onto a grid for FFT-accelerated search.
        """

        # check if requirements are set
        if any(x is None for x in (self.receptor, self.ligand)):
            raise ValueError("Not all requirements are met for a search")

        # if no weights are set for the rotations, make an array with size
        # equal to the number of rotations and set them to one
        if self.weights is None:
            self.weights = np.ones(self.rotations.shape[0], dtype=np.float64)

        # check if the number of weights is equal to the number of rotations
        if len(self.weights) != len(self.rotations):
            raise ValueError("")

        d = self.data

        # determine the size of the grid onto which the receptor and ligand
        # will be projected
        shape = grid_shape(self.receptor.coor, self.ligand.coor, self.voxelspacing)

        # calculate the interaction and core object of the receptor
        # the core object is essentially the van der Waals volume of the
        # molecule, while the interaction object extends the van der Waals
        # radius of each element by the interaction_radius parameter
        vdw_radii = self.receptor.vdw_radius
        radii = vdw_radii + self.interaction_radius
        d['rsurf'] = rsurface(self.receptor.coor, radii, 
                shape, self.voxelspacing)
        d['rcore'] = rsurface(self.receptor.coor, vdw_radii, 
                shape, self.voxelspacing)

        # keep track of some data for later calculations concerning the
        # placement of the grid
        d['origin'] = d['rcore'].origin
        d['shape'] = d['rcore'].shape
        d['start'] = d['rcore'].start
        d['nrot'] = self.rotations.shape[0]

        # ligand center is needed for distance calculations during search
        d['lcenter'] = self.ligand.center

        # set ligand center to the origin of the receptor map
        # and create the scanning chain object projected onto a grid
        radii = self.ligand.vdw_radius
        d['lsurf'] = dilate_points((self.ligand.coor - self.ligand.center \
                + self.receptor.center), radii, volume.zeros_like(d['rcore']))

        # determine the center of image, as this is the point where it will be
        # rotated around during the search
        d['im_center'] = np.asarray((self.receptor.center -
        d['rcore'].origin)/self.voxelspacing, dtype=np.float64)

        # the max_clash and min_interaction volumes were given in angstrom^3,
        # but is now in grid volume
        d['max_clash'] = self.max_clash/self.voxelspacing**3
        d['min_interaction'] = self.min_interaction/self.voxelspacing**3

        # setup the distance restraints
        d['nrestraints'] = len(self.distance_restraints)
        if self.distance_restraints:
            d['restraints'] = grid_restraints(self.distance_restraints, 
                    self.voxelspacing, d['origin'], d['lcenter'])


    def search(self):
        """Main method to perform the search."""

        # initialize the calculations
        self._initialize()

        # if the queue is not set, perform a CPU search, else use
        # GPU-acceleration
        if self.queue is None:
            self._cpu_init()
            self._cpu_search()
        else:
            self._gpu_init()
            self._gpu_search()

        # perform an extra print line if the output is an interactive shell,
        # for proper output
        if _stdout.isatty():
            print()

        # make the accessible interaction space a Volume object, for easier
        # manipulation later
        accessible_interaction_space = \
                volume.Volume(self.data['accessible_interaction_space'], 
                        self.voxelspacing, self.data['origin'])

        return accessible_interaction_space, self.data['accessible_complexes'], self.data['violations']


    def _cpu_init(self):
        """Initialize all the arrays and data required for a CPU search"""

        self.cpu_data = {}
        c = self.cpu_data
        d = self.data

        # create views of data in cpu_data
        c['rcore'] = d['rcore'].array
        c['rsurf'] = d['rsurf'].array
        c['im_lsurf'] = d['lsurf'].array
        c['restraints'] = d['restraints']

        # allocate arrays used for search
        # real arrays
        c['lsurf'] = np.zeros_like(c['rcore'])
        c['clashvol'] = np.zeros_like(c['rcore'])
        c['intervol'] = np.zeros_like(c['rcore'])
        c['interspace'] = np.zeros_like(c['rcore'], dtype=np.int32)
        c['access_interspace'] = np.zeros_like(c['rcore'], dtype=np.int32)
        c['restspace'] = np.zeros_like(c['rcore'], dtype=np.int32)
        c['violations'] = np.zeros((d['nrestraints'], d['nrestraints']), dtype=np.float64)

        # complex arrays
        c['ft_shape'] = list(d['shape'])
        c['ft_shape'][-1] = d['shape'][-1]//2 + 1
        c['ft_lsurf'] = np.zeros(c['ft_shape'], dtype=np.complex128)
        c['ft_rcore'] = np.zeros(c['ft_shape'], dtype=np.complex128)
        c['ft_rsurf'] = np.zeros(c['ft_shape'], dtype=np.complex128)

        c['rotmat'] = np.asarray(self.rotations, dtype=np.float64)
        c['weights'] = np.asarray(self.weights, dtype=np.float64)

        c['nrot'] = d['nrot']
        c['shape'] = d['shape']
        c['max_clash'] = d['max_clash']
        c['min_interaction'] = d['min_interaction']

        # the vlenght is the longest distance from the ligand center to another
        # atom. This makes the rotation of the ligand object cheaper by only
        # rotation the inner part of the array where there is density
        c['vlength'] = int(np.linalg.norm(self.ligand.coor - \
                self.ligand.center, axis=1).max() + \
                self.interaction_radius + 1.5)/self.voxelspacing

        # initial calculations. Calculate the FFT of the fixed chain objects.
        # This only needs to be done once before the search.
        c['ft_rcore'] = rfftn(c['rcore'])
        c['ft_rsurf'] = rfftn(c['rsurf'])


    def _cpu_search(self):
        """Method which performs the exhaustive search using CPU resources"""

        d = self.data
        c = self.cpu_data

        # initialize the number of total sampled complexes and the number of
        # complexes consistent with exactly N restraints
        tot_complex = 0
        list_total_allowed = np.zeros(max(2, d['nrestraints'] + 1), dtype=np.float64)

        # initalize the time
        time0 = _time()

        for n in xrange(c['rotmat'].shape[0]):

            # rotate the scanning chain object. The rotation needs to be
            # inverted, as we are rotating the array, instead of the object.
            rotate_image3d(c['im_lsurf'], c['vlength'], 
                    np.linalg.inv(c['rotmat'][n]), d['im_center'], c['lsurf'])

            # calculate the clashing and interaction volume at every position
            # in space using FFTs.
            np.conj(rfftn(c['lsurf']), c['ft_lsurf'])
            c['clashvol'] = irfftn(c['ft_lsurf'] * c['ft_rcore'], s=c['shape'])
            c['intervol'] = irfftn(c['ft_lsurf'] * c['ft_rsurf'], s=c['shape'])

            # Calculate the accessible interaction space for the current
            # rotation. The clashing volume should not be too high, and the
            # interaction volume of a reasonable size
            np.logical_and(c['clashvol'] < c['max_clash'],
                           c['intervol'] > c['min_interaction'],
                           c['interspace'])

            # Calculate the number of complexes and multiply with the weight
            # for the orientation to correct for rotational/orientational bias
            tot_complex += c['weights'][n] * c['interspace'].sum()

            # if distance-restraints are available
            if self.distance_restraints:
                c['restspace'].fill(0)

                # determine the center of the distance-restraint consistent
                # spheres
                rest_center = d['restraints'][:, :3] - \
                        (np.mat(c['rotmat'][n]) * \
                        np.mat(d['restraints'][:,3:6]).T).T

                mindis = d['restraints'][:,6]
                maxdis = d['restraints'][:,7]
                # Markate the space that is consistent with the distance restraints
                distance_restraint(rest_center, mindis, maxdis, c['restspace'])

                # Multiply the interaction space with the distance-restraint
                # consistent space
                c['interspace'] *= c['restspace']

                # Now count which violation has been violated
                count_violations(rest_center, mindis, maxdis, c['interspace'], c['weights'][n], c['violations'])

            # To visualize the accessible interaction space, keep the maximum
            # number of consistent restraints found at every position in space
            np.maximum(c['interspace'], c['access_interspace'],
                       c['access_interspace'])

            # Keep track of the number of accessible complexes consistent with
            # EXACTLY N restraints. Again, correct for the
            # rotational/orientation bias
            list_total_allowed += c['weights'][n] *\
                        np.bincount(c['interspace'].ravel(),
                        minlength=(max(2, d['nrestraints']+1)))

            # Give the user information on progress if it is used interactively
            if _stdout.isatty():
                self._print_progress(n, c['nrot'], time0)

        # attach the output on the self.data dictionary
        # the accessible interaction space which will be visualized
        d['accessible_interaction_space'] = c['access_interspace']
        # the number of accessible complexes consistent with EXACTLY a certain number of restraints
        # the number of accessible complexes consistent with EXACTLY a certain
        # number of restraints. To account for this, the number of total
        # sampled complexes needs to be reduced by the number of complexes
        # consistent with 1 or more restraints
        d['accessible_complexes'] = [tot_complex - sum(list_total_allowed[1:])] + list(list_total_allowed[1:])
        # the violation matrix
        d['violations'] = c['violations']


    def _print_progress(self, n, total, time0):
        """Method to print the progress of the search"""
        m = n + 1
        pdone = m/total
        t = _time() - time0
        _stdout.write('\r{:d}/{:d} ({:.2%}, ETA: {:d}s)    '\
                .format(m, total, pdone, 
                        int(t/pdone - t)))
        _stdout.flush()


    def _gpu_init(self):
        """Method to initialize all the data for GPU-accelerate search"""

        self.gpu_data = {}
        g = self.gpu_data
        d = self.data
        q = self.queue

        # move data to the GPU. All should be float32, as these is the native
        # lenght for GPUs
        g['rcore'] = cl_array.to_device(q, float32array(d['rcore'].array))
        g['rsurf'] = cl_array.to_device(q, float32array(d['rsurf'].array))
        # Make the scanning chain object an Image, as this is faster to rotate
        g['im_lsurf'] = cl.image_from_array(q.context, float32array(d['lsurf'].array))
        g['sampler'] = cl.Sampler(q.context, False, cl.addressing_mode.CLAMP,
                                  cl.filter_mode.LINEAR)

        if self.distance_restraints:
            g['restraints'] = cl_array.to_device(q, float32array(d['restraints']))

        # Allocate arrays on the GPU
        g['lsurf'] = cl_array.zeros_like(g['rcore'])
        g['clashvol'] = cl_array.zeros_like(g['rcore'])
        g['intervol'] = cl_array.zeros_like(g['rcore'])
        g['interspace'] = cl_array.zeros(q, d['shape'], dtype=np.int32)
        g['restspace'] = cl_array.zeros_like(g['interspace'])
        g['access_interspace'] = cl_array.zeros_like(g['interspace'])
        g['best_access_interspace'] = cl_array.zeros_like(g['interspace'])

        # arrays for counting
        # Reductions are typically tedious on GPU, and we need to define the
        # workgroupsize to allocate the correct amount of data
        WORKGROUPSIZE = 32
        nsubhists = int(np.ceil(g['rcore'].size/WORKGROUPSIZE))
        g['subhists'] = cl_array.zeros(q, (nsubhists, d['nrestraints'] + 1), dtype=np.float32)
        g['viol_counter'] = cl_array.zeros(q, (nsubhists, d['nrestraints'], d['nrestraints']), dtype=np.float32)

        # complex arrays
        g['ft_shape'] = list(d['shape'])
        g['ft_shape'][0] = d['shape'][0]//2 + 1
        g['ft_rcore'] = cl_array.zeros(q, g['ft_shape'], dtype=np.complex64)
        g['ft_rsurf'] = cl_array.zeros_like(g['ft_rcore'])
        g['ft_lsurf'] = cl_array.zeros_like(g['ft_rcore'])
        g['ft_clashvol'] = cl_array.zeros_like(g['ft_rcore'])
        g['ft_intervol'] = cl_array.zeros_like(g['ft_rcore'])

        # other miscellanious data
        g['nrot'] = d['nrot']
        g['max_clash'] = d['max_clash']
        g['min_interaction'] = d['min_interaction']

        # kernels
        g['k'] = Kernels(q.context)
        g['k'].rfftn = pyclfft.RFFTn(q.context, d['shape'])
        g['k'].irfftn = pyclfft.iRFFTn(q.context, d['shape'])

        # initial calculations
        g['k'].rfftn(q, g['rcore'], g['ft_rcore'])
        g['k'].rfftn(q, g['rsurf'], g['ft_rsurf'])



    def _gpu_search(self):
        """Method that actually performs the exhaustive search on the GPU"""

        # make shortcuts
        d = self.data
        g = self.gpu_data
        q = self.queue
        k = g['k']

        # initalize the total number of sampled complexes
        tot_complexes = cl_array.sum(g['interspace'], dtype=np.float32)

        # initialize time
        time0 = _time()

        # loop over all rotations
        for n in xrange(g['nrot']):

            # rotate the scanning chain object
            k.rotate_image3d(q, g['sampler'], g['im_lsurf'],
                    self.rotations[n], g['lsurf'], d['im_center'])

            # perform the FFTs and calculate the clashing and interaction volume
            k.rfftn(q, g['lsurf'], g['ft_lsurf'])
            k.c_conj_multiply(q, g['ft_lsurf'], g['ft_rcore'], g['ft_clashvol'])
            k.irfftn(q, g['ft_clashvol'], g['clashvol'])

            k.c_conj_multiply(q, g['ft_lsurf'], g['ft_rsurf'], g['ft_intervol'])
            k.irfftn(q, g['ft_intervol'], g['intervol'])

            # determine at every position if the conformation is a proper complex
            k.touch(q, g['clashvol'], g['max_clash'],
                    g['intervol'], g['min_interaction'],
                    g['interspace'])

            if self.distance_restraints:
                k.fill(q, g['restspace'], 0)

                # determine the space that is consistent with a number of
                # distance restraints
                k.distance_restraint(q, g['restraints'],
                        self.rotations[n], g['restspace'])

                # get the accessible interaction space also consistent with a
                # certain number of distance restraints
                k.multiply(q, g['restspace'], g['interspace'], g['access_interspace'])


            # calculate the total number of complexes, while taking into
            # account orientational/rotational bias
            tot_complexes += cl_array.sum(g['interspace'], dtype=np.float32)*np.float32(self.weights[n])

            # take at every position in space the maximum number of consistent
            # restraints for later visualization
            cl_array.maximum(g['best_access_interspace'], g['access_interspace'], g['best_access_interspace'])

            # calculate the number of accessable complexes consistent with
            # EXACTLY N distance restraints
            k.histogram(q, g['access_interspace'], g['subhists'], self.weights[n], d['nrestraints'])

            # Count the violations of each restraint for all complexes
            # consistent with EXACTLY N restraints
            k.count_violations(q, g['restraints'], self.rotations[n], 
                    g['access_interspace'], g['viol_counter'], self.weights[n])

            # inform user
            if _stdout.isatty():
                self._print_progress(n, g['nrot'], time0)

        # wait for calculations to finish
        self.queue.finish()

        # transfer the data from GPU to CPU
        # get the number of accessible complexes and reduce the subhistograms
        # to the final histogram
        access_complexes = g['subhists'].get().sum(axis=0)
        # account for the fact that we are counting the number of accessible
        # complexes consistent with EXACTLY N restraints
        access_complexes[0] = tot_complexes.get() - sum(access_complexes[1:])
        d['accessible_complexes'] = access_complexes
        d['accessible_interaction_space'] = g['best_access_interspace'].get()

        # get the violation submatrices and reduce it to the final violation
        # matrix
        d['violations'] = g['viol_counter'].get().sum(axis=0)


def rsurface(points, radius, shape, voxelspacing):
    """Calculate a shape out of the points"""

    dimensions = [x*voxelspacing for x in shape]
    origin = volume_origin(points, dimensions)
    rsurf = volume.zeros(shape, voxelspacing, origin)

    rsurf = dilate_points(points, radius, rsurf)

    return rsurf


def volume_origin(points, dimensions):
    """Determines the origin of the volume in space"""

    center = points.mean(axis=0)
    origin = [(c - d/2.0) for c, d in zip(center, dimensions)]

    return origin
    

def grid_restraints(restraints, voxelspacing, origin, lcenter):
    """Transform the distance restraints given in angstrom to grid
    coordinates"""

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


def grid_shape(points1, points2, voxelspacing):
    """Get the shape of the grid given two point sets"""

    shape = min_grid_shape(points1, points2, voxelspacing)
    # the array-lenght should be a multiply of 2, 3 and 5 to work on the GPU,
    # as this is required for the FFT-plans in clFFT
    shape = [volume.radix235(x) for x in shape]
    return shape


def min_grid_shape(points1, points2, voxelspacing):
    # the minimal grid shape is the size of the fixed protein in 
    # each dimension and the longest diameter is the scanning chain
    dimensions1 = points1.ptp(axis=0)
    dimension2 = longest_distance(points2)

    grid_shape = np.asarray(((dimensions1 + dimension2)/voxelspacing) + 10, dtype=np.int32)[::-1]

    return grid_shape


def float32array(array_like):
    """Get the array in float32 values"""
    return np.asarray(array_like, dtype=np.float32)
