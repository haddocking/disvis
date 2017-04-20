import cython
import numpy as np
cimport numpy as np
from libc.math cimport ceil, exp, floor, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def count_interactions(np.ndarray[np.int32_t, ndim=3] interaction_space,
        np.ndarray[np.float64_t, ndim=2] r_inter_coor,
        np.ndarray[np.float64_t, ndim=2] l_inter_coor,
        double interaction_distance,
        double weight,
        int interaction_restraint_cutoff,
        np.ndarray[np.float64_t, ndim=3] interaction_matrix):

    cdef:
        int x, y, z, i, j, n
        double interaction_distance2 = interaction_distance**2
        double dist2, lx, ly, lz

    for z in range(interaction_space.shape[0]):
        for y in range(interaction_space.shape[1]):
            for x in range(interaction_space.shape[2]):
                n = interaction_space[z, y, x]
                if n < interaction_restraint_cutoff:
                    continue

                for j in range(l_inter_coor.shape[0]):
                    # Move the scanning coordinates
                    lx = l_inter_coor[j, 0] + x
                    ly = l_inter_coor[j, 1] + y
                    lz = l_inter_coor[j, 2] + z

                    for i in range(r_inter_coor.shape[0]):
                        dist2 = (r_inter_coor[i, 0] - lx)**2 +\
                                (r_inter_coor[i, 1] - ly)**2 +\
                                (r_inter_coor[i, 2] - lz)**2

                        if dist2 <= interaction_distance2:
                            interaction_matrix[n - interaction_restraint_cutoff, j, i] += weight


@cython.boundscheck(False)
@cython.wraparound(False)
def count_violations(np.ndarray[np.float64_t, ndim=2] points,
        np.ndarray[np.float64_t, ndim=1] mindis,
        np.ndarray[np.float64_t, ndim=1] maxdis,
        np.ndarray[np.int32_t, ndim=3] interspace,
        double weight,
        np.ndarray[np.float64_t, ndim=2] violations):

    cdef:
        unsigned int x, y, z, i
        double distance2, mindis2, maxdis2

    for z in range(interspace.shape[0]):
        for y in range(interspace.shape[1]):
            for x in range(interspace.shape[2]):

                if interspace[z, y, x] == 0:
                    continue

                for i in range(violations.shape[0]):
                    distance2 = (x - points[i, 0])**2 +\
                            (y - points[i, 1])**2 + (z - points[i, 2])**2
                    mindis2 = mindis[i]**2
                    maxdis2 = maxdis[i]**2

                    if distance2 < mindis2 or distance2 > maxdis2:
                        violations[interspace[z, y, x] - 1, i] += weight
