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
