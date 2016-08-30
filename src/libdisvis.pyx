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


@cython.boundscheck(False)
def dilate_points(np.ndarray[np.float64_t, ndim=2] points,
                  np.ndarray[np.float64_t, ndim=1] radius,
                  np.ndarray[np.float64_t, ndim=3] out,
                  ):
    """

    Parameters
    ----------
    points : ndarray
        Coordinates in (x, y, z)

    radius : float
	The radius of the dilation

    out : 3D-numpy array

    Returns
    -------
    out
    """
    cdef unsigned int n
    cdef int x, y, z, xmin, ymin, zmin, xmax, ymax, zmax
    cdef double radius2
    cdef double dx, x2, dy, x2y2, dz, distance2

    for n in range(points.shape[0]):

        radius2 = radius[n]**2

        xmin = <int> ceil(points[n, 0] - radius[n])
        ymin = <int> ceil(points[n, 1] - radius[n])
        zmin = <int> ceil(points[n, 2] - radius[n])

        xmax = <int> floor(points[n, 0] + radius[n])
        ymax = <int> floor(points[n, 1] + radius[n])
        zmax = <int> floor(points[n, 2] + radius[n])

        for x in range(xmin, xmax+1):
            if (abs(x) >= out.shape[2]):
                continue
            dx = x - points[n, 0]
            x2 = dx**2
            for y in range(ymin, ymax+1):
                if (abs(y) >= out.shape[1]):
                    continue
                dy = y - points[n, 1]
                x2y2 = x2 + dy**2
                for z in range(zmin, zmax+1):
                    if (abs(z) >= out.shape[0]):
                        continue
                    dz = z - points[n, 2]
                    distance2 = x2y2 + dz**2
                    if distance2 <= radius2:
                        out[z,y,x] = 1.0
    return out


@cython.boundscheck(False)
def dilate_points_add(np.ndarray[np.float64_t, ndim=2] points,
                  np.ndarray[np.float64_t, ndim=1] radius,
                  np.ndarray[np.int32_t, ndim=3] out,
                  ):
    """Creates a mask from the points into the volume

    Parameters
    ----------
    points : ndarray

    radius : float

    out : 3D-numpy array

    Returns
    -------
    out
    """
    cdef unsigned int n
    cdef int x, y, z, xmin, ymin, zmin, xmax, ymax, zmax
    cdef double radius2
    cdef double dx, x2, dy, x2y2, dz, distance2

    for n in range(points.shape[0]):

        radius2 = radius[n]**2
        xmin = <int> ceil(points[n, 0] - radius[n])
        ymin = <int> ceil(points[n, 1] - radius[n])
        zmin = <int> ceil(points[n, 2] - radius[n])

        xmax = <int> floor(points[n, 0] + radius[n])
        ymax = <int> floor(points[n, 1] + radius[n])
        zmax = <int> floor(points[n, 2] + radius[n])

        for x in range(xmin, xmax+1):
            dx = x - points[n, 0]
            x2 = dx**2
            for y in range(ymin, ymax+1):
                dy = y - points[n, 1]
                x2y2 = x2 + dy**2
                for z in range(zmin, zmax+1):
                    dz = z - points[n, 2]
                    distance2 = x2y2 + dz**2
                    if distance2 <= radius2:
                        out[z,y,x] += 1

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def binary_erosion(np.ndarray[np.float64_t, ndim=3] image,
                   np.ndarray[np.float64_t, ndim=3] out,
                   ):

    cdef int nz, ny, nx, x, y, z
    nz = <int> image.shape[0]
    ny = <int> image.shape[1]
    nx = <int> image.shape[2]

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if image[z, y, x] > 0:
                    if x > 0:
                        if image[z, y, x - 1] == 0:
                            out[z, y, x] = 0
                            continue
                    if x < nx - 1:
                        if image[z, y, x + 1] == 0:
                            out[z, y, x] = 0
                            continue

                    if y > 0:
                        if image[z, y - 1, x] == 0:
                            out[z, y, x] = 0
                            continue
                    if y < ny - 1:
                        if image[z, y+1, x] == 0:
                            out[z, y, x] = 0
                            continue
                    if z > 0:
                        if image[z - 1, y, x] == 0:
                            out[z, y, x] = 0
                            continue
                    if z < nz - 1:
                        if image[z+1, y, x] == 0:
                            out[z, y, x] = 0
                            continue
                    out[z, y, x] = image[z, y, x]
                else:
                    out[z, y, x] = 0

    return out


@cython.boundscheck(False)
def distance_restraint(np.ndarray[np.float64_t, ndim=2] points,
                  np.ndarray[np.float64_t, ndim=1] mindis,
                  np.ndarray[np.float64_t, ndim=1] maxdis,
                  np.ndarray[np.int32_t, ndim=3] out,
                  ):
    cdef:
        unsigned int n
        int x, y, z, xmin, ymin, zmin, xmax, ymax, zmax
        double mindis2, maxdis2
        double z2, y2z2, x2y2z2

    for n in range(points.shape[0]):

        mindis2 = mindis[n]**2
        maxdis2 = maxdis[n]**2

        xmin = max(<int> ceil(points[n, 0] - maxdis[n]), 0)
        ymin = max(<int> ceil(points[n, 1] - maxdis[n]), 0)
        zmin = max(<int> ceil(points[n, 2] - maxdis[n]), 0)

        xmax = min(<int> floor(points[n, 0] + maxdis[n]), out.shape[2] - 1)
        ymax = min(<int> floor(points[n, 1] + maxdis[n]), out.shape[1] - 1)
        zmax = min(<int> floor(points[n, 2] + maxdis[n]), out.shape[0] - 1)

        for z in range(zmin, zmax+1):

            z2 = (z - points[n, 2])**2
            if z2 > maxdis2:
                continue

            for y in range(ymin, ymax+1):

                y2z2 = (y - points[n, 1])**2 + z2
                if y2z2 > maxdis2:
                    continue

                for x in range(xmin, xmax+1):

                    x2y2z2 = (x - points[n, 0])**2 + y2z2
                    if x2y2z2 <= maxdis2 and x2y2z2 >= mindis2:
                        out[z,y,x] += 1

    return out
