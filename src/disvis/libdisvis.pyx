"""Grid operations of the CPU search, written in Cython.

All grids are indexed as [z, y, x], and all coordinates and distances are in
grid units.
"""
import cython
import numpy as np
cimport numpy as np
from libc.math cimport ceil, floor


@cython.boundscheck(False)
@cython.wraparound(False)
def count_interactions(np.ndarray[np.int32_t, ndim=3] interaction_space,
                       np.ndarray[np.float64_t, ndim=2] r_inter_coor,
                       np.ndarray[np.float64_t, ndim=2] l_inter_coor,
                       double interaction_distance,
                       double weight,
                       int interaction_restraint_cutoff,
                       np.ndarray[np.float64_t, ndim=3] interaction_matrix):
    """Count the interactions between the selected residues.

    For every ligand position in interaction_space consistent with n >=
    interaction_restraint_cutoff restraints, weight is added to
    interaction_matrix[n - interaction_restraint_cutoff, j, i] when ligand
    atom j (l_inter_coor, translated to the position) is within
    interaction_distance of receptor atom i (r_inter_coor).
    """
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
                        dist2 = ((r_inter_coor[i, 0] - lx)**2
                                 + (r_inter_coor[i, 1] - ly)**2
                                 + (r_inter_coor[i, 2] - lz)**2)

                        if dist2 <= interaction_distance2:
                            interaction_matrix[
                                n - interaction_restraint_cutoff, j, i
                            ] += weight


@cython.boundscheck(False)
@cython.wraparound(False)
def count_violations(np.ndarray[np.int32_t, ndim=3] interspace,
                     np.ndarray[np.uint8_t, ndim=4] consistent_bits,
                     double weight,
                     np.ndarray[np.float64_t, ndim=2] violations):
    """Count how often each restraint is violated.

    For every ligand position in interspace consistent with n > 0
    restraints, weight is added to violations[n - 1, i] when bit i of
    consistent_bits at that position (set by distance_restraint) is not set.
    """
    cdef:
        unsigned int x, y, z, i

    for z in range(interspace.shape[0]):
        for y in range(interspace.shape[1]):
            for x in range(interspace.shape[2]):

                if interspace[z, y, x] == 0:
                    continue

                for i in range(violations.shape[0]):
                    if not consistent_bits[z, y, x, i >> 3] & (1 << (i & 7)):
                        violations[interspace[z, y, x] - 1, i] += weight


@cython.boundscheck(False)
def dilate_points(np.ndarray[np.float64_t, ndim=2] points,
                  np.ndarray[np.float64_t, ndim=1] radius,
                  np.ndarray[np.float64_t, ndim=3] out,
                  ):
    """Set out to 1 within radius[n] of each point n.

    points is an (N, 3) array of (x, y, z) coordinates. Negative grid
    indices wrap around, so that a molecule centered on the origin is
    mapped periodically. Returns out.
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

        for x in range(xmin, xmax + 1):
            if (abs(x) >= out.shape[2]):
                continue
            dx = x - points[n, 0]
            x2 = dx**2
            for y in range(ymin, ymax + 1):
                if (abs(y) >= out.shape[1]):
                    continue
                dy = y - points[n, 1]
                x2y2 = x2 + dy**2
                for z in range(zmin, zmax + 1):
                    if (abs(z) >= out.shape[0]):
                        continue
                    dz = z - points[n, 2]
                    distance2 = x2y2 + dz**2
                    if distance2 <= radius2:
                        out[z, y, x] = 1.0
    return out


@cython.boundscheck(False)
def dilate_points_add(np.ndarray[np.float64_t, ndim=2] points,
                      np.ndarray[np.float64_t, ndim=1] radius,
                      np.ndarray[np.int32_t, ndim=3] out,
                      ):
    """Add 1 to out within radius[n] of each point n.

    points is an (N, 3) array of (x, y, z) coordinates; all grid points
    within the radius must lie inside out. Returns out.
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

        for x in range(xmin, xmax + 1):
            dx = x - points[n, 0]
            x2 = dx**2
            for y in range(ymin, ymax + 1):
                dy = y - points[n, 1]
                x2y2 = x2 + dy**2
                for z in range(zmin, zmax + 1):
                    dz = z - points[n, 2]
                    distance2 = x2y2 + dz**2
                    if distance2 <= radius2:
                        out[z, y, x] += 1

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def binary_erosion(np.ndarray[np.float64_t, ndim=3] image,
                   np.ndarray[np.float64_t, ndim=3] out,
                   ):
    """Erode image by one voxel into out and return out.

    A nonzero voxel is kept when none of its six neighbours inside the grid
    is zero; all other voxels are set to 0.
    """
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
                        if image[z, y + 1, x] == 0:
                            out[z, y, x] = 0
                            continue
                    if z > 0:
                        if image[z - 1, y, x] == 0:
                            out[z, y, x] = 0
                            continue
                    if z < nz - 1:
                        if image[z + 1, y, x] == 0:
                            out[z, y, x] = 0
                            continue
                    out[z, y, x] = image[z, y, x]
                else:
                    out[z, y, x] = 0

    return out


@cython.boundscheck(False)
def distance_restraint(np.ndarray[np.float64_t, ndim=2] points,
                       np.ndarray[np.int32_t, ndim=1] offsets,
                       np.ndarray[np.float64_t, ndim=1] mindis,
                       np.ndarray[np.float64_t, ndim=1] maxdis,
                       np.ndarray[np.int32_t, ndim=3] out,
                       np.ndarray[np.uint8_t, ndim=4] consistent_bits,
                       ):
    """Add 1 to out where each restraint n is consistent.

    Restraint n is consistent between mindis[n] and maxdis[n] from any of
    its points, points[offsets[n]] to points[offsets[n + 1] - 1], given as
    (x, y, z). The result counts, for each ligand position, the number of
    consistent distance restraints; bit n of consistent_bits[z, y, x] is set
    where restraint n is consistent, so that it is counted once. Grid points
    outside out are skipped. Returns out.
    """
    cdef:
        unsigned int n, p
        int x, y, z, xmin, ymin, zmin, xmax, ymax, zmax
        unsigned char byte, bit
        double mindis2, maxdis2
        double z2, y2z2, x2y2z2

    for n in range(offsets.shape[0] - 1):

        mindis2 = mindis[n]**2
        maxdis2 = maxdis[n]**2
        byte = n >> 3
        bit = 1 << (n & 7)

        for p in range(offsets[n], offsets[n + 1]):

            xmin = max(<int> ceil(points[p, 0] - maxdis[n]), 0)
            ymin = max(<int> ceil(points[p, 1] - maxdis[n]), 0)
            zmin = max(<int> ceil(points[p, 2] - maxdis[n]), 0)

            xmax = min(<int> floor(points[p, 0] + maxdis[n]), out.shape[2] - 1)
            ymax = min(<int> floor(points[p, 1] + maxdis[n]), out.shape[1] - 1)
            zmax = min(<int> floor(points[p, 2] + maxdis[n]), out.shape[0] - 1)

            for z in range(zmin, zmax + 1):

                z2 = (z - points[p, 2])**2
                if z2 > maxdis2:
                    continue

                for y in range(ymin, ymax + 1):

                    y2z2 = (y - points[p, 1])**2 + z2
                    if y2z2 > maxdis2:
                        continue

                    for x in range(xmin, xmax + 1):

                        x2y2z2 = (x - points[p, 0])**2 + y2z2
                        if (x2y2z2 <= maxdis2 and x2y2z2 >= mindis2
                                and not consistent_bits[z, y, x, byte] & bit):
                            consistent_bits[z, y, x, byte] |= bit
                            out[z, y, x] += 1

    return out
