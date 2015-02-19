import cython
import numpy as np
cimport numpy as np
from libc.math cimport ceil, exp, floor, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def count_violations(np.ndarray[np.float64_t, ndim=2] points,
        np.ndarray[np.float64_t, ndim=1] mindis,
        np.ndarray[np.float64_t, ndim=1] maxdis,
        np.ndarray[np.int32_t, ndim=3] interspace,
        double weight,
        np.ndarray[np.float64_t, ndim=2] violations):

    cdef unsigned int x, y, z, i
    cdef double distance2, mindis2, maxdis2

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
@cython.wraparound(False)
def longest_distance(np.ndarray[np.float64_t, ndim=2] coor):

    cdef unsigned int i, j
    cdef double dis2
    cdef double maxdis2 = 0

    for i in range(coor.shape[0] - 1):
        for j in range(i + 1, coor.shape[0]):

            dis2 = (coor[i, 0] - coor[j, 0])**2 +\
                   (coor[i, 1] - coor[j, 1])**2 +\
                   (coor[i, 2] - coor[j, 2])**2

            if dis2 > maxdis2:
                maxdis2 = dis2

    return sqrt(maxdis2)

@cython.boundscheck(False)
def rotate_image3d(np.ndarray[np.float64_t, ndim=3] image,
                   int vlength,
                   np.ndarray[np.float64_t, ndim=2] rotmat,
                   np.ndarray[np.float64_t, ndim=1] center,
                   np.ndarray[np.float64_t, ndim=3] out,
                   ):
    """Rotate an array around the origin using trilinear interpolation

    Parameters
    ----------
    image : ndarray

    vlenght : unsigned integer
        Vertice length

    rotmat : ndarray
        Rotation matrix.

    out : ndarray
        Output array
    """
    # looping
    cdef int x, y, z

    # rotation
    cdef double xcoor_z, ycoor_z, zcoor_z
    cdef double xcoor_yz, ycoor_yz, zcoor_yz
    cdef double xcoor_xyz, ycoor_xyz, zcoor_xyz

    # interpolation
    cdef int x0, y0, z0, x1, y1, z1
    cdef double dx, dy, dz, dx1, dy1, dz1
    cdef double c00, c01, c10, c11
    cdef double c0, c1, c

    for z in range(-vlength, vlength+1):

        xcoor_z = rotmat[0, 2]*z
        ycoor_z = rotmat[1, 2]*z
        zcoor_z = rotmat[2, 2]*z

        for y in range(-vlength, vlength+1):

            xcoor_yz = rotmat[0, 1]*y + xcoor_z
            ycoor_yz = rotmat[1, 1]*y + ycoor_z
            zcoor_yz = rotmat[2, 1]*y + zcoor_z

            for x in range(-vlength, vlength+1):

                xcoor_xyz = rotmat[0, 0]*x + xcoor_yz + center[0]
                ycoor_xyz = rotmat[1, 0]*x + ycoor_yz + center[1]
                zcoor_xyz = rotmat[2, 0]*x + zcoor_yz + center[2]

                # trilinear interpolation
                x0 = <int> floor(xcoor_xyz)
                y0 = <int> floor(ycoor_xyz)
                z0 = <int> floor(zcoor_xyz)

                x1 = x0 + 1
                y1 = y0 + 1
                z1 = z0 + 1

                dx = xcoor_xyz - <double> x0
                dy = ycoor_xyz - <double> y0
                dz = zcoor_xyz - <double> z0
                dx1 = 1 - dx
                dy1 = 1 - dy
                dz1 = 1 - dz

                c00 = image[z0, y0, x0]*dx1 + image[z0, y0, x1]*dx
                c10 = image[z0, y1, x0]*dx1 + image[z0, y1, x1]*dx
                c01 = image[z1, y0, x0]*dx1 + image[z1, y0, x1]*dx
                c11 = image[z1, y1, x0]*dx1 + image[z1, y1, x1]*dx

                c0 = c00*dy1 + c10*dy
                c1 = c01*dy1 + c11*dy

                c = c0*dz1 + c1*dz

                out[z, y, x] = c

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
            dx = x - points[n, 0]
            x2 = dx**2
            for y in range(ymin, ymax+1):
                dy = y - points[n, 1]
                x2y2 = x2 + dy**2
                for z in range(zmin, zmax+1):
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
    """Creates a mask from the points into the volume

    Parameters
    ----------
    points : ndarray

    out : 3D-numpy array

    Returns
    -------
    out
    """
    cdef unsigned int n
    cdef int x, y, z, xmin, ymin, zmin, xmax, ymax, zmax
    cdef double mindis2, maxdis2
    cdef double z2, y2z2, x2y2z2

    for n in range(points.shape[0]):

        mindis2 = mindis[n]**2
        maxdis2 = maxdis[n]**2

        xmin = <int> ceil(points[n, 0] - maxdis[n])
        ymin = <int> ceil(points[n, 1] - maxdis[n])
        zmin = <int> ceil(points[n, 2] - maxdis[n])

        xmax = <int> floor(points[n, 0] + maxdis[n])
        ymax = <int> floor(points[n, 1] + maxdis[n])
        zmax = <int> floor(points[n, 2] + maxdis[n])

        for z in range(zmin, zmax+1):
            if (z >= out.shape[0]) or (z < 0):
                continue

            z2 = (z - points[n, 2])**2

            for y in range(ymin, ymax+1):
                if (y >= out.shape[1]) or (y < 0):
                    continue

                y2z2 = (y - points[n, 1])**2 + z2

                for x in range(xmin, xmax+1):
                    if (x >= out.shape[2]) or (x < 0):
                        continue

                    x2y2z2 = (x - points[n, 0])**2 + y2z2
                    if mindis2 < x2y2z2 <= maxdis2:
                        out[z,y,x] += 1

    return out
