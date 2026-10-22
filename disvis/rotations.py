"""Rotation matrices and rotational sampling sets."""
import os.path
from math import cos, sin, sqrt
from random import random
import numpy as np


def Rx(theta):
    """Return the matrix of a rotation by theta radians around the x-axis."""
    return np.asarray([[1,          0,           0],
                       [0, cos(theta), -sin(theta)],
                       [0, sin(theta),  cos(theta)]], dtype=np.float64)


def Ry(theta):
    """Return the matrix of a rotation by theta radians around the y-axis."""
    return np.asarray([[cos(theta),  0, sin(theta)],
                       [0,           1,          0],
                       [-sin(theta), 0, cos(theta)]], dtype=np.float64)


def Rz(theta):
    """Return the matrix of a rotation by theta radians around the z-axis."""
    return np.asarray([[cos(theta), -sin(theta), 0],
                       [sin(theta),  cos(theta), 0],
                       [0,                    0, 1]], dtype=np.float64)


def quat_to_rotmat(quaternions, invert=False):
    """Convert quaternions to rotation matrices.

    quaternions is an (N, 4) array with the scalar part first. With invert,
    the inverse rotations are returned. Note that invert changes the vector
    part of quaternions in place. Returns an (N, 3, 3) array, rounded to 8
    decimals.
    """
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    if invert:
        x *= -1
        y *= -1
        z *= -1

    Nq = w**2 + x**2 + y**2 + z**2
    s = np.zeros(Nq.shape, dtype=np.float64)
    s[Nq > 0.0] = 2.0 / Nq[Nq > 0.0]
    s[Nq <= 0.0] = 0

    X = x * s
    Y = y * s
    Z = z * s

    rotmat = np.zeros((quaternions.shape[0], 3, 3), dtype=np.float64)
    rotmat[:, 0, 0] = 1.0 - (y * Y + z * Z)
    rotmat[:, 0, 1] = x * Y - w * Z
    rotmat[:, 0, 2] = x * Z + w * Y

    rotmat[:, 1, 0] = x * Y + w * Z
    rotmat[:, 1, 1] = 1.0 - (x * X + z * Z)
    rotmat[:, 1, 2] = y * Z - w * X

    rotmat[:, 2, 0] = x * Z - w * Y
    rotmat[:, 2, 1] = y * Z + w * X
    rotmat[:, 2, 2] = 1.0 - (x * X + y * Y)

    np.around(rotmat, decimals=8, out=rotmat)

    return rotmat


def random_rotation():
    """Return a uniformly distributed random rotation matrix as a list."""
    while True:
        e1 = random() * 2 - 1
        e2 = random() * 2 - 1
        s1 = e1**2 + e2**2
        if (s1 < 1.0):
            break

    while True:
        e3 = random() * 2 - 1
        e4 = random() * 2 - 1
        s2 = e3**2 + e4**2
        if (s2 < 1.0):
            break

    q0 = e1
    q1 = e2
    q2 = e3 * sqrt((1 - s1) / s2)
    q3 = e4 * sqrt((1 - s1) / s2)

    a11 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    a12 = 2 * (q1 * q2 - q0 * q3)
    a13 = 2 * (q1 * q3 + q0 * q2)
    a21 = 2 * (q1 * q2 + q0 * q3)
    a22 = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
    a23 = 2 * (q2 * q3 - q0 * q1)
    a31 = 2 * (q1 * q3 - q0 * q2)
    a32 = 2 * (q2 * q3 + q0 * q1)
    a33 = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3

    return [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]


def proportional_orientations(angle):
    """Return the rotation set closest to the requested sampling angle.

    The angle is the rotational sampling interval in degrees. Returns the
    quaternions, their weights and the actual sampling angle of the set.
    """
    # orientation sets available: name of file: (Norientations, degree)
    rot_sets = {'E.npy': (1, 360.0),
                'c48u1.npy': (24, 62.8),
                'c600v.npy': (60, 44.48),
                'c48n9.npy': (216, 36.47),
                'c600vc.npy': (360, 27.78),
                'c48u27.npy': (648, 20.83),
                'c48u83.npy': (1992, 16.29),
                'c48u181.npy': (4344, 12.29),
                'c48n309.npy': (7416, 9.72),
                'c48n527.npy': (12648, 8.17),
                'c48u815.npy': (19560, 7.4),
                'c48u1153.npy': (27672, 6.6),
                'c48u1201.npy': (28824, 6.48),
                'c48u1641.npy': (39384, 5.75),
                'c48u2219.npy': (53256, 5.27),
                'c48u2947.npy': (70728, 4.71),
                'c48u3733.npy': (89592, 4.37),
                'c48u4749.npy': (113976, 4.0),
                'c48u5879.npy': (141096, 3.74),
                'c48u7111.npy': (170664, 3.53),
                'c48u8649.npy': (207576, 3.26),
                }

    # determine the apropiate set to use
    smallestdiff = None
    for s, n in rot_sets.items():
        alpha = n[1]
        diff = abs(angle - alpha)
        if smallestdiff is None or diff < smallestdiff:
            smallestdiff = diff
            fname = s

    # read file
    infile = os.path.join(os.path.dirname(__file__), 'data', fname)
    quat_weights = np.load(infile)

    quat = quat_weights[:, :4]
    weights = quat_weights[:, -1]
    alpha = rot_sets[fname][1]

    return quat, weights, alpha
