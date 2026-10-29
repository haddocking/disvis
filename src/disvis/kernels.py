"""OpenCL kernels for the GPU-accelerated search."""

from os.path import dirname, join
from string import Template

import numpy as np
import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel


class Kernels:
    """The OpenCL kernels used by DisVis, compiled for one context.

    The kernels of kernels.cl are compiled after substituting values, a
    dictionary with the grid shape and other constants of the search.
    The methods launch these kernels on a queue; the attributes such as
    multiply_add are elementwise kernels, called as kernel(*args).
    """

    def __init__(self, ctx, values):
        self.context = ctx

        self.source_file = join(dirname(__file__), 'kernels.cl')
        with open(self.source_file, encoding='utf-8') as f:
            t = Template(f.read()).substitute(**values)

        self.program = cl.Program(ctx, t).build()
        # Retrieve every kernel once: each program.<name> access creates a
        # new kernel object.
        self._kernels = {
            k.function_name: k for k in self.program.all_kernels()
        }
        # Global and local workitems for all kernels
        self._gws_rotate_points3d = (8 * 32 * 8 * 8,)
        self._gws_rotate_grid3d = (96, 64, 1)
        self._gws_reduced_interspace = (96 * 256,)

        self._lws_histogram = (96,)
        self._gws_histogram = (96 * 32,)

        self._lws_count_violations = (96, 1, 1)
        self._gws_count_violations = (96, 64, 1)

        self._lws_count_interactions = (48, 1, 1)
        self._gws_count_interactions = (96, 96, 1)

        # Simple elementwise kernels
        self.multiply_add = ElementwiseKernel(
            ctx,
            "int *x, float w, float* y",
            "y[i] += w * x[i];",
        )

        self.multiply_add2 = ElementwiseKernel(
            ctx,
            "float *x, float w, float* y",
            "y[i] += w * x[i];",
        )

        self.multiply_f32 = ElementwiseKernel(
            ctx,
            "float *x, float *y, float *z",
            "z[i] = x[i] * y[i];",
        )

        self.multiply_int32 = ElementwiseKernel(
            ctx,
            "int *x, int *y, int *z",
            "z[i] = x[i] * y[i];",
        )

        self.subtract = ElementwiseKernel(
            ctx,
            "float *in1, float *in2, float *out",
            """out[i] = in1[i] - in2[i];
                """,
        )

        self.conj_multiply = ElementwiseKernel(
            ctx,
            "cfloat_t *in1, cfloat_t *in2, cfloat_t *out",
            "out[i] = cfloat_mul(cfloat_conj(in1[i]), in2[i]);",
        )

        # Interaction space: not clashing and interacting enough, after
        # rounding the volumes to whole voxels.
        self.interaction_space = ElementwiseKernel(
            ctx,
            "float *clashvol, float *intervol, float max_clash, "
            "float min_interaction, int *out",
            "out[i] = ((round(clashvol[i]) <= max_clash) && "
            "(round(intervol[i]) >= min_interaction)) ? 1 : 0;",
        )

        self.conj = ElementwiseKernel(
            ctx,
            "cfloat_t* in, cfloat_t *out",
            "out[i] = cfloat_conj(in[i]);",
        )

        self.cmultiply = ElementwiseKernel(
            ctx,
            "cfloat_t *in1, cfloat_t *in2, cfloat_t *out",
            "out[i] = cfloat_mul(in1[i], in2[i]);",
        )

        self.round = ElementwiseKernel(
            ctx,
            "float *in, float* out",
            "out[i] = round(in[i]);",
        )

        self.less_equal = ElementwiseKernel(
            ctx,
            "float* array, float value, int *out",
            "out[i] = (array[i] <= value) ? 1 : 0;",
        )

        self.greater_equal = ElementwiseKernel(
            ctx,
            "float* array, float value, int *out",
            "out[i] = (array[i] >= value) ? 1 : 0;",
        )

        self.greater_equal_iif = ElementwiseKernel(
            ctx,
            "int* array, int value, float *out",
            "out[i] = (array[i] >= value) ? 1.0f : 0;",
        )

        self.equal = ElementwiseKernel(
            ctx,
            "int *x, int y, float *z",
            "z[i] = (x[i] == y) ? 1.0f : 0;",
        )

        self.logical_and = ElementwiseKernel(
            ctx,
            "int* in1, int *in2, int *out",
            "out[i] = ((in1[i] == 1) && (in2[i] == 1)) ? 1 : 0;",
        )

        self.set_to_f32 = ElementwiseKernel(
            ctx,
            "float value, float *out",
            "out[i] = value;",
        )

        self.set_to_i32 = ElementwiseKernel(
            ctx,
            "int value, int *out",
            "out[i] = value;",
        )

    def rotate_points3d(self, queue, points, rotmat, out):
        """Rotate the points by rotmat (a float16) and store them in out."""
        args = (points.data, np.int32(points.shape[0]), rotmat, out.data)
        self._kernels['rotate_points3d'](
            queue, self._gws_rotate_points3d, None, *args
        )

    def rotate_grid3d(self, queue, grid, rotmat, out):
        """Rotate the grid around its origin into out.

        Only grid points within the ligand radius are rotated, with
        nearest-neighbour interpolation.
        """
        args = (grid.data, rotmat, out.data)
        self._kernels['rotate_grid3d'](
            queue, self._gws_rotate_grid3d, None, *args
        )

    def reduced_interspace(
        self, queue, centers, mindis, maxdis, interspace, red, access
    ):
        """Count the consistent restraints in the interaction space.

        red is set to the number of restraints each translation in
        interspace (nonzero) is consistent with, and 0 elsewhere; access is
        set to the maximum of access and red. centers holds the restraint
        centers in grid units, mindis and maxdis the distance ranges.
        """
        args = (
            centers.data,
            mindis.data,
            maxdis.data,
            interspace.data,
            red.data,
            access.data,
        )
        self._kernels['reduced_interspace'](
            queue, self._gws_reduced_interspace, None, *args
        )

    def histogram(self, queue, data, hist):
        """Add to hist[n - 1] the number of grid points of data equal to n.

        Grid points with value 0 are not counted.
        """
        args = (data.data, hist.data)
        self._kernels['histogram'](
            queue, self._gws_histogram, self._lws_histogram, *args
        )

    def count_violations(
        self, queue, centers, mindis2, maxdis2, interspace, viol
    ):
        """Count how often each restraint is violated, and add it to viol.

        viol[n - 1, i] counts the violations of restraint i in the complexes
        consistent with n restraints. centers holds the restraint centers in
        grid units, mindis2 and maxdis2 the squared distance ranges.
        """
        args = (
            centers.data,
            mindis2.data,
            maxdis2.data,
            interspace.data,
            viol.data,
        )
        self._kernels['count_violations'](
            queue,
            self._gws_count_violations,
            self._lws_count_violations,
            *args,
        )

    def count_interactions(
        self, queue, fixed_coor, scanning_coor, interspace, nconsistent, hist
    ):
        """Count the interactions between the selected residues.

        For the complexes consistent with exactly nconsistent restraints,
        hist[i, j] is incremented for each complex in which residue i of
        the scanning molecule is within the interaction cutoff of residue
        j of the fixed molecule.
        """
        args = (
            fixed_coor.data,
            scanning_coor.data,
            interspace.data,
            np.int32(nconsistent),
            hist.data,
        )
        self._kernels['count_interactions'](
            queue,
            self._gws_count_interactions,
            self._lws_count_interactions,
            *args,
        )
