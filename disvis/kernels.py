from os.path import dirname, join
from string import Template

import numpy as np
import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel

class Kernels():

    def __init__(self, ctx, values):
        self.context = ctx

        self.source_file = join(dirname(__file__), 'kernels.cl')
        with open(self.source_file) as f:
            t = Template(f.read()).substitute(**values)

        self.program = cl.Program(ctx, t).build()
        # Global and local workitems for all kernels
        self._gws_rotate_points3d = (8 * 32 * 8 * 8,)
        self._gws_rotate_grid3d = (96, 64, 1)
        self._gws_dilate_point_add = (32, 32, 1)

        self._lws_histogram = (96,)
        self._gws_histogram = (96 * 32,)

        self._lws_count_violations = (96, 1, 1)
        self._gws_count_violations = (96, 64, 1)

        self._lws_count_interactions = (48, 1, 1)
        self._gws_count_interactions = (96, 96, 1)


        # Simple elementwise kernels
        self.multiply_add = ElementwiseKernel(ctx,
                "int *x, float w, float* y",
                "y[i] += w * x[i];",
                )

        self.multiply_add2 = ElementwiseKernel(ctx,
                "float *x, float w, float* y",
                "y[i] += w * x[i];",
                )

        self.multiply_f32 = ElementwiseKernel(ctx,
                     "float *x, float *y, float *z",
                     "z[i] = x[i] * y[i];",
                     )

        self.multiply_int32 = ElementwiseKernel(ctx,
                     "int *x, int *y, int *z",
                     "z[i] = x[i] * y[i];",
                     )

        self.subtract = ElementwiseKernel(ctx,
                "float *in1, float *in2, float *out",
                """out[i] = in1[i] - in2[i];
                """,
                )

        self.conj = ElementwiseKernel(ctx,
                "cfloat_t* in, cfloat_t *out",
                "out[i] = cfloat_conj(in[i]);",
                )

        self.cmultiply = ElementwiseKernel(ctx,
                "cfloat_t *in1, cfloat_t *in2, cfloat_t *out",
                "out[i] = cfloat_mul(in1[i], in2[i]);",
                )

        self.round = ElementwiseKernel(ctx,
                "float *in, float* out",
                "out[i] = round(in[i]);",
                )

        self.less_equal = ElementwiseKernel(ctx,
                "float* array, float value, int *out",
                "out[i] = (array[i] <= value) ? 1 : 0;",
                )

        self.greater_equal = ElementwiseKernel(ctx,
                "float* array, float value, int *out",
                "out[i] = (array[i] >= value) ? 1 : 0;",
                )

        self.greater_equal_iif = ElementwiseKernel(ctx,
                "int* array, int value, float *out",
                "out[i] = (array[i] >= value) ? 1.0f : 0;",
                )

        self.equal = ElementwiseKernel(ctx,
                "int *x, int y, float *z",
                "z[i] = (x[i] == y) ? 1.0f : 0;",
                )

        self.logical_and = ElementwiseKernel(ctx,
                "int* in1, int *in2, int *out",
                "out[i] = ((in1[i] == 1) && (in2[i] == 1)) ? 1 : 0;",
                )
        
        self.set_to_f32 = ElementwiseKernel(ctx,
                "float value, float *out",
                "out[i] = value;",
                )

        self.set_to_i32 = ElementwiseKernel(ctx,
                "int value, int *out",
                "out[i] = value;",
                )

    def rotate_points3d(self, queue, points, rotmat, out):
        args = (points.data, np.int32(points.shape[0]), rotmat, out.data)
        self.program.rotate_points3d(queue, self._gws_rotate_points3d, None, *args)

    def rotate_grid3d(self, queue, grid, rotmat, out):
        args = (grid.data, rotmat, out.data)
        self.program.rotate_grid3d(queue, self._gws_rotate_grid3d, None, *args)

    def dilate_point_add(self, queue, centers, mindis, maxdis, restspace):
        args = (centers.data, mindis.data, maxdis.data, restspace.data)
        self.program.dilate_point_add(queue, self._gws_dilate_point_add, None, *args)

    def histogram(self, queue, data, hist):
        args = (data.data, hist.data)
        self.program.histogram(queue, self._gws_histogram, self._lws_histogram, *args)

    def count_violations(self, queue, centers, mindis2, maxdis2, interspace, viol):
        args = (centers.data, mindis2.data, maxdis2.data, interspace.data, viol.data)
        self.program.count_violations(queue, self._gws_count_violations,
                self._lws_count_violations, *args)

    def count_interactions(self, queue, fixed_coor, scanning_coor, interspace,
            nconsistent, hist):
        args = (fixed_coor.data, scanning_coor.data, interspace.data,
                np.int32(nconsistent), hist.data)
        self.program.count_interactions(queue, self._gws_count_interactions,
                self._lws_count_interactions, *args)
