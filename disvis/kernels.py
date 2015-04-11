from __future__ import print_function
import numpy as np
import os.path
import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel

class Kernels():

    def __init__(self, ctx):
        self.context = ctx

        self.kernel_file = os.path.join(os.path.dirname(__file__), 'kernels', 'kernels.cl')
        self.kernels = cl.Program(ctx, open(self.kernel_file).read()).build()

        self.kernels.multiply_f32 = ElementwiseKernel(ctx,
                     "float *x, float *y, float *z",
                     "z[i] = x[i] * y[i]",
                     )

        self.kernels.multiply_int32 = ElementwiseKernel(ctx,
                     "int *x, int *y, int *z",
                     "z[i] = x[i] * y[i]",
                     )

        self.kernels.c_conj_multiply = ElementwiseKernel(ctx,
                     "cfloat_t *x, cfloat_t *y, cfloat_t *z",
                     "z[i] = cfloat_mul(cfloat_conj(x[i]),y[i]);",
                     )

        self.kernels.touch = ElementwiseKernel(ctx,
                """float *clashvol, float max_clash, float *intervol, 
                float min_inter, int *interspace""",
                """if ((clashvol[i] < max_clash) && (intervol[i] > min_inter))
                       interspace[i] = 1;
                   else 
                       interspace[i] = 0;
                """,)
                       
        self.kernels.set_to_f = ElementwiseKernel(ctx,
            """float* array, float value""",
            """array[i] = value;""",)

        self.kernels.set_to_i = ElementwiseKernel(ctx,
            """int* array, int value""",
            """array[i] = value;""",)

        
    def touch(self, queue, clashvol, max_clash, intervol, min_inter, out):
        status = self.kernels.touch(clashvol, np.float32(max_clash),
            intervol, np.float32(min_inter), out)
        return status


    def c_conj_multiply(self, queue, array1, array2, out):
        if (array1.dtype == array2.dtype == out.dtype == np.complex64):
            status = self.kernels.c_conj_multiply(array1, array2, out)
        else:
            raise TypeError("Datatype of arrays is not supported")

        return status


    def multiply(self, queue, array1, array2, out):
        if array1.dtype == array2.dtype == out.dtype == np.float32:
            status = self.kernels.multiply_f32(array1, array2, out)
        elif array1.dtype == array2.dtype == out.dtype == np.int32:
            status = self.kernels.multiply_int32(array1, array2, out)
        else:
            raise TypeError("Array type is not supported")
        return status


    def histogram(self, queue, data, subhists, weight, nrestraints):
        
        WORKGROUPSIZE = 32
        kernel = self.kernels.histogram

        local_hist = cl.LocalMemory(4*(nrestraints + 1)*WORKGROUPSIZE)

        gws = (8*WORKGROUPSIZE*8*4,)
        lws = (WORKGROUPSIZE,)

        args = (data.data, subhists.data, local_hist, np.uint32(nrestraints + 1),
                np.float32(weight), np.uint32(data.size))
        status = kernel(queue, gws, lws, *args)

        return status


    def count_violations(self, queue, restraints, rotmat, access_interspace, 
            viol_counter, weight):

        WORKGROUPSIZE = 32
        kernel = self.kernels.count_violations

        rotmat16 = np.zeros(16, dtype=np.float32)
        rotmat16[:9] = rotmat.flatten()[:]
        shape = np.asarray(list(access_interspace.shape) + [access_interspace.size], dtype=np.int32)
        loc_viol = cl.LocalMemory(4*restraints.shape[0]**2*WORKGROUPSIZE)
        # float4
        restraints_center = cl.LocalMemory(4*restraints.shape[0]*4)
        mindist2 = cl.LocalMemory(4*restraints.shape[0])
        maxdist2 = cl.LocalMemory(4*restraints.shape[0])

        kernel.set_args(restraints.data, rotmat16, access_interspace.data, 
                viol_counter.data, loc_viol, restraints_center, mindist2, maxdist2, 
                np.int32(restraints.shape[0]), shape, np.float32(weight))

        gws = (8*WORKGROUPSIZE*8*4,)
        lws = (WORKGROUPSIZE,)
        status = cl.enqueue_nd_range_kernel(queue, kernel, gws, lws)

        return status


    def rotate_image3d(self, queue, sampler, image3d,
            rotmat, array_buffer, center):

        kernel = self.kernels.rotate_image3d

        gws = (8 * 32 *8, )
        shape = np.asarray(list(array_buffer.shape) + [np.product(array_buffer.shape)], dtype=np.int32)

        inv_rotmat = np.linalg.inv(rotmat)
        inv_rotmat16 = np.zeros(16, dtype=np.float32)
        inv_rotmat16[:9] = inv_rotmat.flatten()[:]

        _center = np.zeros(4, dtype=np.float32)
        _center[:3] = center[:]

        kernel.set_args(sampler, image3d, inv_rotmat16, array_buffer.data, _center, shape)
        status = cl.enqueue_nd_range_kernel(queue, kernel, gws, None)

        return status


    def fill(self, queue, array, value):
        if array.dtype == np.float32:
            status = self.kernels.set_to_f(array, np.float32(value))
        elif array.dtype == np.int32:
            status = self.kernels.set_to_i(array, np.int32(value))
        else:
            raise TypeError("Array type ({:s}) is not supported.".format(array.dtype))
        return status


    def distance_restraint(self, queue, restraints, rotmat, restspace):

        kernel = self.kernels.distance_restraint

        shape = np.asarray(list(restspace.shape) + [0], dtype=np.int32)
        nrestraints = np.int32(restraints.shape[0])

        gws = (restspace.shape[0], restspace.shape[1], 1)

        rotmat16 = np.zeros(16, dtype=np.float32)
        rotmat16[:9] = np.asarray(rotmat, dtype=np.float32).flatten()[:]

        kernel.set_args(restraints.data, rotmat16, restspace.data, shape, nrestraints)

        status = cl.enqueue_nd_range_kernel(queue, kernel, gws, None)

        return status
