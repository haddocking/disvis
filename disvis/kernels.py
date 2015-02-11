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

    def rotate_image3d(self, queue, sampler, image3d,
            rotmat, array_buffer, center):

        kernel = self.kernels.rotate_image3d
        compute_units = queue.device.max_compute_units

        work_groups = (compute_units*16*8*4, 1, 1)

        shape = np.asarray(list(array_buffer.shape) + [np.product(array_buffer.shape)], dtype=np.int32)

        inv_rotmat = np.linalg.inv(rotmat)
        inv_rotmat16 = np.zeros(16, dtype=np.float32)
        inv_rotmat16[:9] = inv_rotmat.flatten()[:]

        _center = np.zeros(4, dtype=np.float32)
        _center[:3] = center[:]

        kernel.set_args(sampler, image3d, inv_rotmat16, array_buffer.data, _center, shape)
        status = cl.enqueue_nd_range_kernel(queue, kernel, work_groups, None)

        return status

    def fill(self, queue, array, value):
        if array.dtype == np.float32:
            status = self.kernels.set_to_f(array, np.float32(value))
        elif array.dtype == np.int32:
            status = self.kernels.set_to_i(array, np.int32(value))
        else:
            raise TypeError("Array type ({:s}) is not supported.".format(array.dtype))
        return status

    def dilate_points_add(self, queue, constraints, rotmat, restspace):

        kernel = self.kernels.dilate_points_add

        compute_units = queue.device.max_compute_units
        preferred_work_groups = compute_units*16*8

        shape = np.asarray(list(restspace.shape) + [0], dtype=np.int32)
        nrestraints = np.int32(constraints.shape[0])

        zworkgroups = int(max(min(shape[0], preferred_work_groups), 1))
        yworkgroups = int(max(min(shape[1], preferred_work_groups - zworkgroups), 1))
        xworkgroups = int(max(min(shape[2], preferred_work_groups - zworkgroups - yworkgroups ), 1))
        workgroups = (zworkgroups, yworkgroups, xworkgroups)

        rotmat16 = np.zeros(16, dtype=np.float32)
        rotmat16[:9] = np.asarray(rotmat, dtype=np.float32).flatten()[:]

        kernel.set_args(constraints.data, rotmat16, restspace.data, shape, nrestraints)

        status = cl.enqueue_nd_range_kernel(queue, kernel, workgroups, None)

        return status

    def count(self, queue, interaction_space, accessible_interaction_space,
              weight, counts):

        kernel = self.kernels.count
        compute_units = queue.device.max_compute_units
        workgroups = (compute_units*16*8, )

        size = np.int32(interaction_space.size)
        w = np.float32(weight)

        kernel.set_args(interaction_space.data, accessible_interaction_space.data,
                        w, counts.data, size)
        status = cl.enqueue_nd_range_kernel(queue, kernel, workgroups, None)

        return status


    def distance_restraint(self, queue, constraints, rotmat, restspace):

        kernel = self.kernels.distance_restraint

        compute_units = queue.device.max_compute_units
        preferred_work_groups = compute_units*16*8

        shape = np.asarray(list(restspace.shape) + [0], dtype=np.int32)
        nrestraints = np.int32(constraints.shape[0])

        zworkgroups = int(max(min(shape[0], preferred_work_groups), 1))
        yworkgroups = int(max(min(shape[1], preferred_work_groups - zworkgroups), 1))
        xworkgroups = int(max(min(shape[2], preferred_work_groups - zworkgroups - yworkgroups ), 1))
        workgroups = (zworkgroups, yworkgroups, xworkgroups)

        rotmat16 = np.zeros(16, dtype=np.float32)
        rotmat16[:9] = np.asarray(rotmat, dtype=np.float32).flatten()[:]

        kernel.set_args(constraints.data, rotmat16, restspace.data, shape, nrestraints)

        status = cl.enqueue_nd_range_kernel(queue, kernel, workgroups, None)

        return status
