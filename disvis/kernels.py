import numpy as np
from os.path import dirname
import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel

class Kernels():

    def __init__(self, ctx):
        self.context = ctx

        self.kernel_file = ''.join([dirname(__file__), '/cl_kernels.cl'])
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
                """float *overlap, float *interaction, float *touch, 
                float max_clash, float min_interaction""",
                """if ((overlap[i] < max_clash) && (interaction[i] > min_interaction))
                       touch[i] = 1.0f;
                   else 
                       touch[i] = 0.0f;
                """,)
                       
        self.kernels.set_to_f = ElementwiseKernel(ctx,
            """float* array, float value""",
            """array[i] = value;""",)

        self.kernels.set_to_i = ElementwiseKernel(ctx,
            """int* array, int value""",
            """array[i] = value;""",)

        
    def multiply(self. queue, array1, array2, out):
        if array1.dtype == array2.dtype == out.dtype == np.float32:
            status = self.kernels.multiply_f32(array1, array2, out)
        elif array1.dtype == array2.dtype == out.dtype == np.int32:
            status = self.kernels.multiply_int32(array1, array2, out)
        else:
            raise TypeError("Array type is not supported")
        return status

    def rotate_image(self, queue, image, sampler,
            rotmat, array_buffer):

        kernel = self.kernels.rotate_template
        compute_units = queue.device.max_compute_units
        work_groups = compute_units*64*8
        depth, width, height = array_buffer.shape

        status = kernel(queue, (work_groups,), None,
                        image, sampler, rotations.data, array_buffer.data,
                        np.int32(depth), np.int32(width), np.int32(height),
                        np.int32(n))

        return status

    def fill(self, queue, array, value):
        if array.dtype == np.float32:
            status = self.kernels.set_to_f(array, np.float32(value))
        elif array.dtype == np.int32:
            status = self.kernels.set_to_i(array, np.int32(value))
        else:
            raise TypeError("Array type ({:s}) is not supported.".format(array.dtype))
        return status

    def distance_restraint_energy(self, queue, distance_restraints,
            gcc, weight, rot_mat, nrot):

        kernel = self.kernels.add_distance_restraint_potential

        kernel.set_args(distance_restraints.data, gcc.data, np.float32(weight),
                        np.int32(distance_restraints.shape[0]), rot_mat.data,
                        np.int32(nrot))

        status = cl.enqueue_nd_range_kernel(queue, kernel, gcc.shape, None)

        return status
