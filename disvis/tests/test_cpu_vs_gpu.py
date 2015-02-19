from __future__ import division, print_function
import os.path
import unittest

import numpy as np
import numpy.fft

import pyopencl as cl
import pyopencl.array as cl_array

import disvis
import disvis.helpers
import disvis.rotations
import disvis.pyclfft


class TestCPUvsGPU(unittest.TestCase):
    def setUp(self):

        self.pdb1 = disvis.PDB.fromfile(os.path.join(os.path.dirname(__file__), 'data', 'O14250.pdb'))
        self.pdb2 = disvis.PDB.fromfile(os.path.join(os.path.dirname(__file__), 'data', 'Q9UT97.pdb'))
        q, w, a = disvis.rotations.proportional_orientations(10)
        self.rotations = disvis.rotations.quat_to_rotmat(q)
        self.vlength = int(np.linalg.norm(self.pdb2.coor - self.pdb2.center, axis=1).max() +\
                       3 + 1.5)/1.0

        self.shape = disvis.disvis.grid_shape(self.pdb1.coor, self.pdb2.coor, 1)
        radii = np.zeros(self.pdb1.coor.shape[0], dtype=np.float64)
        radii.fill(1.5 + 3)
        self.rsurf = disvis.disvis.rsurface(self.pdb1.coor, radii, self.shape, 1)
        self.rcore = disvis.volume.erode(self.rsurf, 3)
        self.origin = self.rsurf.origin
        self.voxelspacing = 1

        radii = np.zeros(self.pdb2.coor.shape[0], dtype=np.float64)
        radii.fill(1.5)
        self.im_lsurf = disvis.points.dilate_points(self.pdb2.coor - self.pdb2.center \
            + self.pdb1.center, radii, disvis.volume.zeros_like(self.rcore))
        self.im_center = np.asarray((self.pdb1.center - self.rcore.origin)/1.0, dtype=np.float64)

        self.queue = disvis.helpers.get_queue()
        self.sampler = cl.Sampler(self.queue.context, False, cl.addressing_mode.CLAMP, cl.filter_mode.LINEAR)
        self.kernels = disvis.kernels.Kernels(self.queue.context)
        self.ft_shape = list(self.shape)
        self.ft_shape[0] = self.ft_shape[0]//2 + 1
        self.ft_shape = tuple(self.ft_shape)
        self.kernels.rfftn = disvis.pyclfft.RFFTn(self.queue.context, self.shape)
        self.kernels.irfftn = disvis.pyclfft.iRFFTn(self.queue.context, self.shape)

    def test_rotate_image3d(self):
        # CPU
        NROT = np.random.randint(self.rotations.shape[0] + 1)
        rotmat = self.rotations[NROT]

        #print(rotmat)

        cpu_lsurf = np.zeros_like(self.im_lsurf.array)
        disvis.libdisvis.rotate_image3d(self.im_lsurf.array, self.vlength, np.linalg.inv(rotmat), self.im_center, cpu_lsurf)

        gpu_im_lsurf = cl.image_from_array(self.queue.context, np.asarray(self.im_lsurf.array, dtype=np.float32))
        gpu_lsurf = cl_array.zeros(self.queue, self.shape, dtype=np.float32)
        self.kernels.rotate_image3d(self.queue, self.sampler, gpu_im_lsurf, rotmat, gpu_lsurf, self.im_center)

        self.assertTrue(np.allclose(cpu_lsurf, gpu_lsurf.get(), atol=0.01))

    def test_c_conj_multiply(self):
        
        shape = (5, 6, 8)

        np_array1 = np.zeros(shape, dtype=np.complex64)
        np_array2 = np.zeros(shape, dtype=np.complex64)
        np_out = np.zeros(shape, dtype=np.complex64)

        np_array1.real[:] = np.random.random(shape)
        np_array1.imag[:] = np.random.random(shape)
        np_array2.real[:] = np.random.random(shape)
        np_array2.imag[:] = np.random.random(shape)


        cl_array1 = cl_array.to_device(self.queue, np_array1)
        cl_array2 = cl_array.to_device(self.queue, np_array2)
        cl_out = cl_array.to_device(self.queue, np_out)

        self.kernels.c_conj_multiply(self.queue, cl_array1, cl_array2, cl_out)

        expected = np_array1.conj() * np_array2

        self.assertTrue(np.allclose(expected, cl_out.get()))

    def test_clashvol(self):

        NROT = np.random.randint(self.rotations.shape[0] + 1)
        rotmat = self.rotations[NROT]
        cpu_lsurf = np.zeros_like(self.im_lsurf.array)
        disvis.libdisvis.rotate_image3d(self.im_lsurf.array, self.vlength, np.linalg.inv(rotmat), self.im_center, cpu_lsurf)

        cpu_clashvol = numpy.fft.irfftn(numpy.fft.rfftn(cpu_lsurf).conj() * numpy.fft.rfftn(self.rcore.array), s=self.shape)

        gpu_rcore = cl_array.to_device(self.queue, np.asarray(self.rcore.array, dtype=np.float32))
        gpu_im_lsurf = cl.image_from_array(self.queue.context, np.asarray(self.im_lsurf.array, dtype=np.float32))
        gpu_lsurf = cl_array.zeros(self.queue, self.shape, dtype=np.float32)

        self.kernels.rotate_image3d(self.queue, self.sampler, gpu_im_lsurf, rotmat, gpu_lsurf, self.im_center)

        gpu_ft_lsurf = cl_array.zeros(self.queue, self.ft_shape, dtype=np.complex64)
        gpu_ft_rcore = cl_array.zeros(self.queue, self.ft_shape, dtype=np.complex64)
        gpu_ft_clashvol = cl_array.zeros(self.queue, self.ft_shape, dtype=np.complex64)
        gpu_clashvol = cl_array.zeros(self.queue, self.shape, dtype=np.float32)

        self.kernels.rfftn(self.queue, gpu_rcore, gpu_ft_rcore)
        self.kernels.rfftn(self.queue, gpu_lsurf, gpu_ft_lsurf)
        self.kernels.c_conj_multiply(self.queue, gpu_ft_lsurf, gpu_ft_rcore, gpu_ft_clashvol)
        self.kernels.irfftn(self.queue, gpu_ft_clashvol, gpu_clashvol)

        self.assertTrue(np.allclose(cpu_clashvol, gpu_clashvol.get(), atol=0.8))

    def test_intervol(self):

        NROT = np.random.randint(self.rotations.shape[0] + 1)
        rotmat = self.rotations[NROT]
        cpu_lsurf = np.zeros_like(self.im_lsurf.array)
        disvis.libdisvis.rotate_image3d(self.im_lsurf.array, self.vlength, np.linalg.inv(rotmat), self.im_center, cpu_lsurf)

        cpu_intervol = numpy.fft.irfftn(numpy.fft.rfftn(cpu_lsurf).conj() * numpy.fft.rfftn(self.rsurf.array), s=self.shape)

        gpu_rsurf = cl_array.to_device(self.queue, np.asarray(self.rsurf.array, dtype=np.float32))
        gpu_im_lsurf = cl.image_from_array(self.queue.context, np.asarray(self.im_lsurf.array, dtype=np.float32))
        gpu_lsurf = cl_array.zeros(self.queue, self.shape, dtype=np.float32)

        self.kernels.rotate_image3d(self.queue, self.sampler, gpu_im_lsurf, rotmat, gpu_lsurf, self.im_center)

        gpu_ft_lsurf = cl_array.zeros(self.queue, self.ft_shape, dtype=np.complex64)
        gpu_ft_rsurf = cl_array.zeros(self.queue, self.ft_shape, dtype=np.complex64)
        gpu_ft_intervol = cl_array.zeros(self.queue, self.ft_shape, dtype=np.complex64)
        gpu_intervol = cl_array.zeros(self.queue, self.shape, dtype=np.float32)

        self.kernels.rfftn(self.queue, gpu_rsurf, gpu_ft_rsurf)
        self.kernels.rfftn(self.queue, gpu_lsurf, gpu_ft_lsurf)
        self.kernels.c_conj_multiply(self.queue, gpu_ft_lsurf, gpu_ft_rsurf, gpu_ft_intervol)
        self.kernels.irfftn(self.queue, gpu_ft_intervol, gpu_intervol)

        self.assertTrue(np.allclose(cpu_intervol, gpu_intervol.get(), atol=0.8))

    @unittest.skip('hoi')
    def test_touch(self):

        MAX_CLASH = 100 + 0.9
        MIN_INTER = 300 + 0.9

        NROT = np.random.randint(self.rotations.shape[0] + 1)
        rotmat = self.rotations[0]
        cpu_lsurf = np.zeros_like(self.im_lsurf.array)
        disvis.libdisvis.rotate_image3d(self.im_lsurf.array, self.vlength, np.linalg.inv(rotmat), self.im_center, cpu_lsurf)

        cpu_clashvol = numpy.fft.irfftn(numpy.fft.rfftn(cpu_lsurf).conj() * numpy.fft.rfftn(self.rcore.array))

        gpu_rcore = cl_array.to_device(self.queue, np.asarray(self.rcore.array, dtype=np.float32))
        gpu_im_lsurf = cl.image_from_array(self.queue.context, np.asarray(self.im_lsurf.array, dtype=np.float32))
        gpu_lsurf = cl_array.zeros(self.queue, self.shape, dtype=np.float32)

        self.kernels.rotate_image3d(self.queue, self.sampler, gpu_im_lsurf, rotmat, gpu_lsurf, self.im_center)

        gpu_ft_lsurf = cl_array.zeros(self.queue, self.ft_shape, dtype=np.complex64)
        gpu_ft_rcore = cl_array.zeros(self.queue, self.ft_shape, dtype=np.complex64)
        gpu_ft_clashvol = cl_array.zeros(self.queue, self.ft_shape, dtype=np.complex64)
        gpu_clashvol = cl_array.zeros(self.queue, self.shape, dtype=np.float32)

        self.kernels.rfftn(self.queue, gpu_rcore, gpu_ft_rcore)
        self.kernels.rfftn(self.queue, gpu_lsurf, gpu_ft_lsurf)
        self.kernels.c_conj_multiply(self.queue, gpu_ft_lsurf, gpu_ft_rcore, gpu_ft_clashvol)
        self.kernels.irfftn(self.queue, gpu_ft_clashvol, gpu_clashvol)
        
        cpu_intervol = numpy.fft.irfftn(numpy.fft.rfftn(cpu_lsurf).conj() * numpy.fft.rfftn(self.rsurf.array))

        gpu_rsurf = cl_array.to_device(self.queue, np.asarray(self.rsurf.array, dtype=np.float32))

        gpu_ft_rsurf = cl_array.zeros(self.queue, self.ft_shape, dtype=np.complex64)
        gpu_ft_intervol = cl_array.zeros(self.queue, self.ft_shape, dtype=np.complex64)
        gpu_intervol = cl_array.zeros(self.queue, self.shape, dtype=np.float32)

        cpu_interspace = np.zeros(self.shape, dtype=np.int32)
        gpu_interspace = cl_array.zeros(self.queue, self.shape, dtype=np.int32)

        self.kernels.rfftn(self.queue, gpu_rsurf, gpu_ft_rsurf)
        self.kernels.rfftn(self.queue, gpu_lsurf, gpu_ft_lsurf)
        self.kernels.c_conj_multiply(self.queue, gpu_ft_lsurf, gpu_ft_rsurf, gpu_ft_intervol)
        self.kernels.irfftn(self.queue, gpu_ft_intervol, gpu_intervol)

        self.kernels.touch(self.queue, gpu_clashvol, MAX_CLASH, gpu_intervol, MIN_INTER, gpu_interspace)

        np.logical_and(cpu_clashvol < MAX_CLASH, cpu_intervol > MIN_INTER, cpu_interspace)

        disvis.volume.Volume(cpu_interspace, self.im_lsurf.voxelspacing, self.im_lsurf.origin).tofile('cpu_interspace.mrc')
        disvis.volume.Volume(gpu_interspace.get(), self.im_lsurf.voxelspacing, self.im_lsurf.origin).tofile('gpu_interspace.mrc')
        disvis.volume.Volume(cpu_interspace - gpu_interspace.get(), self.im_lsurf.voxelspacing, self.im_lsurf.origin).tofile('diff.mrc')
        print()
        print(cpu_interspace.sum(), gpu_interspace.get().sum())
        print(np.abs(cpu_interspace - gpu_interspace.get()).sum())
                           

        self.assertTrue(np.allclose(gpu_interspace.get(), cpu_interspace))

if __name__=='__main__':
    unittest.main()
        
