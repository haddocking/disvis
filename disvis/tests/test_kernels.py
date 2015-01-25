from __future__ import print_function
import unittest
import numpy as np
from numpy.random import randint

import pyopencl as cl
import pyopencl.array as cl_array
from disvis.helpers import get_queue
from disvis.kernels import Kernels

class TestKernels(unittest.TestCase):
    
    def setUp(self):
        self.queue = get_queue()
        self.kernels = Kernels(self.queue.context)
        
    def test_rotate_image3d_0(self):
        
        shape = (8, 6, 5)
        np_image = np.zeros(shape, dtype=np.float32)
        np_image[0, 0, 0] = 1
        np_image[0, 0, 1] = 1
        np_image[0, 0, 2] = 1

        rotmat = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]

        np_out = np.zeros_like(np_image)

        cl_image = cl.image_from_array(self.queue.context, np_image)
        cl_out = cl_array.to_device(self.queue, np_out)
        cl_sampler = cl.Sampler(self.queue.context, False, cl.addressing_mode.CLAMP, cl.filter_mode.LINEAR)

        self.kernels.rotate_image3d(self.queue, cl_sampler, cl_image, rotmat, cl_out)

        self.assertTrue(np.allclose(np_image, cl_out.get()))

    def test_rotate_image3d_1(self):
        
        shape = (8, 6, 5)
        np_image = np.zeros(shape, dtype=np.float32)
        np_image[0, 0, 0] = 1
        np_image[0, 0, 1] = 1
        np_image[0, 0, 2] = 1

        # 90 degree rotation around z-axis
        rotmat = [[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]]

        np_out = np.zeros_like(np_image)

        expected = np.zeros_like(np_image)
        expected[0, 0, 0] = 1
        expected[0, 1, 0] = 1
        expected[0, 2, 0] = 1

        cl_image = cl.image_from_array(self.queue.context, np_image)
        cl_out = cl_array.to_device(self.queue, np_out)
        cl_sampler = cl.Sampler(self.queue.context, False, cl.addressing_mode.CLAMP, cl.filter_mode.LINEAR)

        self.kernels.rotate_image3d(self.queue, cl_sampler, cl_image, rotmat, cl_out)

        self.assertTrue(np.allclose(expected, cl_out.get()))

    def test_dilate_points_add_0(self):
        
        shape = (8, 7, 6)

        np_constraints = np.asarray([[3, 3, 3, 0, 0, 0, 1, 0]], dtype=np.float32)
        rotmat = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]
        np_out = np.zeros(shape, dtype=np.int32)

        cl_constraints = cl_array.to_device(self.queue, np_constraints)
        cl_out = cl_array.to_device(self.queue, np_out)

        self.kernels.dilate_points_add(self.queue, cl_constraints, rotmat, cl_out)

        expected = np.zeros_like(np_out)
        expected[3, 3, 2:5] = 1
        expected[3, 2:5, 3] = 1
        expected[2:5, 3, 3] = 1

        self.assertTrue(np.allclose(expected, cl_out.get()))
         
    def test_dilate_points_add_1(self):
        
        shape = (8, 7, 6)

        np_constraints = np.asarray([[3, 3, 3, 1, 1, 1, 1, 0]], dtype=np.float32)
        rotmat = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]
        np_out = np.zeros(shape, dtype=np.int32)

        cl_constraints = cl_array.to_device(self.queue, np_constraints)
        cl_out = cl_array.to_device(self.queue, np_out)

        self.kernels.dilate_points_add(self.queue, cl_constraints, rotmat, cl_out)

        expected = np.zeros_like(np_out)
        expected[2, 2, 1:4] = 1
        expected[2, 1:4, 2] = 1
        expected[1:4, 2, 2] = 1

        self.assertTrue(np.allclose(expected, cl_out.get()))

    def test_dilate_points_add_2(self):
        
        SHAPE = (8, 7, 6)
        CONSTRAINTS = [[3, 3, 3, 1, 1, 1, 1, 0],
                       [3, 3, 3, 1, 1, 1, 1, 0]]

        np_constraints = np.asarray(CONSTRAINTS, dtype=np.float32)
        rotmat = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]
        np_out = np.zeros(SHAPE, dtype=np.int32)

        cl_constraints = cl_array.to_device(self.queue, np_constraints)
        cl_out = cl_array.to_device(self.queue, np_out)

        self.kernels.dilate_points_add(self.queue, cl_constraints, rotmat, cl_out)

        expected = np.zeros_like(np_out)
        expected[2, 2, 1:4] = len(CONSTRAINTS)
        expected[2, 1:4, 2] = len(CONSTRAINTS)
        expected[1:4, 2, 2] = len(CONSTRAINTS)

        self.assertTrue(np.allclose(expected, cl_out.get()))

    def test_count_0(self):
        
        nrepeats = 3
        shape = [5, 5, 5]

        np_interspace = randint(2, size=shape).astype(np.int32)
        np_access_interspace = randint(nrepeats, size=shape).astype(np.int32)
        np_count = np.zeros([nrepeats] + shape, dtype=np.float32)
        weight = 0.5

        expected = np.zeros_like(np_count)
        tmp = expected[0]
        tmp[np_interspace == 1] += weight
        for i in range(1, nrepeats):
            tmp = expected[i]
            tmp[np_access_interspace == i] += weight


        cl_interspace = cl_array.to_device(self.queue, np_interspace)
        cl_access_interspace = cl_array.to_device(self.queue, np_access_interspace)
        cl_count = cl_array.to_device(self.queue, np_count)

        self.kernels.count(self.queue, cl_interspace, cl_access_interspace, weight, cl_count)

        self.assertTrue(np.allclose(expected, cl_count.get()))

    def test_count_1(self):
        
        nrepeats = 3
        shape = [5, 5, 5]

        np_interspace = randint(2, size=shape).astype(np.int32)
        np_access_interspace = randint(nrepeats, size=shape).astype(np.int32)
        np_count = np.ones([nrepeats] + shape, dtype=np.float32)
        weight = 0.5

        expected = np.ones_like(np_count)
        tmp = expected[0]
        tmp[np_interspace == 1] += weight
        for i in range(1, nrepeats):
            tmp = expected[i]
            tmp[np_access_interspace == i] += weight


        cl_interspace = cl_array.to_device(self.queue, np_interspace)
        cl_access_interspace = cl_array.to_device(self.queue, np_access_interspace)
        cl_count = cl_array.to_device(self.queue, np_count)

        self.kernels.count(self.queue, cl_interspace, cl_access_interspace, weight, cl_count)

        self.assertTrue(np.allclose(expected, cl_count.get()))


if __name__=='__main__':
    unittest.main()




        
