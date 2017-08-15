from unittest import TestCase, main

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from disvis.helpers import get_queue
from disvis.kernels import Kernels


class TestKernels(TestCase):
    @classmethod
    def setUpClass(self):
        self.queue = get_queue()
        self.shape = (5, 4, 3)
        self.size = 5 * 4 * 3
        self.values = {'interaction_cutoff': 1,
                       'nrestraints': 3,
                       'shape_x': self.shape[2],
                       'shape_y': self.shape[1],
                       'shape_z': self.shape[0],
                       'llength': 1,
                       'nreceptor_coor': 3,
                       'nligand_coor': 2,
                       }
        self.p = Kernels(self.queue.context, self.values)

    def test_rotate_grid3d(self):
        k = self.p.program.rotate_grid3d
        # Identity rotation
        rotmat = np.asarray([1, 0, 0, 0, 1, 0, 0, 0, 1] + [0] * 7, dtype=np.float32)
        self.cl_grid = cl_array.zeros(self.queue, self.shape, dtype=np.float32)
        self.cl_grid.fill(1)
        self.cl_out = cl_array.zeros(self.queue, self.shape, dtype=np.float32)
        args = (self.cl_grid.data, rotmat, self.cl_out.data)
        gws = tuple([2 * self.values['llength'] + 1] * 3)
        k(self.queue, gws, None, *args)
        answer = [[[1., 1., 1.], [1., 0., 0.], [0., 0., 0.], [1., 0., 0.]],
                  [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                  [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                  [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                  [[1., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]]

        self.assertTrue(np.allclose(answer, self.cl_out.get()))

        # 90 degree rotation around z-axis
        rotmat = np.asarray([0, -1, 0, 1, 0, 0, 0, 0, 1] + [0] * 7, dtype=np.float32)
        grid = np.zeros(self.shape, dtype=np.float32)
        grid[0, 0, 0] = 1
        grid[0, 0, 1] = 1
        self.cl_grid = cl_array.to_device(self.queue, grid)
        self.cl_out.fill(0)
        args = (self.cl_grid.data, rotmat, self.cl_out.data)
        k(self.queue, gws, None, *args)

        answer = np.zeros_like(grid)
        answer[0, 0, 0] = 1
        answer[0, 1, 0] = 1
        self.assertTrue(np.allclose(answer, self.cl_out.get()))

    def test_dilate_point_add(self):
        k = self.p.program.dilate_point_add

        center = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [2, 3, 4, 0]], dtype=np.float32)

        cl_center = cl_array.to_device(self.queue, center)
        cl_mindis = cl_array.zeros(self.queue, 3, dtype=np.float32)
        cl_maxdis = cl_array.to_device(self.queue,
                                       np.array([1, 2, 1], dtype=np.float32))
        cl_out = cl_array.zeros(self.queue, self.shape, dtype=np.int32)

        args = (cl_center.data, cl_mindis.data, cl_maxdis.data, np.int32(0), cl_out.data)
        k(self.queue, (5, 5, 5), None, *args)
        answer = [[[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
        self.assertTrue(np.allclose(answer, cl_out.get()))

        args = (cl_center.data, cl_mindis.data, cl_maxdis.data, np.int32(1), cl_out.data)
        k(self.queue, (5, 5, 5), None, *args)
        answer = [[[2, 2, 1], [2, 1, 0], [1, 0, 0], [0, 0, 0]],
                  [[2, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]],
                  [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
        self.assertTrue(np.allclose(answer, cl_out.get()))

        args = (cl_center.data, cl_mindis.data, cl_maxdis.data, np.int32(2), cl_out.data)
        k(self.queue, (5, 5, 5), None, *args)
        answer = [[[2, 2, 1], [2, 1, 0], [1, 0, 0], [0, 0, 0]],
                  [[2, 1, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0]],
                  [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]]]
        self.assertTrue(np.allclose(answer, cl_out.get()))

    def test_histogram(self):
        k = self.p.program.histogram
        data = np.repeat(range(self.values['nrestraints'] + 1),
                         self.size / (self.values['nrestraints'] + 1)).astype(np.int32)
        cl_data = cl_array.to_device(self.queue, data)
        cl_hist = cl_array.zeros(self.queue, self.values['nrestraints'], dtype=np.int32)
        args = (cl_data.data, cl_hist.data)
        k(self.queue, (12,), (12,), *args)
        answer = [15, 15, 15]
        self.assertTrue(np.allclose(answer, cl_hist.get()))

    def test_count_violations(self):
        k = self.p.program.count_violations

        center = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
        mindis2 = np.asarray([0, 0, 0], dtype=np.float32)
        maxdis2 = np.asarray([1, 1, 1], dtype=np.float32)
        interspace = np.zeros(self.shape, dtype=np.int32)
        interspace[-1, 0, 0] = 1
        interspace[-1, -1, 0] = 2
        interspace[-1, -1, -1] = 2

        cl_center = cl_array.to_device(self.queue, center)
        cl_mindis2 = cl_array.to_device(self.queue, mindis2)
        cl_maxdis2 = cl_array.to_device(self.queue, maxdis2)
        cl_interspace = cl_array.to_device(self.queue, interspace)
        cl_viol = cl_array.zeros(self.queue, self.values['nrestraints'] ** 2, dtype=np.int32)

        args = (cl_center.data, cl_mindis2.data, cl_maxdis2.data,
                cl_interspace.data, cl_viol.data)
        gws = (10, 10, 10)
        lws = (10, 1, 1)
        k(self.queue, gws, lws, *args)
        answer = [[1, 1, 1], [2, 2, 2], [0, 0, 0]]
        self.assertTrue(np.allclose(answer, cl_viol.get().reshape(3, 3)))

    def test_count_interactions(self):
        k = self.p.program.count_interactions
        receptor = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]], dtype=np.float32)
        ligand = np.asarray([[0, 0, 0, 0], [10, 0, 0, 0]], dtype=np.float32)
        interspace = np.zeros(self.shape, dtype=np.int32)
        nconsistent = np.int32(1)
        hist = np.zeros((self.values['nligand_coor'], self.values['nreceptor_coor']),
                        dtype=np.int32)

        interspace[0, 0, 0] = 1
        interspace[0, 0, 1] = 1
        interspace[0, 0, 2] = 2

        cl_receptor = cl_array.to_device(self.queue, receptor)
        cl_ligand = cl_array.to_device(self.queue, ligand)
        cl_interspace = cl_array.to_device(self.queue, interspace)
        cl_hist = cl_array.to_device(self.queue, hist)

        args = (cl_receptor.data, cl_ligand.data, cl_interspace.data,
                nconsistent, cl_hist.data)
        k(self.queue, (10, 10, 10), (10, 1, 1), *args)

        answer = [[2, 2, 1],
                  [0, 0, 0]]
        test = np.allclose(answer, cl_hist.get())
        self.assertTrue(test)


if __name__ == '__main__':
    main()
