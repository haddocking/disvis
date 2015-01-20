from __future__ import print_function
import unittest

import numpy as np
from disvis import volume
from disvis.points import dilate_points

class TestPoints(unittest.TestCase):

    def setUp(self):
        pass

    def test_dilate_points(self):
        shape = (9, 8, 7)
        vol = volume.zeros(shape, voxelspacing=1, origin=(1,0,0))

        dilate_points(np.asarray([[1, 0, 0]]), np.asarray([1]), vol)

        answer = volume.zeros(shape, voxelspacing=1, origin=(1,0,0))
        answer.array[0, 0, 0] = 1
        answer.array[0, 0, 1] = 1
        answer.array[0, 0, -1] = 1
        answer.array[0, 1, 0] = 1
        answer.array[0, -1, 0] = 1
        answer.array[1, 0, 0] = 1
        answer.array[-1, 0, 0] = 1

        self.assertTrue(np.allclose(answer.array, vol.array))
        

if __name__=='__main__':
    unittest.main()
