from __future__ import print_function
import unittest

import numpy as np
from disvis.libdisvis import rotate_image3d, dilate_points, dilate_points_add

class TestPoints(unittest.TestCase):

    def setUp(self):
        pass

    def test_rotate_image3d_id(self):
        
        shape = (8, 6, 5)
        image = np.zeros(shape, dtype=np.float64)
        image[0,0,0] = 1
        image[0, 0, 1] = 1
        image[0, 0, 1] = 1
        out = np.zeros_like(image)
        rotmat = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

        rotate_image3d(image, 2, rotmat, out)

        self.assertTrue(np.allclose(image, out))

    def test_rotate_image3d_z90(self):

        shape = (8, 6, 5)
        image = np.zeros(shape, dtype=np.float64)
        image[0,0,0] = 1
        image[0, 0, 1] = 1
        image[0, 0, 2] = 1
        out = np.zeros_like(image)
        rotmat = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)

        rotate_image3d(image, 2, rotmat, out)
        solution = np.zeros_like(image)
        solution[0,0,0] = 1
        solution[0,1,0] = 1
        solution[0,2,0] = 1

        self.assertTrue(np.allclose(solution, out))
        
if __name__=='__main__':
    unittest.main()
