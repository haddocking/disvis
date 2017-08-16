from unittest import TestCase, main

import numpy as np

from disvis.pdb import PDB
from disvis.disvis import DisVis


class TestDisVis(TestCase):

    def setUp(self):
        self.dv = DisVis()
        self.receptor = PDB.fromfile('data/single-atom.pdb')
        self.ligand = PDB.fromfile('data/single-atom.pdb')

    def test_minimal_volume_parameters(self):
        coor1 = np.asarray([[1, 2, 3], [-2, -3, -4]], dtype=np.float64)
        coor2 = np.asarray([[1, 0, 0], [-1, 0, 0]], dtype=np.float64)
        offset = 0
        voxelspacing = 1
        shape, origin = self.dv._minimal_volume_parameters(coor1, coor2,
                                                           offset, voxelspacing)
        offset = np.linalg.norm(coor2 - coor2.mean(axis=0), axis=1).max()

        self.assertEqual(shape, [9, 8, 5])
        self.assertTrue(np.allclose(origin, [-3.0, -4.0, -5.0]))
        self.assertEqual(offset, 1.0)

    def test_initialize(self):

        dv = self.dv
        # Test the Error raise when things are not set
        with self.assertRaises(ValueError):
            dv._initialize()

        restraint = [self.receptor, self.ligand, 0, 1]

        dv.receptor = self.receptor
        dv.ligand = self.ligand
        dv.add_distance_restraint(*restraint)
        dv._initialize()

        self.assertEqual(dv.interaction_restraints_cutoff, 1)
        self.assertFalse(dv._interaction_analysis)

    def test_cpu_init(self):

        dv = self.dv

        restraint = [self.receptor, self.ligand, 1, 4.5]
        voxelspacing = 2

        dv.receptor = self.receptor
        dv.ligand = self.ligand
        dv.add_distance_restraint(*restraint)
        dv.voxelspacing = voxelspacing
        dv._initialize()
        dv._cpu_init()

        self.assertEqual(dv._mindis[0], 0.5)
        self.assertEqual(dv._maxdis[0], 2.25)

    def test_rotate_lcore(self):
        dv = self.dv

        restraint = [self.receptor, self.ligand, 1, 4.5]
        voxelspacing = 2

        dv.receptor = self.receptor
        dv.ligand = self.ligand
        dv.add_distance_restraint(*restraint)
        dv.voxelspacing = voxelspacing
        dv._initialize()
        dv._cpu_init()

        rotmat = dv.rotations[0]
        dv._rotate_lcore(rotmat)

        # Identity rotation
        self.assertTrue(np.allclose(dv._lcore, dv._rot_lcore))

    def test_get_interaction_space(self):
        dv = self.dv

        restraint = [self.receptor, self.ligand, 1, 4.5]

        dv.receptor = self.receptor
        dv.ligand = self.ligand
        dv.add_distance_restraint(*restraint)
        dv.max_clash = 0.1
        dv.min_interaction = 0.1
        dv._initialize()
        dv._cpu_init()
        
        # Manually set rot_lcore for tests
        dv._rot_lcore = np.zeros_like(dv._lcore)
        dv._rot_lcore[0, 0, 0] = 1

        dv._get_interaction_space()
        
        self.assertTrue(np.allclose(dv._rsurf, dv._intervol))
        self.assertTrue(np.allclose(dv._rcore, dv._clashvol))
        self.assertTrue(np.allclose(dv._intervol, dv._interacting))
        self.assertTrue(np.allclose(dv._clashvol, np.logical_not(dv._not_clashing)))
        self.assertTrue(np.allclose(dv._interspace, dv._interacting - dv._clashvol))

    def test_get_restraints_center(self):
        dv = self.dv

        # 90 degree rotation around Z-axis
        rotmat = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        restraint = [self.receptor, self.ligand, 0, 4.5]
        # Set required attributes to test function
        dv._lrestraints = np.asarray([[2, 0, 0]], dtype=np.float64)
        dv._rot_lrestraints = np.zeros_like(dv._lrestraints)
        dv._restraints_center = np.zeros_like(dv._lrestraints)
        dv._rrestraints = np.asarray([[1, 0, 0]], dtype=np.float64)

        dv._get_restraints_center(rotmat)
        self.assertTrue(np.allclose(dv._rot_lrestraints, np.array([[0, 2, 0]])))
        self.assertTrue(np.allclose(dv._restraints_center, np.array([[1, -2, 0]])))

    def test_get_restraint_space(self):
        dv = self.dv

        dv._restraints_center = np.array([[4, 3, 2]], dtype=np.float64)
        dv._mindis = np.array([1], dtype=np.float64)
        dv._maxdis = np.array([2], dtype=np.float64)
        dv._restspace = np.empty((5, 7, 9), np.int32)

        dv._get_restraint_space()
        # print dv._restspace

    def test_cpu_search(self):
        dv = self.dv
        restraint = [self.receptor, self.ligand, 1, 4.5]
        voxelspacing = 2

        dv.receptor = self.receptor
        dv.ligand = self.ligand
        dv.add_distance_restraint(*restraint)
        dv.voxelspacing = voxelspacing
        dv._initialize()
        dv._cpu_init()
        dv._cpu_search()


if __name__ == '__main__':
    main()
