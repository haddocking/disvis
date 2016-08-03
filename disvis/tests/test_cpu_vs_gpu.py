from unittest import TestCase, main, skipIf
from os.path import join

import numpy as np

from disvis import PDB, Volume
from disvis.disvis import DisVis
from disvis.helpers import get_queue, parse_restraints, parse_interaction_selection

class TestCPUvsGPU(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dv = DisVis()
        cls.gdv = DisVis()

        cls.gdv.queue = get_queue()

        receptor = PDB.fromfile(join('data', 'O14250.pdb'))
        ligand = PDB.fromfile(join('data', 'Q9UT97.pdb'))
        restraints = parse_restraints(join('data', 'restraints.dat'), receptor, ligand)
        rselect, lselect = parse_interaction_selection(join('data',
            'selection.res'), receptor, ligand)

        # Identity rotation and rotation around z-axis
        rotations = np.asarray([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                [[0, -1, 0], [1, 0, 0], [0, 0, 1]]], dtype=np.float64)

        cls.dv.receptor = receptor
        cls.gdv.receptor = receptor
        cls.dv.ligand = ligand
        cls.gdv.ligand = ligand
        cls.dv.distance_restraints = restraints
        cls.gdv.distance_restraints = restraints
        cls.dv.rotations = rotations
        cls.gdv.rotations = rotations

        cls.dv.occupancy_analysis = True
        cls.gdv.occupancy_analysis = True

        cls.dv.receptor_interaction_selection = rselect
        cls.dv.ligand_interaction_selection = lselect
        cls.gdv.receptor_interaction_selection = rselect
        cls.gdv.ligand_interaction_selection = lselect

        cls.dv._initialize()
        cls.dv._cpu_init()
        cls.gdv._initialize()
        cls.gdv._gpu_init()

    def test_all1(self):
        c_rotmat = self.dv.rotations[0]
        g_rotmat = self.gdv._cl_rotations[0]
        weight = np.float64(1)

        self.dv._rotate_lcore(c_rotmat)
        self.gdv._cl_rotate_lcore(g_rotmat)
        test = np.all(self.dv._rot_lcore == self.gdv._cl_rot_lcore.get())
        self.assertTrue(test)

        self.dv._get_interaction_space()
        self.gdv._cl_get_interaction_space()
        test = np.all(self.dv._clashvol == self.gdv._cl_clashvol.get())
        self.assertTrue(test)
        test = np.all(self.dv._intervol == self.gdv._cl_intervol.get())
        self.assertTrue(test)
        test = np.all(self.dv._not_clashing == self.gdv._cl_not_clashing.get())
        self.assertTrue(test)
        test = np.all(self.dv._interacting == self.gdv._cl_interacting.get())
        self.assertTrue(test)
        test = np.all(self.dv._interspace == self.gdv._cl_interspace.get())
        self.assertTrue(test)

        self.dv._get_restraints_center(c_rotmat)
        self.gdv._cl_get_restraints_center(g_rotmat)
        test = np.allclose(self.dv._rot_lrestraints,
                self.gdv._cl_rot_lrestraints.get()[:, :3])
        self.assertTrue(test)
        test = np.allclose(self.dv._restraints_center,
                self.gdv._cl_restraints_center.get()[:, :3])
        self.assertTrue(test)

        self.dv._get_restraint_space()
        self.gdv._cl_get_restraint_space()
        c_restspace = self.dv._restspace
        g_restspace = self.gdv._cl_restspace.get()
        # The restraints space can differ slightly because of roundoff errors
        # in the distance calculation. 
        loc = (c_restspace != g_restspace).nonzero()
        diff = (c_restspace != g_restspace).sum()
        self.assertLessEqual(diff, 1)
        # Cheat here by adjusting the restspace on the CPU version for easier
        # further testing
        for l in zip(loc):
            self.dv._restspace[l] = g_restspace[l]

        self.dv._get_reduced_interspace()
        self.gdv._cl_get_reduced_interspace()
        test = np.all(self.dv._interspace == self.gdv._cl_interspace.get())
        self.assertTrue(test)

        self.dv._count_complexes(weight)
        self.gdv._cl_count_complexes(weight)
        test = np.all(self.dv._consistent_complexes[1:] ==
                self.gdv._cl_consistent_complexes.get())
        self.assertTrue(test)

        self.dv._count_violations(weight)
        self.gdv._cl_count_violations(weight)
        test = np.all(self.dv._violations == self.gdv._cl_violations.get())
        self.assertTrue(test)

        self.dv._get_occupancy_grids(weight)
        self.gdv._cl_get_occupancy_grids(weight)
        for i in (6, 7):
            test = np.all(self.dv._occ_grid[i] == self.gdv._cl_occ_grid[i].get())
            self.assertTrue(test)

        #self.dv._get_interaction_matrix(c_rotmat, weight)
        #self.gdv._cl_get_interaction_matrix(g_rotmat, weight)
        #c_im = self.dv._interaction_matrix[2]
        #g_im = self.gdv._cl_interaction_matrix[2]
        #for i in range(c_im.shape[1]):
        #    print c_im[i, :]
        #    print g_im[i, :]
        #test = np.all(c_im == g_im.get())
        #self.assertTrue(test)
        

        c_rotmat = self.dv.rotations[1]
        g_rotmat = self.gdv._cl_rotations[1]

        self.dv._rotate_lcore(c_rotmat)
        self.gdv._cl_rotate_lcore(g_rotmat)
        test = np.all(self.dv._rot_lcore == self.gdv._cl_rot_lcore.get())
        self.assertTrue(test)

        self.dv._get_interaction_space()
        self.gdv._cl_get_interaction_space()
        test = np.all(self.dv._clashvol == self.gdv._cl_clashvol.get())
        self.assertTrue(test)
        test = np.all(self.dv._intervol == self.gdv._cl_intervol.get())
        self.assertTrue(test)
        test = np.all(self.dv._not_clashing == self.gdv._cl_not_clashing.get())
        self.assertTrue(test)
        test = np.all(self.dv._interacting == self.gdv._cl_interacting.get())
        self.assertTrue(test)
        test = np.all(self.dv._interspace == self.gdv._cl_interspace.get())
        self.assertTrue(test)




if __name__ == '__main__':
    main()
