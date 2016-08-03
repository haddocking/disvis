from unittest import TestCase, main, skipIf
from os.path import join

import numpy as np

from disvis import PDB
from disvis.disvis import DisVis
from disvis.helpers import get_queue, parse_restraints

class TestCPUvsGPU(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dv = DisVis()
        cls.gdv = DisVis()

        cls.gdv.queue = get_queue()

        receptor = PDB.fromfile(join('data', 'O14250.pdb'))
        ligand = PDB.fromfile(join('data', 'Q9UT97.pdb'))
        restraints = parse_restraints(join('data', 'restraints.dat'), receptor, ligand)

        cls.dv.receptor = receptor
        cls.gdv.receptor = receptor
        cls.dv.ligand = ligand
        cls.gdv.ligand = ligand
        cls.dv.distance_restraints = restraints
        cls.gdv.distance_restraints = restraints

        cls.dv._initialize()
        cls.dv._cpu_init()
        cls.gdv._initialize()
        cls.gdv._gpu_init()

    def test_rotate_lcore(self):
        self.dv._rotate_lcore(self.dv.rotations[0])
        self.gdv._cl_rotate_lcore(self.gdv._cl_rotations[0])

        test = np.allclose(self.dv._rot_lcore, self.gdv._cl_rot_lcore.get())
        self.assertTrue(test)

    def test_get_interaction_space(self):
        self.dv._rotate_lcore(self.dv.rotations[0])
        self.gdv._cl_rotate_lcore(self.gdv._cl_rotations[0])

        self.dv._get_interaction_space()
        self.gdv._cl_get_interaction_space()

        test = np.allclose(np.round(self.dv._clashvol), np.round(self.gdv._cl_clashvol.get()))
        print np.round(self.dv._clashvol)
        print np.round(self.gdv._cl_clashvol.get())
        self.assertTrue(test)
        test = np.allclose(self.dv._intervol, self.gdv._cl_intervol.get())
        self.assertTrue(test)

    def test_rotate_restraints(self):
        pass

    def test_get_restraint_space(self):
        pass


if __name__ == '__main__':
    main()
