"""Tests of disvis-complexes and of writing PDB files.

Run from the repository root with python -m unittest.
"""

from os.path import dirname, join
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import numpy as np

from argparse import Namespace

from disvis.complexes import (
    Poses,
    cluster,
    cluster_rmsds,
    read_complexes,
    tune_cutoff,
)
from disvis.helpers import min_consistent
from disvis.main import write_complexes
from disvis.pdb import PDB

DATA = join(dirname(__file__), 'data')


def rotation_z(angle):
    """Return the rotation matrix of angle degrees around z."""
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class TestComplexes(TestCase):
    def test_pose_rmsd(self):
        # The RMSD from rotations and translations equals that of the atoms
        coor = PDB.fromfile(join(DATA, 'O14250.pdb')).coor[:50]
        rotations = np.asarray([rotation_z(10), rotation_z(-35)])
        translations = np.asarray([[1.0, 2, 3], [-4, 0, 2]])
        placed = coor @ rotations.transpose(0, 2, 1) + translations[:, None]
        rmsd = np.sqrt(((placed[0] - placed[1]) ** 2).sum(axis=1).mean())
        poses = Poses(translations, rotations, coor)
        self.assertAlmostEqual(np.sqrt(poses.rmsd2([0], [1])[0, 0]), rmsd)

    def test_cluster(self):
        # Complex 1 is the neighbor of 0 and 2, and the center of their
        # cluster; complex 3 is too far, and complex 4 rotated too much.
        coor = np.asarray([[5.0, 0, 0], [-5, 0, 0], [0, 5, 0]])
        translations = np.asarray(
            [[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [10, 0, 0], [1.5, 0, 0]]
        )
        rotations = np.asarray([rotation_z(a) for a in (0, 0, 0, 0, 30)])
        poses = Poses(translations, rotations, coor)
        clusters = cluster(poses, 2)
        self.assertEqual([c.tolist() for c in clusters], [[1, 0, 2], [3], [4]])
        clusters = cluster(poses, 2, min_population=2)
        self.assertEqual([c.tolist() for c in clusters], [[1, 0, 2]])

    def test_tune_cutoff(self):
        # Poses 1 A apart on a line: with the tuned cutoff, the largest RMSD
        # within clusters is at most the target, not with a larger one.
        translations = np.stack(
            [np.arange(40.0), np.zeros(40), np.zeros(40)], 1
        )
        rotations = np.repeat(np.eye(3)[None], 40, axis=0)
        poses = Poses(translations, rotations, np.eye(3))
        cutoff = tune_cutoff(poses, 2.0)
        self.assertLessEqual(
            cluster_rmsds(poses, cluster(poses, cutoff)).max(), 2 + 1e-6
        )
        self.assertGreater(
            cluster_rmsds(poses, cluster(poses, cutoff + 0.1)).max(), 2
        )

    def test_max_complexes(self):
        # A sample of 3 of 5 complexes is written, in their order, and read.
        rotmat = np.asarray([rotation_z(a) for a in range(0, 50, 10)])
        complexes = (
            np.arange(5),
            np.arange(15.0).reshape(5, 3),
            np.full(5, 2),
        )
        ligand = PDB.fromfile(join(DATA, 'single-atom.pdb'))
        args = Namespace(
            write_complexes=100.0,
            nrestraints=2,
            max_complexes=3,
            receptor='fixed.pdb',
            ligand='scanning.pdb',
            voxelspacing=1.0,
            angle=10.0,
        )
        with TemporaryDirectory() as tmp:
            fname = join(tmp, 'complexes.out')
            write_complexes(fname, complexes, rotmat, ligand, args)
            header, models, translations, rotations = read_complexes(fname)
        self.assertEqual(header['complexes'], '3 written of 5')
        self.assertEqual(models.tolist(), [1, 2, 3])
        index = translations[:, 0].astype(int) // 3
        self.assertTrue(np.all(np.diff(index) > 0))
        self.assertTrue(np.allclose(rotations, rotmat[index], atol=1e-6))

    def test_min_consistent(self):
        # A percentage of the restraints, rounded up
        for percent, nrestraints, n in (
            (50, 43, 22),
            (50, 20, 10),
            (58, 43, 25),
        ):
            self.assertEqual(min_consistent(percent, nrestraints), n)

    def test_tofile(self):
        pdb = PDB.fromfile(join(DATA, 'O14250.pdb'))
        with TemporaryDirectory() as tmp:
            pdb.tofile(join(tmp, 'out.pdb'))
            copy = PDB.fromfile(join(tmp, 'out.pdb'))
        self.assertTrue(np.array_equal(pdb.coor, copy.coor))
        for field in ('atom_id', 'name', 'resn', 'chain', 'resi', 'element'):
            self.assertTrue(np.array_equal(pdb.data[field], copy.data[field]))


if __name__ == '__main__':
    main()
