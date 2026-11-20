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
from disvis.disvis import Complexes
from disvis.main import write_complexes
from disvis.pdb import PDB
from disvis.rotations import Rz

DATA = join(dirname(__file__), 'data')


class TestComplexes(TestCase):
    def test_pose_rmsd(self):
        # The RMSD from rotations and translations equals that of the atoms
        coor = PDB.fromfile(join(DATA, 'O14250.pdb')).coor[:50]
        rotations = np.asarray([Rz(np.radians(10)), Rz(np.radians(-35))])
        translations = np.asarray([[1.0, 2, 3], [-4, 0, 2]])
        placed = coor @ rotations.transpose(0, 2, 1) + translations[:, None]
        rmsd = np.sqrt(((placed[0] - placed[1]) ** 2).sum(axis=1).mean())
        poses = Poses(translations, rotations, coor)
        self.assertAlmostEqual(np.sqrt(poses.rmsd2([0], [1])[0, 0]), rmsd)

    def test_pose_rmsd_models(self):
        # With poses of two models (a distorted copy), the RMSDs equal those
        # of the placed atoms of each model, also in a subset
        first = PDB.fromfile(join(DATA, 'O14250.pdb')).coor[:50]
        rng = np.random.default_rng(0)
        coor = np.stack([first, first + rng.normal(0, 1, first.shape)])
        model = np.asarray([0, 1, 1, 0])
        rotations = np.asarray([Rz(np.radians(a)) for a in (10, -35, 80, 0)])
        translations = np.asarray(
            [[1.0, 2, 3], [-4, 0, 2], [0, 5, -1], [2, 2, 2]]
        )
        placed = (
            coor[model] @ rotations.transpose(0, 2, 1) + translations[:, None]
        )
        rmsd = np.sqrt(
            ((placed[:, None] - placed[None]) ** 2).sum(axis=3).mean(axis=2)
        )
        poses = Poses(translations, rotations, coor, model)
        np.testing.assert_allclose(np.sqrt(poses.rmsd2(np.arange(4))), rmsd)
        subset = poses.subset([1, 3])
        np.testing.assert_allclose(
            np.sqrt(subset.rmsd2([0], [1])), rmsd[[1]][:, [3]]
        )

    def test_cluster(self):
        # Complex 1 is the neighbor of 0 and 2, and the center of their
        # cluster; complex 3 is too far, and complex 4 rotated too much.
        coor = np.asarray([[5.0, 0, 0], [-5, 0, 0], [0, 5, 0]])
        translations = np.asarray(
            [[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [10, 0, 0], [1.5, 0, 0]]
        )
        rotations = np.asarray([Rz(np.radians(a)) for a in (0, 0, 0, 0, 30)])
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
        # The 3 of 5 complexes with the smallest keys are written, in their
        # order, and read.
        rotmat = np.asarray([Rz(np.radians(a)) for a in range(0, 50, 10)])
        complexes = Complexes(
            np.arange(5),
            np.arange(15.0).reshape(5, 3),
            np.full(5, 2),
            np.asarray([4, 0, 3, 1, 2], dtype=np.uint64),
            np.asarray([0, 0, 5]),
        )
        ligand = PDB.fromfile(join(DATA, 'single-atom.pdb'))
        args = Namespace(
            write_complexes=100.0,
            max_complexes=3,
            receptor='fixed.pdb',
            ligand='scanning.pdb',
            voxelspacing=1.0,
            angle=10.0,
        )
        with TemporaryDirectory() as tmp:
            fname = join(tmp, 'complexes.out')
            write_complexes(fname, complexes, rotmat, ligand, 2, args)
            header, models, translations, rotations = read_complexes(fname)
        self.assertEqual(header['complexes'], '3 written of 5')
        self.assertEqual(models.tolist(), [1, 2, 3])
        index = translations[:, 0].astype(int) // 3
        self.assertEqual(index.tolist(), [1, 3, 4])
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
