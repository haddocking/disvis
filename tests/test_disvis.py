"""Tests of the CPU search steps.

Run from the repository root with python -m unittest.
"""

from argparse import ArgumentTypeError
from os.path import dirname, join
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import numpy as np

from disvis.pdb import PDB
from disvis.disvis import DisVis
from disvis.helpers import (
    _TblParser,
    parse_restraints,
    parse_tbl,
    positive_float,
)
from disvis.rotations import proportional_orientations, quat_to_rotmat
from disvis.volume import radix235

DATA = join(dirname(__file__), 'data')


class TestDisVis(TestCase):
    def setUp(self):
        self.dv = DisVis()
        self.receptor = PDB.fromfile(join(DATA, 'single-atom.pdb'))
        self.ligand = PDB.fromfile(join(DATA, 'single-atom.pdb'))

    def test_minimal_volume_parameters(self):
        coor1 = np.asarray([[1, 2, 3], [-2, -3, -4]], dtype=np.float64)
        coor2 = np.asarray([[1, 0, 0], [-1, 0, 0]], dtype=np.float64)
        offset = 0
        voxelspacing = 1
        shape, origin = self.dv._minimal_volume_parameters(
            coor1, coor2, offset, voxelspacing
        )
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
        self.assertTrue(
            np.allclose(dv._clashvol, np.logical_not(dv._not_clashing))
        )
        self.assertTrue(
            np.allclose(dv._interspace, dv._interacting - dv._clashvol)
        )

    def test_get_restraints_center(self):
        dv = self.dv

        # 90 degree rotation around Z-axis
        rotmat = np.asarray(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64
        )
        # Set required attributes to test function
        dv._lrestraints = np.asarray([[2, 0, 0]], dtype=np.float64)
        dv._rot_lrestraints = np.zeros_like(dv._lrestraints)
        dv._restraints_center = np.zeros_like(dv._lrestraints)
        dv._rrestraints = np.asarray([[1, 0, 0]], dtype=np.float64)

        dv._get_restraints_center(rotmat)
        self.assertTrue(
            np.allclose(dv._rot_lrestraints, np.array([[0, 2, 0]]))
        )
        self.assertTrue(
            np.allclose(dv._restraints_center, np.array([[1, -2, 0]]))
        )

    def test_get_restraint_space(self):
        dv = self.dv

        # An ambiguous restraint with two centers, and a restraint with one
        dv._restraints_center = np.array(
            [[4, 3, 2], [6, 3, 2], [0, 0, 0]], dtype=np.float64
        )
        dv._pair_offsets = np.array([0, 2, 3], dtype=np.int32)
        dv._mindis = np.array([0, 0], dtype=np.float64)
        dv._maxdis = np.array([2, 1], dtype=np.float64)
        dv._restspace = np.empty((5, 7, 9), np.int32)
        dv._consistent_bits = np.empty((5, 7, 9, 1), np.uint8)

        dv._get_restraint_space()
        z, y, x = np.indices(dv._restspace.shape)
        dis1 = np.minimum(
            (x - 4) ** 2 + (y - 3) ** 2 + (z - 2) ** 2,
            (x - 6) ** 2 + (y - 3) ** 2 + (z - 2) ** 2,
        )
        answer = (dis1 <= 4).astype(int) + (x**2 + y**2 + z**2 <= 1)
        self.assertTrue(np.array_equal(dv._restspace, answer))

        # Violations of the complexes consistent with 1 restraint
        dv._red_interspace = dv._restspace
        dv._violations = np.zeros((2, 2))
        dv._count_violations(1)
        self.assertEqual(dv._violations[0, 0], 4)
        self.assertEqual(dv._violations[0, 1], (dis1 <= 4).sum())

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

    def test_collect_complexes(self):
        # Every complex collected, placed in A, is consistent with its number
        # of restraints, the highest found or a percentage of it, and
        # they are all collected.
        receptor = PDB.fromfile(join(DATA, 'O14250.pdb'))
        ligand = PDB.fromfile(join(DATA, 'Q9UT97.pdb'))
        restraints = parse_restraints(
            join(DATA, 'restraints.dat'), receptor, ligand
        )
        dv = self.dv
        dv.receptor, dv.ligand = receptor, ligand
        dv.distance_restraints = restraints
        dv.rotations = quat_to_rotmat(proportional_orientations(40)[0])
        dv.voxelspacing = 2
        dv.collect_complexes = True
        dv.search()
        best = dv.complexes[2].max()
        self.assertTrue(np.all(dv.complexes[2] == best))
        # The percentage of the highest number for best - 1 restraints
        dv.complexes_percentage = 100 * (best - 1) / best
        dv._collect_complexes()
        rotation, translation, consistent = dv.complexes
        self.assertEqual(
            len(rotation), dv.accessible_complexes[best - 1 :].sum()
        )
        for n, t, c in zip(rotation, translation, consistent, strict=True):
            placed = 0
            for r, lig, mindis, maxdis in restraints:
                placed_lig = (lig - ligand.center) @ dv.rotations[n].T + t
                dis = np.linalg.norm(r - placed_lig, axis=1).min()
                placed += mindis <= dis <= maxdis
            self.assertEqual(placed, c)


class TestInputChecks(TestCase):
    def test_radix235(self):
        self.assertEqual(radix235(1), 1)
        self.assertEqual(radix235(7), 8)
        self.assertEqual(radix235(31), 32)
        self.assertEqual(radix235(121), 125)
        for n in (0, -3):
            with self.assertRaises(ValueError):
                radix235(n)

    def test_parse_tbl(self):
        pdb1 = PDB.fromfile(join(DATA, 'O14250.pdb'))
        pdb2 = PDB.fromfile(join(DATA, 'Q9UT97.pdb'))
        # The first restraint is the first line of restraints.dat. Residue
        # 3 is only in pdb1, so the second restraint (with keywords
        # shortened as in CNS) is read reversed, its or pair too.
        tbl = """! HADDOCK restraints
assign (segid A and resid 27 and name CA)
       (segid A and resid 18 and name CA) 10.0 10.0 13.0
assi (resi 30 and name CA) (resid 3 and name C*) 2.0 2.5 0.0 ! comment
  or (resid 122:123 and name CA) (resid 125 and not name CA)
"""
        with TemporaryDirectory() as tmp:
            with open(join(tmp, 'restraints.tbl'), 'w') as f:
                f.write(tbl)
            messages = []
            restraints = parse_restraints(f.name, pdb1, pdb2, messages.append)
        reference = parse_restraints(join(DATA, 'restraints.dat'), pdb1, pdb2)

        self.assertEqual(len(restraints), 2)
        for a, b in zip(restraints[0], reference[0], strict=True):
            self.assertTrue(np.array_equal(a, b))
        r1, r2, mindis, maxdis = restraints[1]
        res3 = pdb1.select('resi', 3).data['name']
        n3 = sum(name.startswith('C') for name in res3)
        n125 = pdb1.select('resi', 125).natoms - 1
        self.assertEqual(r1.shape, (n3 + n125 * 2, 3))
        self.assertTrue(
            np.array_equal(
                r2[0], pdb2.select('resi', 30).select('name', 'CA').coor[0]
            )
        )
        self.assertEqual((mindis, maxdis), (0.0, 2.0))
        # Upper distances shorter than min_maxdis are raised to it
        messages = []
        raised = parse_tbl(tbl, pdb1, pdb2, messages.append, min_maxdis=3.9)
        self.assertEqual([r[3] for r in raised], [23.0, 3.9])
        self.assertIn(
            '1 of 2 restraints with an upper distance raised to 3.9 A',
            messages,
        )
        # Both structures are chain A
        self.assertTrue(messages[0].startswith('WARNING'))
        self.assertEqual(
            messages[1],
            '1 of 2 restraints read with the first selection in the '
            'scanning chain',
        )
        # Different chains: no warning, and both pairs read forward
        pdb2b = PDB(pdb2.data.copy())
        pdb2b.data['chain'] = 'B'
        messages = []
        forward = (
            'assign (segid A and resid 27) (segid B and resid 18) 2 2 0 '
            'or (segid A and resid 28) (segid B and resid 19)'
        )
        ((r1, r2, *_),) = parse_tbl(forward, pdb1, pdb2b, messages.append)
        npairs = [
            pdb1.select('resi', i).natoms * pdb2b.select('resi', j).natoms
            for i, j in ((27, 18), (28, 19))
        ]
        self.assertEqual(r1.shape, (sum(npairs), 3))
        self.assertEqual(
            messages,
            [
                '0 of 1 restraints read with the first selection in the '
                'scanning chain'
            ],
        )
        # Residue 3 is not in pdb2: in neither direction for both pairs
        mixed = 'assign (resid 3) (resid 30) 2 2 0 or (resid 30) (resid 3)'
        with self.assertRaises(ValueError):
            parse_tbl(mixed, pdb1, pdb2)

    def test_restraints_file_errors(self):
        pdb = PDB.fromfile(join(DATA, 'single-atom.pdb'))
        with TemporaryDirectory() as tmp:
            for text, message in (
                ('# no restraints\n', 'No restraints found'),
                ('! a HADDOCK comment\n', 'Line 1 of '),
                ('A 1 N A 1 N 0 2x\n', 'Line 1 of '),
                ('A 1 N A 2 N 0 2\n', 'selection of line 1 of '),
            ):
                fname = join(tmp, 'restraints.dat')
                with open(fname, 'w') as f:
                    f.write(text)
                with self.assertRaisesRegex(ValueError, message):
                    parse_restraints(fname, pdb, pdb)

    def test_tbl_selection(self):
        # CNS precedence: not before and before or
        pdb = PDB.fromfile(join(DATA, 'O14250.pdb'))
        resi, ca = pdb.data['resi'], pdb.data['name'] == 'CA'
        chain_a = pdb.data['chain'] == 'A'
        gl = np.char.startswith(pdb.data['resn'].astype(str), 'GL')
        for sel, answer in (
            ('(not (resid 1 or resid 2) and name CA)', (resi > 2) & ca),
            (
                '(resid 1 or resid 2 and name CA)',
                (resi == 1) | (resi == 2) & ca,
            ),
            (
                '((segid A and not resid 10) or (segid B and resid 20))',
                chain_a & (resi != 10),
            ),
            (
                '(resid 5:8 and not (name C* or resname GL*))',
                (resi >= 5)
                & (resi <= 8)
                & ~np.char.startswith(pdb.data['name'].astype(str), 'C')
                & ~gl,
            ),
        ):
            mask = _TblParser(sel).expression()(pdb)
            self.assertTrue(np.array_equal(mask, answer), sel)

    def test_positive_float(self):
        self.assertEqual(positive_float('0.5'), 0.5)
        for value in ('0', '-1', 'nan', 'inf', 'abc'):
            with self.assertRaises(ArgumentTypeError):
                positive_float(value)


if __name__ == '__main__':
    main()
