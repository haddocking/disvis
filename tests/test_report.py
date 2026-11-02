"""Test of the --report option.

Run from the repository root with python -m unittest.
"""

import json
import os
import tempfile
from os.path import dirname, join
from unittest import TestCase, main
from unittest.mock import patch

from disvis import main as disvis_main

DATA = join(dirname(__file__), 'data')


class TestReport(TestCase):
    def test_report(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['disvis'] + [
                join(DATA, f)
                for f in ('O14250.pdb', 'Q9UT97.pdb', 'restraints.dat')
            ]
            argv += ['-a', '40', '-vs', '3', '-np', '-oa', '--report']
            argv += ['-is', join(DATA, 'selection.res'), '-d', directory]
            with patch('sys.argv', argv), patch('builtins.print'):
                disvis_main.main()

            with open(join(directory, 'accessible_complexes.out')) as f:
                rows = [line.split() for line in f if line[0] != '#']
            levels = [r[0] for r in rows if r[0] != '0' and int(r[3]) > 0]
            with open(join(directory, 'state.mvsj')) as f:
                snapshots = json.load(f)['snapshots']
            self.assertEqual(
                [s['metadata']['key'] for s in snapshots],
                ['n' + n for n in reversed(levels)],
            )
            # Mol* loads files relative to the report, all in the directory.
            urls = []
            for s in snapshots:
                urls += [c['params']['url'] for c in s['root']['children']]
            self.assertTrue(set(urls) <= set(os.listdir(directory)))

            with open(join(directory, 'report.html'), encoding='utf-8') as f:
                html = f.read()
            self.assertIn('A27(CA)-A18(CA)', html)
            self.assertIn('ligand residues', html)


if __name__ == '__main__':
    main()
