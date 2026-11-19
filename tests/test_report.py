"""Test of the report (report.html and state.mvsj).

Run from the repository root with python -m unittest.
"""

import json
import re
import os
import shutil
import tempfile
import warnings
from os.path import dirname, join
from unittest import TestCase, main

DATA = join(dirname(__file__), 'data')

with warnings.catch_warnings():
    # molviewspec 1.8 passes a deprecated keyword to pydantic.
    warnings.filterwarnings(
        'ignore', 'Using extra keyword arguments on `Field`'
    )
    from disvis.report import _by_zscore, _restraint_labels, generate_report


class TestReport(TestCase):
    def test_report(self):
        with tempfile.TemporaryDirectory() as directory:
            # The text output of disvis O14250.pdb Q9UT97.pdb restraints.dat
            # -a 40 -vs 3 -oa -is selection.res
            shutil.copytree(
                join(DATA, 'output'), directory, dirs_exist_ok=True
            )
            generate_report(
                directory,
                *[
                    join(DATA, f)
                    for f in ('O14250.pdb', 'Q9UT97.pdb', 'restraints.dat')
                ],
                {},
                join(DATA, 'selection.res'),
            )

            with open(join(directory, 'accessible_complexes.out')) as f:
                rows = [line.split() for line in f if line[0] != '#']
            levels = [r[0] for r in rows if r[0] != '0' and int(r[3]) > 0]
            with open(join(directory, 'state.mvsj')) as f:
                snapshots = json.load(f)['snapshots']
            self.assertEqual(
                [s['metadata']['key'] for s in snapshots],
                ['n' + n for n in reversed(levels)],
            )
            # Mol* loads files relative to the report, in the directory.
            urls = set()
            for s in snapshots:
                urls |= {c['params']['url'] for c in s['root']['children']}
            self.assertEqual(
                urls, {'O14250.pdb', 'accessible_interaction_space.mrc'}
            )
            # The input files are copied, to be downloaded.
            for f in ('Q9UT97.pdb', 'restraints.dat', 'selection.res'):
                self.assertIn(f, os.listdir(directory))

            with open(join(directory, 'report.html'), encoding='utf-8') as f:
                html = f.read()
            self.assertIn('A27(CA)-A18(CA)', html)
            self.assertIn('Most contacted ligand residues', html)
            # All 7 restraints are consistent: their row of violations (0)
            # is shown, and no Z-score flags a false positive.
            self.assertIn('<tr><td>7</td><td>0.0000</td>', html)
            self.assertNotIn('class="high"', html)
            # The Z-scores, highest first (highlighted rows included), and
            # the residues, most contacted (at the highest N) first
            zscores = html.split('<h3>Z-scores')[1].split('</table>')[0]
            self.assertEqual(
                re.findall(r'<tr[^>]*><td>(\d+)</td>', zscores),
                ['2', '1', '4', '3', '5', '6', '7'],
            )
            with open(join(DATA, 'output', 'ligand_interactions.txt')) as f:
                values = [line.split() for line in f][1:]
            most = max(values, key=lambda row: float(row[-1]))[0]
            ligand = html.split('Average interactions per complex, ligand')[1]
            self.assertTrue(
                ligand.split('<td')[1].endswith('>' + most + '</td>')
            )

    def test_by_zscore(self):
        rows = [
            [str(i), '', '', '', z] for i, z in enumerate(('nan', '-1', '2'))
        ]
        self.assertEqual([r[0] for r in _by_zscore(rows)], ['2', '1', '0'])

    def test_delimiter(self):
        # The same run with --delimiter , gives the same report.
        html = []
        for delimiter in (None, ','):
            with tempfile.TemporaryDirectory() as directory:
                shutil.copytree(
                    join(DATA, 'output'), directory, dirs_exist_ok=True
                )
                if delimiter:
                    shutil.copytree(
                        join(DATA, 'output-csv'),
                        directory,
                        dirs_exist_ok=True,
                    )
                fname = generate_report(
                    directory,
                    *[
                        join(DATA, f)
                        for f in ('O14250.pdb', 'Q9UT97.pdb', 'restraints.dat')
                    ],
                    {},
                    delimiter=delimiter,
                )
                with open(fname, encoding='utf-8') as f:
                    html.append(f.read())
        self.assertEqual(html[0], html[1])

    def test_tbl_labels(self):
        # HADDOCK restraints are numbered, also over several lines.
        tbl = """! assign (resid 1) (resid 2) 2.0 2.0 0.0
assign (resid 27) (resid 18) 2.0 2.0 0.0
  or (resid 28) (resid 18)
ASSIGN (resid 29)
       (resid 18) 2.0 2.0 0.0
"""
        with tempfile.TemporaryDirectory() as directory:
            with open(join(directory, 'restraints.tbl'), 'w') as f:
                f.write(tbl)
            self.assertEqual(_restraint_labels(f.name), ['1', '2'])


if __name__ == '__main__':
    main()
