"""Run the test cases and compare the output with the reference results.

The output files of each run must be byte-identical to the reference results
in test-cases/*/example-results/cpu/<grid>, or gpu/<grid> with --gpu. Run
from the repository root, e.g.

    python tests/check_test_cases.py              # coarse, CPU, 4 processes
    python tests/check_test_cases.py --fine -p 8  # fine runs take long
    python tests/check_test_cases.py --gpu

The exit status is 1 if an output file differs from its reference.
"""

import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from argparse import ArgumentParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = {
    'PRE5-PUP2-complex': ('O14250.pdb', 'Q9UT97.pdb'),
    'RNA-polymerase-II/A-E': ('1wcm_A.pdb', '1wcm_E.pdb'),
}
GRIDS = {'coarse': ('9.72', '2'), 'fine': ('5.27', '1')}
OUTPUT = (
    'accessible_complexes.out',
    'violations.out',
    'z-score.out',
    'accessible_interaction_space.mrc',
)


def parse_args():
    """Parse the command-line arguments."""
    p = ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument(
        '--fine', action='store_true', help='run the fine instead of coarse'
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        '-p', '--nproc', type=int, default=4, help='processes (default 4)'
    )
    mode.add_argument(
        '-g', '--gpu', action='store_true', help='run on the GPU'
    )
    p.add_argument(
        '-d', '--directory', help='keep the output in this directory'
    )
    return p.parse_args()


def run_case(case, grid, args, outdir):
    """Run one test case; return the names of the files that differ."""
    receptor, ligand = CASES[case]
    angle, voxelspacing = GRIDS[grid]
    casedir = os.path.join(ROOT, 'test-cases', case)
    # The disvis command of this Python, also outside an activated venv.
    disvis = shutil.which(
        'disvis', path=os.path.dirname(sys.executable)
    ) or shutil.which('disvis')
    command = [
        disvis, receptor, ligand, 'restraints.dat',
        '-a', angle, '-vs', voxelspacing, '-d', outdir,
    ]  # fmt: skip
    command += ['-g'] if args.gpu else ['-p', str(args.nproc)]
    subprocess.run(command, cwd=casedir, check=True, stdout=subprocess.DEVNULL)

    with open(os.path.join(outdir, 'disvis.log'), encoding='utf-8') as f:
        last = f.readlines()[-1][24:].strip()
    print('{:s} {:s}: {:s}'.format(case, grid, last))
    reference = os.path.join(
        casedir, 'example-results', 'gpu' if args.gpu else 'cpu', grid
    )
    differ = []
    for fname in OUTPUT:
        same = filecmp.cmp(
            os.path.join(outdir, fname),
            os.path.join(reference, fname),
            shallow=False,
        )
        print('  {:s}: {:s}'.format(fname, 'identical' if same else 'DIFFERS'))
        if not same:
            differ.append(fname)
    return differ


def main():
    """Run the test cases and report the files that differ."""
    args = parse_args()
    grid = 'fine' if args.fine else 'coarse'
    with tempfile.TemporaryDirectory() as tmpdir:
        base = args.directory or tmpdir
        differ = []
        for case in CASES:
            outdir = os.path.join(base, case.replace('/', '_') + '_' + grid)
            differ += [
                case + ': ' + f for f in run_case(case, grid, args, outdir)
            ]
    if differ:
        print('Different from the reference results:', ', '.join(differ))
        sys.exit(1)
    print('All output files are identical to the reference results.')


if __name__ == '__main__':
    main()
