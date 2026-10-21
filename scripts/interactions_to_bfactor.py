"""Write the average number of interactions into the B-factor column.

Reads a receptor_interactions.txt or ligand_interactions.txt file, written by
`disvis --interaction-selection`, and writes a copy of the corresponding PDB
file in which the B-factor of every atom is the average number of interactions
of its residue, in complexes consistent with at least the given number of
restraints. Residues that were not analysed get a B-factor of 0. Residues are
matched by residue number only, as in DisVis.

The most contacted residues can then be shown by coloring on B-factor, e.g.
`spectrum b, blue_red` in PyMOL or `color bfactor` in ChimeraX.
"""
from argparse import ArgumentParser
import sys

from disvis.helpers import parse_interactions


def parse_args():
    """Parse the command-line arguments."""
    p = ArgumentParser(description=__doc__)

    p.add_argument('pdb',
                   help='PDB file of the receptor or ligand')

    p.add_argument('f_inter',
                   help='Interaction file of the same molecule')

    p.add_argument('nrestraints', type=int,
                   help='Number of consistent restraints')

    p.add_argument('-o', '--out', dest='out', default=None,
                   help='Output PDB file (default: <pdb>_interactions.pdb)')

    args = p.parse_args()
    if args.out is None:
        stem = args.pdb[:-4] if args.pdb.lower().endswith('.pdb') else args.pdb
        args.out = stem + '_interactions.pdb'
    return args


def set_bfactors(lines, bfactors):
    """Return the PDB lines with B-factors taken from bfactors.

    bfactors maps residue numbers to values; atoms of other residues get 0.
    Lines other than ATOM and HETATM records are returned unchanged.
    """
    out = []
    for line in lines:
        if line.startswith(('ATOM  ', 'HETATM')):
            record = line.rstrip('\n').ljust(66)
            bfactor = bfactors.get(int(record[22:26]), 0.0)
            line = '{:s}{:6.2f}{:s}\n'.format(
                record[:60], bfactor, record[66:])
        out.append(line)
    return out


def main():
    """Write the PDB file with the interactions as B-factors."""
    args = parse_args()

    try:
        data = parse_interactions(args.f_inter)
        with open(args.pdb, encoding='utf-8') as fh:
            lines = fh.readlines()
    except (OSError, ValueError) as e:
        sys.exit(str(e))
    if args.nrestraints not in data:
        available = sorted(k for k in data if k != 'residues')
        sys.exit('No interactions for {:d} consistent restraints in {:s}. '
                 'Available: {:s}'.format(
                     args.nrestraints, args.f_inter,
                     ', '.join(str(x) for x in available)))

    bfactors = dict(zip(data['residues'], data[args.nrestraints]))
    pdb_residues = {int(line[22:26]) for line in lines
                    if line.startswith(('ATOM  ', 'HETATM'))}
    missing = sorted(set(bfactors) - pdb_residues)
    if missing:
        sys.exit('Residues of {:s} not found in {:s}: {:s}'.format(
            args.f_inter, args.pdb, ', '.join(str(x) for x in missing)))

    with open(args.out, 'w', encoding='utf-8') as fh:
        fh.writelines(set_bfactors(lines, bfactors))
    print('Wrote {:s} ({:d} residues with interactions, max {:.3f})'.format(
        args.out, len(bfactors), max(bfactors.values(), default=0.0)))


if __name__ == '__main__':
    main()
