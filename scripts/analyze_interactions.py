"""Classify residues as active or passive from a DisVis interaction analysis.

Reads a receptor_interactions.txt or ligand_interactions.txt file, written by
`disvis --interaction-selection`, and selects the residues whose average number
of interactions, in complexes consistent with at least the given number of
restraints, exceeds the active and passive cutoffs.
"""
from argparse import ArgumentParser
import sys

from numpy import argsort, asarray
from disvis.helpers import parse_interactions


# default values
DEFAULT_ACTIVE_CUTOFF = 1.0
DEFAULT_PASSIVE_CUTOFF = 0.1
DEFAULT_OUT = 'average-interactions.out'


def parse_args():
    """Parse the command-line arguments."""
    p = ArgumentParser(description=__doc__)

    # positional
    p.add_argument('f_inter',
                   help='Interaction file')

    p.add_argument('nrestraints', type=int,
                   help='Number of consistent restraints')

    # optional
    p.add_argument('-ac', '--active-cutoff', dest='active_cutoff',
                   default=DEFAULT_ACTIVE_CUTOFF, type=float,
                   help="Minimum number of average interactions for a "
                        "residue to be considered 'active'.")

    p.add_argument('-pc', '--passive-cutoff', dest='passive_cutoff',
                   default=DEFAULT_PASSIVE_CUTOFF, type=float,
                   help="Minimum number of average interactions for a "
                        "residue to be considered 'passive'.")

    p.add_argument('-o', '--out', dest='out',
                   default=DEFAULT_OUT,
                   help='File name of output')

    # flags
    p.add_argument('-s', '--sort', dest='sort', action='store_true',
                   help='Sort the residues by most contacted')

    args = p.parse_args()

    if args.nrestraints < 0:
        raise ValueError("Number of restraints should be >= 0")

    return args


def format_numbers(numbers):
    """Return the numbers as a comma-separated string."""
    return ', '.join(str(x) for x in numbers)


def main():
    args = parse_args()

    try:
        data = parse_interactions(args.f_inter)
    except (OSError, ValueError) as e:
        sys.exit(str(e))
    if args.nrestraints not in data:
        available = sorted(k for k in data if k != 'residues')
        sys.exit('No interactions for {:d} consistent restraints in {:s}. '
                 'Available: {:s}'.format(args.nrestraints, args.f_inter,
                                          format_numbers(available)))

    # extract data into arrays
    residue_id = asarray(data['residues'])
    ave_interactions = asarray(data[args.nrestraints])

    if args.sort:
        order = argsort(ave_interactions)[::-1]
        residue_id = residue_id[order]
        ave_interactions = ave_interactions[order]

    active_res = residue_id[ave_interactions >= args.active_cutoff]

    passive_indices = (ave_interactions >= args.passive_cutoff) & \
        (ave_interactions < args.active_cutoff)
    passive_res = residue_id[passive_indices]

    print('Active residues ({:d}):'.format(active_res.size))
    print(format_numbers(active_res))
    print('Passive residues ({:d}):'.format(passive_res.size))
    print(format_numbers(passive_res))

    with open(args.out, 'w') as fh:
        max_digits = str(len(str(residue_id.max())))
        max_float = str(len(str(int(ave_interactions.max()))) + 3)
        line = '{:<' + max_digits + 'd} {:>' + max_float + '.2f}\n'
        for n in range(residue_id.size):
            fh.write(line.format(residue_id[n], ave_interactions[n]))


if __name__ == '__main__':
    main()
