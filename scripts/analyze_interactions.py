from __future__ import print_function, division
import sys
from argparse import ArgumentParser, FileType

from numpy import argsort, asarray, savetxt
from disvis.helpers import parse_interactions


# default values
DEFAULT_ACTIVE_CUTOFF = 1.0
DEFAULT_PASSIVE_CUTOFF = 0.1
DEFAULT_OUT = open('average-interactions.out', 'w')


def parse_args():

    p = ArgumentParser()

    # positional
    p.add_argument('f_inter', type=file,
        help='Interaction file')

    p.add_argument('nrestraints', type=int,
        help='Number of consistent restraints')

    # optional
    p.add_argument('-ac', '--active-cutoff', dest='active_cutoff',
        default=DEFAULT_ACTIVE_CUTOFF, type=float,
        help="Minimum number of average interactions for a residue to be "
        "considered 'active'.")

    p.add_argument('-pc', '--passive-cutoff', dest='passive_cutoff',
        default=DEFAULT_PASSIVE_CUTOFF, type=float,
        help="Minimum number of average interactions for a residue to be considered 'passive'.")

    p.add_argument('-o', '--out', dest='out', type=FileType('w'),
        default=DEFAULT_OUT,
        help='File name of output')

    # flags
    p.add_argument('-s', '--sort', dest='sort', action='store_true',
        help='Sort the residues by most contacted')

    args = p.parse_args()

    if args.nrestraints < 0:
         raise ValueError("Number of restraints should be > 0")

    return args


def main():

    args = parse_args()

    data = parse_interactions(args.f_inter.name)

    # extract data into arrays
    residue_index = asarray(range(len(data[args.nrestraints]['interactions'])))
    residue_id = asarray(data['residues'])
    ave_interactions = asarray(data[args.nrestraints]['interactions']) /\
        data[args.nrestraints]['total']

    if args.sort:
        order = argsort(ave_interactions)[::-1]
        residue_id = residue_id[order]
        ave_interactions = ave_interactions[order]

    active_res = residue_id[ave_interactions >= args.active_cutoff]

    passive_indices = (ave_interactions >= args.passive_cutoff) &\
        (ave_interactions < args.active_cutoff)
    passive_res = residue_id[passive_indices]

    print('Active residues ({:d}):'.format(active_res.size))
    print(str(list(active_res))[1: -1])
    print('Passive residues ({:d}):'.format(passive_res.size))
    print(str(list(passive_res))[1: -1])

    with args.out as fh:
        max_digits = str(len(str(residue_id.max())))
        max_float = str(len(str(int(ave_interactions.max()))) + 3)
        line = '{:<' + max_digits + 'd} {:>' + max_float + '.2f}\n'
        for n in range(residue_id.size):
            fh.write(line.format(residue_id[n], ave_interactions[n]))


if __name__=='__main__':
    main()
