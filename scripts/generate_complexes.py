"""Generate consistent complexes from extracted translations and rotations."""

from __future__ import division

from argparse import ArgumentParser
from os.path import abspath, getsize, join, splitext
from glob import glob
from sys import exit

import numpy as np

from disvis.pdb import PDB
from disvis.rotations import quat_to_rotmat
from disvis.helpers import mkdir_p


def parse_args():
    """Parse the command-line arguments."""

    p = ArgumentParser(description=__doc__)
    p.add_argument('ligand', type=file, 
            help="File containing the ligand structure.")
    p.add_argument('infile', type=file,
            help="File containing the translations and rotations.")
    p.add_argument('-o', '--output', dest='output', type=abspath, default='.', metavar='<dir>', 
            help="Directory where the structures will be stored.")
    p.add_argument('-f', '--force', dest='force', action='store_true', 
            help="Do not ask permission to write files.")

    args = p.parse_args()
    return args


class Move(object):

    def __init__(self, trans, rotmat):
        self.trans = trans
        self.rotmat = rotmat


def main():

    args = parse_args()

    moves = []
    for line in args.infile:
        values = np.asarray(line.split()[1:], dtype=np.float64)
        moves.append(Move(values[:3], quat_to_rotmat(values[3:].reshape(1, -1))))

    # Give required memory in MBs
    if not args.force:
        print 'Calculating required memory ...'
        file_size = getsize(args.ligand.name)
        ncomplexes = len(moves)
        required_memory = (file_size * ncomplexes / 1000000)
        print 'This operation will require {:.0f} MB.'.format(required_memory)
        proceed = raw_input('Proceed? [y/n]: ')
        if proceed != 'y':
            exit(0)

    # Move ligand to the origin
    ligand = PDB.fromfile(args.ligand)
    ligand.translate(-ligand.coor.mean(axis=0))
    ligand_coor = ligand.coor.copy()

    print 'Writing complexes to file ...'
    mkdir_p(args.output)
    fn = join(args.output, 'sol_{:d}.pdb')
    for n, move in enumerate(moves, 1):
        ligand.rotate(move.rotmat)
        ligand.translate(move.trans)
        ligand.tofile(fn.format(n))
        ligand.data['x'][:] = ligand_coor[:,0]
        ligand.data['y'][:] = ligand_coor[:,1]
        ligand.data['z'][:] = ligand_coor[:,2]


if __name__ == '__main__':
    main()
