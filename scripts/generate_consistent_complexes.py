from __future__ import division

from argparse import ArgumentParser
from os.path import abspath, getsize, join, splitext
from glob import glob
from sys import exit

import numpy as np

from disvis.volume import Volume
from disvis.pdb import PDB
from disvis.rotations import quat_to_rotmat, proportional_orientations


def parse_args():
    """Parse the command-line arguments."""

    p = ArgumentParser(description="Generate consistent complexes.")
    p.add_argument('ligand', type=file, 
        help="File containing the ligand structure.")
    p.add_argument('consistent_restraints', type=int,
        help="Number of required consistent restraints.")
    p.add_argument('-o', '--output', dest='output', type=abspath, default='.', 
        help="Directory where the structures will be stored.")
    p.add_argument('-i', '--input', dest='input', type=abspath, default='.',
        help="Directory where the input files can be found.")
    p.add_argument('-f', '--force', dest='force', action='store_true',
        help="Do not ask permission to write files.")

    args = p.parse_args()
    return args


def get_number_of_complexes(infiles, nrestraints):
    
    ncomplexes = 0
    for f in infiles:
        ncomplexes += (Volume.fromfile(f).array >= nrestraints).sum()
    return ncomplexes

        
def main():
    """Main function that will perform the function of the script."""

    args = parse_args()

    interspace_files = glob(join(args.input, 'red_interspace_*.mrc'))
    nrot = len(interspace_files)
    if nrot == 0:
        raise ValueError("No input files where found to generate complexes.")

    # Give required memory in MBs
    if not args.force:
        print 'Calculating required memory ...'
        file_size = getsize(args.ligand.name)
        ncomplexes = get_number_of_complexes(interspace_files, args.consistent_restraints)
        required_memory = (file_size * ncomplexes / 1000000)
        print 'This operation will require {:.0f} MB. Proceed?'.format(required_memory)
        proceed = raw_input('[y/n]: ')
        if proceed != 'y':
            exit(0)

    # Move ligand to the origin
    ligand = PDB.fromfile(args.ligand)
    ligand.translate(-ligand.coor.mean(axis=0))
    rotmats = quat_to_rotmat(proportional_orientations(nrot, metric='number')[0])

    print 'Writing complexes to file ...'
    n = 0
    for fn in interspace_files:
        job, ind = [int(x) for x in splitext(fn)[0].split('_')[2:]]
        rotmat = rotmats[ind]
        vol = Volume.fromfile(fn)
        # Rotate ligand, and move to origin of the map
        rot_ligand = ligand.duplicate()
        rot_ligand.rotate(rotmat)
        rot_ligand.translate(vol.origin)
        rot_coor = rot_ligand.coor
        trans = np.asarray(vol.array.nonzero()[::-1]).T * vol.voxelspacing
        for t in xrange(trans.shape[0]):
            rot_ligand.translate(trans[t])
            rot_ligand.tofile(join(args.output, 'sol_{:d}.pdb'.format(n)))
            rot_ligand.coor[:] = rot_coor
            n += 1

            if n > 3:
                exit()


if __name__ == '__main__':
    main()
