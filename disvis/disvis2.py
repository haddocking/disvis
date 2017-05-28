"""Quantify and visualize the information content of distance restraints."""

from argparse import ArgumentParser
import os.path
from itertools import izip
import sys
import logging
logger = logging.getLogger(__name__)

import numpy as np

from .pdb import PDB
from .volume import Volume, Volumizer
from .spaces import (InteractionSpace, RestraintSpace, Restraint, 
        AccessibleInteractionSpace, OccupancySpace)
from .helpers import RestraintParser, mkdir_p, DJoiner
from .rotations import proportional_orientations, quat_to_rotmat


def parse_args():

    p = ArgumentParser(description=__doc__)
    p.add_argument("receptor", type=str,
            help="Receptor / fixed chain.")
    p.add_argument("ligand", type=str,
            help="Ligand / scanning chain.")
    p.add_argument("restraints", type=file,
            help="File containing restraints.")

    p.add_argument("-vs", "--voxelspacing", default=2, type=float,
            help="Voxelspacing of grids.")
    p.add_argument("-a", "--angle", default=20, type=float,
            help="Rotational sampling density in degree.")
    p.add_argument("-oa", "--occupancy-analysis", action="store_true",
            help="Perform an occupancy analysis.")
    p.add_argument("-ia", "--interaction-analysis", action="store_true",
            help="Perform an interaction analysis.")
    p.add_argument("-s", "--save", action="store_true",
            help="Save entire accessible interaction space to disk.")
    p.add_argument("-d", "--directory", default='.', type=os.path.abspath,
            help="Directory to store the results.")
    p.add_argument("-v", "--verbose", action="store_true",
            help="Be verbose.")

    args = p.parse_args()
    return args


class DisVisOptions(object):
    minimum_interaction_volume = 300
    maximum_clash_volume = 200
    voxelspacing = 2
    interaction_radius = 3
    save = False
    directory = '.'

    @classmethod
    def fromargs(cls, args):
        for key, value in vars(args).iteritems():
            setattr(cls, key, value)
        return cls


class DisVis(object):

    def __init__(self, receptor, ligand, restraints, options):
        self.receptor = receptor
        self.ligand = ligand
        self.restraints = restraints
        self.options = options
        self._initialized = False
        self._counter = 0

    def initialize(self):

        self._volumizer = Volumizer(
                self.receptor, self.ligand, 
                voxelspacing=self.options.voxelspacing,
                interaction_radius=self.options.interaction_radius,
                )

        rcore = self._volumizer.rcore
        rsurface = self._volumizer.rsurface
        lcore = self._volumizer.lcore
        interaction_space = Volume.zeros_like(rcore, dtype=np.int32)
        self._interaction_space_calc = InteractionSpace(
                interaction_space, rcore, rsurface, lcore,
                max_clash=self.options.maximum_clash_volume,
                min_inter=self.options.minimum_interaction_volume,
                )
        restraint_space= Volume.zeros_like(rcore, dtype=np.int32)
        self._restraint_space_calc = RestraintSpace(
                restraint_space, self.restraints, self.ligand.center
                )
        accessible_interaction_space = Volume.zeros_like(rcore, dtype=np.int32)
        self._ais_calc = AccessibleInteractionSpace(
                accessible_interaction_space, self._interaction_space_calc, 
                self._restraint_space_calc
                )
        if self.options.occupancy_analysis:
            self._occupancy_space = OccupancySpace(
                    self._interaction_space_calc, self._ais_calc)
        self._initialized = True

    def __call__(self, rotmat, weight=1):

        if not self._initialized:
            self.initialize()

        self._volumizer.generate_lcore(rotmat)
        self._interaction_space_calc()
        self._restraint_space_calc(rotmat)
        self._ais_calc(weight=weight)
        if self.options.save:
            fname = os.path.join(self.options.directory, 
                    'ais_{:d}.mrc'.format(self._counter))
            self._ais_calc.consistent_space.tofile(fname)
            self._counter += 1

        if self.options.occupancy_analysis:
            self._occupancy_space(weight=weight)

    @property
    def consistent_complexes(self):
        return self._ais_calc.consistent_complexes()

    @property
    def violation_matrix(self):
        return self._ais_calc.violation_matrix()

    @property
    def max_consistent(self):
        return self._ais_calc.max_consistent


def main():

    args = parse_args()
    mkdir_p(args.directory)

    # Set up logger
    logging_fname = os.path.join(args.directory, 'disvis.log')
    logging.basicConfig(filename=logging_fname, level=logging.INFO)
    if args.verbose:
        console_out = logging.StreamHandler(stream=sys.stdout)
        console_out.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console_out)
    logger.info(sys.argv)

    logger.info("Reading receptor and ligand.")
    receptor = PDB.fromfile(args.receptor)
    ligand = PDB.fromfile(args.ligand)

    # Get the restraint
    logger.info("Parsing restraints.")
    restraint_parser = RestraintParser()
    restraints = []
    for line in args.restraints:
         out = restraint_parser.parse_line(line)
         if out is not None:
             rsel, lsel, min_dis, max_dis = out
             receptor_selection = []
             for sel in rsel:
                 rpart = receptor
                 for key, value in sel:
                     rpart = rpart.select(key, value)
                 receptor_selection.append(rpart)
             lpart = ligand
             ligand_selection = []
             for sel in lsel:
                 lpart = ligand
                 for key, value in sel:
                     lpart = lpart.select(key, value)
                 ligand_selection.append(lpart)
             restraints.append(Restraint(
                 receptor_selection, ligand_selection, min_dis, max_dis)
                 )

    options = DisVisOptions.fromargs(args)

    quat, weights, alpha = proportional_orientations(args.angle)
    rotations = quat_to_rotmat(quat)
    disvis = DisVis(receptor, ligand, restraints, options)
    import time
    time0 = time.time()
    for n, (rotmat, weight) in enumerate(izip(rotations, weights)):
        if args.verbose:
            sys.stdout.write('{:>6d}\r'.format(n))
            sys.stdout.flush()
        disvis(rotmat, weight=weight)
    print 'Time:', time.time() - time0

    print disvis.consistent_complexes
    print disvis.violation_matrix
    disvis.max_consistent.tofile(os.path.join(args.directory, 'consistent_space.mrc'))

    # Write number of correlated consistent restraints
    nrestraints = len(restraints)
    for nconsistent_restraints in xrange(nrestraints + 1):
        mask = disvis._ais_calc._consistent_restraints == nconsistent_restraints
        cons_perm = disvis._ais_calc._consistent_permutations[mask]
        consistent_sets = [bin(x) for x in disvis._ais_calc._indices[mask]]
        for cset, sperm in izip(consistent_sets, cons_perm):
            restraint_flags = ' '.join(list(('{:0>' + str(nrestraints) + 'd}').format(int(cset[2:]))))
            print restraint_flags, sperm
        print '=' * 30

    # Write out occupancy maps
    if options.occupancy_analysis:
        for n, space in enumerate(disvis._occupancy_space.spaces):
            space.tofile('occ_{:}.mrc'.format(n))

