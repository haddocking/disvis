from argparse import ArgumentParser

from .pdb import PDB
from .volume import Volume, Volumizer
from .spaces import InteractionSpace, RestraintSpace, Restraint, AccessibleInteractionSpace


class DisVisOptions(object):
    minimum_interaction_volume = 300
    maximum_clash_volume = 200
    voxelspacing = 1
    interaction_radius = 3


class DisVis(object):

    def __init__(self, receptor, ligand, restraints, options):
        self.receptor = receptor
        self.ligand = ligand
        self.restraints = restraints
        self.rotmat = rotmat
        self.options = options
        self._initialized = False

    def initialize(self):
        self._initialized = True

        self._volumizer = Volumizer(
                self.receptor, self.ligand, 
                voxelspacing=self.options.voxelspacing,
                interaction_radius=self.options.interaction_radius,
                )

        rcore = volumizer.rcore
        rsurface = volumizer.rsurface
        interaction_space = Volume.zeros_like(rcore)
        self._interaction_space_calc = InteractionSpace(
                interaction_space, rcore, rsurface,
                max_clash=self.options.maximum_clash_volume,
                min_inter=self.options.minimum_interaction_volume,
                )
        restraint_space= Volume.zeros_like(rcore)
        self._restraint_space_calc = RestraintSpace(
                restraint_space, self.restraints, self.ligand.center
                )

        accessible_interaction_space = Volume.zeros_like(rcore)
        self._ais_calc = AccessibleInteractionSpace(
                accessible_interaction_space, len(self.restraints),
                )

    def __call__(self, rotmat, weight=1):

        self._volumizer.generate_lcore(rotmat)
        self._interaction_space_calc(self._volumizer.lcore)
        self._restraint_space_calc(rotmat)
        self._ais_calc(
                self._interaction_space_calc.space,
                self._restraint_space_calc.space, 
                weight=weight
                )

    @property
    def consistent_complexes(self):
        return self._ais_calc.consistent_complexes()

    @property
    def violation_matrix(self):
        return self._ais_calc.violation_matrix()


def parse_args():

    p = ArgumentParser(description="Quantify and visualize the information content of distance restraints.")
    p.add_argument("receptor", type=str,
            help="Receptor / fixed chain.")
    p.add_argument("ligand", type=str,
            help="Ligand / scanning chain.")
    p.add_argument("restraints", type=str,
            help="File containing restraints.")

    args = p.parse_args()
    return args


def main():

    args = parse_args()
    recepter = PDB.fromfile(args.receptor)
    ligand = PDB.fromfile(args.ligand)

    options = DisVisOptions

    quat, weights = proportional_orientations(90)
    rotations = quat_to_rotmat(quat)
    disvis = DisVis(receptor, ligand, restraints, options)
    for rotmat in rotations:
        disvis(rotmat)

    print disvis.consistent_complexes
    print disvis.violation_matrix


