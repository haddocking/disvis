"""Calculate the interaction and clash volume of known complexes.

For every pair of `<id>_r_<form>.pdb` (receptor) and `<id>_l_<form>.pdb`
(ligand) files, where form is u (unbound, default) or b (bound),
the receptor core, the receptor interaction shell and the ligand core are
mapped on a grid, in the same way as DisVis does. The interaction volume is the
overlap of the ligand core with the receptor interaction shell, and the clash
volume the overlap of the ligand core with the receptor core. These are the
quantities DisVis compares with its --min-interaction and --max-clash cutoffs.
"""

from argparse import ArgumentParser
from glob import glob
import os.path
import sys

import numpy as np

from disvis import PDB
from disvis.helpers import positive_float
from disvis.libdisvis import dilate_points
from disvis.volume import radix235


def parse_args():
    """Parse the command-line arguments."""
    p = ArgumentParser(description=__doc__)
    p.add_argument(
        'directory',
        nargs='?',
        default='.',
        help="Directory containing the *_r_<form>.pdb and "
        "*_l_<form>.pdb files. Default is the current "
        "directory.",
    )
    p.add_argument(
        '-f',
        '--form',
        choices=('u', 'b'),
        default='u',
        help="Use the unbound (u) or bound (b) structures. Default is u.",
    )
    p.add_argument(
        '-vs',
        '--voxelspacing',
        type=positive_float,
        default=1,
        help="Voxel spacing of the grid in angstrom. Default is 1A.",
    )
    p.add_argument(
        '-ir',
        '--interaction-radius',
        type=float,
        default=3,
        help="Radius of the interaction shell in angstrom. Default is 3A.",
    )
    return p.parse_args()


def find_pairs(directory, form='u'):
    """Return (id, receptor file, ligand file) for every complete pair."""
    receptor_suffix = '_r_{:s}.pdb'.format(form)
    ligand_suffix = '_l_{:s}.pdb'.format(form)
    pairs = []
    for receptor_fname in sorted(
        glob(os.path.join(directory, '*' + receptor_suffix))
    ):
        pdb_id = os.path.basename(receptor_fname)[: -len(receptor_suffix)]
        ligand_fname = receptor_fname[: -len(receptor_suffix)] + ligand_suffix
        if os.path.exists(ligand_fname):
            pairs.append((pdb_id, receptor_fname, ligand_fname))
        else:
            print(
                '{:s}: no ligand file found, skipping'.format(pdb_id),
                file=sys.stderr,
            )
    return pairs


def dilate(coor, radii, origin, voxelspacing, shape):
    """Map spheres around the coordinates onto a new grid."""
    grid = np.zeros(shape, dtype=np.float64)
    dilate_points((coor - origin) / voxelspacing, radii / voxelspacing, grid)
    return grid


def interaction_and_clash_volume(
    receptor, ligand, voxelspacing, interaction_radius
):
    """Return the interaction and clash volume of a complex in A^3."""
    receptor_radii = receptor.vdw_radius
    ligand_radii = ligand.vdw_radius

    # The grid encloses both chains, including their interaction shell, so no
    # atom wraps around the periodic grid boundaries.
    coor = np.vstack((receptor.coor, ligand.coor))
    padding = (
        max(receptor_radii.max(), ligand_radii.max())
        + interaction_radius
        + 2 * voxelspacing
    )
    origin = coor.min(axis=0) - padding
    extent = coor.max(axis=0) + padding - origin
    shape = [radix235(int(np.ceil(x))) for x in extent / voxelspacing][::-1]

    rcore = dilate(receptor.coor, receptor_radii, origin, voxelspacing, shape)
    rsurface = dilate(
        receptor.coor,
        receptor_radii + interaction_radius,
        origin,
        voxelspacing,
        shape,
    )
    lcore = dilate(ligand.coor, ligand_radii, origin, voxelspacing, shape)

    voxel_volume = voxelspacing**3
    interaction_volume = (rsurface * lcore).sum() * voxel_volume
    clash_volume = (rcore * lcore).sum() * voxel_volume
    return interaction_volume, clash_volume


def main():
    """Print the interaction and clash volume of each complex."""
    args = parse_args()

    pairs = find_pairs(args.directory, args.form)
    if not pairs:
        sys.exit(
            'No *_r_{0:s}.pdb / *_l_{0:s}.pdb pairs found in {1:s}'.format(
                args.form, args.directory
            )
        )

    print('{:<8s} {:>12s} {:>12s}'.format('#id', 'interaction', 'clash'))
    interaction_volumes = []
    clash_volumes = []
    for pdb_id, receptor_fname, ligand_fname in pairs:
        try:
            receptor = PDB.fromfile(receptor_fname)
            ligand = PDB.fromfile(ligand_fname)
            interaction_volume, clash_volume = interaction_and_clash_volume(
                receptor, ligand, args.voxelspacing, args.interaction_radius
            )
        except Exception as e:
            print('{:s}: failed: {}'.format(pdb_id, e), file=sys.stderr)
            continue
        interaction_volumes.append(interaction_volume)
        clash_volumes.append(clash_volume)
        print(
            '{:<8s} {:>12.1f} {:>12.1f}'.format(
                pdb_id, interaction_volume, clash_volume
            )
        )

    if interaction_volumes:
        print(
            '# minimum interaction volume: {:.1f} A^3'.format(
                min(interaction_volumes)
            )
        )
        print('# maximum clash volume: {:.1f} A^3'.format(max(clash_volumes)))


if __name__ == '__main__':
    main()
