"""Build PDB models of the complexes written by disvis -wc.

complexes.out gives the rotation and translation of the scanning chain of
each complex. disvis-complexes writes the complexes, fixed and scanning
chain, as PDB files, optionally only the center of each cluster of similar
complexes (on the RMSD of the scanning chain).
"""

import os
import warnings
from argparse import ArgumentParser
from os.path import join

import numpy as np

from disvis.helpers import positive_float
from disvis.pdb import PDB


def parse_args():
    """Parse the command-line arguments."""
    p = ArgumentParser(prog='disvis-complexes', description=__doc__)
    p.add_argument('complexes', help='complexes.out file of disvis -wc.')
    p.add_argument('receptor', help='PDB-file of the fixed chain.')
    p.add_argument('ligand', help='PDB-file of the scanning chain.')
    p.add_argument(
        '-d',
        '--directory',
        default='complexes',
        metavar='<dir>',
        help="Directory of the PDB files (default 'complexes').",
    )
    p.add_argument(
        '-n',
        '--number',
        type=int,
        default=None,
        metavar='<int>',
        help='Number of complexes (or clusters) to write. Default is all.',
    )
    p.add_argument(
        '-c',
        '--cluster',
        action='store_true',
        help=(
            'Cluster the complexes on the RMSD of the scanning chain, and '
            'write the center of each cluster, largest cluster first.'
        ),
    )
    p.add_argument(
        '-mp',
        '--min-population',
        type=int,
        default=2,
        metavar='<int>',
        help='Minimum number of complexes in a cluster (default 2).',
    )
    p.add_argument(
        '-cr',
        '--cluster-rmsd',
        type=positive_float,
        default=None,
        metavar='<float>',
        help=(
            'Largest RMSD (A) of the scanning chain between the center of a '
            'cluster and its members. Default is the cutoff for which the '
            'largest RMSD between the complexes of a cluster is about '
            '--cluster-target.'
        ),
    )
    p.add_argument(
        '-cg',
        '--cluster-target',
        type=positive_float,
        default=6.0,
        metavar='<float>',
        help=(
            'Largest RMSD (A) between the complexes of a cluster, for which '
            'the RMSD cutoff is chosen on a random subset of the complexes '
            '(default 6).'
        ),
    )
    return p.parse_args()


def read_complexes(fname):
    """Read a complexes.out file.

    Returns the header, as a dictionary of its "# key: value" lines, the
    model numbers, the (n, 3) translations and the (n, 3, 3) rotations.
    """
    header = {}
    with open(fname, encoding='utf-8') as f:
        for line in f:
            if not line.startswith('#'):
                break
            key, _, value = line[1:].partition(':')
            header[key.strip()] = value.strip()
    with warnings.catch_warnings():
        # A file without complexes is valid
        warnings.filterwarnings('ignore', 'loadtxt: input contained no data')
        data = np.loadtxt(fname, comments='#', ndmin=2)
    if data.size == 0:
        data = np.zeros((0, 15))
    return (
        header,
        data[:, 0].astype(int),
        data[:, 3:6],
        data[:, 6:15].reshape(-1, 3, 3),
    )


class Poses:
    """The poses of the scanning chain of complexes, to compare them.

    The RMSD between two poses is computed from their rotations R and
    translations t: with the RMSD atoms at R x + t, their centers c = R m + t
    and G the covariance of x - m,
    RMSD^2 = |c1 - c2|^2 + 2 tr(G) - 2 tr(R1 G R2^T).
    """

    def __init__(self, translations, rotations, coor):
        self._inputs = translations, rotations, coor
        mean = coor.mean(axis=0)
        x = coor - mean
        gyration = x.T @ x / len(x)
        self.centers = rotations @ mean + translations
        self.square = (self.centers**2).sum(axis=1)
        self.rotations = rotations.reshape(-1, 9)
        self.rotated = (rotations @ gyration).reshape(-1, 9)
        self.trace2 = 2 * np.trace(gyration)

    def __len__(self):
        """Return the number of poses."""
        return len(self.centers)

    def rmsd2(self, index, other=slice(None)):
        """Return the squared RMSD between the poses index and other."""
        d2 = (
            self.square[index, None]
            + self.square[other]
            - 2 * self.centers[index] @ self.centers[other].T
            + self.trace2
            - 2 * self.rotated[index] @ self.rotations[other].T
        )
        return np.maximum(d2, 0)

    def subset(self, index):
        """Return the poses index, as Poses."""
        translations, rotations, coor = self._inputs
        return Poses(translations[index], rotations[index], coor)


def rmsd_atoms(pdb):
    """Return the coordinates of the CA and P atoms, else of heavy atoms."""
    atoms = pdb.select('name', ('CA', 'P'))
    if not atoms.natoms:
        atoms = pdb.select('element', 'H', '!=')
    return atoms.coor


def cluster(poses, cutoff, min_population=1):
    """Cluster poses on their RMSD.

    Two poses are neighbors when their RMSD is at most cutoff (A). As in the
    clustering of HADDOCK (Daura et al.), the pose with the most neighbors
    not yet in a cluster is the center of the next cluster, with these
    neighbors, until fewer than min_population poses would form one; the
    remaining poses are not clustered. Returns the clusters, arrays of pose
    indices with the center first, largest first.
    """
    # Allow for roundoff at the cutoff
    cutoff2 = cutoff**2 + 1e-6

    def neighbors(index):
        return poses.rmsd2(index) <= cutoff2

    # Count the neighbors in blocks of rows, so that no n x n matrix is
    # needed; each pose is its own neighbor.
    n = len(poses)
    block = max(1, 2**22 // max(n, 1))
    count = np.zeros(n, dtype=np.int64)
    for i in range(0, n, block):
        index = np.arange(i, min(i + block, n))
        count[index] = neighbors(index).sum(axis=1)

    free = np.ones(n, dtype=bool)
    clusters = []
    while free.any():
        center = np.argmax(np.where(free, count, -1))
        if count[center] < min_population:
            break
        members = np.flatnonzero(neighbors([center])[0] & free)
        members = np.concatenate(([center], members[members != center]))
        free[members] = False
        clusters.append(members)
        # Remove the new cluster members from the neighbor counts
        for i in range(0, len(members), block):
            count -= neighbors(members[i : i + block]).sum(axis=0)
    return clusters


def cluster_rmsds(poses, clusters):
    """Return the RMSDs between the poses of the same cluster.

    Clusters of more than 500 poses count with their first 500.
    """
    rmsds = [np.zeros(0)]
    for members in clusters:
        first = members[:500]
        if len(first) > 1:
            rmsd = np.sqrt(poses.rmsd2(first, first))
            rmsds.append(rmsd[np.triu_indices(len(first), 1)])
    return np.concatenate(rmsds)


def tune_cutoff(poses, target, min_population=1, nsubset=5000):
    """Return the RMSD cutoff for which clusters are target (A) wide.

    The cutoff is the largest (to within 0.05 A) for which the largest RMSD
    between the poses of the same cluster, on a random subset of nsubset
    poses (the same for every run), is at most target. On all poses,
    denser than the subset, clusters can be somewhat wider.
    """
    rng = np.random.default_rng(0)
    index = np.sort(rng.permutation(len(poses))[:nsubset])
    subset = poses.subset(index)
    low, high = 0.0, 4 * target
    while high - low > 0.05:
        cutoff = (low + high) / 2
        rmsds = cluster_rmsds(subset, cluster(subset, cutoff, min_population))
        # Allow for roundoff at the target
        if rmsds.size == 0 or rmsds.max() <= target + 1e-6:
            low = cutoff
        else:
            high = cutoff
    return max(low, 0.05)


def main():
    """Write the complexes of a complexes.out file as PDB files."""
    args = parse_args()
    header, models, translations, rotations = read_complexes(args.complexes)
    receptor = PDB.fromfile(args.receptor)
    ligand = PDB.fromfile(args.ligand)
    center = ligand.center
    if not np.allclose(
        center,
        [float(x) for x in header['scanning chain center'].split()],
        atol=1e-3,
    ):
        raise ValueError(
            "The center of {:s} differs from that of the scanning chain of "
            "{:s}.".format(args.ligand, args.complexes)
        )
    print(
        'Read {:d} complexes consistent with {:s} restraints from {:s}'.format(
            len(models),
            header.get('consistent restraints', '?'),
            args.complexes,
        )
    )

    if args.cluster:
        poses = Poses(translations, rotations, rmsd_atoms(ligand) - center)
        cutoff = args.cluster_rmsd
        if cutoff is None:
            cutoff = tune_cutoff(
                poses, args.cluster_target, args.min_population
            )
            print(
                'RMSD cutoff for clusters of at most {:g} A: {:.2f} A'.format(
                    args.cluster_target, cutoff
                )
            )
        clusters = cluster(poses, cutoff, args.min_population)
        if not clusters:
            print(
                'No cluster of at least {:d} complexes within {:.2f} A: '
                'complexes are at least about the voxel spacing apart, '
                'increase --cluster-target or --cluster-rmsd, or decrease '
                '--min-population.'.format(args.min_population, cutoff)
            )
        rmsds = cluster_rmsds(poses, clusters)
        print(
            '{:d} clusters of at least {:d} complexes within {:.2f} A RMSD '
            '(RMSD within clusters: mean {:.2f} A, largest {:.2f} A); {:d} '
            'complexes not clustered'.format(
                len(clusters),
                args.min_population,
                cutoff,
                rmsds.mean() if rmsds.size else 0.0,
                rmsds.max() if rmsds.size else 0.0,
                len(models) - sum(len(c) for c in clusters),
            )
        )
        os.makedirs(args.directory, exist_ok=True)
        fname = join(args.directory, 'clusters.out')
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(
                '# cluster size, then the models of the cluster (the first, '
                'its center, is written)\n'
            )
            for k, members in enumerate(clusters, 1):
                f.write(
                    '{:d} {:d} {:s}\n'.format(
                        k,
                        len(members),
                        ' '.join(str(m) for m in models[members]),
                    )
                )
        print('Writing the clusters to {:s}'.format(fname))
        written = [
            (members[0], 'cluster_{:d}.pdb'.format(k))
            for k, members in enumerate(clusters, 1)
        ]
    else:
        written = [
            (i, 'complex_{:d}.pdb'.format(m)) for i, m in enumerate(models)
        ]

    written = written[: args.number]
    os.makedirs(args.directory, exist_ok=True)
    receptor_lines = receptor.pdb_lines()
    model = ligand.duplicate()
    for i, name in written:
        model.coor = (ligand.coor - center) @ rotations[i].T + translations[i]
        with open(join(args.directory, name), 'w', encoding='utf-8') as f:
            f.writelines(receptor_lines + model.pdb_lines() + ['END\n'])
    print(
        'Writing {:d} complexes to {:s}'.format(len(written), args.directory)
    )


if __name__ == '__main__':
    main()
