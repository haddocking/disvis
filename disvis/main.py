"""The disvis command-line program."""

from os import makedirs, remove
from queue import Empty
from os.path import join, abspath
from sys import stdout, exit
from time import time
import multiprocessing as mp
from argparse import ArgumentParser
import logging

import numpy as np

from disvis import DisVis, PDB, Volume, __version__
from disvis.rotations import proportional_orientations, quat_to_rotmat
from disvis.helpers import (
    dump_args,
    parse_interaction_selection,
    parse_restraints,
    positive_float,
)


def parse_args():
    """Parse the command-line arguments."""
    p = ArgumentParser()

    p.add_argument(
        '--version',
        action='version',
        version='%(prog)s {:s}'.format(__version__),
    )

    p.add_argument('receptor', help='PDB-file containing fixed chain.')

    p.add_argument('ligand', help='PDB-file containing scanning chain.')

    p.add_argument(
        'restraints', help='File containing the distance restraints'
    )

    p.add_argument(
        '-a',
        '--angle',
        dest='angle',
        type=float,
        default=15,
        metavar='<float>',
        help='Rotational sampling density in degrees. Default is 15 degrees.',
    )

    p.add_argument(
        '-vs',
        '--voxelspacing',
        dest='voxelspacing',
        metavar='<float>',
        type=positive_float,
        default=1,
        help='Voxel spacing of search grid in angstrom. Default is 1A.',
    )

    p.add_argument(
        '-ir',
        '--interaction-radius',
        dest='interaction_radius',
        type=float,
        default=3.0,
        metavar='<float>',
        help='Radius of the interaction space for each atom in '
        'angstrom. '
        'Atoms are thus considered interacting if the distance is '
        'larger than the vdW radius and shorther than or equal to '
        'vdW + interaction_radius. Default is 3A.',
    )

    p.add_argument(
        '-cv',
        '--max-clash',
        dest='max_clash',
        type=float,
        default=200,
        metavar='<float>',
        help='Maximum allowed volume of clashes. Increasing this '
        'number results in more allowed complexes. '
        'Default is 200 A^3.',
    )

    p.add_argument(
        '-iv',
        '--min-interaction',
        dest='min_interaction',
        type=float,
        default=300,
        metavar='<float>',
        help='Minimal required interaction volume for a '
        'conformation to be considered a '
        'complex. Increasing this number results in a '
        'stricter counting of complexes. '
        'Default is 300 A^3.',
    )

    p.add_argument(
        '-d',
        '--directory',
        dest='directory',
        metavar='<dir>',
        type=abspath,
        default='.',
        help='Directory where results are written to. '
        'Default is current directory.',
    )

    p.add_argument(
        '-p',
        '--nproc',
        dest='nproc',
        type=int,
        default=1,
        metavar='<int>',
        help='Number of processors used during search.',
    )

    p.add_argument(
        '-g',
        '--gpu',
        dest='gpu',
        action='store_true',
        help='Use GPU-acceleration for search. If not available '
        'the CPU-version will be used with the given number '
        'of processors.',
    )

    help_msg = (
        "File containing residue number for which interactions "
        "will be counted. "
        "The first line holds the receptor residue, "
        "and the second line the ligand residue numbers."
    )
    p.add_argument(
        '-is',
        '--interaction-selection',
        metavar='<file>',
        dest='interaction_selection',
        default=None,
        help=help_msg,
    )

    help_msg = (
        "Number of minimal consistent restraints for which an "
        "interaction "
        "or occupancy analysis will be performed. "
        "Default is number of restraints minus 1."
    )
    p.add_argument(
        '-ic',
        '--interaction-restraints-cutoff',
        metavar='<int>',
        dest='interaction_restraints_cutoff',
        type=int,
        default=None,
        help=help_msg,
    )

    p.add_argument(
        '-oa',
        '--occupancy-analysis',
        dest='occupancy_analysis',
        action='store_true',
        help=(
            "Perform an occupancy analysis, ultimately providing "
            "a volume where each grid point represents the "
            "normalized probability of that spot being "
            "occupied by the ligand."
        ),
    )

    return p.parse_args()


class Joiner:
    """Join file names with an output directory."""

    def __init__(self, directory):
        self.directory = directory

    def __call__(self, fname):
        """Join fname with set directory."""
        return join(self.directory, fname)


class Results:
    """Container for the combined results, like those of DisVis."""

    pass


def run_disvis_instance(
    queue,
    receptor,
    ligand,
    distance_restraints,
    rotmat,
    weights,
    n,
    pdb1_sel,
    pdb2_sel,
    args,
    counter,
):
    """Run a single DisVis instance on a subset of the rotations.

    The results are written to temporary files in args.directory, numbered
    n, and the accessible complexes are put on queue. The shared counter is
    incremented after every rotation, so that the parent process can print
    the combined progress.
    """

    def count_rotation(_n, _total, _time0):
        with counter.get_lock():
            counter.value += 1

    dv = DisVis(print_callback=count_rotation)

    dv.receptor = receptor
    dv.ligand = ligand
    dv.distance_restraints = distance_restraints
    dv.rotations = rotmat
    dv.weights = weights

    dv.voxelspacing = args.voxelspacing
    dv.interaction_radius = args.interaction_radius
    dv.max_clash = args.max_clash
    dv.min_interaction = args.min_interaction
    dv.interaction_restraints_cutoff = args.interaction_restraints_cutoff

    if args.interaction_selection is not None:
        dv.receptor_interaction_selection = pdb1_sel
        dv.ligand_interaction_selection = pdb2_sel
    dv.occupancy_analysis = args.occupancy_analysis

    dv.search()

    # Save results to file, to be combined later
    joiner = Joiner(args.directory)
    fname = joiner('accessible_interaction_space_{:d}.mrc').format(n)
    dv.accessible_interaction_space.tofile(fname)

    fname = joiner('violations_{:d}.npy').format(n)
    np.save(fname, dv.violations)

    if dv.interaction_matrix is not None:
        fname = joiner('interaction_matrix_{:d}.npy'.format(n))
        np.save(fname, dv.interaction_matrix)

    if dv.occupancy_analysis:
        for key, value in dv.occupancy_grids.items():
            fname = joiner('occupancy_{:d}_{:d}.mrc'.format(key, n))
            value.tofile(fname)

    queue.put(dv.accessible_complexes)


def mp_cpu_disvis(
    receptor,
    ligand,
    rotmat,
    weights,
    distance_restraints,
    pdb1_sel,
    pdb2_sel,
    args,
):
    """Run several DisVis instances, each with a subset of all rotations."""
    # multi-threaded CPU version
    try:
        max_cpu = mp.cpu_count()
        jobs = min(max_cpu, args.nproc)
    except NotImplementedError:
        jobs = args.nproc
    # in case more processes are requested than the number
    # of rotations sampled
    nrot = rotmat.shape[0]
    jobs = min(jobs, nrot)
    nrot_per_job = nrot // jobs
    write('Number of processors used: {:d}'.format(jobs))
    write('Number of rotations per job: {:d}'.format(nrot_per_job))

    write('Creating jobs')

    queue = mp.Queue()
    # Number of rotations done by all jobs, to print a combined progress.
    counter = mp.Value('l', 0)
    processes = []
    for n in range(jobs):
        # Determine the rotations that each job needs to sample
        init_rot = n * nrot_per_job
        end_rot = (n + 1) * nrot_per_job
        if n == (jobs - 1):
            end_rot = None

        sub_rotmat = rotmat[init_rot:end_rot]
        sub_weights = weights[init_rot:end_rot]

        disvis_args = (
            queue,
            receptor,
            ligand,
            distance_restraints,
            sub_rotmat,
            sub_weights,
            n,
            pdb1_sel,
            pdb2_sel,
            args,
            counter,
        )
        process = mp.Process(target=run_disvis_instance, args=disvis_args)
        processes.append(process)

    write('Starting jobs')
    time0 = time()
    for p in processes:
        p.start()
    write('Waiting for jobs to finish')

    # Print the combined progress of all jobs every second until they finish.
    progress = stdout.isatty()
    while True:
        alive = [p for p in processes if p.is_alive()]
        done = counter.value
        if progress and done > 0:
            DisVis.print_progress(done - 1, nrot, time0)
        if not alive:
            break
        alive[0].join(timeout=1)
    if progress and 0 < done < nrot:
        # A job failed: end the progress line.
        stdout.write('\n')

    # Check whether the queue is empty, this indicates failure to run on
    # multi-processor runs.
    if queue.empty():
        raise Empty

    write('Searching done. Combining results')

    # Create dummy class with similar results attributes as DisVis class
    results = Results()
    joiner = Joiner(args.directory)

    fname_interspace = joiner('accessible_interaction_space_{:d}.mrc')
    fname_violations = joiner('violations_{:d}.npy')
    fname_intermat = joiner('interaction_matrix_{:d}.npy')

    accessible_complexes = np.asarray(queue.get(), dtype=np.float64)
    accessible_interaction_space = Volume.fromfile(fname_interspace.format(0))
    violations = np.load(fname_violations.format(0))
    for n in range(1, jobs):
        accessible_complexes += np.asarray(queue.get(), dtype=np.float64)
        np.maximum(
            accessible_interaction_space.array,
            Volume.fromfile(fname_interspace.format(n)).array,
            out=accessible_interaction_space.array,
        )
        violations += np.load(fname_violations.format(n))

    # Combine the occupancy grids
    occupancy = None
    if args.occupancy_analysis:
        fname_occupancy = joiner('occupancy_{:d}_{:d}.mrc')
        occupancy = {}
        for consistent_restraints in range(
            args.interaction_restraints_cutoff, len(distance_restraints) + 1
        ):
            occupancy[consistent_restraints] = Volume.fromfile(
                fname_occupancy.format(consistent_restraints, 0)
            )
            total = occupancy[consistent_restraints].array
            for n in range(1, jobs):
                total += Volume.fromfile(
                    fname_occupancy.format(consistent_restraints, n)
                ).array

    # Combine the interaction analysis
    results.interaction_matrix = None
    if args.interaction_selection is not None:
        interaction_matrix = np.load(fname_intermat.format(0))
        for n in range(1, jobs):
            interaction_matrix += np.load(fname_intermat.format(n))
        results.interaction_matrix = interaction_matrix

    # Remove the intermediate files
    write('Cleaning')
    for n in range(jobs):
        remove(fname_interspace.format(n))
        remove(fname_violations.format(n))
        if args.interaction_selection is not None:
            remove(fname_intermat.format(n))
        if args.occupancy_analysis:
            for consistent_restraints in range(
                args.interaction_restraints_cutoff,
                len(distance_restraints) + 1,
            ):
                remove(fname_occupancy.format(consistent_restraints, n))

    results.accessible_interaction_space = accessible_interaction_space
    results.accessible_complexes = accessible_complexes
    results.violations = violations
    results.occupancy_grids = occupancy

    return results


def run_disvis(
    queue,
    receptor,
    ligand,
    rotmat,
    weights,
    distance_restraints,
    pdb1_sel,
    pdb2_sel,
    args,
):
    """Run DisVis in this process, on the GPU if queue is not None."""
    dv = DisVis()

    dv.receptor = receptor
    dv.ligand = ligand
    dv.distance_restraints = distance_restraints
    dv.rotations = rotmat
    dv.weights = weights

    dv.voxelspacing = args.voxelspacing
    dv.interaction_radius = args.interaction_radius
    dv.max_clash = args.max_clash
    dv.min_interaction = args.min_interaction
    dv.queue = queue
    dv.occupancy_analysis = args.occupancy_analysis
    dv.interaction_restraints_cutoff = args.interaction_restraints_cutoff

    if not any([x is None for x in (pdb1_sel, pdb2_sel)]):
        dv.receptor_interaction_selection = pdb1_sel
        dv.ligand_interaction_selection = pdb2_sel
    dv.search()

    return dv


def write(line):
    """Print line and write it to the log."""
    print(line)
    logging.info(line)


def main():
    """Run disvis from the command line and write the results."""
    args = parse_args()

    makedirs(args.directory, exist_ok=True)
    joiner = Joiner(args.directory)
    dump_args(args, joiner('args.json'))

    # force: gpyfft configures the root logger when it is imported.
    logging.basicConfig(
        filename=joiner('disvis.log'),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        force=True,
    )

    time0 = time()

    queue = None
    if args.gpu:
        from disvis.disvis import PYOPENCL

        if not PYOPENCL:
            exit(
                'ERROR: GPU acceleration requires pyopencl, clFFT and '
                'gpyfft. See INSTALLATION.md.'
            )
        from disvis.helpers import get_queue

        queue = get_queue()
        if queue is None:
            raise ValueError("No GPU queue was found.")

    write('Reading fixed model from: {:s}'.format(args.receptor))
    receptor = PDB.fromfile(args.receptor)
    write('Reading scanning model from: {:s}'.format(args.ligand))
    ligand = PDB.fromfile(args.ligand)

    write('Reading in rotations.')
    q, weights, a = proportional_orientations(args.angle)
    rotmat = quat_to_rotmat(q)
    write('Requested rotational sampling density: {:.2f}'.format(args.angle))
    write('Real rotational sampling density: {:.2f}'.format(a))
    write('Number of rotations: {:d}'.format(rotmat.shape[0]))

    write('Reading in restraints from file: {:s}'.format(args.restraints))
    distance_restraints = parse_restraints(args.restraints, receptor, ligand)
    write(
        'Number of distance restraints: {:d}'.format(len(distance_restraints))
    )

    # If the interaction restraints cutoff is not specified, only calculate
    # the interactions and occupancy grids for complexes consistent with at
    # least 1 restraints or more, with a limit of three.
    if args.interaction_restraints_cutoff is None:
        args.interaction_restraints_cutoff = max(
            len(distance_restraints) - 3, 1
        )

    pdb1_sel = pdb2_sel = None
    if args.interaction_selection is not None:
        write(
            'Reading in interaction selection from file: {:s}'.format(
                args.interaction_selection
            )
        )
        pdb1_sel, pdb2_sel = parse_interaction_selection(
            args.interaction_selection, receptor, ligand
        )

        write('Number of receptor residues: {:d}'.format(pdb1_sel.natoms))
        write('Number of ligand residues: {:d}'.format(pdb2_sel.natoms))

    write('Voxel spacing set to: {:.2f}'.format(args.voxelspacing))
    write('Interaction radius set to: {:.2f}'.format(args.interaction_radius))
    write(
        'Minimum required interaction volume: {:.2f}'.format(
            args.min_interaction
        )
    )
    write('Maximum allowed volume of clashes: {:.2f}'.format(args.max_clash))
    if args.occupancy_analysis:
        write('Performing occupancy analysis')

    if queue is None:
        # CPU-version
        if args.nproc > 1:
            try:
                dv = mp_cpu_disvis(
                    receptor,
                    ligand,
                    rotmat,
                    weights,
                    distance_restraints,
                    pdb1_sel,
                    pdb2_sel,
                    args,
                )
            except Empty:
                msg = (
                    'ERROR: Queue.Empty exception raised while processing '
                    'job, stopping execution ...'
                )
                write(msg)
                exit(-1)
        else:
            dv = run_disvis(
                queue,
                receptor,
                ligand,
                rotmat,
                weights,
                distance_restraints,
                pdb1_sel,
                pdb2_sel,
                args,
            )
    else:
        # GPU-version
        write('Using GPU accelerated search.')
        dv = run_disvis(
            queue,
            receptor,
            ligand,
            rotmat,
            weights,
            distance_restraints,
            pdb1_sel,
            pdb2_sel,
            args,
        )

    # write out accessible interaction space
    fname = joiner('accessible_interaction_space.mrc')
    write('Writing accessible interaction space to: {:s}'.format(fname))
    dv.accessible_interaction_space.tofile(fname)

    # write out accessible complexes
    accessible_complexes = dv.accessible_complexes
    norm = sum(accessible_complexes)
    digits = len(str(int(norm))) + 1
    cum_complexes = np.cumsum(np.asarray(accessible_complexes)[::-1])[::-1]
    with open(
        joiner('accessible_complexes.out'), 'w', encoding='utf-8'
    ) as f_accessible_complexes:
        write(
            'Writing number of accessible complexes to: {:s}'.format(
                f_accessible_complexes.name
            )
        )
        header = (
            '# consistent restraints | accessible complexes |'
            + 'relative | cumulative accessible complexes | relative\n'
        )
        f_accessible_complexes.write(header)
        for n, acc in enumerate(accessible_complexes):
            f_accessible_complexes.write(
                '{0:3d} {2:{1}d} {3:8.6f} {4:{1}d} {5:8.6f}\n'.format(
                    n,
                    digits,
                    int(acc),
                    acc / norm,
                    int(cum_complexes[n]),
                    cum_complexes[n] / norm,
                )
            )

    # writing out violations
    violations = dv.violations
    cum_violations = violations[::-1].cumsum(axis=0)[::-1]
    with open(joiner('violations.out'), 'w', encoding='utf-8') as f_viol:
        write('Writing violations to file: {:s}'.format(f_viol.name))
        num_violations = violations.sum(axis=1)
        nrestraints = num_violations.shape[0]
        header = (
            '# row represents the number of consistent restraints\n'
            '# column represents how often that restraint is '
            'violated\n'
        )
        f_viol.write(header)
        header = ('   ' + '{:8d}' * nrestraints + '\n').format(
            *range(1, nrestraints + 1)
        )
        f_viol.write(header)
        for n, line in enumerate(cum_violations):
            f_viol.write('{:<2d} '.format(n + 1))
            for word in line:
                if num_violations[n] > 0:
                    percentage_violated = word / cum_complexes[n + 1]
                else:
                    percentage_violated = 0
                f_viol.write('{:8.4f}'.format(percentage_violated))
            f_viol.write('\n')

    # Give user indication for false positives.
    # Determine minimum number of false positives.
    nrestraints = len(distance_restraints)
    n = 1
    while accessible_complexes[-n] == 0:
        n += 1
    if n > 1:
        msg = (
            'Not all restraints are consistent. '
            'Number of false-positive restraints present '
            'is at least: {:d}'.format(n - 1)
        )
        write(msg)

    # next give possible false-positives based on the percentage of violations
    # and their associated Z-score
    if n == 1:
        n = None
    else:
        n = -n + 1
    percentage_violated = cum_violations[:n] / np.asarray(
        cum_complexes[1:n]
    ).reshape(-1, 1)
    average_restraint_violation = percentage_violated.mean(axis=0)
    std_restraint_violation = percentage_violated.std(axis=0)
    zscore_violations = (
        average_restraint_violation - average_restraint_violation.mean()
    ) / average_restraint_violation.std()
    ind_false_positives = np.flatnonzero(zscore_violations >= 1.0)
    nfalse_positives = ind_false_positives.size
    if nfalse_positives > 0:
        ind_false_positives += 1
        write(
            (
                'Possible false-positive restraints (z-score > 1.0):'
                + ' {:d}' * nfalse_positives
            ).format(*tuple(ind_false_positives))
        )

    with open(joiner('z-score.out'), 'w', encoding='utf-8') as f:
        write('Writing z-score of each restraint to {:s}'.format(f.name))
        for n in range(zscore_violations.shape[0]):
            f.write(
                '{:2d} {:6.3f} {:6.3f} {:6.3f}\n'.format(
                    n + 1,
                    average_restraint_violation[n],
                    std_restraint_violation[n],
                    zscore_violations[n],
                )
            )

    # Write all occupancy grids to MRC-files if requested
    if args.occupancy_analysis:
        for n, vol in dv.occupancy_grids.items():
            # Normalize the occupancy grid
            if cum_complexes[n] > 0:
                vol.array[...] /= cum_complexes[n]
            vol.tofile(joiner('occupancy_{:d}.mrc'.format(n)))

    # Write out interaction analysis
    # the interaction_matrix gives the number of interactions between each
    # residue of the receptor and ligand for complexes consistent with exactly
    # N restraints.
    interaction_matrix = dv.interaction_matrix
    if interaction_matrix is not None:
        # Save interaction matrix
        # f = joiner('interaction_matrix.npy')
        # write('Writing interaction-matrix to: {:s}'.format(f))
        # np.save(f, interaction_matrix)

        # Save contacted receptor and ligand residue interaction for each
        # analyzed number of consistent restraints
        write('Writing contacted receptor residue interactions to files.')
        # Take the cumsum in order to give the number of interactions for
        # complexes with at least N restraints.
        receptor_cum_interactions = interaction_matrix.sum(axis=1)[
            ::-1
        ].cumsum(axis=0)[::-1]
        ligand_cum_interactions = interaction_matrix.sum(axis=2)[::-1].cumsum(
            axis=0
        )[::-1]
        fname = joiner('receptor_interactions.txt')
        with open(fname, 'w', encoding='utf-8') as f:
            # Write header
            f.write('#resi')
            for consistent_restraints in range(
                args.interaction_restraints_cutoff, nrestraints + 1
            ):
                f.write(' {:>6d}'.format(consistent_restraints))
            f.write('\n')

            for n, resi in enumerate(pdb1_sel.data['resi']):
                f.write('{:<5d}'.format(resi))
                for consistent_restraints in range(
                    args.interaction_restraints_cutoff, nrestraints + 1
                ):
                    index = (
                        consistent_restraints
                        - args.interaction_restraints_cutoff
                    )
                    interactions = receptor_cum_interactions[index, n]
                    cum_complex = cum_complexes[consistent_restraints]
                    if cum_complex > 0:
                        relative_interactions = interactions / cum_complex
                    else:
                        relative_interactions = 0
                    f.write(' {:6.3f}'.format(relative_interactions))
                f.write('\n')

        fname = joiner('ligand_interactions.txt')
        with open(fname, 'w', encoding='utf-8') as f:
            # Write header
            f.write('#resi')
            for consistent_restraints in range(
                args.interaction_restraints_cutoff, nrestraints + 1
            ):
                f.write(' {:>6d}'.format(consistent_restraints))
            f.write('\n')

            for n, resi in enumerate(pdb2_sel.data['resi']):
                f.write('{:<5d}'.format(resi))
                for consistent_restraints in range(
                    args.interaction_restraints_cutoff, nrestraints + 1
                ):
                    index = (
                        consistent_restraints
                        - args.interaction_restraints_cutoff
                    )
                    interactions = ligand_cum_interactions[index, n]
                    cum_complex = cum_complexes[consistent_restraints]
                    if cum_complex > 0:
                        relative_interactions = interactions / cum_complex
                    else:
                        relative_interactions = 0
                    f.write(' {:6.3f}'.format(relative_interactions))
                f.write('\n')

    # time indication
    seconds = int(round(time() - time0))
    m, s = divmod(seconds, 60)
    write('Total time passed: {:d}m {:2d}s'.format(m, s))


if __name__ == '__main__':
    main()
