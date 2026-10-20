"""Input parsers and other helper functions."""
import os
import errno
import json
try:
    import pyopencl as cl
except ImportError:
    pass


def get_queue():
    """Return an OpenCL command queue on the first GPU, or None."""
    try:
        platform = cl.get_platforms()[0]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        context = cl.Context(devices=devices)
        queue = cl.CommandQueue(context, device=devices[0])
    except Exception:
        queue = None

    return queue


def mkdir_p(path):
    """Create a directory and its parents, like mkdir -p."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def parse_interactions(f):
    """Parse a receptor_interactions.txt or ligand_interactions.txt file.

    Returns a dictionary with the residue numbers under 'residues' and, for
    each number of consistent restraints, the average number of interactions
    of each residue.
    """
    with open(f) as fh:
        words = fh.readline().split()
        if not words or words[0] != '#resi':
            raise ValueError(
                "{:s} is not a DisVis interaction file.".format(f))
        consistent_restraints = [int(x) for x in words[1:]]

        residues = []
        interactions = []
        for line in fh:
            words = line.split()
            if not words:
                continue
            residues.append(int(words[0]))
            interactions.append([float(x) for x in words[1:]])

    data = {'residues': residues}
    for n, nconsistent in enumerate(consistent_restraints):
        data[nconsistent] = [x[n] for x in interactions]
    return data


def parse_restraints(fid, pdb1, pdb2):
    """Parse a restraints file.

    Each line gives chain, residue number and atom name in pdb1, the same
    in pdb2, and the minimum and maximum distance. Returns a list of
    [pdb1 selection, pdb2 selection, mindis, maxdis].
    """
    if isinstance(fid, str):
        fid = open(fid)

    dist_restraints = []

    for line in fid:
        # ignore comments and empty lines
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        (chain1, resi1, name1,
         chain2, resi2, name2, mindis, maxdis) = line.split()
        pdb1_sel = pdb1.select('chain', chain1).select(
            'resi', int(resi1)).select('name', name1).duplicate()
        pdb2_sel = pdb2.select('chain', chain2).select(
            'resi', int(resi2)).select('name', name2).duplicate()

        if pdb1_sel.natoms == 0 or pdb2_sel.natoms == 0:
            raise ValueError("A restraint selection was not found in line:"
                             "\n{:s}".format(str(line)))

        dist_restraints.append([pdb1_sel, pdb2_sel,
                                float(mindis), float(maxdis)])

    fid.close()
    return dist_restraints


def parse_interaction_selection(fid, pdb1, pdb2):
    """Parse the interaction selection file.

    Its first and second lines list the residue numbers of pdb1 and pdb2
    for which the interactions are analysed. Returns the CA atoms of these
    residues.
    """
    if isinstance(fid, str):
        fid = open(fid)

    resi1 = [int(x) for x in fid.readline().split()]
    resi2 = [int(x) for x in fid.readline().split()]
    fid.close()

    pdb1_sel = pdb1.select('name', 'CA').select('resi', resi1)
    pdb2_sel = pdb2.select('name', 'CA').select('resi', resi2)

    return pdb1_sel, pdb2_sel


def dump_args(args, fname):
    """Dump the command-line arguments to a `.json` file."""
    with open(fname, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
