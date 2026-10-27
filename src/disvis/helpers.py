"""Input parsers and other helper functions."""

from argparse import ArgumentTypeError
import json
from math import isfinite

try:
    import pyopencl as cl
except ImportError:
    pass


def get_queue(device_type='gpu'):
    """Return an OpenCL command queue on the first device, or None.

    device_type is 'gpu' (the default) or 'all', which also accepts other
    OpenCL devices, e.g. pocl on the CPU for testing.
    """
    try:
        types = {'gpu': cl.device_type.GPU, 'all': cl.device_type.ALL}
        platforms = cl.get_platforms()
    except Exception:
        return None
    for platform in platforms:
        try:
            devices = platform.get_devices(device_type=types[device_type])
        except cl.Error:
            # No usable devices of this type on this platform.
            devices = []
        if devices:
            try:
                context = cl.Context(devices=devices[:1])
                return cl.CommandQueue(context, device=devices[0])
            except cl.Error:
                return None
    return None


def positive_float(value):
    """Return value as a float; an argparse type for positive numbers."""
    try:
        number = float(value)
    except ValueError:
        number = None
    if number is None or not isfinite(number) or number <= 0:
        raise ArgumentTypeError("{:s} is not a positive number".format(value))
    return number


def parse_interactions(f):
    """Parse a receptor_interactions.txt or ligand_interactions.txt file.

    Returns a dictionary with the residue numbers under 'residues' and, for
    each number of consistent restraints, the average number of interactions
    of each residue.
    """
    with open(f, encoding='utf-8') as fh:
        words = fh.readline().split()
        if not words or words[0] != '#resi':
            raise ValueError(
                "{:s} is not a DisVis interaction file.".format(f)
            )
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


def parse_interaction_selection(fname, pdb1, pdb2):
    """Parse the interaction selection file.

    Its first and second lines list the residue numbers of pdb1 and pdb2
    for which the interactions are analysed. Returns the CA (protein) or
    O3' (nucleic acid) atoms of these residues.
    """
    with open(fname, encoding='utf-8') as fid:
        resi1 = [int(x) for x in fid.readline().split()]
        resi2 = [int(x) for x in fid.readline().split()]

    pdb1_sel = pdb1.select('name', ('CA', "O3'")).select('resi', resi1)
    pdb2_sel = pdb2.select('name', ('CA', "O3'")).select('resi', resi2)

    if (len(resi1) != pdb1_sel.natoms) or (len(resi2) != pdb2_sel.natoms):
        msg = (
            "Some selected interaction residues were either missing in "
            "the PDB file "
            "or had alternate conformers. Please check your input "
            "residues and remove alternate conformers."
        )
        raise ValueError(msg)

    return pdb1_sel, pdb2_sel


def parse_restraints(fname, pdb1, pdb2):
    """Parse a restraints file.

    Each line gives chain, residue number and atom name in pdb1, the same
    in pdb2, and the minimum and maximum distance. Returns a list of
    [pdb1 selection, pdb2 selection, mindis, maxdis].
    """
    with open(fname, encoding='utf-8') as fid:
        lines = fid.readlines()

    dist_restraints = []

    for raw_line in lines:
        # ignore comments and empty lines
        line = raw_line.strip()
        if line.startswith('#') or not line:
            continue

        (chain1, resi1, name1, chain2, resi2, name2, mindis, maxdis) = (
            line.split()
        )
        pdb1_sel = (
            pdb1.select('chain', chain1)
            .select('resi', int(resi1))
            .select('name', name1)
            .duplicate()
        )
        pdb2_sel = (
            pdb2.select('chain', chain2)
            .select('resi', int(resi2))
            .select('name', name2)
            .duplicate()
        )

        if pdb1_sel.natoms == 0 or pdb2_sel.natoms == 0:
            raise ValueError(
                "A restraint selection was not found in line:\n{:s}".format(
                    str(line)
                )
            )

        dist_restraints.append(
            [pdb1_sel, pdb2_sel, float(mindis), float(maxdis)]
        )

    return dist_restraints


def dump_args(args, fname):
    """Dump the command-line arguments to a `.json` file."""
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
