import os
import errno
try:
    import pyopencl as cl
except ImportError:
    pass
    

def get_queue():
    try:
        platform = cl.get_platforms()[0]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        context = cl.Context(devices=devices)
        queue = cl.CommandQueue(context, device=devices[0])
    except:
        queue = None

    return queue


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def parse_interactions(f):
    data = {}
    with open(f) as f:
        words = f.readline().split()
        data['residues'] = [int(x) for x in words[2:]]

        for line in f:
            words = line.split()

            consistent_restraints = int(words[0])
            data[consistent_restraints] = {}

            data[consistent_restraints]['total'] = int(words[1])
            data[consistent_restraints]['interactions'] = [int(x) for x in words[2:]]
    return data


def parse_restraints(fid, pdb1, pdb2):
    """Parse the restraints file."""

    if isinstance(fid, str):
        fid = open(fid)

    dist_restraints = []

    for line in fid:
        # ignore comments and empty lines
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        chain1, resi1, name1, chain2, resi2, name2, mindis, maxdis = line.split()
        pdb1_sel = pdb1.select('chain', chain1).select('resi',
                int(resi1)).select('name', name1).duplicate()
        pdb2_sel = pdb2.select('chain', chain2).select('resi',
                int(resi2)).select('name', name2).duplicate()

        if pdb1_sel.natoms == 0 or pdb2_sel.natoms == 0:
            raise ValueError("A restraint selection was not found in line:\n{:s}".format(str(line)))

        dist_restraints.append([pdb1_sel, pdb2_sel, float(mindis), float(maxdis)])

    fid.close()
    return dist_restraints


def parse_interaction_selection(fid, pdb1, pdb2):
    """Parse the interaction selection file, i.e. all residues for which an
    interaction analysis is performed."""
    if isinstance(fid, str):
        fid = open(fid)

    resi1 = [int(x) for x in fid.readline().split()]
    resi2 = [int(x) for x in fid.readline().split()]
    fid.close()

    pdb1_sel = pdb1.select('name', 'CA').select('resi', resi1)
    pdb2_sel = pdb2.select('name', 'CA').select('resi', resi2)


    return pdb1_sel, pdb2_sel
