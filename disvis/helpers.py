import os
import errno
import re

try:
    import pyopencl as cl
except ImportError:
    pass
try:
    from pyparsing import (Literal, Word, Combine, Optional, Forward,
                           ZeroOrMore, StringEnd, nums, alphas, alphanums, ParseException,
                           )

    PYPARSING = True
except ImportError:
    PYPARSING = False


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
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class DJoiner(object):
    """Join filenames with a set directory."""

    def __init__(self, directory='.'):
        self.directory = directory

    def __call__(self, fname):
        return os.path.abspath(os.path.join(self.directory, fname))

    def __add__(self, path):
        return DJoiner(os.path.join(self.directory, path))

    def __iadd__(self, path):
        self.directory = os.path.join(self.directory, path)
        return self

    def mkdir(self):
        mkdir_p(self.directory)


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
            msg = "A restraint selection was not found in line:\n{:s}".format(str(line))
            raise ValueError(msg)

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


class RestraintParser(object):
    """Parser for restraint files."""

    if PYPARSING:
        # Define grammar
        point = Literal('.')
        plusorminus = Literal('+') | Literal('-')
        lpar = Literal('(')
        rpar = Literal(')')
        l_or = Literal('or')
        start = Literal('restraint')
        at = Literal('@')
        number = Word(nums)
        integer = Combine(Optional(plusorminus) + number)
        floatnumber = Combine(integer + Optional(point + Optional(number)))

        atomname = Combine(at + Word(alphanums))
        chain = Combine(point + Word(alphanums))
        term = Combine(integer + Optional(chain) + Optional(atomname))
        selection = Forward()
        selection << term + ZeroOrMore(l_or + term)

        pattern = (start + lpar + selection + rpar + lpar + selection +
                   rpar + floatnumber + floatnumber)

    def parse_file(self, fid):
        restraints = []
        with open(fid) as f:
            for line in f:
                restraint = self.parse_line(line)
                if restraint is not None:
                    restraints.append(restraint)
        return restraints

    def parse_line(self, line):
        if not line:
            return

        if line.startswith('#'):
            return
        elif line.startswith('restraint'):
            if not PYPARSING:
                msg = "Ambiguous restraints syntax requires the pyparsing package."
                raise ImportError(msg)
            return self._ambiguous_restraint(line)
        # Ignore empty lines
        elif not line.strip():
            return
        else:
            return self._simple_restraint(line)

    def _simple_restraint(self, line):
        # Old-style restraint
        words = line.split()
        words[1] = int(words[1])
        words[4] = int(words[4])
        receptor_selection = [self._simple_selection(words[:3])]
        ligand_selection = [self._simple_selection(words[3:6])]
        min_dis, max_dis = tuple(float(x) for x in words[6:8])
        return receptor_selection, ligand_selection, min_dis, max_dis

    def _simple_selection(self, words):
        return zip(('chain', 'resi', 'name'), words)

    def _ambiguous_restraint(self, line):
        # Parse ambiguous restraint line
        try:
            args = self.pattern.parseString(line).asList()
        except ParseException:
            print line
            raise

        receptor_start = args.index('(') + 1
        receptor_end = args.index(')')
        ligand_start = receptor_end + 2
        ligand_end = len(args) - 3

        receptor_selection = []
        for sel in args[receptor_start: receptor_end]:
            if sel == 'or':
                continue
            receptor_selection.append(self._ambiguous_selection(sel))

        ligand_selection = []
        for sel in args[ligand_start: ligand_end]:
            if sel == 'or':
                continue
            ligand_selection.append(self._ambiguous_selection(sel))

        min_dis, max_dis = [float(args[-2]), float(args[-1])]

        return receptor_selection, ligand_selection, min_dis, max_dis

    def _ambiguous_selection(self, sel):
        words = re.split('[.@]', sel)
        words[0] = int(words[0])
        # In case the user only specified resi and name.
        if len(words) == 2:
            if '@' in sel:
                return zip(('resi', 'name'), words)
        return zip(('resi', 'chain', 'name'), words)
