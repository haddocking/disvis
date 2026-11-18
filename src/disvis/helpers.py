"""Input parsers and other helper functions."""

from argparse import ArgumentTypeError
from fnmatch import fnmatchcase
import json
from math import isfinite
import re

import numpy as np

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


def parse_restraints(fname, pdb1, pdb2, log=None):
    """Parse a restraints file, in the DisVis or the HADDOCK format.

    In the DisVis format, each line gives chain, residue number and atom
    name in pdb1, the same in pdb2, and the minimum and maximum distance.
    A file with assign statements is read in the HADDOCK format (see
    parse_tbl). Returns a list of [pdb1 points, pdb2 points, mindis,
    maxdis]: a restraint is satisfied when any pair of points (rows of the
    two (npairs, 3) arrays) lies between mindis and maxdis. A DisVis
    restraint has one pair, the centers of its two selections. log, if
    given, is called with the messages of parse_tbl.
    """
    with open(fname, encoding='utf-8') as fid:
        text = fid.read()

    if count_tbl_restraints(text):
        return parse_tbl(text, pdb1, pdb2, log)

    dist_restraints = []

    for n, raw_line in enumerate(text.splitlines(), 1):
        # ignore comments and empty lines
        line = raw_line.strip()
        if line.startswith('#') or not line:
            continue

        words = line.split()
        if len(words) != 8:
            raise ValueError(
                "Line {:d} of {:s} is not a restraint: a line gives the "
                "chain, residue number and atom name in each structure, and "
                "the minimum and maximum distance (HADDOCK restraint files "
                "are recognised by their assign statements):\n{:s}".format(
                    n, fname, line
                )
            )
        chain1, resi1, name1, chain2, resi2, name2, mindis, maxdis = words
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
            [
                pdb1_sel.center.reshape(1, 3),
                pdb2_sel.center.reshape(1, 3),
                float(mindis),
                float(maxdis),
            ]
        )

    if not dist_restraints:
        raise ValueError("No restraints found in {:s}".format(fname))
    return dist_restraints


def count_tbl_restraints(text):
    """Return the number of restraints of a HADDOCK restraints file, or 0.

    A HADDOCK restraints file has a line starting with assign.
    """
    if not re.search(r'^\s*assign\b', text, re.IGNORECASE | re.MULTILINE):
        return 0
    return sum(t.lower() == 'assign' for t in _TblParser(text).tokens)


def parse_tbl(text, pdb1, pdb2, log=None):
    """Parse restraints in the HADDOCK (CNS) format.

    Each restraint is assign (sel1) (sel2) d dminus dplus, optionally
    followed by or (sel3) (sel4) ..., for distances between d - dminus and
    d + dplus; ! starts a comment. Selections combine segid (matched
    against the chain), resid (a number or a range first:last), name and
    resname (* is a wildcard), all, not, and, or and parentheses. sel1 is
    taken in pdb1 and sel2 in pdb2 when every selection pair of the
    restraint has atoms there, else the reverse for the whole restraint; a
    restraint with atoms in neither direction raises a ValueError. Every
    atom of sel1 is paired with every atom of sel2. Returns restraints as
    parse_restraints. log, if given, is called with the number of
    restraints read in the reverse direction and, when pdb1 and pdb2 share
    chain identifiers, a warning.
    """
    shared = sorted(set(pdb1.chain_list) & set(pdb2.chain_list))
    if log and shared:
        log(
            "WARNING: the fixed and scanning chains share the chain "
            "identifier(s) {:s}, so segid cannot tell them apart: the first "
            "selection of a restraint is taken in the fixed chain unless it "
            "has no atoms there. Different chain identifiers are "
            "recommended.".format(', '.join(shared))
        )
    parser = _TblParser(text)
    dist_restraints = []
    nreversed = 0
    while not parser.done():
        pairs, mindis, maxdis = parser.restraint()

        # One direction for the whole restraint: every pair must have atoms
        masks = [(sel1(pdb1), sel2(pdb2)) for sel1, sel2 in pairs]
        reversed_ = not all(m1.any() and m2.any() for m1, m2 in masks)
        if reversed_:
            masks = [(sel2(pdb1), sel1(pdb2)) for sel1, sel2 in pairs]
            if not all(m1.any() and m2.any() for m1, m2 in masks):
                raise ValueError(
                    "The selections of a restraint were not found in the "
                    "fixed and scanning chains, in either direction, in "
                    "restraint:\n{:s}".format(parser.current())
                )

        coor1 = []
        coor2 = []
        for mask1, mask2 in masks:
            n1, n2 = mask1.sum(), mask2.sum()
            coor1.append(np.repeat(pdb1.coor[mask1], n2, axis=0))
            coor2.append(np.tile(pdb2.coor[mask2], (n1, 1)))

        dist_restraints.append(
            [np.concatenate(coor1), np.concatenate(coor2), mindis, maxdis]
        )
        nreversed += reversed_

    if log:
        log(
            "{:d} of {:d} restraints read with the first selection in the "
            "scanning chain".format(nreversed, len(dist_restraints))
        )
    return dist_restraints


def _fnmatch(values, pattern):
    """Return the mask of the values that match pattern (* is a wildcard)."""
    unique, inverse = np.unique(values, return_inverse=True)
    return np.asarray([fnmatchcase(x, pattern) for x in unique], bool)[inverse]


class _TblParser:
    """Read the restraints of a HADDOCK restraints file one by one.

    A selection is returned as a function that gives the atom mask of a
    structure.
    """

    def __init__(self, text):
        self.tokens = re.findall(r'[()]|[^\s()]+', re.sub(r'!.*', '', text))
        self.pos = 0
        self.start = 0

    def done(self):
        return self.pos == len(self.tokens)

    def current(self):
        """Return the text of the restraint read so far."""
        return ' '.join(self.tokens[self.start : self.pos])

    def peek(self):
        return None if self.done() else self.tokens[self.pos].lower()

    def take(self, expected=None):
        if self.done():
            raise ValueError("Restraints file ends in a restraint.")
        token = self.tokens[self.pos]
        self.pos += 1
        if expected is not None and token.lower() != expected:
            raise ValueError(
                "Expected '{:s}' instead of '{:s}' in restraint:\n{:s}".format(
                    expected, token, self.current()
                )
            )
        return token

    def restraint(self):
        """Return the selection pairs, mindis and maxdis of a restraint."""
        self.start = self.pos
        self.take('assign')
        pairs = [(self.expression(), self.expression())]
        d, dminus, dplus = (float(self.take()) for _ in range(3))
        while self.peek() == 'or':
            self.take()
            pairs.append((self.expression(), self.expression()))
        return pairs, max(d - dminus, 0.0), d + dplus

    def expression(self):
        terms = [self.term()]
        while self.peek() == 'or':
            self.take()
            terms.append(self.term())
        return lambda pdb: np.logical_or.reduce([t(pdb) for t in terms])

    def term(self):
        factors = [self.factor()]
        while self.peek() == 'and':
            self.take()
            factors.append(self.factor())
        return lambda pdb: np.logical_and.reduce([f(pdb) for f in factors])

    def factor(self):
        keyword = self.take().lower()
        if keyword == '(':
            sel = self.expression()
            self.take(')')
            return sel
        if keyword == 'not':
            sel = self.factor()
            return lambda pdb: ~sel(pdb)
        if keyword == 'all':
            return lambda pdb: np.ones(pdb.natoms, bool)
        value = self.take()
        if keyword == 'segid':
            return lambda pdb: pdb.data['chain'] == value
        if keyword == 'resid':
            first, _, last = value.partition(':')
            first, last = int(first), int(last or first)
            return lambda pdb: (
                (pdb.data['resi'] >= first) & (pdb.data['resi'] <= last)
            )
        if keyword in ('name', 'resname'):
            field = 'name' if keyword == 'name' else 'resn'
            return lambda pdb: _fnmatch(pdb.data[field], value)
        raise ValueError(
            "Unknown selection keyword '{:s}' in restraint:\n{:s}".format(
                keyword, self.current()
            )
        )


def dump_args(args, fname):
    """Dump the command-line arguments to a `.json` file."""
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
