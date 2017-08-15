from __future__ import absolute_import, print_function, division
import os.path
import operator
from collections import Sequence, defaultdict

import numpy as np

# from .IO.pdb import parse_pdb
from .IO.mmcif import parse_cif
from .elements import ELEMENTS

# records
MODEL = 'MODEL '
ATOM = 'ATOM  '
HETATM = 'HETATM'
TER = 'TER   '

MODEL_LINE = 'MODEL ' + ' ' * 4 + '{:>4d}\n'
ENDMDL_LINE = 'ENDMDL\n'
TER_LINE = 'TER   ' + '{:>5d}' + ' ' * 6 + '{:3s}' + ' ' + '{:1s}' + \
           '{:>4d}' + '{:1s}' + ' ' * 53 + '\n'
ATOM_LINE = '{:6s}' + '{:>5d}' + ' ' + '{:4s}' + '{:1s}' + '{:3s}' + ' ' + \
            '{:1s}' + '{:>4d}' + '{:1s}' + ' ' * 3 + '{:8.3f}' * 3 + '{:6.2f}' * 2 + \
            ' ' * 10 + '{:<2s}' + '{:2s}\n'
END_LINE = 'END   \n'

ATOM_DATA = ('record id name alt resn chain resi i x y z q b ' \
             'e charge').split()
TER_DATA = 'id resn chain resi i'.split()


class PDB(object):
    @classmethod
    def fromfile(cls, pdbfile):
        try:
            fname = pdbfile.name
        except AttributeError:
            fname = pdbfile

        extension = os.path.splitext(fname)[1]

        if extension == '.cif':
            return cls(parse_cif(pdbfile))
        elif extension in ('.pdb', '.ent'):
            return cls(pdb_dict_to_array(parse_pdb(pdbfile)))
        else:
            raise ValueError("Format of file is not recognized")

    def __init__(self, pdbdata):
        self.data = pdbdata

    @property
    def atomnumber(self):
        elements, ind = np.unique(self.data['e'], return_inverse=True)
        atomnumbers = np.asarray([ELEMENTS[e].number for e in elements], dtype=np.float64)
        return atomnumbers[ind]

    @property
    def coor(self):
        return np.asarray([self.data['x'], self.data['y'], self.data['z']]).T

    @coor.setter
    def coor(self, coor_array):
        self.data['x'], self.data['y'], self.data['z'] = coor_array.T

    @property
    def center(self):
        return self.coor.mean(axis=0)

    @property
    def center_of_mass(self):
        mass = self.mass.reshape(-1, 1)
        return (self.coor * mass).sum(axis=0) / mass.sum()

    @property
    def chain_list(self):
        return np.unique(self.data['chain'])

    @property
    def com(self):
        return self.center_of_mass

    @property
    def elements(self):
        return self.data['e']

    @property
    def mass(self):
        elements, ind = np.unique(self.data['e'], return_inverse=True)
        mass = np.asarray([ELEMENTS[e].mass for e in elements], dtype=np.float64)
        return mass[ind]

    @property
    def natoms(self):
        return self.data.shape[0]

    @property
    def sequence(self):
        resids, indices = np.unique(self.data['resi'], return_index=True)
        return self.data['resn'][indices]

    def combine(self, pdb):
        return PDB(np.hstack((self.data, pdb.data)))

    def duplicate(self):
        return PDB(self.data.copy())

    def rmsd(self, pdb):
        return np.sqrt(((self.coor - pdb.coor) ** 2).mean() * 3)

    def rotate(self, rotmat):
        self.data['x'], self.data['y'], self.data['z'] = \
            np.mat(rotmat) * np.mat(self.coor).T

    def translate(self, vector):
        self.data['x'] += vector[0]
        self.data['y'] += vector[1]
        self.data['z'] += vector[2]

    def select(self, identifier, values, loperator='==', return_ind=False):
        """A simple way of selecting atoms"""
        if loperator == '==':
            oper = operator.eq
        elif loperator == '<':
            oper = operator.lt
        elif loperator == '>':
            oper = operator.gt
        elif loperator == '>=':
            oper = operator.ge
        elif loperator == '<=':
            oper = operator.le
        elif loperator == '!=':
            oper = operator.ne
        else:
            raise ValueError('Logic operator not recognized.')

        if not isinstance(values, Sequence) or isinstance(values, basestring):
            values = (values,)

        selection = oper(self.data[identifier], values[0])
        if len(values) > 1:
            for v in values[1:]:
                if loperator == '!=':
                    selection &= oper(self.data[identifier], v)
                else:
                    selection |= oper(self.data[identifier], v)

        if return_ind:
            return selection
        else:
            return PDB(self.data[selection])

    def tofile(self, fid):
        """Write instance to PDB-file"""
        tofile(pdb_array_to_dict(self.data), fid)

    @property
    def vdw_radius(self):
        elements, ind = np.unique(self.data['e'], return_inverse=True)
        rvdw = np.asarray([ELEMENTS[e].vdwrad for e in elements], dtype=np.float64)
        return rvdw[ind]


def parse_pdb(infile):
    if isinstance(infile, file):
        f = infile
    elif isinstance(infile, str):
        f = open(infile)
    else:
        raise TypeError('Input should be either a file or string.')

    pdb = defaultdict(list)
    model_number = 1
    for line in f:
        record = line[:6]
        if record in (ATOM, HETATM):
            pdb['model'].append(model_number)
            pdb['record'].append(record)
            pdb['id'].append(int(line[6:11]))
            name = line[12:16].strip()
            pdb['name'].append(name)
            pdb['alt'].append(line[16])
            pdb['resn'].append(line[17:20].strip())
            pdb['chain'].append(line[21])
            pdb['resi'].append(int(line[22:26]))
            pdb['i'].append(line[26])
            pdb['x'].append(float(line[30:38]))
            pdb['y'].append(float(line[38:46]))
            pdb['z'].append(float(line[46:54]))
            pdb['q'].append(float(line[54:60]))
            pdb['b'].append(float(line[60:66]))
            # Be forgiving when determining the element
            e = line[76:78].strip()
            if not e:
                # If element is not given, take the first non-numeric letter of
                # the name as element.
                for e in name:
                    if e.isalpha():
                        break
            pdb['e'].append(e)
            pdb['charge'].append(line[78: 80].strip())
        elif record == MODEL:
            model_number = int(line[10: 14])
    f.close()
    return pdb


def tofile(pdb, out):
    f = open(out, 'w')

    nmodels = len(set(pdb['model']))
    natoms = len(pdb['id'])
    natoms_per_model = natoms // nmodels

    for nmodel in xrange(nmodels):
        offset = nmodel * natoms_per_model
        # write MODEL record
        if nmodels > 1:
            f.write(MODEL_LINE.format(nmodel + 1))
        prev_chain = pdb['chain'][offset]
        for natom in xrange(natoms_per_model):
            index = offset + natom

            # write TER record
            current_chain = pdb['chain'][index]
            if prev_chain != current_chain:
                prev_record = pdb['record'][index - 1]
                if prev_record == ATOM:
                    line_data = [pdb[data][index - 1] for data in TER_DATA]
                    line_data[0] += 1
                    f.write(TER_LINE.format(*line_data))
                prev_chain = current_chain

            # write ATOM/HETATM record
            line_data = [pdb[data][index] for data in ATOM_DATA]
            # take care of the rules for atom name position
            e = pdb['e'][index]
            name = pdb['name'][index]
            if len(e) == 1 and len(name) != 4:
                line_data[2] = ' ' + name
            f.write(ATOM_LINE.format(*line_data))

        # write ENDMDL record
        if nmodels > 1:
            f.write(ENDMDL_LINE)

    f.write(END_LINE)
    f.close()


def pdb_dict_to_array(pdb):
    dtype = [('record', np.str_, 6), ('id', np.int32),
             ('name', np.str_, 4), ('alt', np.str_, 1),
             ('resn', np.str_, 4), ('chain', np.str_, 2),
             ('resi', np.int32), ('i', np.str_, 1), ('x', np.float64),
             ('y', np.float64), ('z', np.float64),
             ('q', np.float64), ('b', np.float64),
             ('e', np.str_, 2), ('charge', np.str_, 2),
             ('model', np.int32)]

    natoms = len(pdb['id'])
    pdb_array = np.empty(natoms, dtype=dtype)
    for data in ATOM_DATA:
        pdb_array[data] = pdb[data]
    pdb_array['model'] = pdb['model']
    return pdb_array


def pdb_array_to_dict(pdb_array):
    pdb = defaultdict(list)
    for data in ATOM_DATA:
        pdb[data] = pdb_array[data].tolist()
    pdb['model'] = pdb_array['model'].tolist()
    return pdb
