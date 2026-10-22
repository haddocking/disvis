"""An atomic structure read from a PDB or mmCIF file."""
import os.path
import operator
from collections.abc import Iterable

import numpy as np

from .IO.pdb import parse_pdb
from .IO.mmcif import parse_cif
from .elements import ELEMENTS


class PDB:
    """An atomic structure, stored as a NumPy structured array (data)."""

    @classmethod
    def fromfile(cls, pdbfile):
        """Read a structure from a .pdb, .ent or .cif file."""
        try:
            fname = pdbfile.name
        except AttributeError:
            fname = pdbfile

        extension = os.path.splitext(fname)[1]

        if extension == '.cif':
            return cls(parse_cif(pdbfile))
        elif extension in ('.pdb', '.ent'):
            return cls(parse_pdb(pdbfile))
        else:
            raise ValueError("Format of file is not recognized")

    def __init__(self, pdbdata):
        self.data = pdbdata

    @property
    def atomnumber(self):
        """The atomic number of each atom."""
        elements, ind = np.unique(self.data['element'], return_inverse=True)
        atomnumbers = np.asarray([ELEMENTS[e].number for e in elements],
                                 dtype=np.float64)
        return atomnumbers[ind]

    @property
    def coor(self):
        """The (natoms, 3) array of atom coordinates."""
        return np.asarray([self.data['x'], self.data['y'], self.data['z']]).T

    @coor.setter
    def coor(self, coor_array):
        self.data['x'], self.data['y'], self.data['z'] = coor_array.T

    @property
    def center(self):
        """The geometric center of the atoms."""
        return self.coor.mean(axis=0)

    @property
    def center_of_mass(self):
        """The center of mass of the atoms."""
        mass = self.mass.reshape(-1, 1)
        return (self.coor * mass).sum(axis=0) / mass.sum()

    @property
    def chain_list(self):
        """The sorted unique chain identifiers."""
        return np.unique(self.data['chain'])

    @property
    def com(self):
        """Alias of center_of_mass."""
        return self.center_of_mass

    @property
    def elements(self):
        """The element symbol of each atom."""
        return self.data['element']

    @property
    def mass(self):
        """The atomic mass of each atom."""
        elements, ind = np.unique(self.data['element'], return_inverse=True)
        mass = np.asarray([ELEMENTS[e].mass for e in elements],
                          dtype=np.float64)
        return mass[ind]

    @property
    def natoms(self):
        """The number of atoms."""
        return self.data.shape[0]

    @property
    def sequence(self):
        """The residue names, ordered by residue number."""
        resids, indices = np.unique(self.data['resi'], return_index=True)
        return self.data['resn'][indices]

    def combine(self, pdb):
        """Return a new structure with the atoms of self and pdb."""
        return PDB(np.hstack((self.data, pdb.data)))

    def duplicate(self):
        """Return a copy of the structure."""
        return PDB(self.data.copy())

    def rmsd(self, pdb):
        """Return the RMSD to pdb, which must have the same atom order."""
        return np.sqrt(((self.coor - pdb.coor)**2).mean() * 3)

    def rotate(self, rotmat):
        """Rotate the coordinates in place by the 3x3 matrix rotmat."""
        self.data['x'], self.data['y'], self.data['z'] =\
            np.dot(rotmat, self.coor.T)

    def translate(self, vector):
        """Translate the coordinates in place by vector."""
        self.data['x'] += vector[0]
        self.data['y'] += vector[1]
        self.data['z'] += vector[2]

    def select(self, identifier, values, loperator='=='):
        """Return the atoms whose identifier field matches values.

        identifier is a field of data, such as 'chain', 'resi' or 'name'.
        values is a value or a sequence of values, combined with OR (or
        with AND for '!='). loperator is one of '==', '<', '>', '>=', '<='
        and '!='.
        """
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

        if not isinstance(values, Iterable) or isinstance(values, str):
            values = (values,)
        selection = oper(self.data[identifier], values[0])

        if len(values) > 1:
            for v in values[1:]:
                if loperator == '!=':
                    selection &= oper(self.data[identifier], v)
                else:
                    selection |= oper(self.data[identifier], v)

        return PDB(self.data[selection])

    @property
    def vdw_radius(self):
        """The van der Waals radius of each atom."""
        elements, ind = np.unique(self.data['element'], return_inverse=True)
        rvdw = np.asarray([ELEMENTS[e].vdwrad for e in elements],
                          dtype=np.float64)
        return rvdw[ind]
