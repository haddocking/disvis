from __future__ import absolute_import, print_function, division
import os.path
import operator
from collections import Iterable
import numpy as np
from .IO.pdb import parse_pdb, write_pdb
from .IO.mmcif import parse_cif
from .atompar import parameters as atompar

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
            return cls(parse_pdb(pdbfile))
        else:
            raise ValueError("Format of file is not recognized")


    def __init__(self, pdbdata):
        self.data = pdbdata


    @property
    def atomnumber(self):
        elements, ind = np.unique(self.data['element'], return_inverse=True)
        atomnumbers = np.asarray([atompar[e]['Z'] for e in elements], dtype=np.float64)
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
        return (self.coor * mass).sum(axis=0)/mass.sum()


    @property
    def chain_list(self):
        return np.unique(self.data['chain'])


    @property
    def com(self):
        return self.center_of_mass


    @property
    def elements(self):
        return self.data['element']


    @property
    def mass(self):
        elements, ind = np.unique(self.data['element'], return_inverse=True)
        mass = np.asarray([atompar[e]['mass'] for e in elements], dtype=np.float64)
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
        return np.sqrt(((self.coor - pdb.coor)**2).mean()*3)


    def rotate(self, rotmat):
        self.data['x'], self.data['y'], self.data['z'] =\
             np.mat(rotmat) * np.mat(self.coor).T


    def translate(self, vector):
        self.data['x'] += vector[0]
        self.data['y'] += vector[1]
        self.data['z'] += vector[2]


    def select(self, identifier, values, loperator='=='):
        """A simple and probably pretty inefficient way of selection atoms"""
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

        if not isinstance(values, Iterable):
            values = (values,)
        selection = oper(self.data[identifier], values[0])

        if len(values) > 1:
            for v in values[1:]:
                selection |= oper(self.data[identifier], v)

        return PDB(self.data[selection])


    def tofile(self, fid):
        write_pdb(fid, self.data)


    @property
    def vdw_radius(self):
        elements, ind = np.unique(self.data['element'], return_inverse=True)
        rvdw = np.asarray([atompar[e]['rvdW'] for e in elements], dtype=np.float64)
        return rvdw[ind]
