from collections import defaultdict

import numpy as np

def parse_pdb(pdbfile):

    if isinstance(pdbfile, file):
        pass
    elif isinstance(pdbfile, str):
        pdbfile = open(pdbfile)
    else:
        raise TypeError('Input should either be a file or string.')

    ATOM = "ATOM "
    HETATM = "HETATM"
    MODEL = "MODEL "

    pdb = defaultdict(list)

    model_number = 1
    for line in pdbfile:

        if line.startswith(MODEL):
            model_number = int(line[10:14])

        elif line.startswith((ATOM, HETATM)):

            pdb['model'].append(model_number)
            pdb['atom_id'].append(int(line[6:11].strip()))
            name = line[12:16].strip()
            pdb['name'].append(name)
            pdb['alt_loc'].append(line[16])
            pdb['resn'].append(line[17:20])
            pdb['chain'].append(line[21])
            pdb['resi'].append(int(line[22:26]))
            pdb['i_code'].append(line[26])
            pdb['x'].append(float(line[30:38]))
            pdb['y'].append(float(line[38:46]))
            pdb['z'].append(float(line[46:54]))
            pdb['occupancy'].append(float(line[54:60]))
            pdb['bfactor'].append(float(line[60:66]))
            e = line[76:78].strip()
            # Be forgiving if element is not given
            if not e:
                for e in name:
                    if e.isalpha():
                        break
            pdb['element'].append(e)
            pdb['charge'].append(line[78:80])

    natoms = len(pdb['name'])
    dtype = [('atom_id', np.int64), ('name', np.str_, 4), 
             ('resn', np.str_, 4), ('chain', np.str_, 1), 
             ('resi', np.int64), ('x', np.float64),
             ('y', np.float64), ('z', np.float64), 
             ('occupancy', np.float64), ('bfactor', np.float64),
             ('element', np.str_, 2), ('charge', np.str_, 2),
             ('model', np.int64), ('i_code', np.str_, 1),
             ('alt_loc', np.str_, 1)
             ]
             
    pdbdata = np.zeros(natoms, dtype=dtype)
    for key, value in pdb.iteritems():
        pdbdata[key] = value

    return pdbdata
