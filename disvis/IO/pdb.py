import numpy as np
import re

def parse_pdb(pdbfile):

    if isinstance(pdbfile, file):
        pass
    elif isinstance(pdbfile, str):
        pdbfile = open(pdbfile)
    else:
        raise TypeError('Input should either be a file or string.')

    ATOM = "ATOM "
    MODEL = "MODEL "

    model = []
    serial = []
    name = []
    alt_loc = []
    res_name = []
    chain_id = []
    res_seq = []
    i_code = []
    x = []
    y = []
    z = []
    occupancy = []
    temp_factor = []
    element = []
    charge = []

    model_number = 1

    for line in pdbfile:

        if line.startswith(MODEL):
            model_number = int(line[10:14])

        elif line.startswith(ATOM):

            model.append(model_number)
            serial.append(int(line[6:11].strip()))
            name.append(line[12:16].strip())
            alt_loc.append(line[16])
            res_name.append(line[17:20])
            chain_id.append(line[21])
            res_seq.append(int(line[22:26]))
            i_code.append(line[26])
            x.append(float(line[30:38]))
            y.append(float(line[38:46]))
            z.append(float(line[46:54]))
            occupancy.append(float(line[54:60]))
            temp_factor.append(float(line[60:66]))
            e = line[76:78].strip()
            if not e:
                e = line[12:16][re.search("[a-z,A-Z]",line[12:16]).start()]
            element.append(e)
            charge.append(line[78:80])

            tmp = line[76:78].strip()
            if not tmp:
                tmp = line[12:16].strip()[0]

    natoms = len(name)
    dtype = [('atom_id', np.int64), ('name', np.str_, 4), 
             ('resn', np.str_, 4), ('chain', np.str_, 1), 
             ('resi', np.int64), ('x', np.float64),
             ('y', np.float64), ('z', np.float64), 
             ('occupancy', np.float64), ('bfactor', np.float64),
             ('element', np.str_, 2), ('charge', np.str_, 2),
             ('model', np.int64),
             ]
             
    pdbdata = np.zeros(natoms, dtype=dtype)
    pdbdata['atom_id'] = np.asarray(serial, dtype=np.int64)
    pdbdata['name'] = name
    pdbdata['resn'] = res_name
    pdbdata['chain'] = chain_id
    pdbdata['resi'] = res_seq
    pdbdata['x'] = x
    pdbdata['y'] = y
    pdbdata['z'] = z
    pdbdata['occupancy'] = occupancy
    pdbdata['bfactor'] = temp_factor
    pdbdata['element'] = element
    pdbdata['charge'] = charge
    pdbdata['model'] = model

    return pdbdata

def write_pdb(outfile, pdbdata):
    #HETATOM = "HETATM"
    #atom_line = ''.join(['{atom:6s}', '{serial:5d}', ' ', '{name:4s}', 
    #    '{altLoc:1s}', '{resName:3s}', ' ', '{chainID:1s}',
    #    '{resSeq:4d}', '{iCode:1s}', ' '*3, '{x:8.3f}', '{y:8.3f}', '{z:8.3f}', 
    #    '{occupancy:6.2f}', '{tempFactor:6.2f}', ' '*10, '{element:2s}',
    #    '{charge:2s}', '\n'])

    fhandle = open(outfile, 'w')

    # ATOM record
    ATOM_LINE = ''.join(['ATOM  ', '{:5d}', ' ', '{:4s}', 
        '{:1s}', '{:3s}', ' ', '{:1s}',
        '{:4d}', '{:1s}', ' '*3, '{:8.3f}', '{:8.3f}', '{:8.3f}', 
        '{:6.2f}', '{:6.2f}', ' '*10, '{:2s}',
        '{:2s}', '\n'])


    # MODEL record
    nmodels = np.unique(pdbdata['model']).size
    MODEL_LINE = 'ENDMDL\nMODEL     {:>4d}\n'
    current_model = previous_model = 1
    if nmodels > 1:
        fhandle.write(MODEL_LINE.format(current_model))

    # TER record
    TER_LINE = 'TER   ' + '{:4d}' + ' '*6 + '{:>3s}' + ' ' + '{:s}' + '{:>4d}\n'
    previous_chain = pdbdata['chain'][0]

    for n in range(pdbdata.shape[0]):

        # MODEL statement
        current_model = pdbdata['model'][n]
        if current_model != previous_model:
            fhandle.write(MODEL_LINE.format(current_model))
            previous_model = current_model

        # ATOM statement
        fhandle.write(ATOM_LINE.format(pdbdata['atom_id'][n], 
                                       pdbdata['name'][n],
                                       '',
                                       pdbdata['resn'][n],
                                       pdbdata['chain'][n],
                                       pdbdata['resi'][n],
                                       '',
                                       pdbdata['x'][n],
                                       pdbdata['y'][n],
                                       pdbdata['z'][n],
                                       pdbdata['occupancy'][n],
                                       pdbdata['bfactor'][n],
                                       pdbdata['element'][n],
                                       pdbdata['charge'][n],
                                       ))

        # TER record
        current_chain = pdbdata['chain'][n]
        if current_chain != previous_chain:
            fhandle.write(TER_LINE.format(pdbdata['atom_id'][n], pdbdata['resn'][n], pdbdata['chain'][n], pdbdata['resi'][n]))
            previous_chain = current_chain

    # final TER recored
    fhandle.write(TER_LINE.format(pdbdata['atom_id'][n], pdbdata['resn'][n], pdbdata['chain'][n], pdbdata['resi'][n]))

    # END record
    fhandle.write('END')
    fhandle.close()
