import sys
from struct import pack
import numpy as np

def to_mrc(fid, volume, labels=[]):

    voxelspacing = volume.voxelspacing
    with open(fid, 'wb') as out:

        nx, ny, nz = volume.shape
        out.write(pack('i', nx))
        out.write(pack('i', ny))
        out.write(pack('i', nz))

        dtype = volume.data.dtype
        if dtype == np.int8:
            mode = 0
        elif dtype in (np.int16, np.int32):
            mode = 1
        elif dtype in (np.float32, np.float64):
            mode = 2
        else:
            raise TypeError("Data type ({:})is not supported.".format(dtype))
        out.write('i', mode)

        nxstart, nystart, nzstart = [int(x) for x in volume.start]
        out.write(pack('i', nxstart))
        out.write(pack('i', nystart))
        out.write(pack('i', nzstart))

        out.write(pack('i', nx))
        out.write(pack('i', ny))
        out.write(pack('i', nz))

        xl, yl, zl = volume.dimensions
        out.write(pack('f', xl))
        out.write(pack('f', yl))
        out.write(pack('f', zl))

        alpha = beta = gamma = 90.0
        out.write(pack('f', alpha))
        out.write(pack('f', beta))
        out.write(pack('f', gamma))

        mapc, mapr, maps = [1, 2, 3]
        out.write(pack('i', mapc))
        out.write(pack('i', mapr))
        out.write(pack('i', maps))

        out.write(pack('f', volume.data.min()))
        out.write(pack('f', volume.data.max()))
        out.write(pack('f', volume.data.mean()))

        ispg = 1
        out.write(pack('i', ispg))
        nsymbt = 0
        out.write(pack('i', nsymbt))

        lskflg = 0
        out.write(pack('i', lskflg))
        skwmat = [0.0]*9
        for f in skwmat:
            out.write(pack('f', f))
        skwtrn = [0.0]*3
        for f in skwtrn:
            out.write(pack('f', f))

        fut_use = [0.0]*12
        for f in fut_use:
            out.write(pack('f', f))

        for f in volume.origin:
            out.write(pack('f', f))

        str_map = ['M', 'A', 'P', ' ']
        for c in str_map:
            out.write(pack('c', c))

        if sys.byteorder == 'little':
            machst = ['\x44', '\x41' ,'\x00', '\x00']
        elif sys.byteorder == 'big':
            machst = ['\x44', '\x41' ,'\x00', '\x00']
        else:
            raise ValueError("Byteorder {:} is not recognized".format(sys.byteorder))

        for c in machst:
            out.write(pack('c', c))

        out.write(pack('f', volume.data.std()))

        # max 10 labels
        # nlabels = min(len(labels), 10)
        # TODO labels not handled correctly

        #for label in labels:
        #     list_label = [c for c in label]
        #     llabel = len(list_label)
        #     if llabel < 80:
        #         
        #     # max 80 characters
        #     label = min(len(label), 80)

        nlabels = 0
        out.write(pack('i', nlabels))

        labels = [' '] * 800
        for c in labels:
            out.write(pack('c', c))

        # write density
        if mode == 0:
            volume.data.tofile(out)
        if mode == 1:
            volume.data.astype(np.int16).tofile(out)
        if mode == 2:
            volume.data.astype(np.float32).tofile(out)
