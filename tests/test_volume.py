import os.path
import time

from disvis.volume import Volumizer
from disvis.pdb import PDB
from disvis.rotations import proportional_orientations, quat_to_rotmat

receptor_fname = os.path.join('data', 'O14250.pdb')
ligand_fname = os.path.join('data', 'Q9UT97.pdb')

receptor = PDB.fromfile(receptor_fname)
ligand = PDB.fromfile(ligand_fname)

volumizer = Volumizer(receptor, ligand)
print volumizer.shape
print volumizer.origin

volumizer.rcore.tofile('rcore.mrc')
volumizer.rsurface.tofile('rsurface.mrc')
print volumizer.rcore.array.max()

q = proportional_orientations(10)[0]
rotmat = quat_to_rotmat(q)
print 'Rotations:', rotmat.shape[0]
print volumizer.lcore.array.flags

time0 = time.time()
for n, rot in enumerate(rotmat):
    volumizer.generate_lcore(rot)
    # if n == 0:
    #    volumizer.lcore.tofile('lcore.mrc')
    # if n == 10:
    #    volumizer.lcore.tofile('test.mrc')
    #    break

print 'Time:', time.time() - time0
