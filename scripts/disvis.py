from __future__ import print_function
import sys
from disvis import DisVis, PDB 
from disvis.helpers import proportional_orientations, quat_to_rotmat

pdb1 = PDB.fromfile(sys.argv[1])
pdb2 = PDB.fromfile(sys.argv[2])
angle = float(sys.argv[3])


dv = DisVis()
dv.receptor = pdb1
dv.ligand = pdb2
dv.rotations = quat_to_rotmat(proportional_orientations(angle), inverse=True)
interaction_space, complexes = dv.search()
interaction_space.tofile('test.mrc')
for i in complexes:
    print(complexes)
