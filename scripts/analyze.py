from __future__ import print_function
from glob import glob
from disvis import PDB
from disvis.disvis import grid_shape, rsurface
from disvis.points import dilate_points

pdbs = sorted(glob('*_[rl]_u.pdb'))
print(pdbs)
VOXELSPACING = 1

overlaps =[]
cores = []
for n in range(len(pdbs)/2):
    print(pdbs[2*n+1], pdbs[2*n])
    try:
        p1 = PDB.fromfile(pdbs[2*n+1])
        p2 = PDB.fromfile(pdbs[2*n])

        shape = grid_shape(p1.coor, p2.coor, VOXELSPACING)

        vdw_radii = p1.vdw_radius
        radii = vdw_radii + 3.0

        rsurf = rsurface(p1.coor, radii, shape, VOXELSPACING)
        rcore = rsurface(p1.coor, vdw_radii, shape, VOXELSPACING)

        lsurf = rsurf.duplicate()
        lsurf._array.fill(0)
        lsurf = dilate_points(p2.coor, p2.vdw_radius, lsurf)

        #rsurf.tofile('rsurf.mrc')
        #rcore.tofile('rcore.mrc')
        #lsurf.tofile('lsurf.mrc')

        overlap = (rsurf.array*lsurf.array).sum()
        clash = (rcore.array*lsurf.array).sum()
        overlaps.append(overlap)
        cores.append(clash)

        print(pdbs[2*n][:4])
        print('overlap: ', overlap)
        print('clash: ', clash)
        print()
    except:
        pass

print(sorted(overlaps))
print(sorted(cores))
