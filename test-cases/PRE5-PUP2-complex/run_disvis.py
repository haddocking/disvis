from __future__ import print_function, division
from disvis import PDB, DisVis
from disvis.rotations import proportional_orientations, quat_to_rotmat

receptor = PDB.fromfile('O14250.pdb')
ligand = PDB.fromfile('Q9UT97.pdb')
angle = 90

quaternions, weights, angle = proportional_orientations(angle)
rotations = quat_to_rotmat(quaternions, invert=True)
print('Number of rotations sampled: ', rotations.shape[0])


pd = DisVis()
pd.receptor = receptor
pd.ligand = ligand
pd.voxelspacing = 1
pd.weights = weights 
pd.rotations = rotations
pd.surface_radius = 2.5
pd.erosion_iterations = 8
pd.max_clash = 1
pd.min_interaction = 300

# ADH cross links
ADH_DISTANCE = 23
# residue 27 -> 18 (5.9A)
pd.add_distance_restraint(receptor.select('resi', 27).select('name', 'CA'),
                          ligand.select('resi', 18).select('name', 'CA'), 
                          ADH_DISTANCE)
# residue 122 -> 125 (12.1A)
pd.add_distance_restraint(receptor.select('resi', 122).select('name', 'CA'), 
                          ligand.select('resi', 125).select('name', 'CA'), 
                          ADH_DISTANCE)
# residue 122 -> 128 (5.7A)
pd.add_distance_restraint(receptor.select('resi', 122).select('name', 'CA'), 
                          ligand.select('resi', 128).select('name', 'CA'),
                          ADH_DISTANCE)
# residue 122 -> 127 (7.8A)
pd.add_distance_restraint(receptor.select('resi', 122).select('name', 'CA'),
                          ligand.select('resi', 127).select('name', 'CA'), 
                          ADH_DISTANCE)

# ZL cross links
ZL_DISTANCE = 26
# residue 55 -> 169 (10.8A)
pd.add_distance_restraint(receptor.select('resi', 55).select('name', 'CA'),
                          ligand.select('resi', 169).select('name', 'CA'),
                          ZL_DISTANCE)
# residue 55 -> 179 (10.9)
pd.add_distance_restraint(receptor.select('resi', 55).select('name', 'CA'),
                          ligand.select('resi', 179).select('name', 'CA'),
                          ZL_DISTANCE)
# residue 54 -> 179 (9.1A)
pd.add_distance_restraint(receptor.select('resi', 54).select('name', 'CA'),
                          ligand.select('resi', 179).select('name', 'CA'),
                          ZL_DISTANCE)

from time import time
time0 = time()
accessible_interaction_space, accessible_complexes = pd.search()

accessible_interaction_space.tofile('accessible_interaction_space.mrc')

print('Time: ', time()-time0)
total_accessible_complexes = accessible_complexes[0]
for n, accessible in enumerate(accessible_complexes):
    print('Number of accessible complexes complying to {:} restraints: {:} ({:8.4f}%)'.format(n, accessible, accessible/total_accessible_complexes*100.0))
    
