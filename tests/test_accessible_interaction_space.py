import numpy as np

from disvis.spaces import AccessibleInteractionSpace
from disvis.volume import Volume

ais = Volume.zeros((10, 10, 10), dtype=np.int32)
nrestraints = 2

ais_calc = AccessibleInteractionSpace(ais, 2)

interaction_space = Volume.zeros_like(ais)
interaction_space.array.fill(1)
restraint_space= Volume.zeros_like(ais)
restraint_space.array.fill(2)

ais_calc(interaction_space, restraint_space)
print ais_calc.consistent_complexes()
print ais_calc.consistent_matrix()
print ais_calc.violation_matrix()

#print ais_calc.max_consistent.array

