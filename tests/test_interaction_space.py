import numpy as np

from disvis.pdb import PDB
from disvis.volume import Volumizer, Volume
from disvis.spaces import InteractionSpace

rcore = Volume(np.zeros((5, 5, 5), dtype=np.float64))
rsurface = Volume(np.zeros((5, 5, 5), dtype=np.float64))
lcore = Volume(np.zeros((5, 5, 5), dtype=np.float64))

rcore.array[2, 2, 2] = 1
rsurface.array[2, 2, 3] = 1
lcore.array[0, 0, 0] = 1

interaction_space = InteractionSpace(rcore, rsurface, max_clash=0, min_inter=1)
interaction_space(lcore)
print interaction_space.space.array
