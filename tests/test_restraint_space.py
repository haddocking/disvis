import numpy as np

from disvis.spaces import RestraintSpace, Restraint
from disvis.volume import Volume


class Container(object):
    pass


space = Volume.zeros((7, 8, 9), dtype=np.int32)
rsel1 = Container()
lsel1 = Container()
rsel1.coor = np.zeros((1, 3), dtype=np.float64)
lsel1.coor = np.zeros((2, 3), dtype=np.float64)
lsel1.coor[-1] = [-8, -7, -6]

rsel2 = Container()
lsel2 = Container()
rsel2.coor = np.zeros((1, 3), dtype=np.float64)
lsel2.coor = np.zeros((1, 3), dtype=np.float64)

min_dis = 0
max_dis = 5
restraints = [Restraint([rsel1], [lsel1], min_dis, max_dis),
              Restraint([rsel2], [lsel2], min_dis, max_dis),
              ]

E = np.eye(3)

rs = RestraintSpace(space, restraints, (0, 0, 0))
rs(E)
print rs.space.array
