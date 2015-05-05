import sys

import matplotlib.pyplot as plt
from numpy import loadtxt, arange

resids, ave_inter = loadtxt(sys.argv[1], unpack=True)

# we want to have the most interacted residues on top
resids = resids[::-1]
ave_inter = ave_inter[::-1]
res_ind = arange(resids.size)

# make figure
fig = plt.figure()#figsize=(190/25.4/3.0, 250/25.4))

ax1 = plt.gca()

ax1.barh(res_ind, ave_inter, height=1.0, align='center', color='gray')
ax1.set_label('Residue')
ax1.set_yticks(res_ind)
ax1.set_yticklabels(resids)

ax1.tick_params(axis='both', labelsize=7)
ax1.set_ylim([-0.5, res_ind.size - 0.5])

ax1.set_xlabel('Number of average interactions')

fig.tight_layout()
plt.savefig(sys.argv[2], dpi=300, transparent=True)
