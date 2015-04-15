from __future__ import print_function, division
import sys
import matplotlib.pyplot as plt
from numpy import argsort, asarray


def parse_interactions(f):
    data = {}
    with open(f) as f:
        words = f.readline().split()
        data['residues'] = [int(x) for x in words[2:]]

        for line in f:
            words = line.split()

            consistent_restraints = int(words[0])
            data[consistent_restraints] = {}

            data[consistent_restraints]['total'] = int(words[1])
            data[consistent_restraints]['interactions'] = [int(x) for x in words[2:]]
    return data

def main():
    f = sys.argv[1]
    nrestraints = int(sys.argv[2])

    data = parse_interactions(f)

    b_order = False
    y = range(len(data[nrestraints]['interactions']))
    labels = asarray(data['residues'])
    x = asarray(data[nrestraints]['interactions']) / data[nrestraints]['total']
    if b_order:
        order = argsort(data[nrestraints]['interactions'])[::-1]
        lables = labels[order]
        x = x[order]


    active_res = labels[x >= 1]
    passive_res = labels[(x > 0) & (x < 1)]

    print('Active residues:')
    print(str(list(active_res))[1: -1])
    print('Passive residues:')
    print(str(list(passive_res))[1: -1])


    fig = plt.figure(figsize=(190/25.4/3.0, 240/25.4))#, dpi=300)
    ax1 = plt.gca()

    ax1.barh(y, x, height=1.0, align='center')
    ax1.set_ylabel('Residue')
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.tick_params(axis='both', labelsize=7)
    ax1.set_ylim([-0.5, x.shape[0] - 0.5])

    ax1.set_xlabel('Number of interactions')

    fig.tight_layout()
    plt.show()

if __name__=='__main__':
    main()
