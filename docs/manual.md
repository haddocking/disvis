# Manual

--8<-- "README.md:usage"

## Recommended workflow

DisVis is aimed at explorative modelling: checking the consistency and
information content of a set of distance restraints, to design follow-up
experiments or to select the restraints that drive docking, e.g. with
[HADDOCK](https://www.bonvinlab.org/software/).

1. Determine the consistency of the restraints with a quick, coarse search
   (e.g. `-a 15 -vs 2`).
2. Inspect the violations and z-scores (`violations.out`, `z-score.out`) and
   remove the likely false-positive restraints.
3. Once the restraints are consistent, run a finer search (e.g.
   `-a 9.72 -vs 1`), with the occupancy analysis (`-oa`) to approximate the
   position of the scanning chain, and the interaction analysis (`-is`) to find
   the residues most likely at the interface.

To shorten the run time, use the larger molecule as fixed chain and the smaller
one as scanning chain, unless you are interested in the accessible interaction
space of the smaller molecule.

## Web service

DisVis is also available as a web service on the
[WeNMR/CSB portal](https://wenmr.science.uu.nl/disvis). You need an account on
the portal to submit a job.

The submission form takes:

- **Fixed chain** and **scanning chain**: the two structures, in PDB or mmCIF
  format.
- **Restraints file**: the distance restraints, in the format described in
  [Usage](#usage).
- **Scanning mode**: see the table below.
- **Interaction analysis** (optional): a residue selection file, as for the
  `-is` option.
- **Tag** (optional): a name to identify your run.

The scanning modes set the search parameters:

| Mode | Rotational sampling interval | Voxel spacing | Occupancy analysis |
| --- | --- | --- | --- |
| Quick scanning (default) | 15° | 2 Å | no |
| Complete scanning | 9.72° | 1 Å | yes |
| Custom scanning | 5–360° | 1–4 Å | optional |

In the custom mode, the interaction radius (1–4 Å, default 3 Å, `-ir`), the
maximum clash volume (default 200 Å³, `-cv`) and the minimum interaction volume
(default 300 Å³, `-iv`) can also be changed.

As on the command line, start with a quick scan without the interaction
analysis, to check that complexes consistent with all, or all but a few,
restraints exist. The results page shows images of the accessible interaction
space, the accessible complexes, the z-scores, the violations and, with an
interaction analysis, the most contacted residues, and gives all files of the
run as an archive. The results are kept on the server for 14 days.
