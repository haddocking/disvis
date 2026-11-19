# DisVis

[![PyPI - Version](https://img.shields.io/pypi/v/disvis)](https://pypi.org/project/disvis/)
[![DOI](https://zenodo.org/badge/30263463.svg)](https://doi.org/10.5281/zenodo.7775093)
[![ci](https://github.com/haddocking/disvis/actions/workflows/ci.yml/badge.svg)](https://github.com/haddocking/disvis/actions/workflows/ci.yml)
[![SBGrid Badge](https://img.shields.io/badge/DisVis-blue?label=SBGrid&labelColor=%3D)](https://sbgrid.org/software/titles/disvis)
[![bio.tools](https://img.shields.io/badge/bio.tools-disvis-orange)](https://bio.tools/disvis)
[![AI Declaration](https://img.shields.io/badge/AI_Declaration-aidecl.yaml-blue)](https://github.com/haddocking/disvis/blob/main/aidecl.yaml)

Documentation: <https://www.bonvinlab.org/disvis>

<!-- --8<-- [start:about] -->
## About DisVis

**DisVis** is a Python package and command line tool to visualize and quantify 
the accessible interaction space of distance restrained binary biomolecular complexes.
It performs a full and systematic 6 dimensional search of the three translational
and rotational degrees of freedom to determine the number of complexes consistent
with the restraints. In addition, it outputs the percentage of restraints being violated
and a density that represents the center-of-mass position of the scanning chain corresponding 
to the highest number of consistent restraints at every position in space.


## Requirements

* Python 3.11+
* A C compiler (to build the extension modules)

The Python packages DisVis needs are listed in `pyproject.toml` and installed
automatically by *pip*. GPU acceleration also needs an OpenCL 1.2+ enabled GPU
and the optional `gpu` packages (see INSTALLATION.md).

<!-- --8<-- [end:about] -->

## Installation
<!-- --8<-- [start:installation] -->

**DisVis** is best installed in a virtual environment. Open a terminal and type

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install disvis
```

After installation, the command line tool *disvis* should be at your disposal.
To install from the source instead, clone the repository and run `pip install .`
in it.

For GPU acceleration, install the optional GPU requirements with

```bash
pip install 'disvis[gpu]'
```

See INSTALLATION.md for the requirements and for checking the GPU set-up.


### Linux

Make sure git, the Python header files and a C compiler are available, e.g. on
Debian/Ubuntu

```bash
sudo apt-get install git python3-dev python3-venv build-essential
```

If you are working on a Fedora system, use *dnf* and the *python3-devel* and *gcc* packages.
Follow the general instructions above to install **DisVis**.


### MacOSX

Install the Xcode command line tools, which provide git and a C compiler

```bash
xcode-select --install
```

and a recent Python 3, e.g. with [*brew*](https://brew.sh)

```bash
brew install python
```

Follow the general instructions above to install **DisVis**.


### Windows

Install [*git*](https://git-scm.com/download) for Windows, a Python 3
distribution, such as [Anaconda](https://www.anaconda.com/download), and the
[Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
Follow the general instructions above to install **DisVis**.

<!-- --8<-- [end:installation] -->

<!-- --8<-- [start:usage] -->
## Usage

The general pattern to invoke *disvis* is

```bash
disvis <pdb1> <pdb2> <distance-restraints-file>
```

where `<pdb1>` is the fixed chain, `<pdb2>` is the scanning chain and 
`<distance-restraints-file>` is a text-file
containing the distance restraints in the following format

```text
<chainid 1> <resid 1> <atomname 1> <chainid 2> <resid 2> <atomname 2> <mindis> <maxdis>
```

As an example
    
```text
A 18 CA F 27 CB 10.0 20.0
```

This puts a distance restraint between the CA-atom of residue 18 of 
chain A of pdb1 and the CB-atom of residue 27 of chain F of pdb2 that 
should be longer than or equal to 10A and smaller than or equal to 20A.
Comments can be added by starting the line with the pound sign (#) and empty
lines are ignored.

#### HADDOCK restraints

DisVis also reads restraint files in the HADDOCK (CNS) format, recognised by
their `assign` statements, including ambiguous restraints:

```text
! active residue 18 of chain A, passive residues 27 and 30 of chain F
assign (segid A and resid 18) ((segid F and resid 27) or (segid F and resid 30)) 3.9 3.9 0.0
assign (segid A and resid 22 and name CA) (segid F and resid 40:42 and name CA) 15.0 5.0 5.0
  or (segid A and resid 24 and name CA) (segid F and resid 40:42 and name CA)
```

Each restraint `assign (sel1) (sel2) d dminus dplus` should be between
`d - dminus` and `d + dplus`; `or (sel3) (sel4)` lines add further selection
pairs to the restraint, and `!` starts a comment. Selections can use `segid`
(matched against the chain identifier), `resid` (a number or a range such as
`40:42`), `name` and `resname` (`*` is a wildcard), `all`, `not`, `and`, `or`
and parentheses; as in CNS, keywords can be shortened to 4 letters (e.g.
`resi`, `assi`). The first selection of each pair is taken in the fixed
chain and the second in the scanning chain; when one of the selections of
a restraint has no atoms there, the whole restraint is read the other way
round, so restraints can be given in both directions, as in the AIR files
of HADDOCK. A restraint whose pairs are not all found in the same
direction is an error. DisVis reports in the log how many restraints were
read the other way round. Different chain identifiers for the two chains
are recommended: when they share one, `segid` cannot tell them apart, and a
residue missing from one of them can make DisVis read a restraint in the
wrong direction (DisVis then writes a warning to the log).

A restraint is consistent when **any** atom of its first selection and atom
of its second selection lie within the distance range. This differs from
HADDOCK, which combines all atom pairs into an effective distance,
(Σ r<sup>-6</sup>)<sup>-1/6</sup>, shorter than the shortest atom-atom
distance. The upper bounds used in HADDOCK, such as 2 Å for ambiguous
interaction restraints, are therefore too short for DisVis: use instead an
upper bound that two atoms in contact can reach, e.g. 3.9 Å, a
carbon-carbon contact (see the `E2A-HPR` test case). The search time
grows with the number of atom pairs of the restraints; selecting fewer atoms
(e.g. `name CA`) keeps it short.


### Options

To get a help screen with available options and explanation type
            
```bash
disvis --help
```

Some examples to get you started. To perform a 5.27 degree rotational search and store the results
in the directory *results/*

```bash
disvis 1wcm_A.pdb 1wcm_E.pdb restraints.dat -a 5.27 -d results
```

Note that the directory is created if it doesn't exist.

To perform a 9.72 degree rotational search with 16 processors and a voxel spacing of 2A

```bash
disvis O14250.pdb Q9UT97.pdb restraints.dat -a 9.72 -p 16 -vs 2
```

To offload computations to the GPU (if the requirements are met) and increase
the maximum allowed volume of clashes as well as the minimum required volume of
interaction, and set the interaction radius to 2A

```bash
disvis 1wcm_A.pdb 1wcm_E.pdb restraints.dat -g -cv 6000 -iv 7000 -ir 2
```

To perform a grid occupancy analysis for complexes consistent with at least N
restraints, include the `-oa` or `--occupancy-analysis` flag to the command. By
default, the analysis is only performed on complexes consistent with
either *all* and *all - 1* restraints. If you want to perform an occupancy
analysis for complexes consistent with a lower number of restraints, use the
`-ic` or `--interaction-restraints-cutoff` option, followed by the minimum
number of required consistent restraints. Thus the command

```bash
disvis 1wcm_A.pdb 1wcm_E.pdb restraints.dat -oa -ic 5
```

will make an occupancy grid for complexes consistent with at least 5
restraints, and higher.

Finally, to perform an interaction analysis to determine which residues
are most likely at the interface, the `-is` or `--interaction-selection`
options can be used with an additional file giving the residue numbers of the
receptor and ligand for which the interaction analysis will be performed. The
input file consists of two lines, where the two lines are a space separated
sequence of residue numbers for the receptor and ligand, respectively. As the
interaction analysis is particularly expensive, only complexes consistent with
*all*, or *all - 1* are considered. To lower the barrier, the `-ic` option can
again be used. Also limit the selected residues to only the solvent accessible
ones. A simple example for the selection-file

```text
1 2 3 4
101 302 888
```

This will select residue numbers 1, 2, 3, and 4 for the receptor, and 101, 302,
and 888 for the ligand.

#### Writing complexes

With `-wc` or `--write-complexes`, *disvis* writes *complexes.out*, the
rotation and translation of every complex consistent with at least 75% of
the highest number of restraints found, rounded up, or with `-wc P` with at
least *P* percent of it (`-wc 100`: the complexes consistent with the most
restraints; `--max-complexes M` writes at most *M* of them, a random sample
when more are found):

```bash
disvis e2aP_1F3G.pdb hpr_1.pdb e2a-hpr_air.tbl -wc -d results
```

When the restraints include false positives, the correct complex can be
consistent with fewer restraints than the highest number found: in the
`antibody-antigen` test case, the crystal structure is consistent with 25 of
the 43 restraints, the search finds complexes consistent with 32, and `-wc`
writes the 4880 complexes consistent with at least 24 (75% of 32), the
closest at 2.2 Å CA RMSD from the crystal structure. On the five test cases
with a reference structure (coarse searches, with `--max-complexes 50000`
for RNA polymerase II and PRE5-PUP2, with over 100000 complexes), 75% kept
complexes within 2.2 Å of it, and clustering them (see below) gave cluster
centers within 0.8 to 2.9 Å of it.

*disvis-complexes* then writes these complexes, the fixed and scanning chain,
as PDB files (*complex_N.pdb*, in the directory *complexes* by default; `-n`
limits their number):

```bash
disvis-complexes results/complexes.out e2aP_1F3G.pdb hpr_1.pdb -d results/complexes
```

As there can be many complexes, `--cluster` groups similar ones, e.g. to
start a HADDOCK refinement from a set of different complexes, and writes the
center of each cluster (*cluster_N.pdb*, largest cluster first) and the
models of each cluster to *clusters.out*:

```bash
disvis-complexes results/complexes.out e2aP_1F3G.pdb hpr_1.pdb --cluster -d results/clusters
```

The complexes are compared by the RMSD of the scanning chain (its CA and P
atoms, or all its heavy atoms when it has none), computed from their
rotations and translations. As in the clustering of HADDOCK, the complex
with the most neighbors within the RMSD cutoff is the center of the first
cluster, with these neighbors, and so on, while clusters have at least
`--min-population` complexes (default 2); the other complexes are not
clustered. The cutoff (`--cluster-rmsd`) is by default chosen on a random
subset of the complexes so that no two complexes of a cluster are more than
`--cluster-target` apart (default 6 Å; on all complexes, denser than the
subset, clusters can be somewhat wider). As neighboring complexes differ by
at least about one voxel, the target must be larger than the voxel spacing.
On the test cases with many complexes (PRE5-PUP2 and antibody-antigen,
coarse searches), the defaults gave cluster centers within 1.4 Å and 2.2 Å
of the reference structures, and left few complexes close to them
unclustered.

The PDB files keep the chain identifiers of the input files, but not their
segment identifiers, and write HETATM records as ATOM.

*disvis* also writes an HTML report of the results, with a Mol* view of the
accessible interaction space (and of the occupancy grids with `-oa`); see
[Viewing the report](#viewing-the-report).


### Output

Standard *disvis* outputs 5 files:

* *accessible_complexes.out*: a text file containing the number of complexes
  consistent with a number of restraints. The rotations of a set do not
  sample the rotation space perfectly evenly, so each rotation has a weight
  (on average 1) for the part of the rotation space it covers, and each
  complex is counted with the weight of its rotation: the numbers are those
  of evenly sampled rotations, and need not be whole numbers (the file gives
  their integer part).
* *violations.out*: a text file showing how often a specific restraint is
  violated for each number of consistent restraints. This helps in identifying
  which restraint is most likely a false-positive if any.
* *accessible_interaction_space.mrc*: a density file in MRC format. The density
  represents the center of mass of the scanning chain conforming to the maximum
  found consistent restraints at every position in space. The density can be
  inspected by opening it together with the fixed chain in a molecular viewer
  (UCSF Chimera is recommended for its easier manipulation of density
  data, but also PyMol works).
* *z-score.out*: a text file giving the Z-score for each restraint. The higher
  the score, the more likely the restraint is a false-positive. If not all
  restraints are consistent, Z-scores above 1.0 are explicitly mentioned in
  the output.
* *disvis.log*: a log file showing all the parameters used, together with date
  and time indications. It starts with the DisVis version and ends with how to
  view the report.

If an occupancy and/or interaction analysis was requested, *disvis* also
outputs the following files:

* *occupancy_N.mrc*: a volume file giving a normalized indication of how
  frequent a grid point is occupied by the ligand, where *N* indicates the minimal
  required number of consistent restraints that resulted in the occupancy grid.
* *[receptor|ligand]_interactions.txt*: a text file containing the average
  number of interactions per complex each selected residue made, for both the
  receptor and ligand molecule. Each column denotes the minimal number of
  consistent restraints of each complex for which interactions were counted.

With `-wc`, *disvis* also writes:

* *complexes.out*: the complexes consistent with at least 75% (or *P*
  percent) of the highest number of restraints found. The header gives that number, the input
  files and the search parameters; each line gives a model number, the index
  of the rotation, the number of consistent restraints, the position (x, y,
  z, in Å) of the scanning chain center and the rotation matrix R (row by
  row). The atoms *a* of the scanning chain are then at
  R (*a* − center) + (x, y, z), with the center given in the header.
  It lists the complexes one by one, without the weights of their rotations,
  so their number can differ from that of *accessible_complexes.out*: for
  the E2A-HPR test case (coarse search), *complexes.out* lists 18 complexes
  consistent with all 20 restraints, whose rotation weights sum to 21.7, the
  21 complexes of *accessible_complexes.out*.

*disvis* also writes *report.html* and *state.mvsj* (the MolViewSpec state of
the Mol* view), and copies the input files into the output directory, so the
report can show and link them (`--no-report` turns this off).

#### Viewing the report

The report loads the files of the output directory, which browsers do not allow
for a page opened from disk. Serve the output directory with Python instead

```bash
python -m http.server -d <directory>
```

and open <http://localhost:8000/report.html> in a web browser. Stop the server
with `Ctrl-C`. If port 8000 is in use, choose another one, e.g.
`python -m http.server 8080 -d <directory>` and
<http://localhost:8080/report.html>.

To visualize the most contacted residues, write the average number of
interactions into the B-factor column of the receptor or ligand PDB file

```bash
python scripts/interactions_to_bfactor.py receptor.pdb receptor_interactions.txt N
```

where *N* is one of the column numbers of the interaction file. This writes
*receptor_interactions.pdb*, which can be colored on B-factor, e.g. with
`spectrum b, blue_red` in PyMOL or `color bfactor` in ChimeraX.

<!-- --8<-- [end:usage] -->

<!-- --8<-- [start:end] -->
Licensing
---------

If this software was useful to your research please cite us

**Van Zundert, G.C.P. and Bonvin, A.M.J.J.** (2015) DisVis: Visualizing and
quantifying accessible interaction space of distance restrained biomolecular complexes.
*Bioinformatics 31*, 3222-3224.

The software itself can be cited with its Zenodo DOI,
[10.5281/zenodo.7775093](https://doi.org/10.5281/zenodo.7775093), and
[`CITATION.cff`](https://github.com/haddocking/disvis/blob/main/CITATION.cff)
gives both references.

Apache License Version 2.0

The elements.py module is licenced under the MIT License, Copyright Christoph
Gohlke.


## Tested platforms

| Operating System| CPU single | CPU multi | GPU |
| --------------- | ---------- | --------- | --- |
|Linux            | Yes        | Yes       | Yes |
|MacOSX           | Yes        | Yes       | Yes |
|Windows          | Yes        | Yes       | No  |

The GPU version has been tested on:
NVIDIA GeForce GTX 1080 Ti and RTX 6000 Ada for Linux, and Apple M2 Pro for
macOS (and, with the earlier Python 2 version, NVIDIA GeForce GTX 680 and AMD
Radeon HD 7730M for Linux)

Reference results and timings for the test cases, on CPUs and GPUs, are in
`test-cases/README.md`.


## Contributing

See `CONTRIBUTING.md` for the development setup, code style, tests and pull
requests.


## AI Declaration

Large language models (LLM) were used in this repository since
[v3.0.0](https://github.com/haddocking/disvis/blob/main/CHANGELOG.md), for the
port to Python 3 and the later changes. The changes were always reviewed by a
human before being merged. See
[aidecl.yaml](https://github.com/haddocking/disvis/blob/main/aidecl.yaml) for
machine-readable details of the LLM usage.
<!-- --8<-- [end:end] -->
