# DisVis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1037197.svg)](https://doi.org/10.5281/zenodo.1037197)
[![ci](https://github.com/haddocking/disvis/actions/workflows/ci.yml/badge.svg)](https://github.com/haddocking/disvis/actions/workflows/ci.yml)
[![AI Declaration](https://img.shields.io/badge/AI_Declaration-aidecl.yaml-blue)](aidecl.yaml)

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


## Installation

**DisVis** is best installed in a virtual environment. Open a terminal and type

```bash
git clone https://github.com/haddocking/disvis.git
cd disvis
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

After installation, the command line tool *disvis* should be at your disposal.

For GPU acceleration, install the optional GPU requirements with

```bash
pip install '.[gpu]'
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

Note that the directory is created if it doesn't exist. The progress of the
search is printed when the output goes to a terminal; `-np` (`--no-progress`)
turns it off.

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


### Output

Standard *disvis* outputs 5 files:

* *accessible_complexes.out*: a text file containing the number of complexes
  consistent with a number of restraints. 
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
  the score, the more likely the restraint is a false-positive. Z-scores above
  1.0 are explicitly mentioned in the output.
* *disvis.log*: a log file showing all the parameters used, together with date
  and time indications.

If an occupancy and/or interaction analysis was requested, *disvis* also
outputs the following files:

* *occupancy_N.mrc*: a volume file giving a normalized indication of how
  frequent a grid point is occupied by the ligand, where *N* indicates the minimal
  required number of consistent restraints that resulted in the occupancy grid.
* *[receptor|ligand]_interactions.txt*: a text file containing the average
  number of interactions per complex each selected residue made, for both the
  receptor and ligand molecule. Each column denotes the minimal number of
  consistent restraints of each complex for which interactions were counted.

To visualize the most contacted residues, write the average number of
interactions into the B-factor column of the receptor or ligand PDB file

```bash
python scripts/interactions_to_bfactor.py receptor.pdb receptor_interactions.txt N
```

where *N* is one of the column numbers of the interaction file. This writes
*receptor_interactions.pdb*, which can be colored on B-factor, e.g. with
`spectrum b, blue_red` in PyMOL or `color bfactor` in ChimeraX.


Licensing
---------

If this software was useful to your research please cite us

**Van Zundert, G.C.P. and Bonvin, A.M.J.J.** (2015) DisVis: Visualizing and
quantifying accessible interaction space of distance restrained biomolecular complexes.
*Bioinformatics 31*, 3222-3224.

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
[v3.0.0](CHANGELOG.md), for the port to Python 3 and the later changes. The
changes were always reviewed by a human before being merged. See
[aidecl.yaml](aidecl.yaml) for machine-readable details of the LLM usage.
